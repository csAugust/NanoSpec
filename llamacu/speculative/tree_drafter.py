import time
import numpy as np
from .. import C
from ..llama import LLM

import torch

def pack_mask(mask_2d):
    '''
    for static masks, pack them into a uint64 per row
    '''
    mask_2d_packed = torch.zeros((mask_2d.shape[0], 2), dtype=torch.uint32, device="cuda")
    for i in range(mask_2d.shape[0]):
        mask_1 = 0
        mask_2 = 0
        for j in range(i + 1):
            if j < 32:
                mask_1 |= (mask_2d[i][j].item() << j)
            else:
                mask_2 |= (mask_2d[i][j].item() << (j - 32))
        mask_2d_packed[i][0] = mask_1
        mask_2d_packed[i][1] = mask_2
    mask_2d_packed = mask_2d_packed.view(torch.uint64).view(-1)
    return mask_2d_packed

class LLM_with_tree_drafter(LLM):
    def __init__(self,
                 drafter_type, drafter_path, base_path,
                 tree_size,
                 max_context_tokens,
                 **kwargs):
        super().__init__(base_path, **kwargs)

        self.drafter_type = drafter_type
        self.drafter_path = drafter_path
        self.base_path = base_path

        self.tree_size = tree_size
        self.tree_draft_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_gt_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_attn_mask = torch.empty((tree_size), dtype=torch.uint64, device="cuda")
        self.tree_parent = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")

        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

        self.max_context_tokens = max_context_tokens
        self.context_tokens_set = set()
        self.context_tokens_tensor = torch.empty((self.max_context_tokens), dtype=torch.int32, device="cuda")

    def set_token_id_remap(self, token_id_remap: torch.Tensor):
        self.token_id_remap = token_id_remap.clone()

    def load_from_hf(self):
        self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
        super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[], tokenizer=None, is_warmup=False, mode=0):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)

        MODE = mode
        DO_PROFILE_DRAFT = True
        context_length = 0
        if is_warmup:
            MODE = 0
        # print("Running with mode ", mode)
        if MODE != 0:
            # new_tokens_set = set(input_ids.view(-1).tolist())

            topk_indices = torch.topk(logits, k=3, dim=-1).indices
            combined_candidates_gpu = torch.cat([input_ids.view(-1), topk_indices.view(-1)])
            unique_candidates_gpu = torch.unique(combined_candidates_gpu)
            new_tokens_set = set(unique_candidates_gpu.tolist())
            # new_tokens_set = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

            # context_tokens = input_ids[0].tolist()
            # for i in range(prefix_length):
            #     logit_i = logits[i]
            #     topk_tokens = torch.topk(logit_i, k=3).indices
            #     context_tokens += topk_tokens.tolist()
            # context_tokens = sorted(list(set(context_tokens)))
            # new_tokens_set = set(context_tokens)
            new_tokens_tensor_cpu, new_context_length = self.update_context_new(new_tokens_set, MODE)
            if new_tokens_tensor_cpu is not None:
                if MODE >= 2:
                    torch.cuda.nvtx.range_push(f"prefetch")
                    C.trigger_async_prefetch(
                        self.context_tokens_tensor.data_ptr(),  # 目标 GPU buffer 地址
                        new_tokens_tensor_cpu.data_ptr(),       # 源 CPU 数据地址
                        len(new_tokens_tensor_cpu),             # 增量大小
                        context_length,                         # 偏移量 (当前长度)
                    )
                    torch.cuda.nvtx.range_pop()
                context_length = new_context_length

        self.tree_draft_ids[:1].copy_(logits[prefix_length-1].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.tree_draft_ids[0])
        accept_lengths = []
        i = 0
        model_step = 0
        terminal = False

        step_draft_times = []
        self.hit_count = 0
        self.total_count = 0

        COPY = False and not is_warmup
        def _load(name_, dtype_=np.int32, device_=input_ids.device):
            x = np.loadtxt(f'{name_}.txt', dtype=dtype_)
            return torch.from_numpy(x).to(device_)
        if COPY:
            copy_time_stat = []
            start_time = time.time()
            step_draft_tokens_record_tensor = _load('step_draft_tokens')
            step_tree_position_ids_record_tensor = _load('step_tree_position_ids')
            step_tree_attn_masks_record_tensor = _load('step_tree_attn_masks', np.uint64)
            step_tree_parents_record_tensor = _load('step_tree_parents')
            copy_time_stat.append(time.time() - start_time)
            
        SAVE = False and not is_warmup
        step_draft_tokens = []
        step_tree_position_ids = []
        step_tree_attn_masks = []
        step_tree_parents = []

        if DO_PROFILE_DRAFT:
            torch.cuda.synchronize()
            decoding_start_time = time.time()
        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i

            # if DO_PROFILE_DRAFT:
            #     torch.cuda.synchronize()
            # start_time = time.time()
            torch.cuda.nvtx.range_push(f"draft")
            C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                    self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr(),
                    self.context_tokens_tensor.data_ptr(), context_length, MODE)
            torch.cuda.nvtx.range_pop()
            # if DO_PROFILE_DRAFT:
            #     torch.cuda.synchronize()
            #     total_time = time.time() - start_time
            #     step_draft_times.append(total_time)

            if SAVE:
                step_draft_tokens.append(self.tree_draft_ids.tolist())
                step_tree_position_ids.append(self.tree_position_ids.tolist())
                step_tree_attn_masks.append(self.tree_attn_mask.tolist())
                step_tree_parents.append(self.tree_parent.tolist())
            
            if COPY and model_step < step_draft_tokens_record_tensor.shape[0]:
                start_time = time.time()
                self.tree_draft_ids.copy_(step_draft_tokens_record_tensor[model_step])
                self.tree_position_ids.copy_(step_tree_position_ids_record_tensor[model_step])
                self.tree_attn_mask.copy_(step_tree_attn_masks_record_tensor[model_step])
                self.tree_parent.copy_(step_tree_parents_record_tensor[model_step])
                copy_time_stat.append(time.time() - start_time)

            logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
            self.tree_gt_ids.copy_(logits.argmax(dim=-1))

            torch.cuda.nvtx.range_push(f"verify")
            accept_length = C.verify_and_fix(
                self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
            )
            torch.cuda.nvtx.range_pop()

            model_step += 1
            accept_lengths.append(accept_length)
            for temin in teminators:
                if temin in self.tree_draft_ids[:accept_length]:
                    terminal = True
            append_length = min(accept_length, generation_length - 1 - i)

            # self.count_hit_context(MODE, self.tree_draft_ids[:append_length])
            if MODE != 0:
                # topk_indices = torch.topk(logits, k=3, dim=-1).indices
                # combined_candidates_gpu = torch.cat([self.tree_draft_ids.view(-1), topk_indices.view(-1)])
                # unique_candidates_gpu = torch.unique(combined_candidates_gpu)
                # new_tokens_set = set(unique_candidates_gpu.tolist())

                combined_ids_gpu = torch.cat([self.tree_draft_ids.view(-1), self.tree_gt_ids.view(-1)])
                unique_ids_gpu = torch.unique(combined_ids_gpu)
                new_tokens_set = set(unique_ids_gpu.tolist())

                # new_tokens_set = set(self.tree_draft_ids.view(-1).tolist() + self.tree_gt_ids.view(-1).tolist())
                new_tokens_tensor_cpu, new_context_length = self.update_context_new(new_tokens_set, MODE)
                if new_tokens_tensor_cpu is not None:
                    if MODE >= 2:
                        torch.cuda.nvtx.range_push(f"prefetch")
                        C.trigger_async_prefetch(
                            self.context_tokens_tensor.data_ptr(),  # 目标 GPU buffer 地址
                            new_tokens_tensor_cpu.data_ptr(),       # 源 CPU 数据地址
                            len(new_tokens_tensor_cpu),             # 增量大小
                            context_length % self.max_context_tokens,  # 偏移量 (当前长度)
                        )
                        torch.cuda.nvtx.range_pop()
                    context_length = new_context_length

            tokens[1+i:1+i+append_length].copy_(self.tree_draft_ids[:append_length])
            self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
            i += accept_length

        if DO_PROFILE_DRAFT:
            torch.cuda.synchronize()
            decoding_total_time = time.time() - decoding_start_time
            step_draft_times.append(decoding_total_time)
        else:
            step_draft_times.append(0)

        tokens = tokens[:1+i].tolist()
        
        if SAVE:
            def _save(tensor_, name_, dtype_=torch.int32):
                tensor_ = torch.tensor(tensor_, dtype=dtype_).cpu().numpy()
                # tensor_.tofile('step_draft_tokens.bin')
                np.savetxt(f'{name_}.txt', tensor_, fmt='%d')
            _save(step_draft_tokens, "step_draft_tokens")
            _save(step_tree_position_ids, "step_tree_position_ids")
            _save(step_tree_attn_masks, "step_tree_attn_masks", torch.uint64)
            _save(step_tree_parents, "step_tree_parents")
        if COPY:
            print("Copy overhead: ", sum(copy_time_stat))
        
        self.context_tokens_set.clear()
        self.context_tokens_tensor = torch.empty((self.max_context_tokens), dtype=torch.int32, device="cuda")
        
        # print(f"Hit {self.hit_count}, Total {self.total_count}, {self.hit_count / self.total_count:.2f}")
        return tokens, accept_lengths, model_step, step_draft_times
    
    def count_hit_context(self, mode=0, acc_tokens=None):
        if acc_tokens is not None:
            acc_draft_cnt = 0
            acc_draft_str = ''
            full_draft_cnt = 0
            draft_tokens = self.tree_draft_ids.tolist()
            if mode == 0:
                src = self.V
            else:
                src = self.context_tokens_set
            for token in draft_tokens:
                if token in src:
                    full_draft_cnt += 1
            for token in acc_tokens:
                if token.item() in src:
                    acc_draft_cnt += 1
                    acc_draft_str += 'Y'
                else:
                    acc_draft_str += 'N'

            full_occur_rate = full_draft_cnt / len(draft_tokens)
            acc_occur_rate = acc_draft_cnt / len(acc_tokens)
            self.hit_count += acc_draft_cnt
            self.total_count += len(acc_tokens)
            # print(f"full {full_occur_rate:.2f}, acc {acc_occur_rate:.2f}({acc_draft_str}) | acc length {acc_tokens.numel()} | {self.tokenizer.decode(acc_tokens, skip_special_tokens=False)}")

    def update_context_new(self, new_tokens_set, mode=0):
        old_context_length = len(self.context_tokens_set)
        real_new_tokens_set = new_tokens_set.difference(self.context_tokens_set)

        if len(real_new_tokens_set) == 0:
            return None, len(self.context_tokens_set)

        self.context_tokens_set.update(real_new_tokens_set)
        real_new_tokens_tensor = torch.tensor(list(real_new_tokens_set), dtype=torch.int32, device='cpu')

        # enable this if context tokens may exceed self.max_context_tokens
        # self.context_tokens_set = set(list(self.context_tokens_set)[-self.max_context_tokens:])

        new_context_length = len(self.context_tokens_set)
        if mode == 1:
            self.context_tokens_tensor[old_context_length:new_context_length].copy_(real_new_tokens_tensor)
        # print(f"New tokens ({old_context_length} -> {len(self.context_tokens_set)}): {real_new_tokens_set}")
        return real_new_tokens_tensor, new_context_length
    