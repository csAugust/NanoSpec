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

    def load_from_hf(self):
        self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
        super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[], do_copy=False, tokenizer=None):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)

        prefill_topk_tokens = []
        for i in range(prefix_length):
            logit_i = logits[i]
            topk_tokens = torch.topk(logit_i, k=3).indices
            prefill_topk_tokens += topk_tokens.tolist()
        context_tokens = prefill_topk_tokens + input_ids[0].tolist()
        context_tokens = sorted(list(set(context_tokens)))
        context_tokens_tensor = torch.tensor(context_tokens, dtype=torch.int32, device="cuda")
        context_tokens_set = set(context_tokens)
        print(len(context_tokens), ':', context_tokens_tensor)

        self.tree_draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.tree_draft_ids[0])
        accept_lengths = []
        i = 0
        model_step = 0
        terminal = False

        COPY = False and do_copy
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
            
        SAVE = False
        step_draft_tokens = []
        step_tree_position_ids = []
        step_tree_attn_masks = []
        step_tree_parents = []

        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i

            torch.cuda.nvtx.range_push(f"draft")
            C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr(), context_tokens_tensor.data_ptr(), context_tokens_tensor.numel())
            torch.cuda.nvtx.range_pop()

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

            print(f'Vocab: {len(context_tokens_set)}')
            acc_draft_cnt = 0
            acc_draft_str = ''
            full_draft_cnt = 0
            draft_tokens = self.tree_draft_ids.tolist()
            for token in draft_tokens:
                # if token in context_tokens_tensor:
                if token in context_tokens_set:
                    full_draft_cnt += 1
            for token in self.tree_draft_ids[:append_length]:
                # if token in context_tokens_tensor:
                if token.item() in context_tokens_set:
                    acc_draft_cnt += 1
                    acc_draft_str += 'Y'
                else:
                    acc_draft_str += 'N'

            full_occur_rate = full_draft_cnt / len(draft_tokens)
            acc_occur_rate = acc_draft_cnt / len(self.tree_draft_ids[:append_length])
            print(f"Step{model_step}: full {full_occur_rate:.2f}, acc {acc_occur_rate:.2f}({acc_draft_str}) | acc length {accept_length} | {tokenizer.decode(self.tree_draft_ids[:append_length], skip_special_tokens=False)}")
            
            context_tokens_tensor = torch.concat((context_tokens_tensor, self.tree_draft_ids[:append_length]))
            context_tokens_set.update(self.tree_draft_ids.view(-1).tolist())
            context_tokens_set.update(self.tree_gt_ids.view(-1).tolist())

            tokens[1+i:1+i+append_length].copy_(self.tree_draft_ids[:append_length])
            self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
            i += accept_length

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
            
        return tokens, accept_lengths, model_step