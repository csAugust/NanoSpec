from collections import deque
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

        self.context_tokens_fifo = deque(maxlen=self.max_context_tokens)
        self.context_tokens_lookup = set()
        self.current_context_offset = 0
        self.new_tokens_cpu_buffer = torch.empty(
            self.max_context_tokens, dtype=torch.int32, device='cpu'
        ).pin_memory()

    def set_token_id_remap(self, token_id_remap: torch.Tensor):
        self.token_id_remap = token_id_remap.clone()

    def load_from_hf(self):
        self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
        super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[], tokenizer=None, is_warmup=False):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)

        MODE = 2
        context_length = 0
        self.master_context_tensor = torch.empty(0, dtype=torch.int32, device="cuda")
        if is_warmup:
            MODE = 0

        if MODE != 0:
            topk_indices = torch.topk(logits, k=3, dim=-1).indices
            combined_candidates_gpu = torch.cat([input_ids.view(-1), topk_indices.view(-1)])
            combined_candidates_gpu = input_ids.view(-1)
            unique_candidates_gpu = torch.unique(combined_candidates_gpu, sorted=True)
            # new_tokens_set = set(unique_candidates_gpu.tolist())

            # context_tokens = input_ids[0].tolist()
            # for i in range(prefix_length):
            #     logit_i = logits[i]
            #     topk_tokens = torch.topk(logit_i, k=3).indices
            #     context_tokens += topk_tokens.tolist()
            # context_tokens = sorted(list(set(context_tokens)))
            # new_tokens_set = set(context_tokens)

            potential_new_tokens_list = unique_candidates_gpu.tolist()
            # potential_new_tokens_list = list(set(context_tokens))
            new_tokens_tensor_cpu, num_added, write_offset = self.update_context_fifo_with_eviction(potential_new_tokens_list)
            if num_added > 0:
                if MODE >= 2:
                    torch.cuda.nvtx.range_push(f"prefetch")
                    C.trigger_async_prefetch(
                        self.context_tokens_tensor.data_ptr(),  # 目标 GPU buffer 地址
                        new_tokens_tensor_cpu.data_ptr(),       # 源 CPU 数据地址 (已经是 pin_memory 的视图)
                        num_added,                              # 实际新增大小
                        write_offset,                           # 写入的环形偏移量
                    )
                    torch.cuda.nvtx.range_pop()
                # context_length 的更新也要变，用于传递给 draft 函数
                # 现在的 context_length 表示的是“当前 buffer 有多少有效数据”
                # context_length = len(self.context_tokens_fifo)

            # new_tokens_tensor_cpu, new_context_length = self.update_context_new(new_tokens_set)
            # if new_tokens_tensor_cpu is not None:
            #     if MODE >= 2:
            #         torch.cuda.nvtx.range_push(f"prefetch")
            #         C.trigger_async_prefetch(
            #             self.context_tokens_tensor.data_ptr(),  # 目标 GPU buffer 地址
            #             new_tokens_tensor_cpu.data_ptr(),       # 源 CPU 数据地址
            #             len(new_tokens_tensor_cpu),             # 增量大小
            #             context_length,                         # 偏移量 (当前长度)
            #         )
            #         torch.cuda.nvtx.range_pop()
            #     context_length = new_context_length

            # use full vocab
            # assert(hasattr(self, 'V'))
            # self.context_tokens_tensor = torch.range(0, self.V - 1, dtype=torch.int32, device="cuda")

            # use high-freq vocab
            # assert(hasattr(self, 'token_id_remap'))
            # self.context_tokens_tensor = self.token_id_remap

            # print(f'Limited max vocab context: {self.max_context_tokens}')
            # print(f'Limited vocab context: {context_length}')

        self.tree_draft_ids[:1].copy_(logits[prefix_length-1].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.tree_draft_ids[0])
        accept_lengths = []
        step_draft_times = []
        i = 0
        model_step = 0
        terminal = False

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
            
        SAVE = False
        step_draft_tokens = []
        step_tree_position_ids = []
        step_tree_attn_masks = []
        step_tree_parents = []

        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i

            start_time = time.time()
            torch.cuda.nvtx.range_push(f"draft")
            C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                    self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr(),
                    self.context_tokens_tensor.data_ptr(), len(self.context_tokens_fifo), MODE)
            torch.cuda.nvtx.range_pop()
            total_time = time.time() - start_time
            step_draft_times.append(total_time)

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

            if MODE != 0:
                combined_ids_gpu = torch.cat([self.tree_draft_ids.view(-1), self.tree_gt_ids.view(-1)])
                unique_ids_gpu = torch.unique(combined_ids_gpu)

                potential_new_tokens_list = unique_ids_gpu.tolist()
                new_tokens_tensor_cpu, num_added, write_offset = self.update_context_fifo_with_eviction(potential_new_tokens_list)
                if num_added > 0:
                    if MODE >= 2:
                        torch.cuda.nvtx.range_push(f"prefetch")
                        C.trigger_async_prefetch(
                            self.context_tokens_tensor.data_ptr(),
                            new_tokens_tensor_cpu.data_ptr(),
                            num_added,
                            write_offset,  # 直接使用 Python 端计算好的偏移量
                        )
                        torch.cuda.nvtx.range_pop()

                # new_tokens_set = set(unique_ids_gpu.tolist())

                # # new_tokens_set = set(self.tree_draft_ids.view(-1).tolist() + self.tree_gt_ids.view(-1).tolist())
                # new_tokens_tensor_cpu, new_context_length = self.update_context_new(new_tokens_set)
                # if new_tokens_tensor_cpu is not None:
                #     if MODE >= 2:
                #         torch.cuda.nvtx.range_push(f"prefetch")
                #         C.trigger_async_prefetch(
                #             self.context_tokens_tensor.data_ptr(),  # 目标 GPU buffer 地址
                #             new_tokens_tensor_cpu.data_ptr(),       # 源 CPU 数据地址
                #             len(new_tokens_tensor_cpu),             # 增量大小
                #             context_length % self.max_context_tokens,  # 偏移量 (当前长度)
                #         )
                #         torch.cuda.nvtx.range_pop()
                #     context_length = new_context_length
                #     # if context_length > self.max_context_tokens:
                #     #     print(f"{context_length} exceeds {self.max_context_tokens}")

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
        
        self.context_tokens_set.clear()
        self.context_tokens_tensor = torch.empty((self.max_context_tokens), dtype=torch.int32, device="cuda")
        return tokens, accept_lengths, model_step, step_draft_times
    
    def update_context_new(self, new_tokens_set):
        old_context_length = len(self.context_tokens_set)
        real_new_tokens_set = new_tokens_set.difference(self.context_tokens_set)

        if len(real_new_tokens_set) == 0:
            return None, len(self.context_tokens_set)

        self.context_tokens_set.update(real_new_tokens_set)
        real_new_tokens_tensor = torch.tensor(list(real_new_tokens_set), dtype=torch.int32, device='cpu')
        
        # print(f"New tokens ({old_context_length} -> {len(self.context_tokens_set)}): {real_new_tokens_set}")
        return real_new_tokens_tensor, len(self.context_tokens_set)
    
    def update_context_fifo_with_eviction(self, potential_new_tokens_list):
        """
        处理新的候选 token，执行 FIFO 驱逐，并准备用于传输的 CPU Tensor。
        """
        # 1. 过滤掉已经存在的 token
        truly_new_tokens = []
        for token in potential_new_tokens_list:
            if token not in self.context_tokens_lookup:
                truly_new_tokens.append(token)
        
        num_new = len(truly_new_tokens)
        if num_new == 0:
            return None, 0, self.current_context_offset

        current_count = len(self.context_tokens_fifo)
        
        # 2. 计算需要驱逐的数量和实际能添加的数量
        num_to_evict = max(0, current_count + num_new - self.max_context_tokens)
        # 注意：如果 num_new 这个批次本身就比 max_context_tokens 还要大，
        # 我们只取最后 max_context_tokens 个，前面的直接被“驱逐”。
        if num_new > self.max_context_tokens:
             truly_new_tokens = truly_new_tokens[-self.max_context_tokens:]
             num_new = self.max_context_tokens
             num_to_evict = current_count # 现有的全部驱逐
        
        num_actually_added = num_new # 本批次实际添加的数

        # 3. 执行驱逐 (从 FIFO 最旧端移出)
        for _ in range(num_to_evict):
            oldest_token = self.context_tokens_fifo.popleft()
            self.context_tokens_lookup.remove(oldest_token)
            
        # 4. 添加新 token (加到 FIFO 最新端)
        for token in truly_new_tokens:
            self.context_tokens_fifo.append(token)
            self.context_tokens_lookup.add(token)

        # 5. 准备传输用的 CPU Tensor，使用预分配的锁页内存
        # 将数据复制到锁页内存的前 num_actually_added 个位置
        # 使用 non_blocking=True 没有意义，因为源也是 CPU，但拷贝操作很快
        self.new_tokens_cpu_buffer[:num_actually_added].copy_(
            torch.tensor(truly_new_tokens, dtype=torch.int32)
        )
        
        # 创建一个视图指向这一部分数据
        padded_new_tokens_tensor = self.new_tokens_cpu_buffer[:num_actually_added]

        # 6. 计算新的环形偏移量 (用于给 C++ 指定写入起始位置)
        old_offset = self.current_context_offset
        self.current_context_offset = (self.current_context_offset + num_actually_added) % self.max_context_tokens
        
        # 返回：(准备好的 Tensor, 实际新增数量, 写入的起始偏移量)
        return padded_new_tokens_tensor, num_actually_added, old_offset
    