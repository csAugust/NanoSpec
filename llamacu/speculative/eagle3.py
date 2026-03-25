from .. import C
from .tree_drafter import LLM_with_tree_drafter

import torch
from transformers import PretrainedConfig


class Eagle3Config(PretrainedConfig):
    def __init__(
        self,
        num_hidden_layers=1,
        draft_vocab_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eagle3_num_layers = num_hidden_layers
        if draft_vocab_size is not None:
            self.draft_vocab_size = draft_vocab_size
        elif not hasattr(self, 'draft_vocab_size'):
            self.draft_vocab_size = kwargs.get('vocab_size', 32000)


class LLM_with_eagle3(LLM_with_tree_drafter):
    def __init__(self,
                 eagle3_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 max_context_tokens=3072,
                 **kwargs):
        super().__init__(
            "eagle3", eagle3_path, base_path,
            tree_size=tree_size,
            max_context_tokens=max_context_tokens,
            **kwargs
        )

        self.eagle3_path = eagle3_path
        self.eagle3_config = Eagle3Config.from_pretrained(eagle3_path)
        self.draft_vocab_size = self.eagle3_config.draft_vocab_size
        self.valid_target_set = None  # built when d2t is loaded

        C.init_eagle3_model(
            self.eagle3_config.eagle3_num_layers,
            self.eagle3_config.intermediate_size,
            self.eagle3_config.num_attention_heads,
            self.eagle3_config.num_key_value_heads,
            self.eagle3_config.hidden_size // self.eagle3_config.num_attention_heads,  # head_dim
            self.eagle3_config.rms_norm_eps,
            num_iter,
            topk_per_iter,
            self.tree_size,
            self.draft_vocab_size,
            self.dtype_int,
            max_context_tokens,
        )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype

            # Skip embedding (shared with base model)
            if 'embed_tokens' in name:
                return

            # d2t mapping: stored as int64 offsets, load as int32
            if name == 'd2t':
                # Build valid_target_set for Python-side context filtering
                d2t_cpu = param.to(torch.int64)
                indices = torch.arange(len(d2t_cpu), dtype=torch.int64)
                target_ids = indices + d2t_cpu
                self.valid_target_set = set(target_ids.tolist())
                print(f"  eagle3: built valid_target_set with {len(self.valid_target_set)} entries")

                param_i32 = param.to(torch.int32).contiguous()
                C.load_model(f"{cls}.d2t", param_i32.data_ptr())
                return

            # t2d: not needed for inference
            if name == 't2d':
                return

            param = param.contiguous().to(dtype)

            # Weight name mapping from EAGLE-3 checkpoint format to our CUDA format
            if name.startswith('midlayer.self_attn.'):
                mapped = name.replace('midlayer.self_attn.', f'{cls}.layers.0.attn.')
            elif name.startswith('midlayer.hidden_norm.'):
                mapped = name.replace('midlayer.hidden_norm.', f'{cls}.hidden_norm.')
            elif name.startswith('midlayer.input_layernorm.'):
                mapped = name.replace('midlayer.input_layernorm.', f'{cls}.input_layernorm.')
            elif name.startswith('midlayer.post_attention_layernorm.'):
                mapped = name.replace('midlayer.post_attention_layernorm.', f'{cls}.layers.0.post_attention_layernorm.')
            elif name.startswith('midlayer.mlp.'):
                mapped = name.replace('midlayer.mlp.', f'{cls}.layers.0.mlp.')
            elif name.startswith('fc.'):
                mapped = f'{cls}.fc.{name[3:]}'
            elif name.startswith('norm.'):
                mapped = f'{cls}.final_norm.{name[5:]}'
            elif name.startswith('lm_head.'):
                mapped = f'{cls}.lm_head.{name[8:]}'
            else:
                print(f"Warning: skipping unknown EAGLE-3 weight: {name}")
                return

            print(f"  eagle3 load: {name} -> {mapped}  shape={list(param.shape)}")
            C.load_model(mapped, param.data_ptr())
        else:
            super()._load(name, param, dtype)

    def update_context_new(self, new_tokens_set, mode=0):
        # Filter context tokens to only those in EAGLE-3's draft vocabulary
        if self.valid_target_set is not None:
            new_tokens_set = new_tokens_set.intersection(self.valid_target_set)
        return super().update_context_new(new_tokens_set, mode)
