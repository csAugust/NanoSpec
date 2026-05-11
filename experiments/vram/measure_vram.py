"""
Measure VRAM overhead of NanoSpec (mode 2) vs vanilla EAGLE.
Initializes model with given max_context_tokens, reports KV cache budget and memory usage.

Usage:
    python experiments/vram/measure_vram.py \
        --model-path <base_model> \
        --eagle-path <eagle_model> \
        --max-context-tokens 3072 \
        --memory-limit 0.50
"""
import argparse
import json
import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoConfig
from llamacu.speculative.eagle import LLM_with_eagle


def measure(args):
    config = AutoConfig.from_pretrained(args.model_path)

    V = config.vocab_size  # full vocab for fair comparison
    model = LLM_with_eagle(
        base_path=args.model_path,
        eagle_path=args.eagle_path,
        memory_limit=args.memory_limit,
        V=V,
        dtype=str_to_torch_dtype(args.dtype),
        cuda_graph=False,
        num_iter=args.eagle_num_iter,
        tree_size=args.eagle_tree_size,
        max_context_tokens=args.max_context_tokens,
    )

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    model.init_storage()
    budget = model.max_total_length

    # Load identity remap (no FR-Spec, full vocab)
    token_id_remap = torch.arange(V, dtype=torch.int32, device="cpu")
    model._load("token_id_remap", token_id_remap, cls="eagle")
    model.load_from_hf()

    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    # Analytical overhead computation
    hidden_size = config.hidden_size
    dtype_size = 2  # FP16
    max_ctx = args.max_context_tokens

    overhead_context_token_ids = max_ctx * 4                        # int32 per token
    overhead_repack_buffer = max_ctx * hidden_size * dtype_size     # T per element
    overhead_context_tensor = max_ctx * 4                           # Python-side int32
    overhead_total = overhead_context_token_ids + overhead_repack_buffer + overhead_context_tensor

    result = {
        "max_context_tokens": max_ctx,
        "kv_budget": budget,
        "memory_limit": args.memory_limit,
        "mem_before_init_MB": mem_before / 1024 / 1024,
        "mem_after_load_MB": mem_after / 1024 / 1024,
        "mem_peak_MB": mem_peak / 1024 / 1024,
        "hidden_size": hidden_size,
        "vocab_size": V,
        "dtype_size": dtype_size,
        "analytical_overhead_bytes": overhead_total,
        "analytical_overhead_MB": overhead_total / 1024 / 1024,
        "breakdown_context_token_ids_bytes": overhead_context_token_ids,
        "breakdown_repack_buffer_bytes": overhead_repack_buffer,
        "breakdown_context_tensor_bytes": overhead_context_tensor,
    }

    print(f"VRAM_RESULT={json.dumps(result)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eagle-path", type=str, required=True)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--memory-limit", type=float, default=0.50)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--eagle-num-iter", type=int, default=6)
    parser.add_argument("--eagle-tree-size", type=int, default=60)
    args = parser.parse_args()

    measure(args)
