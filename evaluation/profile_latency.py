"""
Latency breakdown profiling for speculative decoding.

Measures per-step timing of:
  - Draft Model Backbone Forward
  - Draft Model LM Head Computation
  - Draft Tree Ops
  - Target Model Verification
  - Weight Gather Overhead (mode 2 only)

Usage:
  python evaluation/profile_latency.py \
    --model-path <base_model> --eagle-path <eagle_model> \
    --mode 0 --V 32768 --num-samples 20
"""

import argparse
import time
import torch
import json
from pathlib import Path
from fastchat.utils import str_to_torch_dtype
from transformers import AutoTokenizer, AutoConfig
from llamacu import C


def load_questions(question_file, num_samples, chat_template="llama-3"):
    """Load spec_bench questions and format as input_ids."""
    questions = []
    with open(question_file) as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj)
            if len(questions) >= num_samples:
                break
    return questions


def profile_eagle(args):
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if "llama-3" in args.chat_template:
        teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        teminators = [tokenizer.eos_token_id]

    # Build model
    if args.drafter == "eagle":
        from llamacu.speculative.eagle import LLM_with_eagle
        FAKE_V = args.V if args.V != -1 else config.vocab_size
        model = LLM_with_eagle(
            base_path=args.model_path,
            eagle_path=args.eagle_path,
            memory_limit=args.memory_limit,
            V=FAKE_V,
            dtype=str_to_torch_dtype(args.dtype),
            cuda_graph=False,  # disable cuda graph for profiling
            num_iter=args.eagle_num_iter,
            tree_size=args.eagle_tree_size,
            max_context_tokens=3072,
        )
        model.init_storage()
        if args.V != -1:
            freq_dir = args.freq_path or args.eagle_path
            with open(f'{freq_dir}/freq_{args.V}.pt', 'rb') as f:
                token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
                token_id_remap = token_id_remap[:FAKE_V]
            print(f'Loaded token_id_remap from {freq_dir}/freq_{args.V}.pt')
        else:
            token_id_remap = torch.arange(config.vocab_size, dtype=torch.int32, device="cpu")
        model._load("token_id_remap", token_id_remap, cls="eagle")
        model.load_from_hf()

    elif args.drafter == "eagle3":
        from llamacu.speculative.eagle3 import LLM_with_eagle3
        V = args.V if args.V != -1 else None
        model = LLM_with_eagle3(
            base_path=args.model_path,
            eagle3_path=args.eagle_path,
            memory_limit=args.memory_limit,
            dtype=str_to_torch_dtype(args.dtype),
            cuda_graph=False,
            num_iter=args.eagle_num_iter,
            tree_size=args.eagle_tree_size,
            V=V,
            max_context_tokens=3072 if args.mode > 0 else 0,
        )
        model.init_storage()
        if args.V != -1:
            freq_dir = args.freq_path or args.eagle_path
            with open(f'{freq_dir}/freq_{args.V}.pt', 'rb') as f:
                token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
                token_id_remap = token_id_remap[:args.V]
            print(f'Loaded token_id_remap from {freq_dir}/freq_{args.V}.pt')
            model.token_id_remap_cpu = token_id_remap
            model._load("token_id_remap", token_id_remap, cls="eagle3")
        model.load_from_hf()

    # Load questions
    questions = load_questions(args.question_file, args.num_samples + args.num_warmup, args.chat_template)

    # Enable C++ profiling
    C.enable_draft_profiling(True)

    # Warmup
    print(f"Warming up with {args.num_warmup} samples...")
    for i in range(min(args.num_warmup, len(questions))):
        q = questions[i]
        conv_text = q.get("turns", [q.get("question", "Hello")])[0] if isinstance(q.get("turns"), list) else q.get("question", "Hello")
        inputs = tokenizer(conv_text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids.int()
        model.generate(input_ids=input_ids, generation_length=128, teminators=teminators, is_warmup=True, mode=0)

    # Reset profiling after warmup
    C.reset_draft_profiling()

    # Profile
    print(f"Profiling {args.num_samples} samples with mode={args.mode}...")
    target_verify_times = []
    total_step_count = 0

    for i in range(args.num_warmup, min(args.num_warmup + args.num_samples, len(questions))):
        q = questions[i]
        conv_text = q.get("turns", [q.get("question", "Hello")])[0] if isinstance(q.get("turns"), list) else q.get("question", "Hello")
        inputs = tokenizer(conv_text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids.int()

        # Run generation and measure target verify time at Python level
        torch.cuda.synchronize()
        tokens, accept_lengths, model_step, draft_time_list = model.generate(
            input_ids=input_ids,
            generation_length=args.max_new_tokens,
            teminators=teminators,
            is_warmup=False,
            mode=args.mode,
        )
        torch.cuda.synchronize()

        total_step_count += model_step
        # draft_time_list[-1] contains total decoding time
        if len(draft_time_list) > 0:
            target_verify_times.append(draft_time_list[-1])

    # Get C++ internal timing
    timing = C.get_draft_timing()
    backbone_ms = timing[0]
    lmhead_ms = timing[1]
    tree_ms = timing[2]
    gather_wait_ms = timing[3]
    num_calls = int(timing[4])

    # Compute averages
    avg_backbone = backbone_ms / num_calls if num_calls > 0 else 0
    avg_lmhead = lmhead_ms / num_calls if num_calls > 0 else 0
    avg_tree = tree_ms / num_calls if num_calls > 0 else 0
    avg_gather = gather_wait_ms / num_calls if num_calls > 0 else 0
    avg_draft_total = avg_backbone + avg_lmhead + avg_tree + avg_gather

    total_decoding_time = sum(target_verify_times) * 1000  # seconds → ms
    avg_total_per_step = total_decoding_time / total_step_count if total_step_count > 0 else 0
    avg_verify = avg_total_per_step - avg_draft_total

    # Output results
    mode_name = {0: "FR-Spec (mode 0)", 1: "Indexed GEMM (mode 1)", 2: "Prefetch/Ours (mode 2)"}
    drafter_name = args.drafter.upper()
    v_str = f"V={args.V}" if args.V != -1 else "Full Vocab"

    print(f"\n{'='*60}")
    print(f"  Latency Breakdown: {drafter_name} {mode_name.get(args.mode, f'mode {args.mode}')} ({v_str})")
    print(f"  {num_calls} draft calls over {total_step_count} decoding steps")
    print(f"{'='*60}")
    print(f"  Draft Backbone Forward:    {avg_backbone:7.3f} ms  ({backbone_ms:10.1f} ms total)")
    print(f"  Draft LM Head:             {avg_lmhead:7.3f} ms  ({lmhead_ms:10.1f} ms total)")
    print(f"  Draft Tree Ops:            {avg_tree:7.3f} ms  ({tree_ms:10.1f} ms total)")
    if args.mode == 2:
        print(f"  Weight Gather Wait:        {avg_gather:7.3f} ms  ({gather_wait_ms:10.1f} ms total)")
    print(f"  ---")
    print(f"  Draft Total:               {avg_draft_total:7.3f} ms")
    print(f"  Target Model Verify:       {avg_verify:7.3f} ms  (estimated)")
    print(f"  Total per step:            {avg_total_per_step:7.3f} ms")
    print(f"{'='*60}")

    # CSV output for plotting
    csv_path = args.output_csv
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("drafter,mode,V,component,avg_ms,total_ms\n")
            f.write(f"{args.drafter},{args.mode},{args.V},backbone,{avg_backbone:.4f},{backbone_ms:.1f}\n")
            f.write(f"{args.drafter},{args.mode},{args.V},lm_head,{avg_lmhead:.4f},{lmhead_ms:.1f}\n")
            f.write(f"{args.drafter},{args.mode},{args.V},tree_ops,{avg_tree:.4f},{tree_ms:.1f}\n")
            f.write(f"{args.drafter},{args.mode},{args.V},gather_wait,{avg_gather:.4f},{gather_wait_ms:.1f}\n")
            f.write(f"{args.drafter},{args.mode},{args.V},verify,{avg_verify:.4f},0\n")
            f.write(f"{args.drafter},{args.mode},{args.V},total,{avg_total_per_step:.4f},{total_decoding_time:.1f}\n")
        print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eagle-path", type=str, required=True)
    parser.add_argument("--drafter", type=str, default="eagle", choices=["eagle", "eagle3"])
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--V", type=int, default=-1)
    parser.add_argument("--freq-path", type=str, default=None)
    parser.add_argument("--memory-limit", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--chat-template", type=str, default="llama-3")
    parser.add_argument("--eagle-num-iter", type=int, default=6)
    parser.add_argument("--eagle-tree-size", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--question-file", type=str, default="data/spec_bench/question.jsonl")
    parser.add_argument("--output-csv", type=str, default=None)

    args = parser.parse_args()
    profile_eagle(args)
