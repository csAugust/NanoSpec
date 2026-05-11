"""
Ablation study inference for NanoSpec.
Runs three ablation modes (full, ctx_only, ext_only) and collects per-step profiling data.
Outputs a JSON report with coverage, accept length, and vocab evolution per question per mode.
Supports both spec_bench (mt_bench format) and human_eval benchmarks.
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from fastchat.utils import str_to_torch_dtype
from fastchat.llm_judge.common import load_questions
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from llamacu.speculative.eagle import LLM_with_eagle
from experiments.ablation.profiling_generate import generate_with_profiling


def _build_prompt_mt_bench(question, tokenizer):
    """Build prompt for spec_bench / mt_bench format (has 'turns')."""
    messages = [{"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."}]
    messages.append({"role": "user", "content": question["turns"][0]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, question.get("question_id", "unknown"), question.get("category", "unknown")


def _build_prompt_human_eval(question, tokenizer):
    """Build prompt for human_eval format (has 'prompt' and 'task_id')."""
    messages = [{"role": "system",
                 "content": "Please complete the following Python code without providing any additional tasks such as testing or explanations."}]
    messages.append({"role": "user", "content": question["prompt"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, question.get("task_id", "unknown"), "Code"


def run_ablation(model, tokenizer, questions, teminators, max_new_tokens, max_length,
                 ablation_modes, build_prompt_fn, output_path):
    """Run ablation across modes, collect per-step stats."""

    results = []

    # warmup (mode 0, no profiling)
    question = questions[0]
    for wm_i in range(3):
        prompt, _, _ = build_prompt_fn(question, tokenizer)
        inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids.int()
        prefill_length = len(input_ids[0])
        mnew = min(max_new_tokens, max_length - prefill_length)
        model.generate(input_ids=input_ids, generation_length=mnew,
                       teminators=teminators, tokenizer=tokenizer, is_warmup=True, mode=0)
        print(f"warmup {wm_i} done")
    print("Warmup done")

    for ablation_mode in ablation_modes:
        print(f"\n{'='*60}")
        print(f"Running ablation mode: {ablation_mode}")
        print(f"{'='*60}")

        mode_results = []
        for question in tqdm(questions, desc=ablation_mode):
            prompt, qid, category = build_prompt_fn(question, tokenizer)
            inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids.int()
            prefill_length = len(input_ids[0])
            mnew = min(max_new_tokens, max_length - prefill_length)

            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, accept_lengths, model_step, draft_time_list, step_stats = \
                generate_with_profiling(
                    model, input_ids, generation_length=mnew,
                    teminators=teminators, tokenizer=tokenizer,
                    ablation_mode=ablation_mode)
            torch.cuda.synchronize()
            total_time = time.time() - start_time

            new_token = len(output_ids)
            decoding_time = draft_time_list[-1] if draft_time_list else total_time
            gen_speed = new_token / total_time if total_time > 0 else 0
            dec_speed = new_token / decoding_time if decoding_time > 0 else 0
            avg_accept = np.mean(accept_lengths) if accept_lengths else 0

            # per-step vocab sizes and coverages
            vocab_sizes = [s["vocab_size"] for s in step_stats]
            coverages = [s["coverage"] for s in step_stats]
            step_accept_lens = [s["accept_length"] for s in step_stats]

            mode_results.append({
                "question_id": qid,
                "category": category,
                "ablation_mode": ablation_mode,
                "new_tokens": new_token,
                "model_steps": model_step,
                "total_time": total_time,
                "generate_speed": gen_speed,
                "decoding_speed": dec_speed,
                "avg_accept_length": float(avg_accept),
                "avg_coverage": float(np.mean(coverages)) if coverages else 0,
                "step_vocab_sizes": vocab_sizes,
                "step_coverages": coverages,
                "step_accept_lengths": step_accept_lens,
            })

        results.extend(mode_results)

        # print summary
        speeds = [r["generate_speed"] for r in mode_results]
        acc_lens = [r["avg_accept_length"] for r in mode_results]
        covs = [r["avg_coverage"] for r in mode_results]
        print(f"  [{ablation_mode}] Gen speed: {np.mean(speeds):.1f} tok/s, "
              f"Accept len: {np.mean(acc_lens):.2f}, "
              f"Coverage: {np.mean(covs):.3f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eagle-path", type=str, required=True)
    parser.add_argument("--eagle-num-iter", type=int, default=6)
    parser.add_argument("--eagle-tree-size", type=int, default=60)
    parser.add_argument("--memory-limit", type=float, default=0.8)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--bench-name", type=str, default="spec_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--max-length", type=int, default=100000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--chat-template", type=str, default="llama-3")
    parser.add_argument("--output", type=str, default="logs/ablation/ablation_results.json")
    parser.add_argument("--ablation-modes", type=str, nargs="+",
                        default=["full", "ctx_only", "ext_only"],
                        choices=["full", "ctx_only", "ext_only"])
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    max_length = min(args.max_length, config.max_position_embeddings)

    model = LLM_with_eagle(
        base_path=args.model_path,
        eagle_path=args.eagle_path,
        memory_limit=args.memory_limit,
        V=config.vocab_size,
        dtype=str_to_torch_dtype(args.dtype),
        cuda_graph=args.cuda_graph,
        num_iter=args.eagle_num_iter,
        tree_size=args.eagle_tree_size,
        max_context_tokens=3072,
    )
    model.init_storage()
    token_id_remap = torch.arange(config.vocab_size, dtype=torch.int32, device="cpu")
    model._load("token_id_remap", token_id_remap, cls="eagle")
    model.load_from_hf()
    model.set_token_id_remap(token_id_remap)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "llama-3" in args.chat_template:
        teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        teminators = [tokenizer.eos_token_id]

    if args.bench_name == "gsm8k":
        question_file = f"data/gsm8k/gsm8k/main"
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    questions = load_questions(question_file, args.question_begin, args.question_end)

    # Select prompt builder based on benchmark
    if args.bench_name == "human_eval":
        build_prompt_fn = _build_prompt_human_eval
    else:
        build_prompt_fn = _build_prompt_mt_bench

    run_ablation(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        teminators=teminators,
        max_new_tokens=args.max_new_tokens,
        max_length=max_length,
        ablation_modes=args.ablation_modes,
        build_prompt_fn=build_prompt_fn,
        output_path=args.output,
    )
