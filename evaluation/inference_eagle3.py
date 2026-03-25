import argparse
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.mt_bench.eval import run_eval as run_eval_mt_bench
from evaluation.he_local.eval import run_eval as run_eval_human_eval
from evaluation.gsm8k.eval import run_eval as run_eval_gsm8k
from transformers import AutoTokenizer, AutoConfig
from llamacu.speculative.eagle3 import LLM_with_eagle3

MODE = 0

def eagle3_forward(inputs, model, tokenizer, max_new_tokens, max_length, teminators, is_warmup=False):
    input_ids = inputs.input_ids.int()

    prefill_length = len(input_ids[0])
    max_new_tokens = min(max_new_tokens, max_length - prefill_length)

    output_ids, accept_length_list, model_step, draft_time_list = model.generate(
        input_ids=input_ids,
        generation_length=max_new_tokens,
        teminators=teminators,
        tokenizer=tokenizer,
        is_warmup=is_warmup,
        mode=MODE,
    )

    new_token = len(output_ids)
    return output_ids, new_token, model_step, accept_length_list, draft_time_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eagle3-path", type=str, required=True)
    parser.add_argument("--eagle-num-iter", type=int, default=6)
    parser.add_argument("--eagle-tree-size", type=int, default=60)
    parser.add_argument("--memory-limit", type=float, default=0.8)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--model-id", type=str, default="eagle3")
    parser.add_argument("--bench-name", type=str, default="spec_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-length", type=int, default=100000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--chat-template", type=str, default="llama-3")
    parser.add_argument("--mode", type=int, default=0,
                        help="Draft mode: 0=d2t baseline, 1=indexed_gemm, 2=prefetch")

    args = parser.parse_args()
    MODE = args.mode

    if args.bench_name == "gsm8k":
        question_file = f"data/gsm8k/gsm8k/main"
    else:
        question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    config = AutoConfig.from_pretrained(args.model_path)
    max_length = min(args.max_length, config.max_position_embeddings)

    model = LLM_with_eagle3(
        base_path=args.model_path,
        eagle3_path=args.eagle3_path,
        memory_limit=args.memory_limit,
        dtype=str_to_torch_dtype(args.dtype),
        cuda_graph=args.cuda_graph,
        num_iter=args.eagle_num_iter,
        tree_size=args.eagle_tree_size,
        max_context_tokens=3072 if args.mode > 0 else 0,
    )
    model.init_storage()
    model.load_from_hf()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if "llama-3" in args.model_id or "llama_3" in args.model_id or "llama-3" in args.chat_template:
        teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        teminators = [tokenizer.eos_token_id]

    if args.bench_name == "mt_bench" or args.bench_name == "spec_bench":
        run_eval = run_eval_mt_bench
    elif args.bench_name == "human_eval":
        run_eval = run_eval_human_eval
    elif args.bench_name == "gsm8k":
        run_eval = run_eval_gsm8k
    else:
        raise ValueError(f"Unknown benchmark name: {args.bench_name}")

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=eagle3_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        num_choices=args.num_choices,
        chat_template=args.chat_template,
        teminators=teminators,
    )
