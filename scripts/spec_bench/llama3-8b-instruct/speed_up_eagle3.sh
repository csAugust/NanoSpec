tokenizer_path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct

baseline="data/spec_bench/model_answer/llama-3-8b-instruct/baseline.jsonl"
eagle3_original="data/spec_bench/model_answer/llama-3-8b-instruct/eagle3-original.jsonl"
eagle3_fr_spec="data/spec_bench/model_answer/llama-3-8b-instruct/eagle3-fr-spec.jsonl"
eagle3_ours="data/spec_bench/model_answer/llama-3-8b-instruct/eagle3-ours.jsonl"

echo "EAGLE3 ORIGINAL (d2t baseline, mode 0)"
python evaluation/mt_bench/speed_mt_bench.py \
    --file-path $eagle3_original \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path

echo "EAGLE3 FR-SPEC (d2t baseline = FR-Spec equivalent)"
python evaluation/mt_bench/speed_mt_bench.py \
    --file-path $eagle3_fr_spec \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path

echo "EAGLE3 OURS (mode 2, dynamic pruning with prefetch)"
python evaluation/mt_bench/speed_mt_bench.py \
    --file-path $eagle3_ours \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path
