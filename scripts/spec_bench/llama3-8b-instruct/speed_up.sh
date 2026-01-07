tokenizer_path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Vocab=32768

baseline="data/spec_bench/model_answer/llama-3-8b-instruct/baseline.jsonl"
eagle_original="data/spec_bench/model_answer/llama-3-8b-instruct/eagle-original-new-model.jsonl"
eagle_fr_spec="data/spec_bench/model_answer/llama-3-8b-instruct/eagle-fr-spec-$Vocab.jsonl"

echo "EAGLE ORIGINAL"
python evaluation/mt_bench/speed_mt_bench.py \
    --file-path $eagle_original \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path

echo "EAGLE FR-SPEC"
python evaluation/mt_bench/speed_mt_bench.py \
    --file-path $eagle_fr_spec \
    --base-path $baseline \
    --checkpoint-path $tokenizer_path