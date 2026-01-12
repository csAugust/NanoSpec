export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
Model_id="llama-3-8b-instruct"
Bench_name="spec_bench"
Vocab=32768

python3 evaluation/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id $Model_id/eagle-fr-spec-$Vocab \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --V $Vocab

