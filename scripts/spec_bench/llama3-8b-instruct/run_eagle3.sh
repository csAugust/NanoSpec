export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE3-LLaMA3.1-Instruct-8B
Model_id="llama-3-8b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_eagle3.py \
    --model-path $Model_Path \
    --eagle3-path $Eagle3_Path \
    --cuda-graph \
    --model-id $Model_id/eagle3-original \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --mode 0 \
    > logs/eagle3_original.log 2>&1
