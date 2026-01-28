export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Qwen2-7B-Instruct
Model_id="qwen2-7b-instruct"
Bench_name="human_eval"

python3 evaluation/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id $Model_id/baseline \
    --memory-limit 0.8 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "qwen2" \
    --question-end 1000 \
    --max-new-tokens 1024

