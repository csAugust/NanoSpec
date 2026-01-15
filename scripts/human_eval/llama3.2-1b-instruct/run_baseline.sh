export CUDA_VISIBLE_DEVICES=4
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Llama-3.2-1B-Instruct
Model_id="llama-3.2-1b-instruct"
Bench_name="human_eval"

python3 evaluation/inference_baseline.py \
    --model-path $Model_Path \
    --cuda-graph \
    --model-id ${Model_id}/baseline \
    --memory-limit 0.80 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --max-new-tokens 1024