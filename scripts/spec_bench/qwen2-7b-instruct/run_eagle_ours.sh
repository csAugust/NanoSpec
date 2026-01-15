export CUDA_VISIBLE_DEVICES=1
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Qwen2-7B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-Qwen2-7B-Instruct
Model_id="qwen2-7b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_eagle.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --model-id $Model_id/eagle-ours \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "qwen2" \
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --mode 2

