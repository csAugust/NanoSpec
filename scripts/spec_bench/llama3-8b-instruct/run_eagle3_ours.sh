export CUDA_VISIBLE_DEVICES=2
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.1-8b-eagle3-sharegpt-online/epoch_9_step_20380
Model_id="llama-3-8b-instruct"
Bench_name="spec_bench"

python3 evaluation/inference_eagle3.py \
    --model-path $Model_Path \
    --eagle3-path $Eagle3_Path \
    --cuda-graph \
    --model-id $Model_id/eagle3-ours \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --mode 2 \
    > results/extra/logs/eagle3_ours.log 2>&1
