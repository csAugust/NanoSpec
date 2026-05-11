export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
# Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.1-8b-eagle3-sharegpt-online/epoch_9_step_20380
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.1-8b-eagle3-sharegpt-online-smallvocab/epoch_9_step_2550
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
    > results/extra/logs/eagle3_original.log 2>&1
