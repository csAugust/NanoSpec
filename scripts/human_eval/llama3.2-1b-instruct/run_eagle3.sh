export CUDA_VISIBLE_DEVICES=3
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Llama-3.2-1B-Instruct
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.2-1b-eagle3-sharegpt-online/epoch_9_step_5100
Model_id="llama-3.2-1b-instruct"
Bench_name="human_eval"

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
    > logs/eagle3_original_1b_he.log 2>&1
