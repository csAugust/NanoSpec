export CUDA_VISIBLE_DEVICES=4
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Llama-3.2-1B-Instruct
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.2-1b-eagle3-sharegpt-online/epoch_9_step_5100
Freq_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
Model_id="llama-3.2-1b-instruct"
Bench_name="spec_bench"
Vocab=32768

python3 evaluation/inference_eagle3.py \
    --model-path $Model_Path \
    --eagle3-path $Eagle3_Path \
    --freq-path $Freq_Path \
    --cuda-graph \
    --model-id $Model_id/eagle3-fr-spec-$Vocab \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --V $Vocab \
    --mode 0 \
    > results/extra/logs/eagle3_fr_spec_1b.log 2>&1
