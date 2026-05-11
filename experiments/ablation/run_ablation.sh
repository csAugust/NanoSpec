export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
Bench_name="spec_bench"

python3 experiments/ablation/inference_ablation.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --cuda-graph \
    --memory-limit 0.50 \
    --bench-name $Bench_name \
    --dtype "float16" \
    --chat-template "llama-3" \
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --ablation-modes full ctx_only ext_only \
    --output results/extra/logs/ablation/ablation_results.json \
    > results/extra/logs/ablation/ablation_run.log 2>&1
