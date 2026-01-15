export CUDA_VISIBLE_DEVICES=4
Model_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/Llama-3.2-1B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/LLaMA3.2-Instruct-1B-FR-Spec
Model_id="llama-3.2-1b-instruct"
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
    --eagle-num-iter 6 \
    --eagle-tree-size 60 \
    --question-end 1000 \
    --max-new-tokens 1024 \
    --V $Vocab \
    --mode 0

