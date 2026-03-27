#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.1-8b-eagle3-sharegpt-online/epoch_9_step_20380
Freq_Path=$Eagle_Path

NUM_SAMPLES=20
NUM_WARMUP=3
MAX_TOKENS=512

mkdir -p logs/profile

COMMON_ARGS="--model-path $Model_Path --memory-limit 0.50 --dtype float16 --chat-template llama-3 --eagle-num-iter 6 --eagle-tree-size 60 --max-new-tokens $MAX_TOKENS --num-samples $NUM_SAMPLES --num-warmup $NUM_WARMUP"

echo "=== EAGLE-2: Full Vocab (mode 0, no V) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle_Path --drafter eagle \
    --mode 0 \
    --output-csv logs/profile/eagle2_fullvocab.csv \
    > logs/profile/eagle2_fullvocab.log 2>&1

echo "=== EAGLE-2: FR-Spec (mode 0, V=32768) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle_Path --drafter eagle \
    --mode 0 --V 32768 \
    --output-csv logs/profile/eagle2_frspec.csv \
    > logs/profile/eagle2_frspec.log 2>&1

echo "=== EAGLE-2: Indexed GEMM (mode 1) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle_Path --drafter eagle \
    --mode 1 \
    --output-csv logs/profile/eagle2_mode1.csv \
    > logs/profile/eagle2_mode1.log 2>&1

echo "=== EAGLE-2: Prefetch/Ours (mode 2) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle_Path --drafter eagle \
    --mode 2 \
    --output-csv logs/profile/eagle2_mode2.csv \
    > logs/profile/eagle2_mode2.log 2>&1

echo "=== EAGLE-3: Full Vocab (mode 0, no V) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle3_Path --drafter eagle3 \
    --mode 0 \
    --output-csv logs/profile/eagle3_fullvocab.csv \
    > logs/profile/eagle3_fullvocab.log 2>&1

echo "=== EAGLE-3: FR-Spec (mode 0, V=32768) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle3_Path --drafter eagle3 \
    --mode 0 --V 32768 --freq-path $Freq_Path \
    --output-csv logs/profile/eagle3_frspec.csv \
    > logs/profile/eagle3_frspec.log 2>&1

echo "=== EAGLE-3: Prefetch/Ours (mode 2) ==="
python3 evaluation/profile_latency.py $COMMON_ARGS \
    --eagle-path $Eagle3_Path --drafter eagle3 \
    --mode 2 \
    --output-csv logs/profile/eagle3_mode2.csv \
    > logs/profile/eagle3_mode2.log 2>&1

echo "=== All profiling done. Results in logs/profile/ ==="
echo ""
echo "Summary (tail -5 each log):"
for f in logs/profile/*.log; do
    echo "--- $(basename $f) ---"
    tail -8 "$f"
    echo ""
done
