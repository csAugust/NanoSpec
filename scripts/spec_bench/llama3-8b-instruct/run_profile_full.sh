#!/bin/bash
# Full latency breakdown profiling for rebuttal.
# Runs all configurations in parallel across GPUs.

source /opt/venv/bin/activate
cd /mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec

Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
Eagle3_Path=/mnt/user-ssd/chenzhiyang1/workspace/Train/DFlash/SpecForge/outputs/llama3.1-8b-eagle3-sharegpt-online/epoch_9_step_20380
Freq_Path=$Eagle_Path

NUM_SAMPLES=80
NUM_WARMUP=3
MAX_TOKENS=512

mkdir -p logs/profile

COMMON="--model-path $Model_Path --memory-limit 0.50 --dtype float16 --chat-template llama-3 --eagle-num-iter 6 --eagle-tree-size 60 --max-new-tokens $MAX_TOKENS --num-samples $NUM_SAMPLES --num-warmup $NUM_WARMUP"

# EAGLE-2 configs (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle_Path --drafter eagle --mode 0 \
    --output-csv logs/profile/eagle2_fullvocab.csv \
    > logs/profile/eagle2_fullvocab.log 2>&1 &
PID_E2_FULL=$!

CUDA_VISIBLE_DEVICES=1 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle_Path --drafter eagle --mode 0 --V 32768 \
    --output-csv logs/profile/eagle2_frspec.csv \
    > logs/profile/eagle2_frspec.log 2>&1 &
PID_E2_FR=$!

CUDA_VISIBLE_DEVICES=2 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle_Path --drafter eagle --mode 1 \
    --output-csv logs/profile/eagle2_mode1.csv \
    > logs/profile/eagle2_mode1.log 2>&1 &
PID_E2_M1=$!

CUDA_VISIBLE_DEVICES=3 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle_Path --drafter eagle --mode 2 \
    --output-csv logs/profile/eagle2_mode2.csv \
    > logs/profile/eagle2_mode2.log 2>&1 &
PID_E2_M2=$!

# Wait for EAGLE-2 to finish, then run EAGLE-3 on same GPUs
wait $PID_E2_FULL $PID_E2_FR $PID_E2_M1 $PID_E2_M2
echo "EAGLE-2 profiling done."

# EAGLE-3 configs (GPUs 0-2)
CUDA_VISIBLE_DEVICES=0 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle3_Path --drafter eagle3 --mode 0 \
    --output-csv logs/profile/eagle3_fullvocab.csv \
    > logs/profile/eagle3_fullvocab.log 2>&1 &
PID_E3_FULL=$!

CUDA_VISIBLE_DEVICES=1 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle3_Path --drafter eagle3 --mode 0 --V 32768 --freq-path $Freq_Path \
    --output-csv logs/profile/eagle3_frspec.csv \
    > logs/profile/eagle3_frspec.log 2>&1 &
PID_E3_FR=$!

CUDA_VISIBLE_DEVICES=2 python3 evaluation/profile_latency.py $COMMON \
    --eagle-path $Eagle3_Path --drafter eagle3 --mode 2 \
    --output-csv logs/profile/eagle3_mode2.csv \
    > logs/profile/eagle3_mode2.log 2>&1 &
PID_E3_M2=$!

wait $PID_E3_FULL $PID_E3_FR $PID_E3_M2
echo "EAGLE-3 profiling done."

echo ""
echo "============================================"
echo "  ALL PROFILING COMPLETE"
echo "============================================"
echo ""

# Print summary table
for f in logs/profile/eagle2_fullvocab.log logs/profile/eagle2_frspec.log logs/profile/eagle2_mode1.log logs/profile/eagle2_mode2.log logs/profile/eagle3_fullvocab.log logs/profile/eagle3_frspec.log logs/profile/eagle3_mode2.log; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f .log) ---"
        grep -A 12 "Latency Breakdown" "$f" 2>/dev/null || tail -12 "$f"
        echo ""
    fi
done
