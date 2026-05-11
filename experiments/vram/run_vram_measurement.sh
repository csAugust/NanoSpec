#!/bin/bash
# Measure VRAM overhead of NanoSpec vs Vanilla EAGLE-2
# Runs two separate processes: max_context_tokens=0 (baseline) and 3072 (NanoSpec)
export CUDA_VISIBLE_DEVICES=0
Model_Path=/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct
Eagle_Path=/mnt/user-ssd/chenzhiyang1/workspace/Models/EAGLE-LLaMA3.1-Instruct-8B
LOG=results/extra/logs/vram_measurement.log

mkdir -p results/extra/logs
> $LOG

echo "=== Measuring Vanilla EAGLE (max_context_tokens=0) ===" >> $LOG
python3 experiments/vram/measure_vram.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --max-context-tokens 0 \
    --memory-limit 0.50 \
    --dtype "float16" \
    >> $LOG 2>&1

echo "=== Measuring NanoSpec (max_context_tokens=3072) ===" >> $LOG
python3 experiments/vram/measure_vram.py \
    --model-path $Model_Path \
    --eagle-path $Eagle_Path \
    --max-context-tokens 3072 \
    --memory-limit 0.50 \
    --dtype "float16" \
    >> $LOG 2>&1

echo "=== Analysis ===" >> $LOG
python3 experiments/vram/analyze_vram.py --log $LOG >> $LOG 2>&1

echo "Done. Results in $LOG"
