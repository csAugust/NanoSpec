"""
Parse VRAM measurement logs and generate formatted table for paper.

Usage:
    python experiments/vram/analyze_vram.py --log results/extra/logs/vram_measurement.log
"""
import argparse
import json
import re


def parse_log(log_path):
    results = []
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'VRAM_RESULT=(.+)', line)
            if m:
                results.append(json.loads(m.group(1)))
    return results


def format_table(results):
    if not results:
        print("No results found in log file.")
        return

    # Sort by max_context_tokens
    results.sort(key=lambda r: r["max_context_tokens"])

    baseline = results[0]  # max_context_tokens=0
    hidden_size = baseline["hidden_size"]
    dtype_size = baseline["dtype_size"]
    vocab_size = baseline["vocab_size"]

    print("=" * 72)
    print("VRAM Overhead Analysis: NanoSpec vs Vanilla EAGLE-2")
    print("=" * 72)
    print(f"Model hidden_size={hidden_size}, vocab_size={vocab_size}, dtype=FP{dtype_size*8}")
    print(f"memory_limit={baseline['memory_limit']}")
    print()

    # Table header
    header = f"{'Config':<25} {'max_ctx':>8} {'KV Budget':>10} {'Overhead':>12} {'% of Peak':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        max_ctx = r["max_context_tokens"]
        budget = r["kv_budget"] if r["kv_budget"] is not None else 0
        overhead_mb = r["analytical_overhead_MB"]
        pct = overhead_mb / r["mem_peak_MB"] * 100 if r["mem_peak_MB"] > 0 else 0

        if max_ctx == 0:
            label = "Vanilla EAGLE"
        else:
            label = f"NanoSpec (ctx={max_ctx})"

        print(f"{label:<25} {max_ctx:>8} {budget:>10} {overhead_mb:>9.2f} MB {pct:>9.2f}%")

    print()

    # Budget difference
    if len(results) >= 2:
        budget_diff = baseline["kv_budget"] - results[1]["kv_budget"]
        print(f"KV cache budget reduction: {budget_diff} tokens "
              f"({budget_diff / baseline['kv_budget'] * 100:.2f}% of baseline)")

    # Overhead breakdown (for the NanoSpec config)
    nano = [r for r in results if r["max_context_tokens"] > 0]
    if nano:
        r = nano[0]
        print()
        print("Overhead breakdown:")
        print(f"  context_token_ids:      {r['breakdown_context_token_ids_bytes'] / 1024:>8.1f} KB  (GPU token set mirror)")
        print(f"  repack_buffer:          {r['breakdown_repack_buffer_bytes'] / 1024 / 1024:>8.2f} MB  (LM-head weight repack)")
        print(f"  context_tokens_tensor:  {r['breakdown_context_tensor_bytes'] / 1024:>8.1f} KB  (Python GPU tensor)")
        print(f"  Total:                  {r['analytical_overhead_MB']:>8.2f} MB")

    # Scaling formula
    print()
    print(f"Scaling formula: overhead = max_ctx * (8 + {hidden_size} * {dtype_size}) bytes")
    print()
    for ctx in [1024, 2048, 3072, 4096, 8192]:
        oh = ctx * (8 + hidden_size * dtype_size)
        oh_mb = oh / 1024 / 1024
        print(f"  max_ctx={ctx:<5} -> {oh_mb:>7.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="logs/vram_measurement.log")
    args = parser.parse_args()

    results = parse_log(args.log)
    format_table(results)
