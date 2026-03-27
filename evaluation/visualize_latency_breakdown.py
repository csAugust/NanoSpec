#!/usr/bin/env python3
"""
Visualize latency breakdown per decoding step for rebuttal.
Reads CSV files from logs/profile/ and produces a stacked bar chart.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── data ─────────────────────────────────────────────────────────────────

# Profiling results: (config_label, drafter, csv_file)
CONFIGS = [
    ("Full Vocab",        "EAGLE-2", "logs/profile/eagle2_fullvocab.csv"),
    ("FR-Spec",           "EAGLE-2", "logs/profile/eagle2_frspec.csv"),
    ("NanoSpec\n(Gather)", "EAGLE-2", "logs/profile/eagle2_mode1.csv"),
    ("NanoSpec\n(Prefetch)","EAGLE-2", "logs/profile/eagle2_mode2.csv"),
    ("Full Vocab",        "EAGLE-3", "logs/profile/eagle3_fullvocab.csv"),
    ("FR-Spec",           "EAGLE-3", "logs/profile/eagle3_frspec.csv"),
    ("NanoSpec\n(Gather)", "EAGLE-3", "logs/profile/eagle3_mode1.csv"),
    ("NanoSpec\n(Prefetch)","EAGLE-3", "logs/profile/eagle3_mode2.csv"),
]

COMPONENTS = ['backbone', 'lm_head', 'tree_ops', 'gather_wait', 'verify']
COMPONENT_LABELS = {
    'backbone':     'Draft Backbone',
    'lm_head':      'Draft LM Head',
    'tree_ops':     'Draft Tree Ops',
    'gather_wait':  'Weight Gather',
    'verify':       'Target Verify',
}
COMPONENT_COLORS = {
    'backbone':     '#4C72B0',
    'lm_head':      '#DD8452',
    'tree_ops':     '#55A868',
    'gather_wait':  '#C44E52',
    'verify':       '#8172B3',
}


def load_csv(path):
    """Load a profile CSV and return dict of component -> avg_ms."""
    if not os.path.exists(path):
        print(f"  [SKIP] {path}")
        return None
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        result[row['component']] = row['avg_ms']
    return result


def plot_breakdown(out_path):
    # Load data grouped by drafter
    drafters = ['EAGLE-2', 'EAGLE-3']
    drafter_configs = {d: [] for d in drafters}

    for label, drafter, csv_path in CONFIGS:
        data = load_csv(csv_path)
        if data is None:
            continue
        drafter_configs[drafter].append((label, data))

    n_cols = len(drafters)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for col, drafter in enumerate(drafters):
        ax = axes[col]
        configs = drafter_configs[drafter]
        if not configs:
            continue

        labels = [c[0] for c in configs]
        n = len(labels)
        x = np.arange(n)
        bar_width = 0.55

        bottoms = np.zeros(n)
        for comp in COMPONENTS:
            vals = []
            for _, data in configs:
                v = data.get(comp, 0)
                vals.append(max(v, 0))
            vals = np.array(vals)
            if vals.sum() < 0.001:
                continue
            bars = ax.bar(x, vals, bar_width, bottom=bottoms,
                          label=COMPONENT_LABELS[comp],
                          color=COMPONENT_COLORS[comp],
                          edgecolor='white', linewidth=0.5)

            # Add value labels on each segment (only if > 0.05ms)
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0.05:
                    ax.text(x[i], b + v / 2, f'{v:.2f}',
                            ha='center', va='center', fontsize=9,
                            fontweight='bold', color='white')
            bottoms += vals

        # Total time label on top of each bar
        for i, total in enumerate(bottoms):
            ax.text(x[i], total + 0.15, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(drafter, fontsize=15, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Latency per Step (ms)' if col == 0 else '', fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', ls='--', alpha=0.35, linewidth=0.5)
        ax.set_ylim(0, max(bottoms) * 1.15)

    # Shared legend at bottom
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center',
               bbox_to_anchor=(0.5, -0.06),
               ncol=len(COMPONENTS), fontsize=12, frameon=True,
               borderpad=0.5, columnspacing=1.2)

    fig.suptitle('Latency Breakdown per Decoding Step — Llama-3.1-8B on H20',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    plot_breakdown('figs/latency_breakdown.pdf')
    plot_breakdown('figs/latency_breakdown.png')
