#!/usr/bin/env python3
"""
Visualization for ablation study results.
Produces two figures:

Figure 1 (ablation_bar.pdf): 3-panel grouped bar chart
  - Panel A: Average Acceptance Length per category per ablation mode
  - Panel B: Coverage (%) per category per ablation mode
  - Panel C: Generate Speed (tokens/s) per category per ablation mode

Figure 2 (vocab_evolution.pdf): Vocab size & coverage evolution over decode steps
  - Left: vocab size vs decode step (line plot, one line per ablation mode)
  - Right: per-step coverage vs decode step
  Averaged across all questions within selected categories.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MERGE_CATS = ['writing', 'roleplay', 'reasoning', 'math', 'coding',
              'extraction', 'stem', 'humanities']

CAT_RENAME = {
    'translation': 'MT', 'conversation': 'Conv.',
    'math_reasoning': 'Math', 'qa': 'QA',
    'rag': 'RAG', 'summarization': 'Summ.',
}

MODE_LABELS = {
    'full': 'Ctx+Ext',
    'ctx_only': 'Ctx Only',
    'ext_only': 'Ext Only',
}

MODE_COLORS = {
    'Ctx+Ext': '#2196F3',
    'Ctx Only': '#FF9800',
    'Ext Only': '#4CAF50',
}


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # rename categories
    df.loc[df['category'].isin(MERGE_CATS), 'category'] = 'Conv.'
    df['category'] = df['category'].map(lambda c: CAT_RENAME.get(c, c))
    df['mode_label'] = df['ablation_mode'].map(MODE_LABELS)
    return df


def add_bar_labels(ax, detail=False, spacing=2):
    for p in ax.patches:
        h = p.get_height()
        if h == 0 or not np.isfinite(h):
            continue
        fmt = f'{h:.2f}' if detail else f'{h:.0f}'
        ax.annotate(fmt,
                    xy=(p.get_x() + p.get_width() / 2, h),
                    xytext=(0, spacing), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    rotation=90 if p.get_width() < 0.12 else 0)


def plot_ablation_bars(df, out_path):
    """3-panel grouped bar chart: accept length, coverage, gen speed."""
    metrics = [
        ('avg_accept_length', 'Accept Length', True),
        ('avg_coverage', 'Coverage', True),
        # ('generate_speed', 'Gen Speed (tok/s)', False),
    ]

    # aggregate by category and mode
    agg = df.groupby(['category', 'mode_label']).agg(
        avg_accept_length=('avg_accept_length', 'mean'),
        avg_coverage=('avg_coverage', 'mean'),
        generate_speed=('generate_speed', 'mean'),
        avg_accept_length_sem=('avg_accept_length', 'sem'),
        avg_coverage_sem=('avg_coverage', 'sem'),
        generate_speed_sem=('generate_speed', 'sem'),
    ).reset_index()

    # also compute overall average
    overall = df.groupby(['category', 'mode_label']).agg(
        avg_accept_length=('avg_accept_length', 'mean'),
        avg_coverage=('avg_coverage', 'mean'),
        generate_speed=('generate_speed', 'mean'),
    ).groupby('mode_label').mean().reset_index()
    overall['category'] = 'Avg.'
    agg = pd.concat([agg, overall], ignore_index=True)

    methods = ['Ctx Only', 'Ext Only', 'Ctx+Ext']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, (metric, ylabel, detail) in enumerate(metrics):
        ax = axes[ax_idx]
        pivot = agg.pivot(index='category', columns='mode_label', values=metric).reindex(columns=methods)
        cats = pivot.index.tolist()
        n_c = len(cats)
        n_m = len(methods)
        x = np.arange(n_c)
        bw = 0.78 / n_m

        for i, m in enumerate(methods):
            if m not in pivot.columns:
                continue
            vals = pivot[m].fillna(0).values
            # scale coverage to percentage
            if metric == 'avg_coverage':
                vals = vals * 100
            off = (i - (n_m - 1) / 2) * bw
            ax.bar(x + off, vals, width=bw,
                   label=m, color=MODE_COLORS[m],
                   edgecolor='white', linewidth=0.4)

        if metric == 'avg_coverage':
            ax.set_ylabel(f'{ylabel} (%)', fontsize=13)
        else:
            ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', ls='--', alpha=0.4, linewidth=0.5)
        add_bar_labels(ax, detail=detail, spacing=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(methods), fontsize=13, frameon=True,
               borderpad=0.6, columnspacing=1.5)

    fig.suptitle('Ablation: Effect of Ctx and Ext Components',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _smooth(arr, window=10):
    """Simple centered moving average with edge handling."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # pad edges to avoid shrinkage
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def plot_vocab_evolution(df, out_path, max_steps=100, smooth_window=5):
    """
    2-panel line plot showing vocab size and coverage evolution over decode steps.

    Only includes questions whose step count >= max_steps, so the sample set is
    fixed across all steps and the mean doesn't jump when shorter sequences drop out.

    Coverage is cumulative hit rate per question, then averaged.
    """
    methods = ['Ctx Only', 'Ext Only', 'Ctx+Ext']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for mode_label in methods:
        subset = df[df['mode_label'] == mode_label]
        if subset.empty:
            continue

        # Filter: only keep questions with enough steps
        filtered_vocab = []
        filtered_cum_cov = []
        for _, row in subset.iterrows():
            vocab_sizes = row['step_vocab_sizes']
            coverages = row['step_coverages']
            accept_lens = row['step_accept_lengths']

            if len(vocab_sizes) < max_steps:
                continue

            filtered_vocab.append(vocab_sizes[:max_steps])

            # Compute cumulative coverage
            cum_hits = 0.0
            cum_total = 0
            cum_cov_seq = []
            for cov, alen in zip(coverages[:max_steps], accept_lens[:max_steps]):
                cum_hits += cov * alen
                cum_total += alen
                cum_cov_seq.append(cum_hits / cum_total if cum_total > 0 else 0.0)
            filtered_cum_cov.append(cum_cov_seq)

        n = len(filtered_vocab)
        if n == 0:
            print(f"  Warning: no questions with >= {max_steps} steps for {mode_label}")
            continue
        print(f"  {mode_label}: {n} questions with >= {max_steps} steps")

        vocab_arr = np.array(filtered_vocab, dtype=float)       # [n, max_steps]
        cov_arr = np.array(filtered_cum_cov, dtype=float) * 100  # to %

        mean_vocab = np.mean(vocab_arr, axis=0)
        mean_cov = np.mean(cov_arr, axis=0)

        mean_vocab_s = _smooth(mean_vocab, smooth_window)
        mean_cov_s = _smooth(mean_cov, smooth_window)

        steps = np.arange(1, max_steps + 1)
        color = MODE_COLORS[mode_label]

        axes[0].plot(steps, mean_vocab_s, label=mode_label, color=color, linewidth=1.8)
        axes[1].plot(steps, mean_cov_s, label=mode_label, color=color, linewidth=1.8)

    axes[0].set_xlabel('Decode Step', fontsize=13)
    axes[0].set_ylabel('Active Vocab Size', fontsize=13)
    axes[0].set_title('Vocab Size Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, ls='--', alpha=0.4)
    axes[0].tick_params(labelsize=11)
    axes[0].set_xlim(1, max_steps)

    axes[1].set_xlabel('Decode Step', fontsize=13)
    axes[1].set_ylabel('Cumulative Coverage (%)', fontsize=13)
    axes[1].set_title('Cumulative Coverage Over Decoding', fontsize=14, fontweight='bold')
    axes[1].grid(True, ls='--', alpha=0.4)
    axes[1].tick_params(labelsize=11)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlim(1, max_steps)

    axes[0].legend(fontsize=12)

    fig.suptitle('Dynamic Vocabulary Evolution During Decoding',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _collect_evolution(subset, max_steps):
    """Collect vocab size and cumulative coverage arrays for questions with >= max_steps steps."""
    filtered_vocab = []
    filtered_cum_cov = []
    for _, row in subset.iterrows():
        vocab_sizes = row['step_vocab_sizes']
        coverages = row['step_coverages']
        accept_lens = row['step_accept_lengths']
        if len(vocab_sizes) < max_steps:
            continue
        filtered_vocab.append(vocab_sizes[:max_steps])
        cum_hits = 0.0
        cum_total = 0
        cum_cov_seq = []
        for cov, alen in zip(coverages[:max_steps], accept_lens[:max_steps]):
            cum_hits += cov * alen
            cum_total += alen
            cum_cov_seq.append(cum_hits / cum_total if cum_total > 0 else 0.0)
        filtered_cum_cov.append(cum_cov_seq)
    if not filtered_vocab:
        return None, None, 0
    vocab_arr = np.array(filtered_vocab, dtype=float)
    cov_arr = np.array(filtered_cum_cov, dtype=float) * 100
    return vocab_arr, cov_arr, len(filtered_vocab)


def plot_vocab_evolution_detail(df, out_path, smooth_window=5):
    """
    Per-category vocab evolution with min/avg/max bands.
    Each category gets a 2-row subplot (vocab size + coverage).
    Three methods shown as colored bands (min-max fill) + avg line.
    Adaptive max_steps per category (25th percentile of step counts).
    """
    methods = ['Ctx Only', 'Ext Only', 'Ctx+Ext']
    categories = sorted(df['category'].unique().tolist())

    n_cats = len(categories)
    fig, axes = plt.subplots(2, n_cats, figsize=(3.5 * n_cats, 7), squeeze=False)

    for col, cat in enumerate(categories):
        cat_df = df[df['category'] == cat]

        # Determine adaptive max_steps: 25th percentile of step counts across all modes
        all_lens = []
        for _, row in cat_df.iterrows():
            all_lens.append(len(row['step_vocab_sizes']))
        if not all_lens:
            continue
        adaptive_max = max(int(np.percentile(all_lens, 25)), 5)

        for mode_label in methods:
            subset = cat_df[cat_df['mode_label'] == mode_label]
            if subset.empty:
                continue
            vocab_arr, cov_arr, n = _collect_evolution(subset, adaptive_max)
            if vocab_arr is None:
                continue

            steps = np.arange(1, adaptive_max + 1)
            color = MODE_COLORS[mode_label]

            # Vocab size (top row)
            ax_v = axes[0][col]
            mean_v = _smooth(np.mean(vocab_arr, axis=0), smooth_window)
            min_v = _smooth(np.min(vocab_arr, axis=0), smooth_window)
            max_v = _smooth(np.max(vocab_arr, axis=0), smooth_window)
            ax_v.fill_between(steps, min_v, max_v, color=color, alpha=0.15)
            ax_v.plot(steps, mean_v, color=color, linewidth=1.5, label=mode_label)

            # Coverage (bottom row)
            ax_c = axes[1][col]
            mean_c = _smooth(np.mean(cov_arr, axis=0), smooth_window)
            min_c = _smooth(np.min(cov_arr, axis=0), smooth_window)
            max_c = _smooth(np.max(cov_arr, axis=0), smooth_window)
            ax_c.fill_between(steps, min_c, max_c, color=color, alpha=0.15)
            ax_c.plot(steps, mean_c, color=color, linewidth=1.5, label=mode_label)

        # Formatting
        axes[0][col].set_title(f'{cat} (typical decode steps={adaptive_max})',
                               fontsize=11, fontweight='bold')
        axes[0][col].set_xlim(1, adaptive_max)
        axes[0][col].grid(True, ls='--', alpha=0.3)
        axes[0][col].tick_params(labelsize=9)

        axes[1][col].set_xlim(1, adaptive_max)
        axes[1][col].set_ylim(0, 105)
        axes[1][col].grid(True, ls='--', alpha=0.3)
        axes[1][col].tick_params(labelsize=9)
        axes[1][col].set_xlabel('Step', fontsize=10)

    axes[0][0].set_ylabel('Vocab Size', fontsize=11)
    axes[1][0].set_ylabel('Cum. Coverage (%)', fontsize=11)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=len(methods),
               fontsize=11, frameon=True)

    fig.suptitle('Per-Category Vocabulary Evolution\n Solid line: avg; Upper bind: max; Lower bind: min',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table(df):
    """Print a summary table of ablation results."""
    print(f"\n{'='*80}")
    print("  Ablation Summary (averaged across categories)")
    print(f"{'='*80}")

    # per-category-then-average (to weight categories equally)
    cat_means = df.groupby(['category', 'mode_label']).agg(
        accept_len=('avg_accept_length', 'mean'),
        coverage=('avg_coverage', 'mean'),
        gen_speed=('generate_speed', 'mean'),
    )
    overall = cat_means.groupby('mode_label').mean()

    for metric, label in [('accept_len', 'Accept Length'), ('coverage', 'Coverage (%)'),
                           ('gen_speed', 'Gen Speed (tok/s)')]:
        print(f"\n-- {label} --")
        pivot = cat_means[metric].unstack()
        if metric == 'coverage':
            pivot = pivot * 100
            overall_row = overall[metric] * 100
        else:
            overall_row = overall[metric]
        pivot.loc['Average'] = overall_row
        print(pivot.to_string(float_format='{:.2f}'.format))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/extra/logs/ablation/ablation_results.json")
    parser.add_argument("--out-dir", type=str, default="results/extra/figs/ablation")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max decode steps to show in evolution plot")
    args = parser.parse_args()

    df = load_results(args.input)
    print(f"Loaded {len(df)} records, modes={df['mode_label'].unique().tolist()}, "
          f"categories={df['category'].unique().tolist()}")

    print_summary_table(df)
    plot_ablation_bars(df, os.path.join(args.out_dir, 'ablation_bar.pdf'))
    plot_vocab_evolution(df, os.path.join(args.out_dir, 'vocab_evolution.pdf'),
                         max_steps=args.max_steps)
    plot_vocab_evolution_detail(df, os.path.join(args.out_dir, 'vocab_evolution_detail.pdf'))
