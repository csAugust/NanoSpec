#!/usr/bin/env python3
"""
Combined visualization & report for EAGLE-3 experiments.
Merges spec_bench + human_eval (as 'Code' category) for each model.
Produces a single 3×2 figure (rows=metrics, cols=models) with shared legend.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── data loading ─────────────────────────────────────────────────────────

def load_single_jsonl(file_path, method_label):
    extracted = []
    if not os.path.exists(file_path):
        print(f"  [SKIP] {file_path}")
        return []
    print(f"  [LOAD] {method_label}: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            category = d.get("category", "Code")
            choices = d.get("choices", [])
            if not choices:
                continue
            c = choices[0]
            speed_list = c.get("generate_speed", [])
            speed = np.mean(speed_list) if speed_list else None
            dec_list = c.get("decoding_speed", [])
            dec = np.mean(dec_list) if dec_list else None
            avg_len = c.get("avg_accept_length")
            if speed is None or avg_len is None:
                continue
            if not np.isfinite(speed) or not np.isfinite(avg_len):
                continue
            extracted.append({
                'method': method_label,
                'category': category,
                'generate_speed': float(speed),
                'decoding_speed': float(dec) if dec and np.isfinite(dec) else 0,
                'avg_accept_length': float(avg_len),
            })
    return extracted

METHODS = [
    ("eagle3-original.jsonl",       "EAGLE3"),
    ("eagle3-fr-spec-32768.jsonl",  "FRSpec"),
    ("eagle3-ours.jsonl",           "NanoSpec"),
]

MERGE_CATS = ['writing', 'roleplay', 'reasoning', 'math', 'coding',
              'extraction', 'stem', 'humanities']

CAT_RENAME = {
    'translation': 'MT', 'conversation': 'Conv.',
    'math_reasoning': 'Math', 'qa': 'QA',
    'rag': 'RAG', 'summarization': 'Summ.',
    'human-eval': 'Code',
}

def load_model_data(model_dir):
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    base = os.path.abspath(base)
    all_data = []
    for bench in ['spec_bench', 'human_eval']:
        d = os.path.join(base, bench, 'logs', model_dir)
        for fname, label in METHODS:
            rows = load_single_jsonl(os.path.join(d, fname), label)
            all_data.extend(rows)
    df = pd.DataFrame(all_data)
    if df.empty:
        return df
    df.loc[df['category'].isin(MERGE_CATS), 'category'] = 'Conv.'
    df['category'] = df['category'].map(lambda c: CAT_RENAME.get(c, c))
    return df

# ── summary table ────────────────────────────────────────────────────────

def print_summary(df, model_name):
    print(f"\n{'='*80}")
    print(f"  {model_name}")
    print(f"{'='*80}")
    metrics = ['generate_speed', 'avg_accept_length', 'decoding_speed']
    labels  = ['Gen Speed', 'Accept Len', 'Dec Speed']
    method_cat_mean = df.groupby(['category', 'method'])[metrics].mean()
    overall = method_cat_mean.groupby('method').mean()
    for metric, lbl in zip(metrics, labels):
        print(f"\n── {lbl} ──")
        pivot = df.groupby(['category', 'method'])[metric].mean().unstack()
        pivot.loc['Average'] = overall[metric]
        print(pivot.to_string(float_format='{:.2f}'.format))
    print()

# ── plotting (single combined figure) ────────────────────────────────────

def add_labels(ax, spacing=3, detail=False):
    for p in ax.patches:
        h = p.get_height()
        if h == 0 or not np.isfinite(h):
            continue
        fmt = f'{h:.2f}' if detail else f'{h:.0f}'
        ax.annotate(fmt,
                    xy=(p.get_x() + p.get_width()/2, h),
                    xytext=(0, spacing), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    rotation=90 if p.get_width() < 0.12 else 0)

METRICS_SPEC = [
    ('generate_speed',   'tokens/s',      'Generate Speed',  False),
    ('avg_accept_length', '',  'Accept Length',   True),
    ('decoding_speed',   'tokens/s',      'Decoding Speed',  False),
]

MODEL_DIRS = [
    ("llama-3-8b-instruct_eagle3",   "LLaMA-3.1-8B-Instruct"),
    ("llama-3.2-1b-instruct_eagle3", "LLaMA-3.2-1B-Instruct"),
]

def plot_combined(model_dfs, out_path):
    """
    model_dfs: list of (model_name, df) tuples, one per model (column).
    Produces a 3-row × N-col figure.
    """
    n_rows = len(METRICS_SPEC)
    n_cols = len(model_dfs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    # consistent method order & colors across all subplots
    all_methods = sorted({m for _, df in model_dfs for m in df['method'].unique()})
    cmap = plt.get_cmap('tab10')
    colors = {m: cmap(i) for i, m in enumerate(all_methods)}

    for col, (model_name, df) in enumerate(model_dfs):
        agg = df.groupby(['category', 'method']).agg(['mean', 'sem'])
        methods = sorted(df['method'].unique())

        for row, (metric, ylabel, title, detail) in enumerate(METRICS_SPEC):
            ax = axes[row][col]
            means = agg[metric]['mean'].unstack(fill_value=0)
            sems  = agg[metric]['sem'].unstack(fill_value=0)
            cats = means.index.tolist()
            n_c, n_m = len(cats), len(methods)
            x = np.arange(n_c)
            bw = 0.78 / n_m

            for i, m in enumerate(methods):
                if m not in means.columns:
                    continue
                off = (i - (n_m - 1) / 2) * bw
                ax.bar(x + off, means[m].values, width=bw,
                       yerr=sems[m].fillna(0).values,
                       label=m, color=colors[m], capsize=2,
                       edgecolor='white', linewidth=0.4)

            # column title (model name) on top row only
            if row == 0:
                ax.set_title(model_name, fontsize=15, fontweight='bold', pad=8)

            # row metric label on left column only
            if col == 0:
                if ylabel != '':
                    ax.set_ylabel(f'{title}\n({ylabel})', fontsize=14)
                else:
                    ax.set_ylabel(f'{title}', fontsize=14)
            else:
                ax.set_ylabel('')

            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=14)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(axis='y', ls='--', alpha=0.4, linewidth=0.5)
            add_labels(ax, spacing=2, detail=detail)

    # shared legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.01),
               ncol=len(all_methods), fontsize=14, frameon=True,
               borderpad=0.6, columnspacing=1.5)

    # footnote about error bars
    fig.text(0.5, -0.05,
             'Error bars show ±1 SEM (standard error of the mean) across samples within each category.',
             ha='center', va='top', fontsize=14, fontstyle='italic', color='gray')

    fig.suptitle('Evaluation of NanoSpec on EAGLE-3',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45, wspace=0.22)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {out_path}")

# ── main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs('figs', exist_ok=True)
    model_dfs = []
    all_summaries = []

    for model_dir, model_name in MODEL_DIRS:
        print(f"\n{'#'*60}")
        print(f"# Loading: {model_name} ({model_dir})")
        print(f"{'#'*60}")
        df = load_model_data(model_dir)
        if df.empty:
            print(f"  No data found for {model_dir}, skipping.")
            continue
        print(f"  Loaded {len(df)} records, methods={df['method'].unique().tolist()}, "
              f"categories={df['category'].unique().tolist()}")
        print_summary(df, model_name)
        model_dfs.append((model_name, df))

        method_cat = df.groupby(['category', 'method'])[['generate_speed', 'avg_accept_length']].mean()
        overall = method_cat.groupby('method').mean()
        overall['model'] = model_name
        all_summaries.append(overall)

    # combined figure
    if model_dfs:
        plot_combined(model_dfs, 'results/main/figs/eagle3_combined.pdf')

    # cross-model summary
    if all_summaries:
        print(f"\n{'='*80}")
        print("  Cross-Model Summary (category-averaged means)")
        print(f"{'='*80}")
        combined = pd.concat(all_summaries).reset_index()
        for metric, lbl in [('generate_speed', 'Generate Speed (tokens/s)'),
                            ('avg_accept_length', 'Accept Length')]:
            print(f"\n── {lbl} ──")
            pivot = combined.pivot(index='model', columns='method', values=metric)
            print(pivot.to_string(float_format='{:.2f}'.format))
        print()
