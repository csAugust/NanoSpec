import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np

CATEGORY_MAP = {
    # 你的数据category : 论文表格列名
    'translation': 'MT',
    'conversation': 'Conv.',
    'math_reasoning': 'Math',
    'qa': 'QA',
    'rag': 'RAG',
    'summarization': 'Summ.',
    'human-eval': 'Code'
}

# 论文报告的 DynaSpec 接受长度
# PAPER_ACCEPT_LENGTH = {
#     'MT': 3.51, 'Conv.': 4.07, 'RAG': 3.93, 'Math': 4.23,
#     'QA': 3.38, 'Summ.': 3.55, 'Code': 3.85, 'Average': 3.79
# }

# Qwen
PAPER_ACCEPT_LENGTH = {
    'MT': 2.86, 'Conv.': 3.72, 'RAG': 3.32, 'Math': 4.18,
    'QA': 2.97, 'Summ.': 3.24, 'Code': 3.96, 'Average': 3.46
}

# 论文报告的速度数据 (用于计算比例)
# 使用 DynaSpec-27k / FR-Spec-32k 来计算加速比例
PAPER_SPEED_DYNASPEC = {
    'MT': 89.05, 'Conv.': 104.38, 'RAG': 82.5, 'Math': 107.15,
    'QA': 84.43, 'Summ.': 82.82, 'Code': 92.92, 'Average': 91.89
}
PAPER_SPEED_FRSPEC = {
    'MT': 87.56, 'Conv.': 101.85, 'RAG': 81.81, 'Math': 107.91,
    'QA': 84.61, 'Summ.': 83.46, 'Code': 81.60, 'Average': 89.83
}

def generate_simulated_baseline_data(existing_df, baseline_label_param, ratio_factor=1.0, sim_accept_length=1.0, new_label="DynaSpec (Simulated)"):
    """
    基于论文数据和已有的基线数据，生成模拟的 Baseline 数据行。
    
    Args:
        existing_df (pd.DataFrame): 已加载的所有数据的 DataFrame。
        baseline_label_param (str): 用于在 existing_df 中查找真实基线的标签子串 (例如 "baseline")。
        new_label (str): 新生成的模拟方法的标签名称。
    Returns:
        list of dict: 生成的新模拟数据列表。
    """
    simulated_data = []
    
    # 1. 从现有数据中找到用于参考的基线数据
    # 假设只要标签里包含传入的字符串 (比如 "Baseline A (v1)" 包含 "baseline") 就认为是参考基线
    reference_df = existing_df[existing_df['method'].str.contains(baseline_label_param, case=False, na=False)]

    if reference_df.empty:
        print(f"Warning: Cannot generate simulated baseline. No existing method label contains '{baseline_label_param}'.")
        return []

    print(f"Generating simulated baseline '{new_label}' based on reference methods: {reference_df['method'].unique()}")
    
    # 2. 计算这个参考基线在每个 category 下的平均速度
    # 这里的均值用于代表该基线的“典型性能”
    ref_avg_speed = reference_df.groupby('category')['generate_speed'].mean()
    ref_avg_decoding_speed = reference_df.groupby('category')['decoding_speed'].mean()
    ref_avg_acc_length = reference_df.groupby('category')['avg_accept_length'].mean()
    
    # 获取所有涉及的 category
    categories = ref_avg_speed.index.tolist()

    for cat in categories:
        if new_label == "DynaSpec":
            # 3. 获取论文对应的列名
            paper_col = CATEGORY_MAP.get(cat, 'Average') 
            
            # 获取论文中的接受长度
            sim_accept_length = PAPER_ACCEPT_LENGTH.get(paper_col, PAPER_ACCEPT_LENGTH['Average'])
            
            # 4. 计算速度比例因子 (DynaSpec / FR-Spec)
            speed_dyna = PAPER_SPEED_DYNASPEC.get(paper_col, PAPER_SPEED_DYNASPEC['Average'])
            speed_fr = PAPER_SPEED_FRSPEC.get(paper_col, PAPER_SPEED_FRSPEC['Average'])
            
            if speed_fr != 0:
                ratio_factor = speed_dyna / speed_fr
        
        # 获取参考基线的实际平均速度
        base_speed = ref_avg_speed.get(cat)
        base_decoding_speed = ref_avg_decoding_speed.get(cat)
        base_acc_length = ref_avg_acc_length.get(cat)
        
        if base_speed is not None and np.isfinite(base_speed):
            # 5. 计算模拟速度 = 实际基线速度 * 论文比例因子
            sim_speed = base_speed * ratio_factor
            sim_decoding_speed = base_decoding_speed * ratio_factor
            sim_accept_length = sim_accept_length
            
            # 添加一条模拟数据记录
            # 注意：这里只添加了一条代表均值的记录。
            # 绘图时，这条记录的标准误(SEM)将为 0 (因为它没有 variance)。
            simulated_data.append({
                'method': new_label,
                'category': cat,
                'generate_speed': float(sim_speed),
                'decoding_speed': float(sim_decoding_speed),
                'avg_accept_length': float(sim_accept_length),
                'total_draft_time': 0
            })

    print(f"Generated {len(simulated_data)} simulated records for '{new_label}'.")
    return simulated_data

def add_labels_to_bars(ax, spacing=5, detail=True):
    """
    在柱状图上方添加每个柱子的具体数值。
    这个函数基于 matplotlib 的 patches 工作，因此可以直接复用。
    """
    for patch in ax.patches:
        # 获取柱子的高度和位置
        xy = patch.get_xy()
        width = patch.get_width()
        height = patch.get_height()
        
        # 忽略高度为 0, NaN 或无穷大的柱子
        if height in (0, float('nan')) or not np.isfinite(height):
            continue

        # 在柱子顶部中心绘制文本
        ax.annotate(
            text=f'{height:.2f}' if detail else f'{height:.0f}',          # 格式化为两位小数
            xy=(xy[0] + width / 2, height),# 文本的坐标点（柱子顶部中间）
            xytext=(0, spacing),           # 偏移量（向上移动 spacing 个点）
            textcoords='offset points',    # 坐标系
            ha='center', va='bottom',      # 水平和垂直居中对齐
            fontsize=8, fontweight='bold', color='black', # 稍微调小字体以适应密集数据
            rotation=90 if width < 0.1 else 0 # 如果柱子很窄，旋转数字以防重叠
        )

def load_single_jsonl(file_path, method_label):
    """读取单个文件，并为每一行数据打上方法标签 (保持不变)"""
    extracted_data = []
    print(f"Load file [{method_label}]: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    data_dict = json.loads(line)
                    
                    category = data_dict.get("category", "human-eval")
                    choices = data_dict.get("choices", [])
                    
                    if choices and len(choices) > 0:
                        first_choice = choices[0]
                        speed_list = first_choice.get("generate_speed", [])
                        speed = np.mean(speed_list) if speed_list else None
                        decoding_speed_list = first_choice.get("decoding_speed", [])
                        decoding_speed = np.mean(decoding_speed_list) if decoding_speed_list else None
                        total_draft_time = first_choice.get("total_draft_time(ms)", 0)
                        avg_len = first_choice.get("avg_accept_length")

                        if speed is not None and avg_len is not None:
                            # 如果数据中有 NaN 或 inf，转换为 None 以便后续处理
                            if not np.isfinite(speed): speed = None
                            if not np.isfinite(avg_len): avg_len = None

                            if speed is not None and avg_len is not None:
                                extracted_data.append({
                                    'method': method_label,
                                    'category': category,
                                    'generate_speed': float(speed),
                                    'avg_accept_length': float(avg_len),
                                    'decoding_speed': float(decoding_speed) if decoding_speed else 0,
                                    'total_draft_time': float(total_draft_time) if total_draft_time else 0
                                })
                            
                except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError) as e:
                    print(f"Error: {file_path} {e}")
                    return [] 
                     
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
        
    return extracted_data

def plot_grouped_bars(ax, df_pivot, metric_name, ylabel_text, title_text, colors_map, detail=False):
    """
    使用纯 matplotlib 绘制分组柱状图的核心辅助函数。
    df_pivot: index是category, columns是metric的mean和sem的高等dataframe
    """
    # 提取均值和误差(SEM: 标准误)数据，并unstack成宽格式：行=Category, 列=Method
    means_df = df_pivot[metric_name]['mean'].unstack(fill_value=0)
    sems_df = df_pivot[metric_name]['sem'].unstack(fill_value=0)
    
    categories = means_df.index.tolist()
    methods = means_df.columns.tolist()
    
    n_cats = len(categories)
    n_methods = len(methods)
    
    # 计算X轴基础坐标
    x_base = np.arange(n_cats)
    
    # 定义分组柱状图的总宽度和单个柱子宽度
    total_width = 0.85
    bar_width = total_width / n_methods
    
    # 循环绘制每个方法的柱子
    for i, method in enumerate(methods):
        # 计算当前方法的柱子在X轴上的偏移量
        # (i - (n_methods - 1) / 2) 这部分确保柱子组是以刻度线为中心对称的
        offset = (i - (n_methods - 1) / 2) * bar_width
        x_coords = x_base + offset
        
        means = means_df[method].values
        # 确保误差棒没有 NaN 值，否则 matplotlib 会报错
        yerrs = sems_df[method].fillna(0).values 

        ax.bar(
            x_coords, 
            means, 
            width=bar_width, 
            yerr=yerrs,           # 添加误差棒 (标准误 SEM)
            label=method, 
            color=colors_map[method],
            capsize=4,            # 误差棒顶部横线宽度
            edgecolor='white',    # 增加白色边框使柱子区分更明显
            linewidth=0.7
        )

    # 设置样式
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories, rotation=90, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调用标签函数
    # 传入更紧凑的 spacing，如果柱子很多，可能还需要调整 label 的字体大小
    add_labels_to_bars(ax, spacing=3, detail=detail) 


def visualize_comparison_matplotlib(df, model_id, bench_id):
    """接收合并后的 DataFrame，使用纯 Matplotlib 绘制对比柱状图"""
    
    # --- 1. 数据准备 (聚合) ---
    # 按 category 和 method 分组，同时计算均值(mean)和标准误(sem)
    # sem (Standard Error of the Mean) 常用于误差棒，表示均值的估计精确度 (类似于 seaborn 的 ci=68，若要 ci=95大约是 1.96*sem)
    agg_df = df.groupby(['category', 'method']).agg(['mean', 'sem'])

    # 获取所有唯一的方法名称，并为它们分配颜色
    unique_methods = sorted(df['method'].unique())
    num_methods = len(unique_methods)
    # 使用 tab10 或 tab20 颜色板，以此支持多达 10-20 种不同的 baseline
    colormap = plt.get_cmap('tab10' if num_methods <= 10 else 'tab20')
    colors_map = {method: colormap(i) for i, method in enumerate(unique_methods)}

    # --- 2. 绘图设置 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 使用 matplotlib 内置的类似风格
    fig, axes = plt.subplots(2, 2, figsize=(20 if bench_id == 'spec_bench' else 8, 8)) # 稍微增加高度给图例

    # --- 3. 绘制子图 1: Generate Speed ---
    plot_grouped_bars(
        ax=axes[0,0],
        df_pivot=agg_df,
        metric_name='generate_speed',
        ylabel_text='Avg Generate Speed (tokens/s)',
        title_text='Average Generate Speed',
        colors_map=colors_map
    )

    # --- 4. 绘制子图 2: Avg Accept Length ---
    plot_grouped_bars(
        ax=axes[0,1],
        df_pivot=agg_df,
        metric_name='avg_accept_length',
        ylabel_text='Avg Accept Length',
        title_text='Average Accept Length',
        colors_map=colors_map,
        detail=True
    )

    # --- 4. 绘制子图 2: Avg Accept Length ---
    plot_grouped_bars(
        ax=axes[1,0],
        df_pivot=agg_df,
        metric_name='decoding_speed',
        ylabel_text='Decoding Speed (tokens/s)',
        title_text='Decoding Speed',
        colors_map=colors_map
    )

    # --- 4. 绘制子图 2: Avg Accept Length ---
    plot_grouped_bars(
        ax=axes[1,1],
        df_pivot=agg_df,
        metric_name='total_draft_time',
        ylabel_text='Total Draft Time (ms)',
        title_text='Total Draft Time',
        colors_map=colors_map
    )

    # --- 5. 全局图例和布局调整 ---
    # 获取第一个子图的句柄和标签用于全局图例
    handles, labels = axes[0,0].get_legend_handles_labels()
    
    fig.legend(
        handles, 
        labels, 
        loc='lower center',         
        bbox_to_anchor=(0.5, 0.02), # 放置在底部
        ncol=min(num_methods, 5),   # 图例列数，最多5列，多了换行
        frameon=True,               
        fontsize=11,
        borderpad=0.8
    )
    plt.suptitle(f'{model_id} {bench_id}')

    # 调整布局，留出底部空间
    plt.tight_layout()
    # 根据方法数量动态调整底部边距。方法越多，图例可能占越多行。
    bottom_margin = 0.15 + (num_methods // 6) * 0.05 
    plt.subplots_adjust(bottom=bottom_margin)

    output_file = f'figs/{model_id}_{bench_id}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"-"*30)
    print(f"柱状图高度表示平均值 (Mean)。")
    print(f"黑色工字线表示标准误 (SEM)。")
    print(f"图片已保存至: {output_file}")
    print("各分类数据概览 (均值):")
    print(df.groupby(['category', 'method']).mean(numeric_only=True))
    print("-" * 30)
    print(df.groupby('method').mean(numeric_only=True))
    print("方法最终平均值 (先计算每个category的均值，再对这些均值求平均):")
    # 先计算每个category和method的平均值
    category_method_means = df.groupby(['category', 'method']).mean(numeric_only=True)
    # 再对每个method计算所有category平均值的平均值
    method_final_means = category_method_means.groupby('method').mean()
    print(method_final_means)
    print("-" * 30)


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_config = [
        {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct_eagle3/eagle3-original.jsonl', "label": "EAGLE3"},
        {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct_eagle3/eagle3-fr-spec-32768.jsonl', "label": "FRSpec"},
        {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct_eagle3/eagle3-ours.jsonl', "label": "Ours"},
    ]
    # file_config = [
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/model_answer/llama-3-8b-instruct/eagle3-fr-spec.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/model_answer/llama-3-8b-instruct/eagle3-ours.jsonl', "label": "Ours"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-ours-noasync.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-ours-1024.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-ours-2048.jsonl', "label": "Ours1"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-ours-4096.jsonl', "label": "Ours2"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3-8b-instruct/eagle-ours-8192.jsonl', "label": "Ours3"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3-8b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3-8b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3-8b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3-8b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/gsm8k/logs/llama-3-8b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/gsm8k/logs/llama-3-8b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/gsm8k/logs/llama-3-8b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3.2-1b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3.2-1b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3.2-1b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/llama-3.2-1b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3.2-1b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3.2-1b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3.2-1b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/llama-3.2-1b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/qwen2-7b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/qwen2-7b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/qwen2-7b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/spec_bench/logs/qwen2-7b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]
    # file_config = [
    #     # 替换为你真实的文件路径
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/qwen2-7b-instruct/baseline.jsonl', "label": "AR"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/qwen2-7b-instruct/eagle-fr-spec-32768.jsonl', "label": "FRSpec"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/qwen2-7b-instruct/eagle-ours.jsonl', "label": "Ours"},
    #     {"path": '/mnt/user-ssd/chenzhiyang1/workspace/Inference/FR-Spec/data/human_eval/logs/qwen2-7b-instruct/eagle-original-new-model.jsonl', "label": "EAGLE"},
    # ]

    # 2. 根据配置读取所有文件
    model_id = file_config[0]["path"].split('/')[-2]
    bench_id = file_config[0]["path"].split('/')[-4]
    all_data = []
    files_found = 0
    for config in file_config:
        if os.path.exists(config["path"]):
             data_list = load_single_jsonl(config["path"], config["label"])
             all_data.extend(data_list)
             files_found += 1
        else:
             print(f"Warning: File config skipped due to missing file: {config['path']}")

    # 3. 转换为 Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Merge specific categories into 'conversation' as requested
    categories_to_merge = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
    if not df.empty and 'category' in df.columns:
        df.loc[df['category'].isin(categories_to_merge), 'category'] = 'conversation'

    # 4. 数据校验与绘图
    if df.empty:
        print("-" * 30)
        print("错误：未提取到任何有效数据，无法绘图。请检查路径是否正确以及文件是否包含目标字段。")
        if files_found == 0:
             print("提示：没有找到任何配置文件中指定的文件。")
    else:
        print("-" * 30)
        print(f"成功从 {files_found} 个文件中加载数据。共 {len(df)} 条记录。")
        print("包含的方法 (Methods):", df['method'].unique())
        print("包含的分类 (Categories):", df['category'].unique())
        print("-" * 30)

        simulated_dynaspec_data = generate_simulated_baseline_data(
            existing_df=df,
            baseline_label_param="FRSpec",
            ratio_factor=1.02 if bench_id == 'spec_bench' else 1.07,
            sim_accept_length=3.79 if bench_id == 'spec_bench' else 3.85,
            new_label="DynaSpec"
        )
        simulated_coral_data = generate_simulated_baseline_data(
            existing_df=df,
            baseline_label_param="EAGLE",
            ratio_factor=1.06 if bench_id == 'spec_bench' else 1.12,
            sim_accept_length=3.92  if bench_id == 'spec_bench' else 5.03,
            new_label="Coral"
        )


        # 将模拟数据合并到主数据列表中
        # 使用 pd.concat 将两个 DataFrame 纵向合并
        # if simulated_dynaspec_data:
        #     df_sim = pd.DataFrame(simulated_dynaspec_data)
        #     df = pd.concat([df, df_sim], ignore_index=True)
        # if simulated_coral_data:
        #     df_sim = pd.DataFrame(simulated_coral_data)
        #     df = pd.concat([df, df_sim], ignore_index=True)
       
        # 执行新的 Matplotlib 可视化函数
        visualize_comparison_matplotlib(df, model_id, bench_id)
