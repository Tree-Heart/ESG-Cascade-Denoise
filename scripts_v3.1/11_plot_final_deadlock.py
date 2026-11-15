"""
11_plot_final_deadlock.py
生成终极帕累托死锁图：保留所有散点 + 高亮帕累托前沿
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 路径设置
results_dir = r'D:\BCI-invention-ZBL\results_v3.1'
sub_csv = os.path.join(results_dir, 'v3.1_Oracle_Exhaustive_60.csv')
hyb_csv = os.path.join(results_dir, 'v3.1_Hybrid_Laws_Exhaustive.csv')
plot_path = os.path.join(results_dir, 'v3.1_Final_Deadlock_Frontier.png')

df_sub = pd.read_csv(sub_csv)
df_hyb = pd.read_csv(hyb_csv)

df_sub = df_sub[df_sub['Chain'] != 'NO_OP']

# 提取帕累托前沿的函数（不删数据，只用来高亮）
def get_pareto_frontier(df, x_col, y_col):
    df_sorted = df.sort_values(by=[x_col, y_col], ascending=[False, False])
    frontier = []
    max_y = -float('inf')
    for _, row in df_sorted.iterrows():
        if row[y_col] > max_y:
            frontier.append(row)
            max_y = row[y_col]
    return pd.DataFrame(frontier)

frontier_sub = get_pareto_frontier(df_sub, 'CC_GT_contaminated', 'SSIM_clean')
frontier_hyb = get_pareto_frontier(df_hyb, 'CC_GT_contaminated', 'SSIM_clean')

plt.figure(figsize=(10, 7))

# 1. 画背景：所有散点全保留，透明度调高，展示数据全貌
plt.scatter(df_sub['CC_GT_contaminated'], df_sub['SSIM_clean'], 
            c='blue', alpha=0.25, marker='o', s=40, label='All Subtraction Cascades')
plt.scatter(df_hyb['CC_GT_contaminated'], df_hyb['SSIM_clean'], 
            c='red', alpha=0.25, marker='X', s=40, label='All Hybrid Cascades')

# 2. 画前景：高亮出帕累托前沿边界
plt.plot(frontier_sub['CC_GT_contaminated'], frontier_sub['SSIM_clean'], 
         color='blue', linewidth=2.5, marker='o', label='Subtraction Pareto Frontier')
plt.plot(frontier_hyb['CC_GT_contaminated'], frontier_hyb['SSIM_clean'], 
         color='red', linewidth=2.5, marker='X', linestyle='--', label='Hybrid Laws Pareto Frontier')

# 画及格线
plt.axhline(0.5, color='black', linestyle=':', alpha=0.7, label='SSIM = 0.5 (Morphology Pass)')
plt.axvline(0.5, color='black', linestyle=':', alpha=0.7, label='CC = 0.5 (Artifact Pass)')

plt.title('Exhaustive Search: Linear Operations hit the Morphological Wall', fontsize=14, fontweight='bold')
plt.xlabel('CC_GT_contaminated (Artifact Removal) - Higher is Better', fontsize=12)
plt.ylabel('SSIM_clean (Morphological Fidelity) - Higher is Better', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ 全量数据及高亮前沿图已保存至: {plot_path}")