import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from pathlib import Path
import os

BASE_DIR = r'D:\BCI-invention-ZBL'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

print("正在读取并合并所有 Cross-Task 数据...")
all_csvs = list(Path(RESULTS_DIR).glob('batch_ri_results_*.csv'))

dfs = []
for f in all_csvs:
    df = pd.read_csv(f)
    filename = f.stem
    subject_id = filename.replace('batch_ri_results_', '')
    
    if '_median' in subject_id:
        task = 'Median'
        subject = subject_id.replace('_median', '')
    else:
        task = 'Tibial'
        subject = subject_id
        
    df['Subject'] = subject
    df['Task'] = task
    dfs.append(df)

if not dfs:
    print("❌ 没有找到任何结果 CSV 文件。")
    exit()

big_df = pd.concat(dfs, ignore_index=True)
print(f"✅ 成功合并 {len(dfs)} 个文件！")
print(f"📊 总计样本数: {len(big_df)} 个通道 (来自 {len(big_df['Subject'].unique())} 个独立受试者)")

print("\n--- 全局统计学检验 (Wilcoxon 符号秩检验) ---")
stat_ssp, p_ssp = wilcoxon(big_df['RI_SSP'], big_df['RI_Cascade'])
stat_ats, p_ats = wilcoxon(big_df['RI_ATS'], big_df['RI_Cascade'])
print(f"级联 vs 仅SSP 的 p 值: {p_ssp:.2e}")
print(f"级联 vs 仅ATS 的 p 值: {p_ats:.2e}")

df_melted = big_df.melt(
    id_vars=['Channel', 'Subject', 'Task'], 
    value_vars=['RI_SSP', 'RI_ATS', 'RI_Cascade'], 
    var_name='Method', value_name='RI'
)

df_melted['Method'] = df_melted['Method'].replace({
    'RI_SSP': 'SSP Only',
    'RI_ATS': 'ATS Only',
    'RI_Cascade': 'Cascade (Ours)'
})

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

sns.boxplot(x='Method', y='RI', data=df_melted, palette="Set2", width=0.5, 
            showfliers=False, ax=ax)

# 核心修正：分别绘制 Tibial 和 Median 的散点，用不同 marker 区分
sns.stripplot(x='Method', y='RI', data=df_melted[df_melted['Task'] == 'Tibial'], 
              color='black', marker='o', size=5, alpha=0.6, jitter=True, dodge=False, 
              ax=ax, label='Tibial')

sns.stripplot(x='Method', y='RI', data=df_melted[df_melted['Task'] == 'Median'], 
              color='dimgray', marker='X', size=5, alpha=0.6, jitter=True, dodge=False, 
              ax=ax, label='Median')

plt.title('Cross-Task Validation: Overall RI Comparison (3 Subjects, 2 Tasks, 234 Channels)', fontsize=16, pad=15)
plt.xlabel('Denoising Methods', fontsize=14)
plt.ylabel('Residual Index (RI) - Lower is Better', fontsize=14)

plt.ylim(0, 0.6) 
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 处理图例，自动去重
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # 利用字典键唯一的特性去重
ax.legend(by_label.values(), by_label.keys(), title='Stimulation Task', loc='upper right')

# 补上保存图片的代码
plt.tight_layout()
save_plot_path = os.path.join(RESULTS_DIR, 'Fig3_Cross_Task_Validation.png')
plt.savefig(save_plot_path)
print(f"\n✅ 终极交叉验证箱线图已保存到: {save_plot_path}")