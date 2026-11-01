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

print("正在读取并合并所有受试者数据...")
# 自动寻找 results 文件夹下的所有结果
all_csvs = Path(RESULTS_DIR).glob('batch_ri_results_*.csv')

dfs = []
for f in all_csvs:
    df = pd.read_csv(f)
    # 从文件名中提取受试者 ID，例如 batch_ri_results_sub-001.csv -> sub-001
    subject_id = f.stem.replace('batch_ri_results_', '')
    df['Subject'] = subject_id
    dfs.append(df)

if not dfs:
    print("❌ 没有找到任何结果 CSV 文件。")
    exit()

big_df = pd.concat(dfs, ignore_index=True)
print(f"✅ 成功合并 {len(dfs)} 个受试者的数据，共 {len(big_df)} 个通道样本！")

# 统计学检验
stat_ssp, p_ssp = wilcoxon(big_df['RI_SSP'], big_df['RI_Cascade'])
stat_ats, p_ats = wilcoxon(big_df['RI_ATS'], big_df['RI_Cascade'])

print("\n--- 终极统计学检验 (Wilcoxon 符号秩检验) ---")
print(f"级联 vs 仅SSP 的 p 值: {p_ssp:.2e}")
print(f"级联 vs 仅ATS 的 p 值: {p_ats:.2e}")

# 画图
df_melted = big_df.melt(id_vars=['Channel', 'Subject'], 
                    value_vars=['RI_SSP', 'RI_ATS', 'RI_Cascade'], 
                    var_name='Method', value_name='RI')

df_melted['Method'] = df_melted['Method'].replace({
    'RI_SSP': 'SSP Only',
    'RI_ATS': 'ATS Only',
    'RI_Cascade': 'Cascade (Ours)'
})

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

sns.boxplot(x='Method', y='RI', data=df_melted, palette="Set2", width=0.5, 
            showfliers=False, ax=ax) 

sns.stripplot(x='Method', y='RI', data=df_melted, hue='Subject', 
              size=4, alpha=0.6, jitter=True, dodge=True, ax=ax)

plt.title('Overall Residual Iteration (RI) Comparison\nAcross Subjects and Spinal Channels', fontsize=16, pad=15)
plt.xlabel('Denoising Methods', fontsize=14)
plt.ylabel('Residual Iteration (RI) - Lower is Better', fontsize=14)

plt.ylim(0, 0.5) 
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
save_plot_path = os.path.join(RESULTS_DIR, 'Fig3_Ultimate_Comparison.png')
plt.savefig(save_plot_path)
print(f"\n✅ 终极箱线图已保存到: {save_plot_path}")