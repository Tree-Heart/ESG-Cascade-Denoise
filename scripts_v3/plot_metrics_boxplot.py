import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR = r'D:\BCI-invention-ZBL'
RESULTS_DIR = os.path.join(BASE_DIR, 'results_v3')
csv_path = os.path.join(RESULTS_DIR, 'v3.0_All_228_Channels_Metrics.csv')

print("📊 正在读取数据并生成 7 个指标的箱线图...")
df_final = pd.read_csv(csv_path)

# 方法名映射（去掉 Pure, Clean, v2.1 等后缀，保持和论文一致）
method_map = {
    'Pure SSP': 'SSP', 
    'Clean ATS': 'ATS', 
    'Cascade v2.1': 'Cascade', 
    'Slew-Rate v3.0': 'Slew-Rate'
}
df_final['Method_Display'] = df_final['Method'].map(method_map)

# 修复：按行数统计真实的样本量（228），而不是去重后的通道名（38）
n_channels = len(df_final[df_final['Method_Display'] == 'SSP'])

methods_order = ['SSP', 'ATS', 'Cascade', 'Slew-Rate']
metrics = ['RI', 'CC_GT', 'CC_WB', 'RE', 'SD', 'KR2', 'SSIM']
colors = ['green', 'orange', 'blue', 'purple']

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    data_to_plot = [df_final[df_final['Method_Display'] == m][metric].dropna().values for m in methods_order]
    bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=methods_order, showfliers=False)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    # 🚨 修复：标题动态显示 n=228，且方法名标准化
    ax.set_title(f'{metric} (n={n_channels})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=15)
    
    if metric in ['CC_GT', 'CC_WB', 'SSIM']: ax.axhline(0, color='black', linestyle=':', alpha=0.7)
    if metric == 'KR2': ax.axhline(1, color='black', linestyle=':', alpha=0.7)
# 统计 SSP 方法有多少行数据（也就是真正的样本量）
n_channels = len(df_final[df_final['Method_Display'] == 'SSP'])
# 删掉多余的两个空子图
fig.delaxes(axes[7])
fig.delaxes(axes[8])

plt.tight_layout()
# 文件名也同步改一下，防止和旧的混淆
plot_path = os.path.join(RESULTS_DIR, 'v3.0_All_228_Channels_Boxplot_Final.png')
plt.savefig(plot_path, dpi=300)
print(f"✅ 最终版箱线图已保存至: {plot_path}")