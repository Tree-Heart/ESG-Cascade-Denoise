import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

csv_path = r'D:\BCI-invention-ZBL\results\all_channel_sep_retention.csv'
df = pd.read_csv(csv_path)

def get_group(ch):
    if ch == 'LM': return 'Drop'
    if ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')): return 'ESG'
    if ch in ['BreathBelt', 'ECG', 'EP', 'Biceps', 'Thumb', 'Toe', 'KneeM', 'Knee1', 'Knee2', 'Knee3', 'Knee4']: return 'Drop'
    return 'EEG'

df['Group'] = df['Channel'].apply(get_group)
df = df[df['Group'] != 'Drop'] 

df_melted = df.melt(
    id_vars=['Channel', 'Group'], 
    value_vars=['SSP_Retention_%', 'ATS_Retention_%', 'Cascade_Retention_%'], 
    var_name='Method', value_name='Retention'
)
df_melted['Method'] = df_melted['Method'].replace({
    'SSP_Retention_%': 'SSP Only',
    'ATS_Retention_%': 'ATS Only',
    'Cascade_Retention_%': 'Cascade'
})

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 图 4：EEG =================
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
eeg_data = df_melted[df_melted['Group'] == 'EEG']
sns.boxplot(x='Method', y='Retention', data=eeg_data, palette="Pastel1", width=0.5, showfliers=False, ax=ax1)
sns.stripplot(x='Method', y='Retention', data=eeg_data, color="black", size=4, alpha=0.6, jitter=True, ax=ax1)
ax1.set_title('EEG Channels: Signal Preservation', fontsize=14, pad=15)
ax1.set_ylabel('SEP Amplitude Retention (%)', fontsize=12)
ax1.set_xlabel('Denoising Methods', fontsize=12)
ax1.set_ylim(90, 105)
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path_eeg = r'D:\BCI-invention-ZBL\results\Fig4_EEG_Retention.png'
plt.savefig(save_path_eeg)
print(f"✅ Figure 4 (EEG) 已保存至: {save_path_eeg}")

# ================= 图 5：ESG =================
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
esg_data = df_melted[df_melted['Group'] == 'ESG']
sns.boxplot(x='Method', y='Retention', data=esg_data, palette="Set2", width=0.5, showfliers=False, ax=ax2)
sns.stripplot(x='Method', y='Retention', data=esg_data, color="black", size=5, alpha=0.7, jitter=True, ax=ax2)
ax2.set_title('ESG Channels: Severe Signal Attenuation', fontsize=14, pad=15)
ax2.set_ylabel('SEP Amplitude Retention (%)', fontsize=12)
ax2.set_xlabel('Denoising Methods', fontsize=12)
ax2.set_ylim(0, 110)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path_esg = r'D:\BCI-invention-ZBL\results\Fig5_ESG_Retention.png'
plt.savefig(save_path_esg)
print(f"✅ Figure 5 (ESG) 已保存至: {save_path_esg}")