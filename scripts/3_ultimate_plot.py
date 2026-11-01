import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
import numpy as np
from pathlib import Path
import os
import mne

BASE_DIR = r'D:\BCI-invention-ZBL'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ==========================================
# 第一部分：画箱线图 (Figure 1)
# ==========================================
print("正在读取并合并所有受试者数据...")
all_csvs = Path(RESULTS_DIR).glob('batch_ri_results_*.csv')

dfs = []
for f in all_csvs:
    df = pd.read_csv(f)
    subject_id = f.stem.replace('batch_ri_results_', '')
    df['Subject'] = subject_id
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)
print(f"✅ 成功合并 {len(dfs)} 个受试者的数据，共 {len(big_df)} 个通道样本！")

df_melted = big_df.melt(
    id_vars=['Channel', 'Subject'],
    value_vars=['RI_SSP', 'RI_ATS', 'RI_Cascade'],
    var_name='Method', value_name='RI'
)
df_melted['Method'] = df_melted['Method'].replace({
    'RI_SSP': 'SSP Only',
    'RI_ATS': 'ATS Only',
    'RI_Cascade': 'Cascade (Ours)'
})

print("\n--- 统计学检验 (Wilcoxon) ---")
stat_ssp, p_ssp = wilcoxon(big_df['RI_SSP'], big_df['RI_Cascade'])
stat_ats, p_ats = wilcoxon(big_df['RI_ATS'], big_df['RI_Cascade'])
print(f"级联 vs 仅SSP 的 p 值: {p_ssp:.2e}")
print(f"级联 vs 仅ATS 的 p 值: {p_ats:.2e}")

print("\n--- 线性混合效应模型 (LMM) ---")
md = smf.mixedlm("RI ~ Method", df_melted, groups=df_melted["Subject"])
mdf = md.fit()
print(mdf.summary())

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
sns.boxplot(x='Method', y='RI', data=df_melted, palette="Set2", width=0.5, showfliers=False, ax=ax)
sns.stripplot(x='Method', y='RI', data=df_melted, color="black", size=4, alpha=0.6, jitter=True, dodge=True, ax=ax)

plt.title('Overall Residual Index (RI) Comparison\nAcross Subjects and Spinal Channels', fontsize=16, pad=15)
plt.xlabel('Denoising Methods', fontsize=14)
plt.ylabel('Residual Index (RI) - Lower is Better', fontsize=14)
plt.ylim(0, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
save_plot_path = os.path.join(RESULTS_DIR, 'Fig3_Ultimate_Comparison.png')
plt.savefig(save_plot_path)
print(f"\n✅ 终极箱线图已保存到: {save_plot_path}")


# ==========================================
# 第二部分：画波形图 (Figure 2)
# ==========================================
print("\n正在生成 Figure 2 (波形对比图)...")
# 我们拿 sub-001 的 S17 通道来做个示范
fif_path = os.path.join(BASE_DIR, 'data', 'sub-001', 'sub-001_clean.fif')
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
out = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)
ecg_events = out[0]

target_ch = 'S17'
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []
raw_filt.info['bads'] = []

# 1. SSP
projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()

# 2. ATS
event_id_val = int(ecg_events[0, 2])
epochs = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject_by_annotation=False, reject=None, flat=None)
template = epochs.average().data
raw_ats = raw_neural.copy()
data = raw_ats._data
sfreq = raw_ats.info['sfreq']
n_pre = int(round(0.2 * sfreq))
n_times = template.shape[1]
ch_indices = [raw_ats.ch_names.index(ch) for ch in epochs.average().ch_names]
data_subset = data[ch_indices, :]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= data_subset.shape[1]:
        data_subset[:, start:stop] -= template
data[ch_indices, :] = data_subset
raw_ats._data = data

# 3. Cascade
epochs_ssp = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject_by_annotation=False, reject=None, flat=None)
template_ssp = epochs_ssp.average().data
raw_cascade = raw_ssp.copy()
data_c = raw_cascade._data
ch_indices_ssp = [raw_cascade.ch_names.index(ch) for ch in epochs_ssp.average().ch_names]
data_subset_ssp = data_c[ch_indices_ssp, :]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= data_subset_ssp.shape[1]:
        data_subset_ssp[:, start:stop] -= template_ssp
data_c[ch_indices_ssp, :] = data_subset_ssp
raw_cascade._data = data_c

# 截取 0-5 秒的数据进行可视化
tmin, tmax = 0, 5
start_samp, end_samp = int(tmin * sfreq), int(tmax * sfreq)
times = np.arange(start_samp, end_samp) / sfreq

idx_ch = raw_neural.ch_names.index(target_ch)

fig2, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axs[0].plot(times, raw_neural._data[idx_ch, start_samp:end_samp], color='black', linewidth=0.8)
axs[0].set_title('Original (Raw)', loc='left', fontsize=12)
axs[1].plot(times, raw_ssp._data[idx_ch, start_samp:end_samp], color='blue', linewidth=0.8)
axs[1].set_title('SSP Only', loc='left', fontsize=12)
axs[2].plot(times, raw_ats._data[idx_ch, start_samp:end_samp], color='green', linewidth=0.8)
axs[2].set_title('ATS Only', loc='left', fontsize=12)
axs[3].plot(times, raw_cascade._data[idx_ch, start_samp:end_samp], color='red', linewidth=0.8)
axs[3].set_title('Cascade (SSP -> ATS)', loc='left', fontsize=12)

for ax in axs:
    for event in ecg_events:
        event_time = event[0] / sfreq
        if tmin <= event_time <= tmax:
            ax.axvline(x=event_time, color='orange', linestyle='--', alpha=0.7)
    ax.set_ylabel('Amplitude')

plt.xlabel('Time (s)')
plt.tight_layout()
save_fig2_path = os.path.join(RESULTS_DIR, 'Fig2_Waveform_Comparison.png')
plt.savefig(save_fig2_path)
print(f"✅ 波形图已保存到: {save_fig2_path}")
