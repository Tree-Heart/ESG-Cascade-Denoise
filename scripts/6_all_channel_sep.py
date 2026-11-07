import mne
import numpy as np
import pandas as pd
import os

fif_path = r'D:\BCI-invention-ZBL\data\sub-001-median\sub-001_clean.fif'
tmin_sep, tmax_sep = 0.010, 0.060

print("开始全通道(除LM) SEP 保留率计算...")
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

# 找心跳和刺激事件
out = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)
ecg_events = out[0]
event_id_val = int(ecg_events[0, 2])
stim_events, _ = mne.events_from_annotations(raw_filt)

# ================= 1. 生成四种数据状态 =================
# A. Raw (已经存在 raw_neural)
# B. Pure SSP
projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()

# C. Pure ATS
epochs_ats = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_ats = epochs_ats.average().data
raw_ats = raw_neural.copy()
data_a = raw_ats._data
n_pre = int(round(0.2 * raw_ats.info['sfreq']))
n_times = template_ats.shape[1]
ch_idx_a = [raw_ats.ch_names.index(ch) for ch in epochs_ats.average().ch_names]
sub_a = data_a[ch_idx_a, :]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= sub_a.shape[1]:
        sub_a[:, start:stop] -= template_ats
data_a[ch_idx_a, :] = sub_a
raw_ats._data = data_a

# D. Cascade (SSP -> ATS)
epochs_casc = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_casc = epochs_casc.average().data
raw_cascade = raw_ssp.copy()
data_c = raw_cascade._data
ch_idx_c = [raw_cascade.ch_names.index(ch) for ch in epochs_casc.average().ch_names]
sub_c = data_c[ch_idx_c, :]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= sub_c.shape[1]:
        sub_c[:, start:stop] -= template_casc
data_c[ch_idx_c, :] = sub_c
raw_cascade._data = data_c

# ================= 2. 批量切出所有通道的 SEP =================
def get_sep_amplitudes(raw_data):
    # 带着所有通道去切 SEP，这样不会触发投影器报错
    epochs = mne.Epochs(raw_data, stim_events, tmin=tmin_sep, tmax=tmax_sep, baseline=None, preload=True, reject=None, flat=None)
    sep_data = epochs.average().data # shape: (n_channels, n_times)
    # 计算每个通道的峰谷差 (ptp)
    amps = np.ptp(sep_data, axis=1)
    return amps, epochs.ch_names

print(" -> 提取 Raw SEP...")
amp_raw, ch_names = get_sep_amplitudes(raw_neural)
print(" -> 提取 SSP SEP...")
amp_ssp, _ = get_sep_amplitudes(raw_ssp)
print(" -> 提取 ATS SEP...")
amp_ats, _ = get_sep_amplitudes(raw_ats)
print(" -> 提取 Cascade SEP...")
amp_casc, _ = get_sep_amplitudes(raw_cascade)

# ================= 3. 计算保留率并保存 =================
results = []
# 使用 zip 同时遍历 5 个列表，长度会自动对齐，绝不越界
for ch, r_amp, s_amp, a_amp, c_amp in zip(ch_names, amp_raw, amp_ssp, amp_ats, amp_casc):
    if ch == 'ECG': continue # 跳过心电通道
    
    if r_amp < 1e-8: # 防止除以0
        continue
        
    results.append({
        'Channel': ch,
        'Raw_Amp_uV': round(r_amp * 1e6, 3),
        'SSP_Retention_%': round((s_amp / r_amp) * 100, 2),
        'ATS_Retention_%': round((a_amp / r_amp) * 100, 2),
        'Cascade_Retention_%': round((c_amp / r_amp) * 100, 2)
    })

df = pd.DataFrame(results)

# 🌟 补上这两句，把明细数据存下来
save_path = r'D:\BCI-invention-ZBL\results\all_channel_sep_retention.csv'
df.to_csv(save_path, index=False)
print(f"✅ 全通道(除LM)明细表已保存至: {save_path}")

# ==========================================
# 4. 分层输出：全通道 vs ESG vs EEG
# ==========================================
# 1. 筛选出脊髓/边缘通道 (ESG)
esg_df = df[df['Channel'].str.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) & (df['Channel'] != 'LM')]

# 2. 筛选出脑电通道 (EEG) —— 排除掉脊髓通道和其他非脑电通道
eeg_df = df[~df['Channel'].isin(esg_df['Channel'])]
# 剔除已知的非脑电通道（比如 BreathBelt, ECG, EP, Biceps, 等）
non_eeg_list = ['BreathBelt', 'ECG', 'EP', 'Biceps', 'Thumb', 'Toe', 'KneeM', 'Knee1', 'Knee2', 'Knee3', 'Knee4']
eeg_df = eeg_df[~eeg_df['Channel'].isin(non_eeg_list)]

print("\n" + "="*50)
print("📊 1. 全通道 (All Channels) 中位数保留率")
print(f"   SSP:     {df['SSP_Retention_%'].median():.2f}%")
print(f"   ATS:     {df['ATS_Retention_%'].median():.2f}%")
print(f"   Cascade: {df['Cascade_Retention_%'].median():.2f}%")

print("\n" + "="*50)
print("🧠 2. 纯脑电通道 (EEG Channels) 中位数保留率")
print(f"   SSP:     {eeg_df['SSP_Retention_%'].median():.2f}%")
print(f"   ATS:     {eeg_df['ATS_Retention_%'].median():.2f}%")
print(f"   Cascade: {eeg_df['Cascade_Retention_%'].median():.2f}%")

print("\n" + "="*50)
print("⚡ 3. 纯脊髓通道 (ESG Channels) 中位数保留率")
print(f"   SSP:     {esg_df['SSP_Retention_%'].median():.2f}%")
print(f"   ATS:     {esg_df['ATS_Retention_%'].median():.2f}%")
print(f"   Cascade: {esg_df['Cascade_Retention_%'].median():.2f}%")
print("="*50 + "\n")

# 5. 把分层中位数保存成一个单独的表格，并附带标准差和四分位数
def get_stats(sub_df, method):
    return pd.Series({
        f'{method}_Mean': sub_df[f'{method}_Retention_%'].mean(),
        f'{method}_Std': sub_df[f'{method}_Retention_%'].std(),
        f'{method}_Median': sub_df[f'{method}_Retention_%'].median(),
        f'{method}_Q1': sub_df[f'{method}_Retention_%'].quantile(0.25),
        f'{method}_Q3': sub_df[f'{method}_Retention_%'].quantile(0.75)
    })

# 分别计算三组的统计量
stats_all = pd.concat([get_stats(df, 'SSP'), get_stats(df, 'ATS'), get_stats(df, 'Cascade')])
stats_eeg = pd.concat([get_stats(eeg_df, 'SSP'), get_stats(eeg_df, 'ATS'), get_stats(eeg_df, 'Cascade')])
stats_esg = pd.concat([get_stats(esg_df, 'SSP'), get_stats(esg_df, 'ATS'), get_stats(esg_df, 'Cascade')])

summary_df = pd.DataFrame({
    'All': stats_all,
    'EEG': stats_eeg,
    'ESG': stats_esg
}).T.round(2)

summary_path = r'D:\BCI-invention-ZBL\results\sep_retention_stats.csv'
summary_df.to_csv(summary_path)
print(f"✅ 统计学汇总表已保存至: {summary_path}")