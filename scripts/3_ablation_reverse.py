import mne
import numpy as np
import pandas as pd
import os

fif_path = r'D:\BCI-invention-ZBL\data\sub-001-median\sub-001_clean.fif'
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
out = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)
ecg_events = out[0]
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

esg_names = [ch for ch in raw_filt.ch_names 
             if ch.startswith('S') or ch.startswith('L') or ch in ('AC', 'AL', 'IZ', 'SC1', 'SC6')]
event_id_val = int(ecg_events[0, 2])

# 算 RI 的函数
def calc_ri(raw_data, target_ch):
    ep = mne.Epochs(raw_data, ecg_events, event_id=event_id_val,
                    tmin=-0.2, tmax=0.6, baseline=None, preload=True, reject=None, flat=None)
    x = ep.get_data()
    ch_idx = raw_data.ch_names.index(target_ch)
    return np.mean(np.sqrt(np.mean(x[:, ch_idx:ch_idx+1, :]**2, axis=2)))

print("正在执行全通道消融实验，请稍候...")

# --- 正向：先 SSP 后 ATS ---
projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()
epochs_ssp = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_ssp = epochs_ssp.average().data
raw_forward = raw_ssp.copy()
data_f = raw_forward._data
ch_idx_f = [raw_forward.ch_names.index(ch) for ch in epochs_ssp.average().ch_names]
n_pre = int(round(0.2 * raw_forward.info['sfreq']))
n_times = template_ssp.shape[1]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= data_f.shape[1]:
        data_f[ch_idx_f, start:stop] -= template_ssp
raw_forward._data = data_f

# --- 反向：先 ATS 后 SSP ---
epochs_ats = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_ats = epochs_ats.average().data
raw_ats = raw_neural.copy()
data_a = raw_ats._data
ch_idx_a = [raw_ats.ch_names.index(ch) for ch in epochs_ats.average().ch_names]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= data_a.shape[1]:
        data_a[ch_idx_a, start:stop] -= template_ats
raw_ats._data = data_a
# 在 ATS 结果上算 SSP
projs_rev, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_reverse = raw_ats.copy().add_proj(projs_rev).apply_proj()

# --- 全通道遍历 ---
results = []
base = {ch: calc_ri(raw_neural, ch) for ch in esg_names}

for ch in esg_names:
    ri_f = calc_ri(raw_forward, ch) / base[ch]
    ri_r = calc_ri(raw_reverse, ch) / base[ch]
    results.append({
        'Channel': ch,
        'RI_Forward(SSP->ATS)': round(ri_f, 4),
        'RI_Reverse(ATS->SSP)': round(ri_r, 4),
        'Difference': round(abs(ri_f - ri_r), 4)
    })

df = pd.DataFrame(results)
csv_path = r'D:\BCI-invention-ZBL\results\ablation_sub001_median.csv'
df.to_csv(csv_path, index=False)

print("\n✅ 全通道消融实验完成！结果已保存至:", csv_path)
print("\n--- 前 10 行数据预览 ---")
print(df.head(10))
print("\n最大差异值:", df['Difference'].max())