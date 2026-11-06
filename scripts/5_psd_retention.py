import mne
import numpy as np
import os

fif_path = r'D:\BCI-invention-ZBL\data\sub-001-median\sub-001_clean.fif'
target_ch = 'S17'
tmin_sep, tmax_sep = 0.010, 0.060

print("开始计算三种方法的 SEP 波幅保留率...")
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

# 提取心跳和刺激事件
out = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)
ecg_events = out[0]
event_id_val = int(ecg_events[0, 2])
stim_events, _ = mne.events_from_annotations(raw_filt)

# 通用的 SEP 保留率计算函数
def calc_sep_retention(raw_data, name):
    epochs = mne.Epochs(raw_data, stim_events, tmin=tmin_sep, tmax=tmax_sep, 
                        baseline=None, preload=True, reject=None, flat=None)
    sep_data = epochs.average().data
    ch_idx = epochs.ch_names.index(target_ch)
    amp = np.ptp(sep_data[ch_idx, :])
    print(f"   -> {name} 的 SEP 峰谷差: {amp*1e6:.2f} uV")
    return amp

# --- 路线 1：纯 SSP ---
print("\n1. 处理纯 SSP...")
projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()

# --- 路线 2：纯 ATS ---
print("2. 处理纯 ATS...")
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

# --- 路线 3：级联 (SSP -> ATS) ---
print("3. 处理级联...")
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

# --- 开始对比 ---
print("\n--- 最终结果对比 ---")
amp_raw = calc_sep_retention(raw_neural, "原始数据(Raw)")
amp_ssp = calc_sep_retention(raw_ssp, "纯 SSP")
amp_ats = calc_sep_retention(raw_ats, "纯 ATS")
amp_casc = calc_sep_retention(raw_cascade, "级联(Cascade)")

print("\n" + "="*40)
print(f"纯 SSP 保留率:  {(amp_ssp/amp_raw)*100:.2f}%")
print(f"纯 ATS 保留率:  {(amp_ats/amp_raw)*100:.2f}%")
print(f"级联保留率:    {(amp_casc/amp_raw)*100:.2f}%")
print("="*40)