import mne
import numpy as np
import pandas as pd
import os

SUBJECTS = ['sub-001', 'sub-012', 'sub-018']
BASE_DIR = r'D:\BCI-invention-ZBL'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 确保 results 文件夹存在
os.makedirs(RESULTS_DIR, exist_ok=True)

def locked_rms(raw_data, events, event_id_val, target_ch):
    ep = mne.Epochs(raw_data, events, event_id=event_id_val,
                    tmin=-0.2, tmax=0.6, baseline=None,
                    preload=True, reject_by_annotation=False,
                    reject=None, flat=None)
    x = ep.get_data()
    ch_idx = raw_data.ch_names.index(target_ch)
    x = x[:, ch_idx:ch_idx+1, :] 
    return np.mean(np.sqrt(np.mean(x**2, axis=2)))

for sub in SUBJECTS:
    print(f"\n========== 正在处理受试者: {sub} ==========")
    fif_path = os.path.join(BASE_DIR, 'data', sub, f'{sub}_clean.fif')
    csv_path = os.path.join(RESULTS_DIR, f'batch_ri_results_{sub}.csv')

    if not os.path.exists(fif_path):
        print(f"⚠️ 找不到 {sub} 的 fif 文件，跳过。")
        continue

    raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
    
    # 1. 检测心跳
    out = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)
    ecg_events = out[0]

    # 提取脊髓通道
    esg_names = [ch for ch in raw_filt.ch_names 
                 if ch.startswith('S') or ch.startswith('L') or ch in ('AC', 'AL', 'IZ', 'SC1', 'SC6')]
    print(f"发现 {len(esg_names)} 个脊髓通道...")

    # 剥离 ECG
    raw_filt.info['bads'] = []
    raw_neural = raw_filt.copy().drop_channels(['ECG'])
    raw_neural.info['bads'] = []

    # 路线 A：SSP
    projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
    raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()

    # 路线 B：ATS
    event_id_val = int(ecg_events[0, 2])
    epochs = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val,
                        tmin=-0.2, tmax=0.6, baseline=(-0.2, 0),
                        preload=True, reject_by_annotation=False, reject=None, flat=None)
    template = epochs.average().data

    raw_ats = raw_neural.copy()
    data = raw_ats._data
    sfreq = raw_ats.info['sfreq']
    n_pre = int(round(0.2 * sfreq))
    n_times = template.shape[1]

    template_ch_names = epochs.average().ch_names
    ch_indices = [raw_ats.ch_names.index(ch) for ch in template_ch_names]
    data_subset = data[ch_indices, :]

    n_applied, n_skipped = 0, 0
    for samp in ecg_events[:, 0]:
        start = samp - n_pre
        stop = start + n_times
        if start >= 0 and stop <= data_subset.shape[1]:
            data_subset[:, start:stop] -= template
            n_applied += 1
        else:
            n_skipped += 1
    data[ch_indices, :] = data_subset
    raw_ats._data = data

    # 路线 C：级联
    epochs_ssp = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val,
                            tmin=-0.2, tmax=0.6, baseline=(-0.2, 0),
                            preload=True, reject_by_annotation=False, reject=None, flat=None)
    template_ssp = epochs_ssp.average().data

    raw_cascade = raw_ssp.copy()
    data_c = raw_cascade._data
    template_ssp_ch_names = epochs_ssp.average().ch_names
    ch_indices_ssp = [raw_cascade.ch_names.index(ch) for ch in template_ssp_ch_names]
    data_subset_ssp = data_c[ch_indices_ssp, :]

    for samp in ecg_events[:, 0]:
        start = samp - n_pre
        stop = start + n_times
        if start >= 0 and stop <= data_subset_ssp.shape[1]:
            data_subset_ssp[:, start:stop] -= template_ssp
    data_c[ch_indices_ssp, :] = data_subset_ssp
    raw_cascade._data = data_c

    # 计算 RI
    results = []
    for i, ch in enumerate(esg_names):
        base = locked_rms(raw_neural, ecg_events, event_id_val, ch)
        ri_ssp = locked_rms(raw_ssp, ecg_events, event_id_val, ch) / base
        ri_ats = locked_rms(raw_ats, ecg_events, event_id_val, ch) / base
        ri_cas = locked_rms(raw_cascade, ecg_events, event_id_val, ch) / base
        
        results.append({
            'Channel': ch,
            'RI_SSP': round(ri_ssp, 4),
            'RI_ATS': round(ri_ats, 4),
            'RI_Cascade': round(ri_cas, 4)
        })

    df = pd.DataFrame(results)
    def get_best(row):
        vals = {'SSP': row['RI_SSP'], 'ATS': row['RI_ATS'], 'Cascade': row['RI_Cascade']}
        return min(vals, key=vals.get)
    df['Best_Method'] = df.apply(get_best, axis=1)

    df.to_csv(csv_path, index=False)
    print(f"✅ {sub} 处理完成，结果已保存至 {csv_path}")