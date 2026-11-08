import mne
import numpy as np
import pandas as pd  # 引入 pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

fif_path = r'D:\BCI-invention-ZBL\data\sub-001-median\sub-001_clean.fif'
results_dir = r'D:\BCI-invention-ZBL\results'
os.makedirs(results_dir, exist_ok=True)

print("1. 加载数据并执行 SSP...")
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

ecg_events = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)[0]
event_id_val = int(ecg_events[0, 2])
stim_events, _ = mne.events_from_annotations(raw_filt)

projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()

# 锁死 v2.1 窗口
tmin_ecg, tmax_ecg = -0.2, 0.6
tmin_sep, tmax_sep = 0.010, 0.060
sfreq = raw_ssp.info['sfreq']
n_pre = int(round(0.2 * sfreq))

epochs_ats = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_ats = epochs_ats.average().data
ch_names = epochs_ats.average().ch_names
ch_indices = [raw_ssp.ch_names.index(ch) for ch in ch_names]
n_times = template_ats.shape[1]

epochs_casc = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_casc = epochs_casc.average().data

ecg_times = np.linspace(tmin_ecg, tmax_ecg, template_casc.shape[1])
idx_start = np.searchsorted(ecg_times, tmin_sep)
idx_end = np.searchsorted(ecg_times, tmax_sep)

# 评价指标（返回每个通道的独立结果，而不是只有中位数）
def get_sep_amps(raw_data):
    ep = mne.Epochs(raw_data, stim_events, tmin=tmin_sep, tmax=tmax_sep, baseline=None, preload=True, reject=None, flat=None)
    return np.ptp(ep.average().data, axis=1)

def get_ri_single(raw_data, target_ch):
    ep = mne.Epochs(raw_data, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True, reject=None, flat=None)
    x = ep.get_data()
    ch_idx = raw_data.ch_names.index(target_ch)
    return np.mean(np.sqrt(np.mean(x[:, ch_idx:ch_idx+1, :]**2, axis=2)))

mask = np.array([ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) and ch != 'LM' for ch in ch_names])
esg_names = [ch for i, ch in enumerate(ch_names) if mask[i]]
base_ri = {ch: get_ri_single(raw_neural, ch) for ch in esg_names}
base_sep = get_sep_amps(raw_neural)

# 建立存储所有明细数据的列表
all_data_records = []

def process_method(raw_data, method_name, w_val):
    # 计算 SEP
    amps = get_sep_amps(raw_data)
    for i, ch in enumerate(ch_names):
        if mask[i]:
            sep_ret = (amps[i] / base_sep[i]) * 100
            ri_val = get_ri_single(raw_data, ch) / base_ri[ch]
            all_data_records.append({
                'Method': method_name,
                'Weight': w_val,
                'Channel': ch,
                'SEP_Retention_%': sep_ret,
                'RI': ri_val
            })

print("2. 计算基准与网格搜索...")
# SSP Only
process_method(raw_ssp, 'SSP Only', None)

# ATS Only
raw_ats = raw_neural.copy()
data_a = raw_ats._data
sub_a = data_a[ch_indices, :]
for samp in ecg_events[:, 0]:
    start, stop = samp - n_pre, samp - n_pre + n_times
    if start >= 0 and stop <= sub_a.shape[1]:
        sub_a[:, start:stop] -= template_ats
data_a[ch_indices, :] = sub_a
raw_ats._data = data_a
process_method(raw_ats, 'ATS Only', None)

# Gated Cascade
weights = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
for w in weights:
    weight_mask = np.ones(n_times)
    weight_mask[idx_start:idx_end] = w 
    
    raw_gated = raw_ssp.copy()
    data_g = raw_gated._data
    sub_g = data_g[ch_indices, :]
    for samp in ecg_events[:, 0]:
        start, stop = samp - n_pre, samp - n_pre + n_times
        if start >= 0 and stop <= sub_g.shape[1]:
            sub_g[:, start:stop] -= template_casc * weight_mask
            
    data_g[ch_indices, :] = sub_g
    raw_gated._data = data_g
    
    name = "Cascade (v2.1)" if w == 1.0 else f"Gated (w={w:.2f})"
    process_method(raw_gated, name, w)

# ================= 3. 将所有明细数据保存为 CSV =================
df = pd.DataFrame(all_data_records)
csv_path = os.path.join(results_dir, 'Grid_Search_Detailed_Results.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 完整通道级明细数据已保存至: {csv_path}")

# 生成汇总表打印在终端
summary_df = df.groupby('Method').agg(
    SEP_Median=('SEP_Retention_%', 'median'),
    RI_Median=('RI', 'median')
).reset_index()

# 为了排序好看，按字典序自定义排序
order = ['SSP Only', 'ATS Only'] + [f"Gated (w={w:.2f})" for w in weights[:-1]] + ['Cascade (v2.1)']
summary_df['Method'] = pd.Categorical(summary_df['Method'], categories=order, ordered=True)
summary_df = summary_df.sort_values('Method')

print("\n" + "="*50)
print(summary_df.to_string(index=False))

# ================= 4. 画图 =================
plt.figure(figsize=(8, 6))
gated_df = summary_df[summary_df['Method'].str.contains('Gated|Cascade')]
plt.plot(gated_df['RI_Median'], gated_df['SEP_Median'], marker='o', linestyle='-', color='purple', label='Gated Cascade Curve')

for idx, row in gated_df.iterrows():
    plt.annotate(row['Method'].split('w=')[-1].replace(')', ''), (row['RI_Median'], row['SEP_Median']), textcoords="offset points", xytext=(0,10), ha='center')

ssp_row = summary_df[summary_df['Method'] == 'SSP Only'].iloc[0]
ats_row = summary_df[summary_df['Method'] == 'ATS Only'].iloc[0]
plt.scatter(ssp_row['RI_Median'], ssp_row['SEP_Median'], color='green', marker='s', s=100, label='SSP Only')
plt.scatter(ats_row['RI_Median'], ats_row['SEP_Median'], color='orange', marker='^', s=100, label='ATS Only')

plt.title('Pareto Frontier: SEP vs Denoising (v2.1 Window)')
plt.xlabel('Residual Index (RI) - Lower is Better')
plt.ylabel('SEP Retention (%) - Higher is Better')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

save_path = os.path.join(results_dir, 'Pareto_Frontier_Final.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 帕累托曲线图已保存至: {save_path}")