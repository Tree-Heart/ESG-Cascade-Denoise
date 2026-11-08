import mne
import numpy as np
import pandas as pd
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

epochs_casc = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True, reject=None, flat=None)
template_casc = epochs_casc.average().data
ch_names = epochs_casc.average().ch_names
ch_indices = [raw_ssp.ch_names.index(ch) for ch in ch_names]
n_times = template_casc.shape[1]

ecg_times = np.linspace(tmin_ecg, tmax_ecg, template_casc.shape[1])
idx_start = np.searchsorted(ecg_times, tmin_sep)
idx_end = np.searchsorted(ecg_times, tmax_sep)

# 评价指标函数
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

def process_method(raw_data, method_name, w_val):
    amps = get_sep_amps(raw_data)
    retentions = []
    ris = []
    for i, ch in enumerate(ch_names):
        if mask[i]:
            retentions.append((amps[i] / base_sep[i]) * 100)
            ris.append(get_ri_single(raw_data, ch) / base_ri[ch])
    return np.median(retentions), np.median(ris)

# ================= 核心：从 0.0 到 1.0，步长 0.05 =================
weights = np.arange(0.0, 1.05, 0.05)
results_list = []

print("2. 开始全量扫描 (w=0.0 到 1.0)...")
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
    
    sep_g, ri_g = process_method(raw_gated, f"w={w:.2f}", w)
    results_list.append({'Weight': w, 'SEP': sep_g, 'RI': ri_g})
    print(f"  w={w:.2f} | SEP: {sep_g:.2f}% | RI: {ri_g:.4f}")

df = pd.DataFrame(results_list)
csv_path = os.path.join(results_dir, 'Grid_0_to_1_Detailed.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 数据已保存至: {csv_path}")

# ================= 画图 =================
# 图1：帕累托散点图
plt.figure(figsize=(8, 6))
plt.scatter(df['RI'], df['SEP'], c=df['Weight'], cmap='viridis', s=80, edgecolors='black', zorder=3)
for idx, row in df.iterrows():
    if idx % 2 == 0 or idx == len(df)-1: # 每隔一个标一个，防止重叠
        plt.annotate(f"{row['Weight']:.2f}", (row['RI'], row['SEP']), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(df['Weight'].min(), df['Weight'].max())), label='Weight (w)')
plt.title('Pareto Frontier (w=0.0 to 1.0)')
plt.xlabel('Residual Index (RI) - Lower is Better')
plt.ylabel('SEP Retention (%) - Higher is Better')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(results_dir, 'Pareto_Scatter_0_to_1.png'), dpi=300, bbox_inches='tight')
plt.close()

# 图2：双 Y 轴折线图
fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = 'tab:purple'
ax1.set_xlabel('Weight (w) - Proportion of ATS subtraction in SEP window')
ax1.set_ylabel('SEP Retention (%)', color=color1)
ax1.plot(df['Weight'], df['SEP'], marker='o', color=color1, label='SEP')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Residual Index (RI)', color=color2)
ax2.plot(df['Weight'], df['RI'], marker='s', color=color2, linestyle='--', label='RI')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Trade-off Analysis: SEP Retention vs. Denoising (w=0.0 to 1.0)')
fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'Tradeoff_Line_0_to_1.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 帕累托散点图与双Y轴折线图已保存至 results 文件夹。")