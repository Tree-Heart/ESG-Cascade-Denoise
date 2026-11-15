import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d

# 路径设置
BASE_DIR = r'D:\BCI-invention-ZBL'
fif_path = os.path.join(BASE_DIR, 'data', 'sub-001-median', 'sub-001_clean.fif')
results_dir = os.path.join(BASE_DIR, 'results_v3.1')
os.makedirs(results_dir, exist_ok=True)

def calc_ssim_1d(x, y):
    L = max(np.ptp(x), np.ptp(y), 1e-6)
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    mu_x, mu_y = uniform_filter1d(x, size=5), uniform_filter1d(y, size=5)
    sigma_x = uniform_filter1d(x**2, size=5) - mu_x**2
    sigma_y = uniform_filter1d(y**2, size=5) - mu_y**2
    sigma_xy = uniform_filter1d(x*y, size=5) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return np.mean(ssim_map)

print("1. 加载数据与构建 Pseudo-GT...")
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

ecg_events = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)[0]
stim_events, _ = mne.events_from_annotations(raw_filt)
sfreq = raw_neural.info['sfreq']
event_id_val = int(ecg_events[0, 2])
n_pre = int(round(0.2 * sfreq))

safe_stims = [[stim, 0, 1] for stim in stim_events[:, 0] if np.min(np.abs(ecg_events[:, 0] - stim)) > int(0.2 * sfreq)]
tmin_gt, tmax_gt = -0.02, 0.08
epochs_gt = mne.Epochs(raw_neural, np.array(safe_stims), tmin=tmin_gt, tmax=tmax_gt, baseline=(-0.02, 0.0), preload=True)
ep_gt = epochs_gt.average()
ep_gt_filtered = ep_gt.copy().filter(l_freq=10, h_freq=100)
idx_10 = int((0.01 - tmin_gt) * sfreq)
idx_60 = int((0.06 - tmin_gt) * sfreq)

ch_names = ep_gt.ch_names
mask_esg = np.array([ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) and ch != 'LM' for ch in ch_names])
esg_names = [ch for i, ch in enumerate(ch_names) if mask_esg[i]]
esg_idx = [i for i, ch in enumerate(ch_names) if mask_esg[i]]

# 获取标准 ATS 模板
tmin_ecg, tmax_ecg = -0.2, 0.6
epochs_ecg = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
template_ecg = epochs_ecg.average().data
ch_indices = [raw_neural.ch_names.index(ch) for ch in epochs_ecg.average().ch_names]

# 截取模板中对应 [-0.02, 0.08] 的片段，用于校准
start_idx = n_pre - int(abs(tmin_gt) * sfreq)
end_idx = n_pre + int(tmax_gt * sfreq) + 1

weights = np.linspace(0, 1.5, 31)
all_results = []

print("\n2. 开始遍历所有 ESG 通道进行 Oracle 校准...")
for ch in esg_names:
    idx_ch = ch_names.index(ch)
    gt_single = ep_gt_filtered.data[idx_ch, idx_10:idx_60]
    
    # 安全刺激的原始平均波形
    safe_avg = epochs_gt.average().data[idx_ch, :]
    
    # 该通道的专用心电模板
    idx_ch_ecg = epochs_ecg.average().ch_names.index(ch)
    template_full = template_ecg[idx_ch_ecg, :]
    template_cropped = template_full[start_idx:end_idx]
    
    best_w = 0
    best_corr = -2
    
    # 搜索最优权重
    for w in weights:
        test_avg = safe_avg - w * template_cropped
        test_seg = test_avg[idx_10:idx_60]
        try:
            corr, _ = pearsonr(gt_single, test_seg)
        except:
            corr = -1
        if corr > best_corr:
            best_corr = corr
            best_w = w
            
    # 将最优权重应用到全信号
    raw_cal = raw_neural.copy()
    data_c = raw_cal._data
    sub_c = data_c[ch_indices, :]
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template_full.shape[0]
        if st >= 0 and en <= sub_c.shape[1]:
            sub_c[idx_ch_ecg, st:en] -= (template_full * best_w)
    data_c[ch_indices, :] = sub_c
    raw_cal._data = data_c
    
    # 评估该通道的最终指标
    ep_test = mne.Epochs(raw_cal, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
    ep_test_bp = ep_test.copy().filter(l_freq=10, h_freq=100)
    test_data_gt = ep_test_bp.data[idx_ch, idx_10:idx_60]
    
    try: final_cc, _ = pearsonr(gt_single, test_data_gt)
    except: final_cc = 0
    try: final_ssim = calc_ssim_1d(gt_single, test_data_gt)
    except: final_ssim = 0
    
    def locked_rms(rd, tc):
        ep = mne.Epochs(rd, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
        x = ep.get_data()
        ch_idx_l = rd.ch_names.index(tc)
        return np.mean(np.sqrt(np.mean(x[:, ch_idx_l:ch_idx_l+1, :]**2, axis=2)))
        
    base_ri = locked_rms(raw_neural, ch)
    final_ri = locked_rms(raw_cal, ch) / base_ri
    
    all_results.append({
        'Channel': ch,
        'Best_w': best_w,
        'CC_GT': final_cc,
        'SSIM': final_ssim,
        'RI': final_ri
    })
    print(f"  -> {ch}: 最优 w = {best_w:.2f}, CC_GT = {final_cc:.4f}, RI = {final_ri:.4f}")

df = pd.DataFrame(all_results)
csv_path = os.path.join(results_dir, 'v3.1_Oracle_All_Channels.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 全通道 Oracle 数据已保存至: {csv_path}")

# ================= 画图 =================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：w 的分布直方图
axes[0].hist(df['Best_w'], bins=15, color='purple', alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='w = 0 (No subtraction)')
axes[0].set_title('Distribution of Optimal Subtraction Weight (w)', fontsize=14)
axes[0].set_xlabel('Optimal w', fontsize=12)
axes[0].set_ylabel('Number of Channels', fontsize=12)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# 图2：RI vs CC_GT 散点图
scatter = axes[1].scatter(df['RI'], df['CC_GT'], c=df['Best_w'], cmap='plasma', s=80, edgecolors='black')
plt.colorbar(scatter, ax=axes[1], label='Optimal w')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Oracle Calibration: RI vs CC_GT', fontsize=14)
axes[1].set_xlabel('Residual Index (RI)', fontsize=12)
axes[1].set_ylabel('CC_GT (Correlation with Pseudo-GT)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'v3.1_Oracle_All_Channels_Summary.png')
plt.savefig(plot_path, dpi=300)
print(f"✅ 汇总图已保存至: {plot_path}")