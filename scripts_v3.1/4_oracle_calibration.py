import mne
import numpy as np
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

# 构建 Pseudo-GT
safe_stims = [[stim, 0, 1] for stim in stim_events[:, 0] if np.min(np.abs(ecg_events[:, 0] - stim)) > int(0.2 * sfreq)]
tmin_gt, tmax_gt = -0.02, 0.08
epochs_gt = mne.Epochs(raw_neural, np.array(safe_stims), tmin=tmin_gt, tmax=tmax_gt, baseline=(-0.02, 0.0), preload=True)
ep_gt = epochs_gt.average()
ep_gt_filtered = ep_gt.copy().filter(l_freq=10, h_freq=100)
idx_10 = int((0.01 - tmin_gt) * sfreq)
idx_60 = int((0.06 - tmin_gt) * sfreq)

ch_plot = 'S17'
ch_idx = ep_gt.ch_names.index(ch_plot)
gt_data_single_ch = ep_gt_filtered.data[ch_idx, idx_10:idx_60]

print(f"2. 针对通道 {ch_plot} 进行 Oracle 级联校准...")
# 获取标准 ATS 模板
tmin_ecg, tmax_ecg = -0.2, 0.6
epochs_ecg = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
template_ecg = epochs_ecg.average().data
ch_indices = [raw_neural.ch_names.index(ch) for ch in epochs_ecg.average().ch_names]

# 我们只提取 S17 的模板
idx_s17 = epochs_ecg.average().ch_names.index(ch_plot)
template_s17_full = template_ecg[idx_s17, :]

# 🚨 核心修复：将 801 长度的模板截取到与 101 长度的安全波形相同的窗口 [-0.02, 0.08]
start_idx = n_pre - int(abs(tmin_gt) * sfreq)
end_idx = n_pre + int(tmax_gt * sfreq) + 1 # 加 1 确保包含最后一个采样点
template_s17_cropped = template_s17_full[start_idx:end_idx]

# 安全刺激的原始平均波形（用于校准）
epochs_safe = mne.Epochs(raw_neural, np.array(safe_stims), tmin=tmin_gt, tmax=tmax_gt, baseline=(-0.02, 0.0), preload=True)
safe_avg_s17 = epochs_safe.average().data[idx_s17, :]

# 搜索权重
weights = np.linspace(0, 1.5, 31) # 搜索 0 到 1.5 的减法权重
best_w = 0
best_corr = -2

print("   -> 正在用金波形搜索最优减法权重 w...")
for w in weights:
    # 模拟：在安全波形上减去 w * 模板
    test_avg = safe_avg_s17 - w * template_s17_cropped
    
    # 提取 10-60ms 窗口
    test_seg = test_avg[idx_10:idx_60]
    
    try:
        corr, _ = pearsonr(gt_data_single_ch, test_seg)
    except:
        corr = -1
        
    if corr > best_corr:
        best_corr = corr
        best_w = w

print(f"   -> 找到最优权重 w = {best_w:.2f}, 此时在安全数据上的 CC_GT = {best_corr:.4f}")

# ================= 应用校准后的权重到全信号 =================
print("3. 应用校准后的权重，进行全信号去噪...")
raw_calibrated = raw_neural.copy()
data_c = raw_calibrated._data
sub_c = data_c[ch_indices, :]

for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + template_s17_full.shape[0]
    if st >= 0 and en <= sub_c.shape[1]:
        # 只对 S17 通道减去 w * 模板
        sub_c[idx_s17, st:en] -= (template_s17_full * best_w)

data_c[ch_indices, :] = sub_c
raw_calibrated._data = data_c

# ================= 评估全信号指标 =================
print("4. 评估最终指标...")
ep_test = mne.Epochs(raw_calibrated, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
ep_test_bp = ep_test.copy().filter(l_freq=10, h_freq=100)
test_data_gt = ep_test_bp.data[ch_idx, idx_10:idx_60]

cc_gt_final, _ = pearsonr(gt_data_single_ch, test_data_gt)
ssim_final = calc_ssim_1d(gt_data_single_ch, test_data_gt)

def locked_rms(rd, tc):
    ep = mne.Epochs(rd, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
    x = ep.get_data()
    ch_idx_l = rd.ch_names.index(tc)
    return np.mean(np.sqrt(np.mean(x[:, ch_idx_l:ch_idx_l+1, :]**2, axis=2)))

base_ri = locked_rms(raw_neural, ch_plot)
ri_final = locked_rms(raw_calibrated, ch_plot) / base_ri

print("\n" + "="*50)
print(f"🏆 S17 通道 Oracle 校准结果")
print("="*50)
print(f"最优权重: w = {best_w:.2f}")
print(f"CC_GT:   {cc_gt_final:.4f}")
print(f"SSIM:    {ssim_final:.4f}")
print(f"RI:      {ri_final:.4f}")
print("="*50)

# ================= 画图对比 =================
ep_raw = mne.Epochs(raw_neural, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()

plt.figure(figsize=(10, 6))
scale = 1e6
plt.plot(ep_raw.times, ep_raw.data[ch_idx]*scale, label='Raw', color='black', linewidth=2)
plt.plot(ep_test.times, ep_test.data[ch_idx]*scale, label=f'Calibrated ATS (w={best_w:.2f})', color='red', linestyle='--')
plt.plot(ep_gt_filtered.times, ep_gt_filtered.data[ch_idx]*scale, label='Pseudo-GT', color='green', linestyle=':')
plt.axvspan(0.01, 0.06, color='yellow', alpha=0.2, label='SEP Window')
plt.legend()
plt.title(f"Oracle Calibrated ATS: {ch_plot}")
plt.savefig(os.path.join(results_dir, 'v3.1_Oracle_Calibrated.png'), dpi=300)
print(f"✅ 对比图已保存至: {os.path.join(results_dir, 'v3.1_Oracle_Calibrated.png')}")