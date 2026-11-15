import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

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
gt_data = ep_gt_filtered.data[:, idx_10:idx_60]

ch_names = ep_gt.ch_names
mask_esg = np.array([ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) and ch != 'LM' for ch in ch_names])
esg_names = [ch for i, ch in enumerate(ch_names) if mask_esg[i]]
esg_idx = [i for i, ch in enumerate(ch_names) if mask_esg[i]]

print("2. 重新生成 PCA 和 CCA 的基础数据...")
tmin_ecg, tmax_ecg = -0.2, 0.6
epochs_ecg = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
data_ecg = epochs_ecg.get_data() 
ecg_epochs_signal = mne.Epochs(raw_filt, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True).get_data(picks=['ECG'])[:, 0, :] 
X_flat = data_ecg.transpose(1, 0, 2).reshape(data_ecg.shape[1], -1).T 
Y_flat = ecg_epochs_signal.transpose(1, 0).reshape(1, -1).T 

ch_indices = [raw_neural.ch_names.index(ch) for ch in epochs_ecg.ch_names]

# PCA
pca = PCA(n_components=5)
pca.fit(X_flat)
X_pca = pca.transform(X_flat) 
corrs = [abs(pearsonr(X_pca[:, i], Y_flat[:, 0])[0]) for i in range(5)]
target_comp = np.argmax(corrs)
X_pca_corrected = X_pca.copy()
X_pca_corrected[:, target_comp] = 0
X_pca_clean = pca.inverse_transform(X_pca_corrected).T 
data_pca = data_ecg.copy()
for i in range(data_ecg.shape[0]):
    start_idx = i * data_ecg.shape[2]
    end_idx = start_idx + data_ecg.shape[2]
    data_pca[i] = X_pca_clean[:, start_idx:end_idx].reshape(data_ecg.shape[1], -1)
raw_pca = raw_neural.copy()
template_pca = np.mean(data_pca, axis=0)

# 基础 PCA 减法
data_pca_base = raw_pca._data
sub_pca_base = data_pca_base[ch_indices, :]
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + template_pca.shape[1]
    if st >= 0 and en <= sub_pca_base.shape[1]:
        sub_pca_base[:, st:en] -= template_pca
data_pca_base[ch_indices, :] = sub_pca_base
raw_pca._data = data_pca_base

# CCA
cca = CCA(n_components=1) 
cca.fit(X_flat, Y_flat)
X_c, Y_c = cca.transform(X_flat, Y_flat)
X_c_clean = np.zeros_like(X_c)
X_cca_clean = cca.inverse_transform(X_c_clean).T 
data_cca = data_ecg.copy()
for i in range(data_ecg.shape[0]):
    start_idx = i * data_ecg.shape[2]
    end_idx = start_idx + data_ecg.shape[2]
    data_cca[i] = X_cca_clean[:, start_idx:end_idx].reshape(data_ecg.shape[1], -1)
raw_cca = raw_neural.copy()
template_cca = np.mean(data_cca, axis=0)

# 基础 CCA 减法
data_cca_base = raw_cca._data
sub_cca_base = data_cca_base[ch_indices, :]
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + template_cca.shape[1]
    if st >= 0 and en <= sub_cca_base.shape[1]:
        sub_cca_base[:, st:en] -= template_cca
data_cca_base[ch_indices, :] = sub_cca_base
raw_cca._data = data_cca_base

# ================= 评估指标函数 =================
def eval_method(raw_obj):
    ep_test = mne.Epochs(raw_obj, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
    ep_test_bp = ep_test.copy().filter(l_freq=10, h_freq=100)
    test_data_gt = ep_test_bp.data[esg_idx, idx_10:idx_60]
    
    cc_gt_list, ssim_list = [], []
    for i in range(len(esg_names)):
        try: cc_gt, _ = pearsonr(gt_data[esg_idx[i]], test_data_gt[i])
        except: cc_gt = 0
        try: ssim = calc_ssim_1d(gt_data[esg_idx[i]], test_data_gt[i])
        except: ssim = 0
        cc_gt_list.append(cc_gt)
        ssim_list.append(ssim)
    
    def locked_rms(rd, tc):
        ep = mne.Epochs(rd, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
        x = ep.get_data()
        ch_idx = rd.ch_names.index(tc)
        return np.mean(np.sqrt(np.mean(x[:, ch_idx:ch_idx+1, :]**2, axis=2)))
    
    base_ri = {ch: locked_rms(raw_neural, ch) for ch in esg_names}
    ri_list = [locked_rms(raw_obj, ch) / base_ri[ch] for ch in esg_names]
    
    return np.median(cc_gt_list), np.median(ssim_list), np.median(ri_list)

# ================= 级联地狱模式 =================
print("\n4. 开启级联地狱模式...")

# 组合 1: PCA -> ATS
print("   -> 正在执行 PCA -> ATS...")
raw_pca_ats = raw_pca.copy()
epochs_pca_ats = mne.Epochs(raw_pca, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True)
# 🚨 终极修复：使用 average().ch_names 以确保通道完全对齐
avg_pca_ats = epochs_pca_ats.average()
template_pca_ats = avg_pca_ats.data
ch_idx_pca_ats = [raw_pca_ats.ch_names.index(ch) for ch in avg_pca_ats.ch_names]

data_pca_ats = raw_pca_ats._data
sub_pca_ats = data_pca_ats[ch_idx_pca_ats, :]
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + template_pca_ats.shape[1]
    if st >= 0 and en <= sub_pca_ats.shape[1]:
        sub_pca_ats[:, st:en] -= template_pca_ats
data_pca_ats[ch_idx_pca_ats, :] = sub_pca_ats
raw_pca_ats._data = data_pca_ats

# 组合 2: CCA -> ATS
print("   -> 正在执行 CCA -> ATS...")
raw_cca_ats = raw_cca.copy()
epochs_cca_ats = mne.Epochs(raw_cca, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True)
avg_cca_ats = epochs_cca_ats.average()
template_cca_ats = avg_cca_ats.data
ch_idx_cca_ats = [raw_cca_ats.ch_names.index(ch) for ch in avg_cca_ats.ch_names]

data_cca_ats = raw_cca_ats._data
sub_cca_ats = data_cca_ats[ch_idx_cca_ats, :]
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + template_cca_ats.shape[1]
    if st >= 0 and en <= sub_cca_ats.shape[1]:
        sub_cca_ats[:, st:en] -= template_cca_ats
data_cca_ats[ch_idx_cca_ats, :] = sub_cca_ats
raw_cca_ats._data = data_cca_ats

# 评估
pca_ats_cc, pca_ats_ssim, pca_ats_ri = eval_method(raw_pca_ats)
cca_ats_cc, cca_ats_ssim, cca_ats_ri = eval_method(raw_cca_ats)

print("\n" + "="*50)
print("🏆 级联测试结果 (CC_GT, SSIM, RI)")
print("="*50)
print(f"PCA -> ATS:   CC_GT = {pca_ats_cc:.4f}, SSIM = {pca_ats_ssim:.4f}, RI = {pca_ats_ri:.4f}")
print(f"CCA -> ATS:   CC_GT = {cca_ats_cc:.4f}, SSIM = {cca_ats_ssim:.4f}, RI = {cca_ats_ri:.4f}")
print("="*50)

# ================= 画图直观对比 =================
ch_plot = 'S17'
ch_idx = ch_names.index(ch_plot)
ep_raw = mne.Epochs(raw_neural, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
ep_pca_ats = mne.Epochs(raw_pca_ats, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
ep_cca_ats = mne.Epochs(raw_cca_ats, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()

plt.figure(figsize=(10, 6))
scale = 1e6
plt.plot(ep_raw.times, ep_raw.data[ch_idx]*scale, label='Raw', color='black', linewidth=2)
plt.plot(ep_pca_ats.times, ep_pca_ats.data[ch_idx]*scale, label='PCA -> ATS', color='purple', linestyle='-.')
plt.plot(ep_cca_ats.times, ep_cca_ats.data[ch_idx]*scale, label='CCA -> ATS', color='orange', linestyle='--')
plt.axvspan(0.01, 0.06, color='yellow', alpha=0.2, label='SEP Window')
plt.legend()
plt.title(f"v3.1 Cascade Test: {ch_plot}")
plt.savefig(os.path.join(results_dir, 'v3.1_Cascade_Test.png'), dpi=300)
print(f"✅ 级联对比图已保存至: {os.path.join(results_dir, 'v3.1_Cascade_Test.png')}")