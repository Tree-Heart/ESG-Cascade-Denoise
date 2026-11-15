import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from itertools import permutations

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

tmin_ecg, tmax_ecg = -0.2, 0.6
n_times_ecg = int((tmax_ecg - tmin_ecg) * sfreq)

# ================= 纯 Numpy 切数据 =================
def get_numpy_epochs(raw_obj):
    data = []
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + n_times_ecg
        if st >= 0 and en <= raw_obj._data.shape[1]:
            data.append(raw_obj._data[:, st:en])
    return np.array(data) # (n_epochs, n_channels, n_times)

print("2. 预计算全通道对齐的 ECG 模板与死通道掩码...")
data_epochs_init = get_numpy_epochs(raw_neural)
stds = np.std(data_epochs_init, axis=(0, 2))
valid_ch_mask_global = stds > 1e-12
# 🚨 记录哪些通道被踢出了拟合
dead_ch_names = [raw_neural.ch_names[i] for i, is_alive in enumerate(valid_ch_mask_global) if not is_alive]
dead_ch_str = ", ".join(dead_ch_names) if dead_ch_names else "None"
print(f"⚠️ 警告：发现 {len(dead_ch_names)} 个死通道，PCA/CCA 拟合时将排除 -> [{dead_ch_str}]")

# 🚨 核心修复：纯 Numpy 切 ECG，保证和上面的 data_epochs_init 长度 100% 一致
ecg_data_raw = raw_filt.get_data(picks=['ECG'])[0]
ecg_sig_list = []
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + n_times_ecg
    if st >= 0 and en <= len(ecg_data_raw):
        ecg_sig_list.append(ecg_data_raw[st:en])
ecg_sig = np.array(ecg_sig_list).flatten() # 完美对齐 (n_epochs * n_times,)

epochs_ecg = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
template_ecg_avg = epochs_ecg.average()
template_ecg_data = template_ecg_avg.data
ecg_ch_names = template_ecg_avg.ch_names

template_ecg_full = np.zeros((raw_neural._data.shape[0], template_ecg_data.shape[1]))
for i, ch in enumerate(ecg_ch_names):
    if ch in raw_neural.ch_names:
        idx = raw_neural.ch_names.index(ch)
        template_ecg_full[idx] = template_ecg_data[i]

def get_base_ri(rd, tc):
    ep = mne.Epochs(rd, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
    x = ep.get_data()
    if tc not in ep.ch_names: return 1.0
    ch_idx = ep.ch_names.index(tc)
    return np.mean(np.sqrt(np.mean(x[:, ch_idx:ch_idx+1, :]**2, axis=2)))

BASE_RI = {ch: get_base_ri(raw_neural, ch) for ch in esg_names}

# ================= 算法定义 =================
def apply_SSP(raw_obj):
    projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
    return raw_obj.copy().add_proj(projs).apply_proj()

def apply_PCA(raw_obj, w=1.0):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    
    stds = np.std(data_epochs, axis=(0, 2))
    valid_ch_mask = stds > 1e-12
    n_valid = np.sum(valid_ch_mask)
    
    if n_valid == 0: return raw_obj
    
    data_flat = data_epochs[:, valid_ch_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    pca = PCA(n_components=min(5, n_valid)).fit(data_flat)
    X_pca = pca.transform(data_flat)
    
    corrs = [abs(pearsonr(X_pca[:, i], ecg_sig)[0]) for i in range(X_pca.shape[1])]
    X_pca[:, np.argmax(corrs)] = 0
    clean_data = pca.inverse_transform(X_pca).T
    
    epochs_clean = clean_data.T.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_valid = np.mean(epochs_clean, axis=0)
    
    template_full = np.zeros((n_channels, n_times))
    template_full[valid_ch_mask] = template_valid
    
    raw_out = raw_obj.copy()
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template_full.shape[1]
        if st >= 0 and en <= raw_out._data.shape[1]:
            raw_out._data[:, st:en] -= (template_full * w)
    return raw_out

def apply_CCA(raw_obj, w=1.0):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    
    stds = np.std(data_epochs, axis=(0, 2))
    valid_ch_mask = stds > 1e-12
    n_valid = np.sum(valid_ch_mask)
    
    if n_valid == 0: return raw_obj
    
    data_flat = data_epochs[:, valid_ch_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    
    target_ch_name = 'S17' if 'S17' in [raw_obj.ch_names[i] for i, v in enumerate(valid_ch_mask) if v] else [raw_obj.ch_names[i] for i, v in enumerate(valid_ch_mask) if v][0]
    target_local_idx = [raw_obj.ch_names[i] for i, v in enumerate(valid_ch_mask) if v].index(target_ch_name)
    
    cca = CCA(n_components=1)
    cca.fit(data_flat, data_flat[:, [target_local_idx]])
    X_c, _ = cca.transform(data_flat, data_flat[:, [target_local_idx]])
    X_c_clean = np.zeros_like(X_c)
    clean_data = cca.inverse_transform(X_c_clean).T
    
    epochs_clean = clean_data.T.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_valid = np.mean(epochs_clean, axis=0)
    
    template_full = np.zeros((n_channels, n_times))
    template_full[valid_ch_mask] = template_valid
    
    raw_out = raw_obj.copy()
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template_full.shape[1]
        if st >= 0 and en <= raw_out._data.shape[1]:
            raw_out._data[:, st:en] -= (template_full * w)
    return raw_out

def apply_ATS(raw_obj, w=1.0, mode='subtract'):
    raw_out = raw_obj.copy()
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template_ecg_full.shape[1]
        if st >= 0 and en <= raw_out._data.shape[1]:
            if mode == 'subtract':
                raw_out._data[:, st:en] -= (template_ecg_full * w)
            elif mode == 'divide':
                divider = 1.0 + (template_ecg_full * w)
                divider[np.abs(divider) < 0.1] = 0.1
                raw_out._data[:, st:en] /= divider
    return raw_out

def run_cascade(raw_obj, sequence, w, mode='subtract'):
    raw_temp = raw_obj.copy()
    for step in sequence:
        if step == 'SSP': raw_temp = apply_SSP(raw_temp)
        elif step == 'PCA': raw_temp = apply_PCA(raw_temp, w)
        elif step == 'CCA': raw_temp = apply_CCA(raw_temp, w)
        elif step == 'ATS': raw_temp = apply_ATS(raw_temp, w, mode)
    return raw_temp

def eval_final_metrics(raw_obj):
    ep_test = mne.Epochs(raw_obj, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
    ep_test_bp = ep_test.copy().filter(l_freq=10, h_freq=100)
    
    valid_esg_names = [ch for ch in esg_names if ch in ep_test.ch_names]
    idx_test = [ep_test.ch_names.index(ch) for ch in valid_esg_names]
    idx_gt = [ep_gt_filtered.ch_names.index(ch) for ch in valid_esg_names]
    
    test_data_gt = ep_test_bp.data[idx_test, idx_10:idx_60]
    gt_data_aligned = ep_gt_filtered.data[idx_gt, idx_10:idx_60]
    
    cc_list, ssim_list = [], []
    for i in range(len(valid_esg_names)):
        try: cc, _ = pearsonr(gt_data_aligned[i], test_data_gt[i])
        except: cc = 0
        try: ssim = calc_ssim_1d(gt_data_aligned[i], test_data_gt[i])
        except: ssim = 0
        cc_list.append(cc)
        ssim_list.append(ssim)
        
    ri_list = [get_base_ri(raw_obj, ch) / BASE_RI[ch] for ch in valid_esg_names]
    return np.median(cc_list) if cc_list else 0, np.median(ssim_list) if ssim_list else 0, np.median(ri_list) if ri_list else 1.0

# ================= 主循环 =================
methods = ['SSP', 'ATS', 'PCA', 'CCA']
all_combs = []
for r in range(2, 5):
    all_combs.extend(list(permutations(methods, r)))

results = []
weights = np.linspace(0.0, 1.5, 16)

print(f"\n🚀 开启快速版地狱模式：{len(all_combs)} 种级联 × 加减法对决...")
for idx_combo, combo in enumerate(all_combs):
    seq_name = " -> ".join(combo)
    print(f"  [{idx_combo+1}/{len(all_combs)}] 处理: {seq_name}", flush=True)
    
    if 'ATS' in combo:
        best_corr_sub, best_w_sub = -2, 0
        best_corr_div, best_w_div = -2, 0
        
        for w in weights:
            if w == 0: continue
            
            temp_sub = run_cascade(raw_neural, combo, w, 'subtract')
            ep_temp_sub = mne.Epochs(temp_sub, safe_stims, tmin=tmin_gt, tmax=tmax_gt, baseline=(-0.02, 0.0), preload=True).average().filter(l_freq=10, h_freq=100)
            valid_esg_temp = [ch for ch in esg_names if ch in ep_temp_sub.ch_names]
            idx_temp = [ep_temp_sub.ch_names.index(ch) for ch in valid_esg_temp]
            idx_gt_temp = [ep_gt_filtered.ch_names.index(ch) for ch in valid_esg_temp]
            corr_sub = np.median([pearsonr(ep_gt_filtered.data[i, idx_10:idx_60], ep_temp_sub.data[j, idx_10:idx_60])[0] for i, j in zip(idx_gt_temp, idx_temp)])
            if corr_sub > best_corr_sub: best_corr_sub, best_w_sub = corr_sub, w
            
            temp_div = run_cascade(raw_neural, combo, w, 'divide')
            ep_temp_div = mne.Epochs(temp_div, safe_stims, tmin=tmin_gt, tmax=tmax_gt, baseline=(-0.02, 0.0), preload=True).average().filter(l_freq=10, h_freq=100)
            valid_esg_temp = [ch for ch in esg_names if ch in ep_temp_div.ch_names]
            idx_temp = [ep_temp_div.ch_names.index(ch) for ch in valid_esg_temp]
            idx_gt_temp = [ep_gt_filtered.ch_names.index(ch) for ch in valid_esg_temp]
            corr_div = np.median([pearsonr(ep_gt_filtered.data[i, idx_10:idx_60], ep_temp_div.data[j, idx_10:idx_60])[0] for i, j in zip(idx_gt_temp, idx_temp)])
            if corr_div > best_corr_div: best_corr_div, best_w_div = corr_div, w
            
        final_sub = run_cascade(raw_neural, combo, best_w_sub, 'subtract')
        final_div = run_cascade(raw_neural, combo, best_w_div, 'divide')
        
        cc_sub, ssim_sub, ri_sub = eval_final_metrics(final_sub)
        cc_div, ssim_div, ri_div = eval_final_metrics(final_div)
        
        results.append({'Chain': seq_name, 'Mode': 'Subtract', 'Best_w': best_w_sub, 'CC_GT': cc_sub, 'SSIM': ssim_sub, 'RI': ri_sub, 'Excluded_Channels': dead_ch_str})
        results.append({'Chain': seq_name, 'Mode': 'Divide', 'Best_w': best_w_div, 'CC_GT': cc_div, 'SSIM': ssim_div, 'RI': ri_div, 'Excluded_Channels': dead_ch_str})
    else:
        final_raw = run_cascade(raw_neural, combo, 0)
        cc, ssim, ri = eval_final_metrics(final_raw)
        results.append({'Chain': seq_name, 'Mode': 'None', 'Best_w': None, 'CC_GT': cc, 'SSIM': ssim, 'RI': ri, 'Excluded_Channels': dead_ch_str})

df = pd.DataFrame(results)
df.to_csv(os.path.join(results_dir, 'v3.1_Exhaustive_Oracle_v2.csv'), index=False)

print("\n" + "="*60)
print("🏆 终极排名 (按 CC_GT 降序)")
print("="*60)
print(df.sort_values(by='CC_GT', ascending=False).head(15).to_string(index=False))
print("="*60)

# ================= 画帕累托散点图 =================
plt.figure(figsize=(10, 8))
shapes = {'Subtract': 'o', 'Divide': 'X', 'None': '^'}
colors = {'Subtract': 'blue', 'Divide': 'red', 'None': 'green'}

for mode in df['Mode'].unique():
    sub_df = df[df['Mode'] == mode]
    plt.scatter(sub_df['RI'], sub_df['CC_GT'], marker=shapes[mode], c=colors[mode], s=80, alpha=0.7, label=mode, edgecolors='black')

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.1, color='gray', linestyle='--', alpha=0.5)

plt.title(f"v3.1 Exhaustive Search (Excluded: {dead_ch_str})", fontsize=14, fontweight='bold')
plt.xlabel('Residual Index (RI) - Lower is Better', fontsize=12)
plt.ylabel('CC_GT (Correlation with Pseudo-GT) - Higher is Better', fontsize=12)
plt.legend(title="Operation Mode")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

fig_path = os.path.join(results_dir, 'v3.1_Exhaustive_Oracle_Pareto.png')
plt.savefig(fig_path, dpi=300)
print(f"✅ 帕累托散点图已保存至: {fig_path}")