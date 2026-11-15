"""
8_division_test.py
修正波形快照：蓝线改用混合试次，选方差最大的通道，文件名防冲突。
"""
import mne
import numpy as np
import pandas as pd
import os
import copy
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from itertools import permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = r'D:\BCI-invention-ZBL'
fif_path = os.path.join(BASE_DIR, 'data', 'sub-001-median', 'sub-001_clean.fif')
results_dir = os.path.join(BASE_DIR, 'results_v3.1')
os.makedirs(results_dir, exist_ok=True)

MODE = 'blind' 
# 防断点续传误读旧文件，重命名
csv_path = os.path.join(results_dir, f'v4_Division_{MODE}_normalized.csv')
sub_csv_path = os.path.join(results_dir, 'v3.1_Oracle_Exhaustive_60.csv')

def calc_ssim_1d(x, y):
    L = max(np.ptp(x), np.ptp(y), 1e-6)
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    mu_x, mu_y = uniform_filter1d(x, size=5), uniform_filter1d(y, size=5)
    sigma_x = uniform_filter1d(x**2, size=5) - mu_x**2
    sigma_y = uniform_filter1d(y**2, size=5) - mu_y**2
    sigma_xy = uniform_filter1d(x*y, size=5) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return np.mean(ssim_map)

print("1. 加载数据...")
raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
raw_filt.info['bads'] = []
raw_neural = raw_filt.copy().drop_channels(['ECG'])
raw_neural.info['bads'] = []

ecg_events = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)[0]
stim_events, _ = mne.events_from_annotations(raw_filt)
sfreq = raw_neural.info['sfreq']
event_id_val = int(ecg_events[0, 2])
n_pre = int(round(0.2 * sfreq))

TMIN, TMAX, BASELINE = -0.02, 0.08, (-0.02, 0.0)
idx_10 = int((0.01 - TMIN) * sfreq)
idx_60 = int((0.06 - TMIN) * sfreq)

THRESH = n_pre
safe_stims = np.array([[s, 0, 1] for s in stim_events[:, 0] if np.min(np.abs(ecg_events[:, 0] - s)) > THRESH])
contaminated_stims = np.array([[s, 0, 1] for s in stim_events[:, 0] if np.min(np.abs(ecg_events[:, 0] - s)) <= THRESH])
USE_CONTAMINATED = len(contaminated_stims) >= 10

epochs_gt = mne.Epochs(raw_neural, safe_stims, tmin=TMIN, tmax=TMAX, baseline=BASELINE, preload=True)
ep_gt_filtered = epochs_gt.average().copy().filter(l_freq=10, h_freq=100)

ch_names = ep_gt_filtered.ch_names
mask_esg = np.array([ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) and ch != 'LM' for ch in ch_names])

tmin_ecg, tmax_ecg = -0.2, 0.6
n_times_ecg = int((tmax_ecg - tmin_ecg) * sfreq)

def get_numpy_epochs(raw_obj):
    data = []
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + n_times_ecg
        if st >= 0 and en <= raw_obj._data.shape[1]:
            ep = raw_obj._data[:, st:en].copy()
            ep -= ep[:, :n_pre].mean(axis=1, keepdims=True)
            data.append(ep)
    return np.array(data)

data_epochs_init = get_numpy_epochs(raw_neural)
valid_ch_mask_global = np.std(data_epochs_init, axis=(0, 2)) > 1e-12
valid_ch_dict = {raw_neural.ch_names[i]: valid_ch_mask_global[i] for i in range(len(raw_neural.ch_names))}
esg_names = [ch for i, ch in enumerate(ch_names) if mask_esg[i] and valid_ch_dict.get(ch, False)]

ecg_data_raw = raw_filt.get_data(picks=['ECG'])[0]
ecg_sig_list = []
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + n_times_ecg
    if st >= 0 and en <= len(ecg_data_raw):
        ep = ecg_data_raw[st:en].copy()
        ecg_sig_list.append(ep - ep[:n_pre].mean())
ECG_SIG = np.array(ecg_sig_list).flatten()

def get_all_ri(raw_obj):
    ep = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
    x = ep.get_data()
    ch_to_idx = {ch: i for i, ch in enumerate(ep.ch_names)}
    return {ch: np.mean(np.sqrt(np.mean(x[:, ch_to_idx[ch], :]**2, axis=1))) for ch in esg_names if ch in ch_to_idx}

BASE_RI = get_all_ri(raw_neural)
PROJS, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)

def normalize_template(template):
    # 全局归一化，保留通道间相对幅度
    max_val = np.max(np.abs(template))
    if max_val < 1e-12: return template
    return template / max_val

def apply_div(raw_obj, template, w):
    raw_out = raw_obj.copy()
    data = raw_out._data
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= data.shape[1]:
            divider = 1.0 + w * template
            # 修复除以0的致命Bug：把0强制视为正向
            signs = np.sign(divider)
            signs[signs == 0] = 1.0
            divider = signs * np.maximum(np.abs(divider), 0.1)
            data[:, st:en] /= divider
    raw_out._data = data
    return raw_out

def apply_SSP(raw_obj):
    raw_out = raw_obj.copy()
    new_projs = []
    for proj in PROJS:
        p = copy.deepcopy(proj)
        col_names = p['data']['col_names']
        vec = p['data']['data']
        idx = [col_names.index(ch) for ch in raw_out.ch_names if ch in col_names]
        if not idx: continue
        if vec.ndim == 1:
            p['data']['data'] = vec[idx]
            p['data']['n_row'] = 1
        else:
            p['data']['data'] = vec[:, idx]
            p['data']['n_row'] = vec.shape[0]
        p['data']['col_names'] = [col_names[i] for i in idx]
        p['data']['n_cols'] = len(idx)
        new_projs.append(p)
    return raw_out.add_proj(new_projs).apply_proj()

def apply_ATS_div(raw_obj, w):
    epochs = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
    template = epochs.average().data[:, :n_times_ecg]
    template = normalize_template(template)
    ch_idx = [raw_obj.ch_names.index(ch) for ch in epochs.average().ch_names]
    template_full = np.zeros((raw_obj._data.shape[0], n_times_ecg))
    for i, idx in enumerate(ch_idx): template_full[idx] = template[i]
    return apply_div(raw_obj, template_full, w)

def apply_PCA_div(raw_obj, w):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    valid_mask = np.std(data_epochs, axis=(0, 2)) > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0: return raw_obj
    data_flat = data_epochs[:, valid_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    pca = PCA(n_components=min(5, n_valid)).fit(data_flat)
    X_pca = pca.transform(data_flat)
    corrs = [abs(pearsonr(X_pca[:, i], ECG_SIG)[0]) for i in range(X_pca.shape[1])]
    target = np.argmax(corrs)
    X_art = np.zeros_like(X_pca); X_art[:, target] = X_pca[:, target]
    art_flat = pca.inverse_transform(X_art)
    art_epochs = art_flat.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_full = np.zeros((n_channels, n_times))
    template_full[valid_mask] = np.mean(art_epochs, axis=0)
    template_full = normalize_template(template_full)
    return apply_div(raw_obj, template_full, w)

def apply_CCA_div(raw_obj, w):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    valid_mask = np.std(data_epochs, axis=(0, 2)) > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0: return raw_obj
    data_flat = data_epochs[:, valid_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    Y = ECG_SIG.reshape(-1, 1)
    cca = CCA(n_components=1); cca.fit(data_flat, Y)
    X_c, _ = cca.transform(data_flat, Y)
    X_art, _ = cca.inverse_transform(X_c, np.zeros_like(Y))
    art_epochs = X_art.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_full = np.zeros((n_channels, n_times))
    template_full[valid_mask] = np.mean(art_epochs, axis=0)
    template_full = normalize_template(template_full)
    return apply_div(raw_obj, template_full, w)

def run_cascade_div(raw_obj, combo, w):
    raw_temp = raw_obj.copy()
    for step in combo:
        if step == 'SSP': raw_temp = apply_SSP(raw_temp)
        elif step == 'ATS': raw_temp = apply_ATS_div(raw_temp, w)
        elif step == 'PCA': raw_temp = apply_PCA_div(raw_temp, w)
        elif step == 'CCA': raw_temp = apply_CCA_div(raw_temp, w)
    return raw_temp

def _eval_one_set(raw_obj, stim_set, with_ssim=False):
    ep = mne.Epochs(raw_obj, stim_set, tmin=TMIN, tmax=TMAX, baseline=BASELINE, preload=True).average().filter(l_freq=10, h_freq=100)
    ccs, ssims = [], []
    for ch in esg_names:
        if ch not in ep.ch_names: continue
        gt = ep_gt_filtered.data[ep_gt_filtered.ch_names.index(ch), idx_10:idx_60]
        y = ep.data[ep.ch_names.index(ch), idx_10:idx_60]
        if np.std(y) > 1e-12:
            ccs.append(pearsonr(gt, y)[0])
            if with_ssim: ssims.append(calc_ssim_1d(gt, y))
        else:
            ccs.append(0.0)
            if with_ssim: ssims.append(0.0)
    return (np.median(ccs) if ccs else 0.0, np.median(ssims) if ssims else 0.0)

def eval_metrics(raw_obj):
    cc_art, _ = _eval_one_set(raw_obj, stim_events, False)
    cc_clean, ssim_clean = _eval_one_set(raw_obj, safe_stims, True)
    cc_cont, _ = _eval_one_set(raw_obj, contaminated_stims, False) if USE_CONTAMINATED else (np.nan, np.nan)
    ri_map = get_all_ri(raw_obj)
    ri = np.median([ri_map[ch] / (BASE_RI[ch] + 1e-15) for ch in esg_names if ch in ri_map])
    return cc_art, cc_clean, ssim_clean, ri, cc_cont

def search_w_div(combo, weights):
    if not any(m in combo for m in ['ATS', 'PCA', 'CCA']): return 0.0
    best_corr, best_w = -2, 0.0
    for w in weights:
        temp = run_cascade_div(raw_neural, combo, w)
        cc_art, _ = _eval_one_set(temp, stim_events, False)
        if np.isfinite(cc_art) and cc_art > best_corr:
            best_corr, best_w = cc_art, w
    if not np.isfinite(best_corr):
        print(f"    [search_w_div] WARNING: {combo} 所有 w 的 cc_art 非 finite，退回 w=0")
    return best_w

# ================= 核心：跑全量前的波形快照 (3链 x 2通道) =================
print("\n1.5 多链多通道波形快照测试...")
test_combos = [('ATS', 'CCA'), ('SSP', 'ATS'), ('CCA', 'PCA')]
test_w = 1.0
N_cont = len(contaminated_stims)
N_safe = len(safe_stims)

# 找出方差最大的 2 个通道
ch_stds = [(ch, np.std(ep_gt_filtered.data[ep_gt_filtered.ch_names.index(ch), idx_10:idx_60])) for ch in esg_names]
ch_stds.sort(key=lambda x: -x[1])
top_chs = [ch_stds[0][0], ch_stds[1][0]]

fig, axes = plt.subplots(len(test_combos), len(top_chs), figsize=(15, 12))
times = ep_gt_filtered.times

for row, combo in enumerate(test_combos):
    combo_str = " -> ".join(combo)  # 修复：提前定义变量
    final_div = run_cascade_div(raw_neural, combo, test_w)
    
    # 性能优化：把 Epochs 计算提到通道循环外面
    ep_mixed = mne.Epochs(raw_neural, stim_events, tmin=TMIN, tmax=TMAX, baseline=BASELINE, preload=True).average().filter(l_freq=10, h_freq=100)
    ep_div = mne.Epochs(final_div, stim_events, tmin=TMIN, tmax=TMAX, baseline=BASELINE, preload=True).average().filter(l_freq=10, h_freq=100)
    
    for col, ch in enumerate(top_chs):
        ax = axes[row, col]
        idx_ch = ep_gt_filtered.ch_names.index(ch)
        
        gt_data = ep_gt_filtered.data[idx_ch, :]
        mixed_data = ep_mixed.data[idx_ch, :]
        div_data = ep_div.data[idx_ch, :]

        ax.plot(times, gt_data, 'k-', label='GT (Clean Target)', lw=2)
        ax.plot(times, mixed_data, 'b--', alpha=0.6, label='Original (Contaminated Mix)')
        ax.plot(times, div_data, 'r-', alpha=0.9, label=f'Div ({combo_str})')  # 修复：使用 combo_str
        
        ax.axvline(0, color='gray', ls=':')
        ax.axvspan(0.01, 0.06, alpha=0.1, color='yellow')
        
        # 动态标题
        ax.set_title(f'{combo_str} | {ch} | N_safe={N_safe}, N_cont={N_cont}', fontsize=10)
        if row == 0 and col == 0:
            ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
save_path = os.path.join(results_dir, 'v4_Waveform_Check_MultiGrid.png')
plt.savefig(save_path, dpi=150)
print(f"  多链快照已保存至 {save_path}")
print(f"  👉 请打开图查看：\n"
      f"     1. 蓝线和黑线是否分开？\n"
      f"     2. 红线在非尖峰区是否向黑线靠拢？\n"
      f"     3. 红线有没有在尖峰处爆炸？\n")

# ================= 主循环 =================
print(f"2. 开始除法逻辑全量穷举 ({MODE} 模式)...")
methods = ['SSP', 'ATS', 'PCA', 'CCA']
all_combs = []
for r in range(2, 5):
    all_combs.extend(list(permutations(methods, r)))

weights = [0.1, 0.2, 0.5, 1.0] if MODE == 'oracle' else [1.0]
completed = set()
if os.path.exists(csv_path):
    try: completed = set(pd.read_csv(csv_path)['Chain'].unique().tolist())
    except: pass

for idx, combo in enumerate(all_combs):
    seq_name = " -> ".join(combo)
    if seq_name in completed: continue

    print(f"[{idx+1}/{len(all_combs)}] {seq_name}", flush=True)
    if MODE == 'oracle':
        best_w = search_w_div(combo, weights)
    else:
        best_w = 1.0
        
    final = run_cascade_div(raw_neural, combo, best_w)
    cc_art, cc_clean, ssim_clean, ri, cc_cont = eval_metrics(final)
    
    pd.DataFrame([{
        'Chain': seq_name, 'Length': len(combo), 'Best_w': best_w,
        'CC_GT_artifact': cc_art, 'CC_GT_clean': cc_clean,
        'CC_GT_contaminated': cc_cont, 'SSIM_clean': ssim_clean, 'RI': ri
    }]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"   w={best_w:.2f} | clean={cc_clean:.4f} | cont={cc_cont:.4f} | SSIM={ssim_clean:.4f} | RI={ri:.4f}")

print(f"\n✅ 除法结果已保存至 {csv_path}")