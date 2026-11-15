"""
10_hybrid_laws_cascade.py
终极线性审判：SSP / ATS / PCA / CCA 的三种法则（sub/div/mul）级联全排列
修复：减法必须用原始幅度模板，除法和乘法在应用时现场归一化。
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
from tqdm import tqdm

BASE_DIR = r'D:\BCI-invention-ZBL'
fif_path = os.path.join(BASE_DIR, 'data', 'sub-001-median', 'sub-001_clean.fif')
results_dir = os.path.join(BASE_DIR, 'results_v3.1')
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, 'v3.1_Hybrid_Laws_Exhaustive.csv')

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

# ================= 核心：三种法则的运算 (已修复归一化Bug) =================
def apply_law(raw_obj, template, w, law):
    raw_out = raw_obj.copy()
    data = raw_out._data
    
    # 🚨 修复：只有 div 和 mul 需要归一化模板，sub 需要原始幅度
    if law in ['div', 'mul']:
        max_val = np.max(np.abs(template))
        if max_val > 1e-12:
            template = template / max_val
            
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= data.shape[1]:
            if law == 'sub':
                data[:, st:en] -= (template * w)
            elif law == 'div':
                divider = 1.0 + w * template
                signs = np.sign(divider)
                signs[signs == 0] = 1.0
                divider = signs * np.maximum(np.abs(divider), 0.1)
                data[:, st:en] /= divider
            elif law == 'mul':
                factor = 1.0 - w * template
                factor = np.clip(factor, 0.2, 1.0)
                data[:, st:en] *= factor
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

# 🚨 修复：返回原始幅度模板，不在这里归一化
def get_ATS_template(raw_obj):
    epochs = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
    template = epochs.average().data[:, :n_times_ecg]
    ch_idx = [raw_obj.ch_names.index(ch) for ch in epochs.average().ch_names]
    template_full = np.zeros((raw_obj._data.shape[0], n_times_ecg))
    for i, idx in enumerate(ch_idx): template_full[idx] = template[i]
    return template_full

def get_PCA_template(raw_obj):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    valid_mask = np.std(data_epochs, axis=(0, 2)) > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0: return np.zeros((n_channels, n_times))
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
    return template_full

def get_CCA_template(raw_obj):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    valid_mask = np.std(data_epochs, axis=(0, 2)) > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0: return np.zeros((n_channels, n_times))
    data_flat = data_epochs[:, valid_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    Y = ECG_SIG.reshape(-1, 1)
    cca = CCA(n_components=1); cca.fit(data_flat, Y)
    X_c, _ = cca.transform(data_flat, Y)
    X_art, _ = cca.inverse_transform(X_c, np.zeros_like(Y))
    art_epochs = X_art.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_full = np.zeros((n_channels, n_times))
    template_full[valid_mask] = np.mean(art_epochs, axis=0)
    return template_full

# ================= 级联运行引擎 =================
def run_hybrid_cascade(raw_obj, combo, w):
    raw_temp = raw_obj.copy()
    for step in combo:
        method, law = step
        if method == 'SSP':
            raw_temp = apply_SSP(raw_temp)
        elif method == 'ATS':
            template = get_ATS_template(raw_temp)
            raw_temp = apply_law(raw_temp, template, w, law)
        elif method == 'PCA':
            template = get_PCA_template(raw_temp)
            raw_temp = apply_law(raw_temp, template, w, law)
        elif method == 'CCA':
            template = get_CCA_template(raw_temp)
            raw_temp = apply_law(raw_temp, template, w, law)
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

def search_w_hybrid(combo, weights):
    if not any(step[1] is not None for step in combo): return 0.0
    best_corr, best_w = -2, 0.0
    for w in weights:
        temp = run_hybrid_cascade(raw_neural, combo, w)
        cc_art, _ = _eval_one_set(temp, stim_events, False)
        if np.isfinite(cc_art) and cc_art > best_corr:
            best_corr, best_w = cc_art, w
    return best_w

# ================= 主循环 =================
print(f"\n2. 开始混合法则级联穷举...")

operators = []
for m in ['ATS', 'PCA', 'CCA']:
    for l in ['sub', 'div', 'mul']:
        operators.append((m, l))
operators.append(('SSP', None)) 

all_combs = []
for r in range(2, 4): 
    all_combs.extend(list(permutations(operators, r)))

# 提前去重：去掉连续两次相同操作的组合 (如 ATS_sub -> ATS_div)
filtered_combs = []
for combo in all_combs:
    if len(combo) == 3 and combo[0] == combo[1]: continue
    filtered_combs.append(combo)

print(f"   共 {len(filtered_combs)} 种有效混合级联组合。")
weights = [0.5, 1.0, 1.5] 

completed = set()
if os.path.exists(csv_path):
    try: completed = set(pd.read_csv(csv_path)['Chain'].unique().tolist())
    except: pass

for combo in tqdm(filtered_combs, desc="Cascades"):
    seq_name = " -> ".join([f"{m}_{l}" if l else m for m, l in combo])
    if seq_name in completed: continue

    best_w = search_w_hybrid(combo, weights)
    final = run_hybrid_cascade(raw_neural, combo, best_w)
    cc_art, cc_clean, ssim_clean, ri, cc_cont = eval_metrics(final)
    
    pd.DataFrame([{
        'Chain': seq_name, 'Length': len(combo), 'Best_w': best_w,
        'CC_GT_artifact': cc_art, 'CC_GT_clean': cc_clean,
        'CC_GT_contaminated': cc_cont, 'SSIM_clean': ssim_clean, 'RI': ri
    }]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    
print(f"\n✅ 终极混合法则结果已保存至 {csv_path}")