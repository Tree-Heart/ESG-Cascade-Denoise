"""
6_cascade_oracle_math_new.py
v3.1 Oracle 校准版（全量 60 种）：SSP/ATS/PCA/CCA 长度 2-4 全排列 × 每级联搜最优 w
口径：B+（三指标）
  - CC_GT_artifact     : stim_events 全 trial，混合诊断列
  - CC_GT_clean        : safe_stims 纯干净，保真度维度
  - CC_GT_contaminated : contaminated_stims 仅污染，去伪迹维度
  - SSIM_clean         : 保真度形态指标
  - RI                 : 能量抑制
  - N_contaminated     : 污染 trial 数量
搜 w 口径：cc_art（保守版）
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

BASE_DIR = r'D:\BCI-invention-ZBL'
fif_path = os.path.join(BASE_DIR, 'data', 'sub-001-median', 'sub-001_clean.fif')
results_dir = os.path.join(BASE_DIR, 'results_v3.1')
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, 'v3.1_Oracle_Exhaustive_60.csv')
ONLY_LENGTH_2 = False  # 跑全量

def calc_ssim_1d(x, y):
    L = max(np.ptp(x), np.ptp(y), 1e-6)
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    mu_x, mu_y = uniform_filter1d(x, size=5), uniform_filter1d(y, size=5)
    sigma_x = uniform_filter1d(x**2, size=5) - mu_x**2
    sigma_y = uniform_filter1d(y**2, size=5) - mu_y**2
    sigma_xy = uniform_filter1d(x*y, size=5) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return np.mean(ssim_map)

# ================= 1. 加载 =================
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
safe_stims = np.array([[s, 0, 1] for s in stim_events[:, 0]
                       if np.min(np.abs(ecg_events[:, 0] - s)) > THRESH])
contaminated_stims = np.array([[s, 0, 1] for s in stim_events[:, 0]
                               if np.min(np.abs(ecg_events[:, 0] - s)) <= THRESH])
assert len(safe_stims) > 0, "safe_stims 为空"
print(f"   stim_events={len(stim_events)}, safe={len(safe_stims)}, contaminated={len(contaminated_stims)}")
USE_CONTAMINATED = len(contaminated_stims) >= 10

epochs_gt = mne.Epochs(raw_neural, safe_stims, tmin=TMIN, tmax=TMAX, baseline=BASELINE, preload=True)
ep_gt_filtered = epochs_gt.average().copy().filter(l_freq=10, h_freq=100)

# 诊断通道差异
print(f"   raw_neural 通道数: {len(raw_neural.ch_names)}, ep_gt_filtered 通道数: {len(ep_gt_filtered.ch_names)}")
if len(raw_neural.ch_names) != len(ep_gt_filtered.ch_names):
    print("   [提示] 通道数不一致，大概率是 MNE 自动 drop 了 stim 通道。")

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
dead_names = [raw_neural.ch_names[i] for i, v in enumerate(valid_ch_mask_global) if not v]
if dead_names:
    print(f"   WARNING: 死通道 {len(dead_names)} 个: {dead_names[:10]}{'...' if len(dead_names)>10 else ''}")

# 修复：使用字典映射，杜绝索引错位
valid_ch_dict = {raw_neural.ch_names[i]: valid_ch_mask_global[i] for i in range(len(raw_neural.ch_names))}

esg_names = [ch for i, ch in enumerate(ch_names)
             if mask_esg[i] and valid_ch_dict.get(ch, False)]
print(f"   ESG 通道数（过滤后）: {len(esg_names)}")

ecg_data_raw = raw_filt.get_data(picks=['ECG'])[0]
ecg_sig_list = []
for samp in ecg_events[:, 0]:
    st, en = samp - n_pre, samp - n_pre + n_times_ecg
    if st >= 0 and en <= len(ecg_data_raw):
        ep = ecg_data_raw[st:en].copy()
        ecg_sig_list.append(ep - ep[:n_pre].mean())
ECG_SIG = np.array(ecg_sig_list).flatten()

def get_all_ri(raw_obj):
    ep = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val,
                    tmin=-0.2, tmax=0.6, baseline=None, preload=True)
    x = ep.get_data()
    ch_to_idx = {ch: i for i, ch in enumerate(ep.ch_names)}
    return {ch: np.mean(np.sqrt(np.mean(x[:, ch_to_idx[ch], :]**2, axis=1)))
            for ch in esg_names if ch in ch_to_idx}

print("2. 预计算 BASE_RI...")
BASE_RI = get_all_ri(raw_neural)

print("3. 预计算 SSP 投影...")
PROJS, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)

# ================= 2. 算子 =================
def apply_SSP(raw_obj):
    # SSP：使用全局固定投影（compute_proj_ecg 一次），每个 raw_obj 上只裁剪通道
    raw_out = raw_obj.copy()
    new_projs = []
    for proj in PROJS:
        p = copy.deepcopy(proj)
        col_names = p['data']['col_names']
        vec = p['data']['data']
        idx = [col_names.index(ch) for ch in raw_out.ch_names if ch in col_names]
        if len(idx) == 0:
            continue
        if vec.ndim == 1:
            p['data']['data'] = vec[idx]
            p['data']['n_row'] = 1
        else:
            p['data']['data'] = vec[:, idx]
            p['data']['n_row'] = vec.shape[0]
        p['data']['col_names'] = [col_names[i] for i in idx]
        p['data']['n_cols'] = len(idx)
        new_projs.append(p)
    raw_out = raw_out.add_proj(new_projs).apply_proj()
    delta = np.std(raw_out._data - raw_obj._data)
    if delta < 1e-12:
        print(f"    [SSP] WARNING: no-op! delta_std={delta:.3e}")
    return raw_out

def apply_ATS(raw_obj, w=1.0):
    epochs = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val,
                        tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
    template = epochs.average().data[:, :n_times_ecg]
    ch_idx = [raw_obj.ch_names.index(ch) for ch in epochs.average().ch_names]
    raw_out = raw_obj.copy()
    data = raw_out._data
    sub = data[ch_idx, :]
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= sub.shape[1]:
            sub[:, st:en] -= template * w
    data[ch_idx, :] = sub
    raw_out._data = data
    return raw_out

def apply_PCA(raw_obj, w=1.0):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    stds = np.std(data_epochs, axis=(0, 2))
    valid_mask = stds > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0:
        return raw_obj

    data_flat = data_epochs[:, valid_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    pca = PCA(n_components=min(5, n_valid)).fit(data_flat)
    X_pca = pca.transform(data_flat)
    corrs = [abs(pearsonr(X_pca[:, i], ECG_SIG)[0]) for i in range(X_pca.shape[1])]
    target = np.argmax(corrs)

    X_artifact = np.zeros_like(X_pca)
    X_artifact[:, target] = X_pca[:, target]
    artifact_flat = pca.inverse_transform(X_artifact)

    artifact_epochs = artifact_flat.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_valid = np.mean(artifact_epochs, axis=0)

    template_full = np.zeros((n_channels, n_times))
    template_full[valid_mask] = template_valid

    raw_out = raw_obj.copy()
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + n_times
        if st >= 0 and en <= raw_out._data.shape[1]:
            raw_out._data[:, st:en] -= template_full * w
    return raw_out

def apply_CCA(raw_obj, w=1.0):
    data_epochs = get_numpy_epochs(raw_obj)
    n_epochs, n_channels, n_times = data_epochs.shape
    stds = np.std(data_epochs, axis=(0, 2))
    valid_mask = stds > 1e-12
    n_valid = np.sum(valid_mask)
    if n_valid == 0:
        return raw_obj

    data_flat = data_epochs[:, valid_mask, :].transpose(1, 0, 2).reshape(n_valid, -1).T
    Y = ECG_SIG.reshape(-1, 1)

    cca = CCA(n_components=1)
    cca.fit(data_flat, Y)
    X_c, Y_c = cca.transform(data_flat, Y)

    X_artifact, _ = cca.inverse_transform(X_c, np.zeros_like(Y_c))

    artifact_epochs = X_artifact.reshape(n_epochs, n_times, n_valid).transpose(0, 2, 1)
    template_valid = np.mean(artifact_epochs, axis=0)

    template_full = np.zeros((n_channels, n_times))
    template_full[valid_mask] = template_valid

    raw_out = raw_obj.copy()
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + n_times
        if st >= 0 and en <= raw_out._data.shape[1]:
            raw_out._data[:, st:en] -= template_full * w
    return raw_out

def run_cascade(raw_obj, combo, w):
    raw_temp = raw_obj.copy()
    for step in combo:
        if step == 'SSP': raw_temp = apply_SSP(raw_temp)
        elif step == 'ATS': raw_temp = apply_ATS(raw_temp, w)
        elif step == 'PCA': raw_temp = apply_PCA(raw_temp, w)
        elif step == 'CCA': raw_temp = apply_CCA(raw_temp, w)
    return raw_temp

# ================= 3. 评估 =================
def _eval_one_set(raw_obj, stim_set, with_ssim=False):
    if stim_set is None or len(stim_set) == 0:
        return (np.nan, np.nan)
    ep = mne.Epochs(raw_obj, stim_set, tmin=TMIN, tmax=TMAX,
                    baseline=BASELINE, preload=True).average().filter(l_freq=10, h_freq=100)
    ccs, ssims = [], []
    for ch in esg_names:
        if ch not in ep.ch_names:
            continue
        idx_gt = ep_gt_filtered.ch_names.index(ch)
        gt = ep_gt_filtered.data[idx_gt, idx_10:idx_60]
        y = ep.data[ep.ch_names.index(ch), idx_10:idx_60]
        if np.std(y) > 1e-12:
            ccs.append(pearsonr(gt, y)[0])
            if with_ssim:
                try: ssims.append(calc_ssim_1d(gt, y))
                except Exception: ssims.append(0.0)
        else:
            ccs.append(0.0)
            if with_ssim: ssims.append(0.0)
    return (np.median(ccs) if ccs else 0.0,
            np.median(ssims) if ssims else 0.0)

def _eval_cc_art_only(raw_obj):
    return _eval_one_set(raw_obj, stim_events, with_ssim=False)[0]

def eval_metrics(raw_obj):
    cc_art, _ = _eval_one_set(raw_obj, stim_events, with_ssim=False)
    cc_clean, ssim_clean = _eval_one_set(raw_obj, safe_stims, with_ssim=True)
    cc_cont = _eval_one_set(raw_obj, contaminated_stims, with_ssim=False)[0] if USE_CONTAMINATED else np.nan
    ri_map = get_all_ri(raw_obj)
    ri = np.median([ri_map[ch] / (BASE_RI[ch] + 1e-15) for ch in esg_names if ch in ri_map])
    return cc_art, cc_clean, ssim_clean, ri, cc_cont

def search_w(combo, weights):
    if not any(m in combo for m in ['ATS', 'PCA', 'CCA']):
        return 0.0
    best_corr, best_w = -2, 0.0
    for w in weights:
        temp = run_cascade(raw_neural, combo, w)
        cc_art = _eval_cc_art_only(temp)
        if np.isfinite(cc_art) and cc_art > best_corr:
            best_corr, best_w = cc_art, w
    if not np.isfinite(best_corr):
        print(f"    [search_w] WARNING: {combo} 所有 w 的 cc_art 非 finite，退回 w=0")
    return best_w

# ================= 4. NO_OP 基线 =================
print("\n3.5 NO_OP 基线自检...")
completed = set()
if os.path.exists(csv_path):
    try:
        completed = set(pd.read_csv(csv_path)['Chain'].unique().tolist())
    except pd.errors.EmptyDataError:
        completed = set()

N_CONT = len(contaminated_stims)

if 'NO_OP' not in completed:
    n_art, n_clean, n_ssim, n_ri, n_cont = eval_metrics(raw_neural)
    print(f"   NO_OP: art={n_art:.4f}, clean={n_clean:.4f} (应≈1.0), "
          f"cont={n_cont if np.isfinite(n_cont) else float('nan'):.4f}, "
          f"SSIM={n_ssim:.4f}, RI={n_ri:.4f}")
    if abs(n_clean - 1.0) > 1e-6:
        raise RuntimeError(
            f"NO_OP.cc_clean={n_clean:.6f} != 1.0。"
            f"评估函数与 ep_gt_filtered 不一致，请先修 _eval_one_set，不要继续全量跑。"
        )
    pd.DataFrame([{
        'Chain': 'NO_OP', 'Length': 0, 'Best_w': 0.0,
        'CC_GT_artifact': n_art, 'CC_GT_clean': n_clean,
        'CC_GT_contaminated': n_cont, 'SSIM_clean': n_ssim, 'RI': n_ri,
        'N_contaminated': N_CONT,
    }]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
else:
    print("   NO_OP 已存在，跳过")

# ================= 5. 主循环 =================
print("\n4. 开始 Oracle 穷举...")
methods = ['SSP', 'ATS', 'PCA', 'CCA']
if ONLY_LENGTH_2:
    all_combs = list(permutations(methods, 2))
    print(f"   sanity check：长度 2，共 {len(all_combs)} 种")
else:
    all_combs = []
    for r in range(2, 5):
        all_combs.extend(list(permutations(methods, r)))
    print(f"   全量：共 {len(all_combs)} 种")

weights = np.linspace(0.0, 1.5, 16)

for idx, combo in enumerate(all_combs):
    seq_name = " -> ".join(combo)
    if seq_name in completed:
        print(f"[{idx+1}/{len(all_combs)}] SKIP {seq_name}")
        continue

    print(f"[{idx+1}/{len(all_combs)}] {seq_name}", flush=True)
    best_w = search_w(combo, weights)
    final = run_cascade(raw_neural, combo, best_w)
    cc_art, cc_clean, ssim_clean, ri, cc_cont = eval_metrics(final)

    pd.DataFrame([{
        'Chain': seq_name, 'Length': len(combo), 'Best_w': best_w,
        'CC_GT_artifact': cc_art, 'CC_GT_clean': cc_clean,
        'CC_GT_contaminated': cc_cont,
        'SSIM_clean': ssim_clean, 'RI': ri,
        'N_contaminated': N_CONT,
    }]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"   w={best_w:.2f} | art={cc_art:.4f} | clean={cc_clean:.4f} | "
          f"cont={cc_cont if np.isfinite(cc_cont) else float('nan'):.4f} | "
          f"SSIM={ssim_clean:.4f} | RI={ri:.4f}", flush=True)

print(f"\n✅ 完成，{csv_path}")

# ================= 6. 诊断 =================
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("\n全表：")
    print(df.sort_values('CC_GT_artifact', ascending=False).to_string(index=False))

    print("\n" + "="*70)
    print("诊断")
    print("="*70)

    print(f"  数据规模: stim={len(stim_events)}, safe={len(safe_stims)}, "
          f"contaminated={N_CONT} (USE_CONTAMINATED={USE_CONTAMINATED})")
    if not USE_CONTAMINATED:
        print(f"  ⚠ contaminated < 10，CC_GT_contaminated 全 NaN，二维帕累托证据弱")
    elif N_CONT < 20:
        print(f"  ⚠ contaminated={N_CONT} < 20，CC_GT_contaminated 绝对值不可跨链比较，只看相对高低")
    else:
        print(f"  ✓ contaminated={N_CONT} >= 20，CC_GT_contaminated 可跨链比较")

    chains = df[df['Chain'] != 'NO_OP']
    if len(chains) == 0:
        print("  未完成任何 chain（可能中途崩溃），跳过后续诊断")
    else:
        print(f"\n  best_w 分布: {chains['Best_w'].value_counts().sort_index().to_dict()}")
        n_zero = (chains['Best_w'] == 0.0).sum()
        pct = n_zero / len(chains) * 100
        print(f"  best_w=0.0 占比: {n_zero}/{len(chains)} = {pct:.0f}%")
        if pct >= 99:
            print("  ⚠ best_w 全 0，cc_art 搜索退化，w 搜索无效。")
            print("     下一步：切 search_w 用 cc_cont 跑激进版。")
        elif pct <= 60:
            print("  ✓ w 搜索有效。")

        noop = df[df['Chain'] == 'NO_OP']
        if len(noop):
            print(f"\n  NO_OP 对照:")
            print(f"    CC_clean={noop['CC_GT_clean'].values[0]:.4f} (基准=1.0)")
            print(f"    CC_art  ={noop['CC_GT_artifact'].values[0]:.4f}")
            if USE_CONTAMINATED:
                print(f"    CC_cont ={noop['CC_GT_contaminated'].values[0]:.4f}  ← 污染 trial 原始保真度")

        print(f"\n  语义诊断:")
        d = (chains['CC_GT_artifact'] - chains['CC_GT_clean']).abs()
        print(f"    |CC_art - CC_clean| : median={d.median():.4f}, max={d.max():.4f}")
        if USE_CONTAMINATED:
            gap = chains['CC_GT_clean'] - chains['CC_GT_contaminated']
            print(f"    CC_clean - CC_cont  : median={gap.median():.4f}, max={gap.max():.4f}")
            if gap.median() > 0.2:
                print("    → 二维帕累托成立")
            elif gap.median() > 0.1:
                print("    → 弱证据，论文口径需说明")
            else:
                print("    → 退化为单维权衡，论文口径需调整")

        print(f"\n  全量边界:")
        print(f"    SSIM_clean max = {chains['SSIM_clean'].max():.4f} "
              f"({chains.loc[chains['SSIM_clean'].idxmax(), 'Chain']})")
        print(f"    CC_GT_clean max = {chains['CC_GT_clean'].max():.4f} "
              f"({chains.loc[chains['CC_GT_clean'].idxmax(), 'Chain']})")
        if USE_CONTAMINATED:
            print(f"    CC_GT_contaminated max = {chains['CC_GT_contaminated'].max():.4f} "
                  f"({chains.loc[chains['CC_GT_contaminated'].idxmax(), 'Chain']})")
    print("="*70)

    if len(chains) > 0:
        print("\n顺序 Sanity Check（长度 2 的六对）")
        print("-"*70)
        for a, b in [('SSP -> ATS', 'ATS -> SSP'), ('SSP -> PCA', 'PCA -> SSP'),
                     ('SSP -> CCA', 'CCA -> SSP'), ('ATS -> PCA', 'PCA -> ATS'),
                     ('ATS -> CCA', 'CCA -> ATS'), ('PCA -> CCA', 'CCA -> PCA')]:
            ra, rb = df[df['Chain'] == a], df[df['Chain'] == b]
            if len(ra) == 0 or len(rb) == 0:
                print(f"  {a:20s} vs {b:20s}  缺失，跳过")
                continue
            d = abs(ra['CC_GT_artifact'].values[0] - rb['CC_GT_artifact'].values[0])
            note = "（数学可交换）" if d < 1e-6 and {a, b} == {'SSP -> ATS', 'ATS -> SSP'} else ""
            status = "✓" if d > 1e-6 or note else "✗ 顺序失效"
            print(f"  {a:20s} vs {b:20s}  Δ={d:.4e}  {status}{note}")
        print("-"*70)