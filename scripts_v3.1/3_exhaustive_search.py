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
from itertools import permutations, combinations

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

tmin_ecg, tmax_ecg = -0.2, 0.6
epochs_ecg = mne.Epochs(raw_neural, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
data_ecg = epochs_ecg.get_data() 
ecg_epochs_signal = mne.Epochs(raw_filt, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True).get_data(picks=['ECG'])[:, 0, :] 
X_flat = data_ecg.transpose(1, 0, 2).reshape(data_ecg.shape[1], -1).T 
Y_flat = ecg_epochs_signal.transpose(1, 0).reshape(1, -1).T 

ch_indices = [raw_neural.ch_names.index(ch) for ch in epochs_ecg.ch_names]

# ================= 定义 4 种基础算法的函数 =================
def apply_SSP(raw_obj):
    projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
    return raw_obj.copy().add_proj(projs).apply_proj()

def apply_ATS(raw_obj):
    epochs = mne.Epochs(raw_obj, ecg_events, event_id=event_id_val, tmin=tmin_ecg, tmax=tmax_ecg, baseline=(-0.2, 0), preload=True)
    avg = epochs.average()
    template = avg.data
    ch_idx = [raw_obj.ch_names.index(ch) for ch in avg.ch_names]
    raw_out = raw_obj.copy()
    data = raw_out._data
    sub = data[ch_idx, :]
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= sub.shape[1]:
            sub[:, st:en] -= template
    data[ch_idx, :] = sub
    raw_out._data = data
    return raw_out

def apply_PCA(raw_obj):
    pca = PCA(n_components=5)
    pca.fit(X_flat)
    X_pca = pca.transform(X_flat) 
    corrs = [abs(pearsonr(X_pca[:, i], Y_flat[:, 0])[0]) for i in range(5)]
    target = np.argmax(corrs)
    X_c = X_pca.copy()
    X_c[:, target] = 0
    X_clean = pca.inverse_transform(X_c).T 
    data_pca = data_ecg.copy()
    for i in range(data_ecg.shape[0]):
        st, en = i * data_ecg.shape[2], (i+1) * data_ecg.shape[2]
        data_pca[i] = X_clean[:, st:en].reshape(data_ecg.shape[1], -1)
    template = np.mean(data_pca, axis=0)
    
    raw_out = raw_obj.copy()
    data = raw_out._data
    sub = data[ch_indices, :]
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= sub.shape[1]:
            sub[:, st:en] -= template
    data[ch_indices, :] = sub
    raw_out._data = data
    return raw_out

def apply_CCA(raw_obj):
    cca = CCA(n_components=1) 
    cca.fit(X_flat, Y_flat)
    X_c, _ = cca.transform(X_flat, Y_flat)
    X_c_clean = np.zeros_like(X_c)
    X_clean = cca.inverse_transform(X_c_clean).T 
    data_cca = data_ecg.copy()
    for i in range(data_ecg.shape[0]):
        st, en = i * data_ecg.shape[2], (i+1) * data_ecg.shape[2]
        data_cca[i] = X_clean[:, st:en].reshape(data_ecg.shape[1], -1)
    template = np.mean(data_cca, axis=0)
    
    raw_out = raw_obj.copy()
    data = raw_out._data
    sub = data[ch_indices, :]
    for samp in ecg_events[:, 0]:
        st, en = samp - n_pre, samp - n_pre + template.shape[1]
        if st >= 0 and en <= sub.shape[1]:
            sub[:, st:en] -= template
    data[ch_indices, :] = sub
    raw_out._data = data
    return raw_out

# ================= 评估函数 =================
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

# ================= 开启终极穷举 =================
methods = ['SSP', 'ATS', 'PCA', 'CCA']
funcs = {'SSP': apply_SSP, 'ATS': apply_ATS, 'PCA': apply_PCA, 'CCA': apply_CCA}

# 生成所有长度为 2, 3, 4 的排列组合
all_combs = []
for r in range(2, 5):
    for combo in permutations(methods, r):
        all_combs.append(combo)

print(f"\n🚀 开始穷举所有级联情况，共计 {len(all_combs)} 种...")
results = []

for combo in all_combs:
    combo_str = " -> ".join(combo)
    print(f"正在执行: {combo_str}...")
    
    # 依次执行级联
    raw_temp = raw_neural.copy()
    for m in combo:
        raw_temp = funcs[m](raw_temp)
        
    cc, ssim, ri = eval_method(raw_temp)
    
    results.append({
        'Method_Chain': combo_str,
        'Length': len(combo),
        'CC_GT': cc,
        'SSIM': ssim,
        'RI': ri
    })

# 保存表格
df = pd.DataFrame(results)
csv_path = os.path.join(results_dir, 'v3.1_Exhaustive_Cascade_Results.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 全量数据已保存至: {csv_path}")

# 打印按 CC_GT 排序的前 10 名
print("\n" + "="*60)
print("🏆 终极排名 (按 CC_GT 降序，证明保真度)")
print("="*60)
print(df.sort_values(by='CC_GT', ascending=False).head(10).to_string(index=False))
print("="*60)

# ================= 画终极帕累托散点图 =================
plt.figure(figsize=(10, 8))
# 用颜色深浅表示级联的长度
scatter = plt.scatter(df['RI'], df['CC_GT'], c=df['Length'], cmap='viridis', s=100, edgecolors='black', alpha=0.8)

# 给每个点打上标签
for idx, row in df.iterrows():
    if row['CC_GT'] > 0 or row['Length'] == 2: # 只标注表现较好的，防止文字重叠
        plt.annotate(row['Method_Chain'], (row['RI'], row['CC_GT']), 
                     textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

plt.colorbar(scatter, label='Cascade Length')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.1, color='gray', linestyle='--', alpha=0.5)

plt.title('v3.1 Exhaustive Search: The Inescapable Pareto Deadlock')
plt.xlabel('Residual Index (RI) - Lower is Better')
plt.ylabel('CC_GT (Correlation with Pseudo-GT) - Higher is Better')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

scatter_path = os.path.join(results_dir, 'v3.1_Exhaustive_Pareto_Frontier.png')
plt.savefig(scatter_path, dpi=300)
print(f"✅ 终极帕累托图已保存至: {scatter_path}")