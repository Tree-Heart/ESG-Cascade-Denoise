import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.signal import iirnotch, filtfilt, welch
from scipy.stats import pearsonr, kurtosis
from scipy.ndimage import uniform_filter1d

BASE_DIR = r'D:\BCI-invention-ZBL'
RESULTS_DIR = os.path.join(BASE_DIR, 'results_v3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 5个受试者，6个session
sessions_info = [
    {'sub': 'sub-001', 'task': 'Tibial', 'path': r'D:\BCI-invention-ZBL\data\sub-001\sub-001_clean.fif'},
    {'sub': 'sub-012', 'task': 'Tibial', 'path': r'D:\BCI-invention-ZBL\data\sub-012\sub-012_clean.fif'},
    {'sub': 'sub-018', 'task': 'Tibial', 'path': r'D:\BCI-invention-ZBL\data\sub-018\sub-018_clean.fif'},
    {'sub': 'sub-001', 'task': 'Median', 'path': r'D:\BCI-invention-ZBL\data\sub-001-median\sub-001_clean.fif'},
    {'sub': 'sub-009', 'task': 'Median', 'path': r'D:\BCI-invention-ZBL\data\sub-009-median\sub-009_clean.fif'},
    {'sub': 'sub-029', 'task': 'Median', 'path': r'D:\BCI-invention-ZBL\data\sub-029-median\sub-029_clean.fif'}
]

def calc_ssim_1d(x, y):
    L = max(np.ptp(x), np.ptp(y), 1e-6)
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    mu_x, mu_y = uniform_filter1d(x, size=5), uniform_filter1d(y, size=5)
    sigma_x = uniform_filter1d(x**2, size=5) - mu_x**2
    sigma_y = uniform_filter1d(y**2, size=5) - mu_y**2
    sigma_xy = uniform_filter1d(x*y, size=5) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return np.mean(ssim_map)

# 用于存储所有通道的总表
all_results = []

print("🚀 开始全量受试者评估...")

for session in sessions_info:
    sub, task, fif_path = session['sub'], session['task'], session['path']
    print(f"\n▶️ 正在处理: {sub} - {task}")

    if not os.path.exists(fif_path):
        print(f"❌ 跳过：{fif_path} 不存在")
        continue

    try:
        # ================= 1. 基础加载 =================
        raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
        raw_filt.info['bads'] = []
        raw_neural = raw_filt.copy().drop_channels(['ECG'])
        raw_neural.info['bads'] = []

        ecg_events = mne.preprocessing.find_ecg_events(raw_filt, ch_name='ECG', l_freq=5, h_freq=35)[0]
        stim_events, _ = mne.events_from_annotations(raw_filt)
        sfreq = raw_neural.info['sfreq']
        event_id_val = int(ecg_events[0, 2])
        n_pre = int(round(0.2 * sfreq))

        # ================= 2. 构建纯净金波形 (Clean GT) =================
        safe_stims = []
        for stim in stim_events[:, 0]:
            if np.min(np.abs(ecg_events[:, 0] - stim)) > int(0.2 * sfreq):
                safe_stims.append([stim, 0, 1])
        
        tmin_gt, tmax_gt = -0.02, 0.08
        baseline_gt = (-0.02, 0.0)
        epochs_gt = mne.Epochs(raw_neural, np.array(safe_stims), tmin=tmin_gt, tmax=tmax_gt, baseline=baseline_gt, preload=True)
        ep_gt = epochs_gt.average()
        ep_gt_filtered = ep_gt.copy().filter(l_freq=10, h_freq=100)
        
        idx_10 = int((0.01 - tmin_gt) * sfreq)
        idx_60 = int((0.06 - tmin_gt) * sfreq)
        gt_data = ep_gt_filtered.data[:, idx_10:idx_60]
        
        # 获取真实的 ESG 通道名和索引
        ch_names = ep_gt.ch_names
        mask_esg = np.array([ch.startswith(('S', 'L', 'AC', 'AL', 'IZ', 'SC')) and ch != 'LM' for ch in ch_names])
        esg_names = [ch for i, ch in enumerate(ch_names) if mask_esg[i]]
        esg_idx = [i for i, ch in enumerate(ch_names) if mask_esg[i]]

        # ================= 3. 生成 5 种去噪方法 =================
        # 纯净模板
        pure_ecg_events = [[s, 0, event_id_val] for s in ecg_events[:, 0] if np.min(np.abs(stim_events[:, 0] - s)) > int(0.2 * sfreq)]
        epochs_pure = mne.Epochs(raw_neural, np.array(pure_ecg_events), event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True)
        template_pure = epochs_pure.average().data
        ch_indices = [raw_neural.ch_names.index(ch) for ch in epochs_pure.average().ch_names]
        n_times = template_pure.shape[1]

        # 标准模板 & 级联
        projs, _ = mne.preprocessing.compute_proj_ecg(raw_filt, n_grad=0, n_mag=0, n_eeg=2, ch_name='ECG', reject=None)
        raw_ssp = raw_neural.copy().add_proj(projs).apply_proj()
        epochs_casc = mne.Epochs(raw_ssp, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True)
        template_casc = epochs_casc.average().data

        # 方法字典
        methods = {}
        methods['Pure SSP'] = raw_ssp.copy()
        
        # Clean ATS
        raw_ats = raw_neural.copy()
        for samp in ecg_events[:, 0]:
            st, en = samp - n_pre, samp - n_pre + n_times
            if st >= 0 and en <= raw_ats._data.shape[1]: raw_ats._data[ch_indices, st:en] -= template_pure
        methods['Clean ATS'] = raw_ats

        # Cascade
        raw_casc = raw_ssp.copy()
        for samp in ecg_events[:, 0]:
            st, en = samp - n_pre, samp - n_pre + n_times
            if st >= 0 and en <= raw_casc._data.shape[1]: raw_casc._data[ch_indices, st:en] -= template_casc
        methods['Cascade v2.1'] = raw_casc

        # Slew-Rate
        rr = np.diff(ecg_events[:, 0]) / sfreq
        raw_slew = raw_neural.copy()
        for freq in [(1.0/np.mean(rr)) * i for i in range(1, 10)]:
            b, a = iirnotch(freq, Q=30, fs=sfreq)
            raw_slew._data = filtfilt(b, a, raw_slew._data, axis=1)
        d_slew = raw_slew._data
        win_len = int(0.005 * sfreq)
        for ch in range(d_slew.shape[0]):
            diff_sig = np.abs(np.diff(d_slew[ch, :]))
            spike_idx = np.where(diff_sig > np.percentile(diff_sig, 99.5))[0]
            for idx in spike_idx:
                st, en = max(0, idx - win_len), min(d_slew.shape[1] - 1, idx + win_len)
                if en > st + 1: d_slew[ch, st:en] = np.linspace(d_slew[ch, st], d_slew[ch, en], en-st)
        raw_slew._data = d_slew
        methods['Slew-Rate v3.0'] = raw_slew

        # ================= 4. 计算所有指标 =================
        for name, raw_obj in methods.items():
            # 宽带信号
            ep_test_wb = mne.Epochs(raw_obj, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average()
            # 10-100Hz 神经频段信号 (用于 GT 对比)
            ep_test_bp = ep_test_wb.copy().filter(l_freq=10, h_freq=100)
            
            test_data_gt = ep_test_bp.data[esg_idx, idx_10:idx_60]
            raw_data_wb = mne.Epochs(raw_neural, stim_events, tmin=0.0, tmax=0.08, baseline=(0.0, 0.01), preload=True).average().data[esg_idx, int(0.01*sfreq):int(0.06*sfreq)]
            test_data_wb = ep_test_wb.data[esg_idx, int(0.01*sfreq):int(0.06*sfreq)]

            for i, ch in enumerate(esg_names):
                # 1. 与纯净金波形相关性
                try: cc_gt, _ = pearsonr(gt_data[esg_idx[i]], test_data_gt[i])
                except: cc_gt = np.nan
                
                # 2. 宽带 CC
                try: cc_wb, _ = pearsonr(raw_data_wb[i], test_data_wb[i])
                except: cc_wb = np.nan

                # 3. RE (相对误差)
                re = np.sum((test_data_wb[i] - raw_data_wb[i])**2) / (np.sum(raw_data_wb[i]**2) + 1e-15)
                
                # 4. SD (频谱失真)
                _, psd_r = welch(raw_data_wb[i], fs=sfreq, nperseg=min(32, len(raw_data_wb[i])))
                _, psd_t = welch(test_data_wb[i], fs=sfreq, nperseg=min(32, len(test_data_wb[i])))
                sd = np.sqrt(np.mean((10*np.log10(psd_t+1e-15) - 10*np.log10(psd_r+1e-15))**2))
                
                # 5. KR2 (峰度比)
                kr = kurtosis(test_data_wb[i]) / (kurtosis(raw_data_wb[i]) + 1e-15)
                
                # 6. SSIM
                try: ssim = calc_ssim_1d(raw_data_wb[i], test_data_wb[i])
                except: ssim = np.nan
                
                # 7. RI (能量)
                def locked_rms(rd, tc):
                    ep = mne.Epochs(rd, ecg_events, event_id=event_id_val, tmin=-0.2, tmax=0.6, baseline=None, preload=True)
                    x = ep.get_data()
                    return np.mean(np.sqrt(np.mean(x[:, rd.ch_names.index(tc):rd.ch_names.index(tc)+1, :]**2, axis=2)))
                base_ri = locked_rms(raw_neural, ch)
                ri = locked_rms(raw_obj, ch) / (base_ri + 1e-15)

                all_results.append({
                    'Subject': sub, 'Task': task, 'Channel': ch, 'Method': name,
                    'CC_GT': cc_gt, 'CC_WB': cc_wb, 'RE': re, 'SD': sd, 'KR2': kr, 'SSIM': ssim, 'RI': ri
                })
    except Exception as e:
        print(f"❌ Error processing {sub} - {task}: {e}")

# 保存总表
df_final = pd.DataFrame(all_results)
# 修复：文件名从 234 改为 228_ESG，符合真实样本量
csv_path = os.path.join(RESULTS_DIR, 'v3.0_All_228_ESG_Channels_Metrics.csv')
df_final.to_csv(csv_path, index=False)
print(f"\n✅ 全量数据已保存至: {csv_path}")



# # ================= 5. 汇总并画箱线图 =================
# print("📊 正在生成 7 个指标的箱线图...")
# methods_order = ['Pure SSP', 'Clean ATS', 'Cascade v2.1', 'Slew-Rate v3.0']
# metrics = ['RI', 'CC_GT', 'CC_WB', 'RE', 'SD', 'KR2', 'SSIM']
# colors = ['green', 'orange', 'blue', 'purple']

# fig, axes = plt.subplots(3, 3, figsize=(20, 15))
# axes = axes.flatten()

# for i, metric in enumerate(metrics):
#     ax = axes[i]
#     data_to_plot = [df_final[df_final['Method'] == m][metric].dropna().values for m in methods_order]
#     bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=methods_order, showfliers=False)
    
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
        
#     ax.set_title(f'{metric} (n=234)', fontsize=14)
#     ax.grid(axis='y', linestyle='--', alpha=0.5)
#     ax.tick_params(axis='x', rotation=15)
    
#     if metric in ['CC_GT', 'CC_WB', 'SSIM']: ax.axhline(0, color='black', linestyle=':', alpha=0.7)
#     if metric == 'KR2': ax.axhline(1, color='black', linestyle=':', alpha=0.7)

# # 删掉多余的两个空子图
# fig.delaxes(axes[7])
# fig.delaxes(axes[8])

# plt.tight_layout()
# plot_path = os.path.join(RESULTS_DIR, 'v3.0_All_234_Channels_Boxplot.png')
# plt.savefig(plot_path, dpi=300)
# print(f"✅ 箱线图已保存至: {plot_path}")

# # 打印汇总中位数
# summary = df_final.groupby('Method')[metrics].median().reindex(methods_order)
# print("\n" + "="*85)
# print("🏆 终极全量对比 (Median, n=234)")
# print("="*85)
# print(summary.round(4).to_string())