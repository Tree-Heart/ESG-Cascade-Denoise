import mne
import os

# 处理的三个 median 受试者
SUBJECTS = ['sub-009', 'sub-029', 'sub-001']
BASE_DIR = r'D:\BCI-invention-ZBL'

print("="*50)
print("🚀 开始执行 Median 任务数据预处理流水线")
print(f"🎯 目标受试者: {', '.join(SUBJECTS)}")
print("="*50)

for sub in SUBJECTS:
    print(f"\n▶️ 正在检查受试者: {sub}")
    
    # 1. 路径定义（针对 median 任务和 run-03）
    # 注意：你的文件夹命名是 sub-xxx-median
    file_path = os.path.join(BASE_DIR, 'data', f'{sub}-median', f'{sub}_task-median_run-03_eeg.set')
    output_path = os.path.join(BASE_DIR, 'data', f'{sub}-median', f'{sub}_clean.fif')
    
    # 2. 防御性检查：原始文件是否存在？
    if not os.path.exists(file_path):
        print(f"  ❌ 错误: 找不到原始数据文件 {file_path}，请检查是不是漏下了 .set 文件！跳过。")
        continue
        
    # 3. 防御性检查：如果已经处理过，直接跳过
    if os.path.exists(output_path):
        print(f"  ⏭️ 提示: 已经存在处理好的 .fif 文件，完美跳过以节省时间。")
        continue
        
    # 4. 开始处理
    try:
        print(f"  [1/4] 正在读取原始 .set 数据...")
        raw = mne.io.read_raw_eeglab(file_path, preload=False)
        
        print(f"  [2/4] 正在降采样到 1000 Hz (耐心等待 15-20 分钟)...")
        raw.resample(1000)
        
        print(f"  [3/4] 正在载入内存并执行 1-100 Hz 带通滤波...")
        raw.load_data()
        raw.filter(l_freq=1, h_freq=100)
        
        print(f"  [4/4] 正在保存为轻量级 .fif 文件...")
        raw.save(output_path, overwrite=True)
        
        print(f"  ✅ 受试者 {sub} 预处理完成！保存路径: {output_path}")
        
    except Exception as e:
        print(f"  ❌ 处理 {sub} 时发生严重错误: {e}")

print("\n" + "="*50)
print("🎉 所有 Median 受试者预处理阶段结束！")