import mne
import os

# --- 配置参数区 ---
SUBJECTS = ['sub-001', 'sub-012', 'sub-018']
BASE_DIR = r'D:\BCI-invention-ZBL'

print("="*50)
print("🚀 开始执行数据预处理流水线 (Preprocessing Pipeline)")
print(f"🎯 目标受试者: {', '.join(SUBJECTS)}")
print("="*50)

for sub in SUBJECTS:
    print(f"\n▶️ 正在检查受试者: {sub}")
    
    # 1. 路径定义
    file_path = os.path.join(BASE_DIR, 'data', sub, f'{sub}_task-tibial_run-02_eeg.set')
    output_path = os.path.join(BASE_DIR, 'data', sub, f'{sub}_clean.fif')
    
    # 2. 防御性检查：原始文件是否存在？
    if not os.path.exists(file_path):
        print(f"  ❌ 错误: 找不到原始数据文件 {file_path}，跳过该受试者。")
        continue
        
    # 3. 防御性检查：如果已经处理过，就直接跳过
    if os.path.exists(output_path):
        print(f"  ⏭️ 提示: 已经存在处理好的 .fif 文件，跳过以节省时间。")
        continue
        
    # 4. 开始处理
    try:
        print(f"  [1/4] 正在读取原始 .set 数据 (文件较大，可能需要十几秒)...")
        # preload=False 意味着先只读文件头，不把所有数据塞进内存，防止瞬间爆满
        raw = mne.io.read_raw_eeglab(file_path, preload=False)
        
        print(f"  [2/4] 正在降采样到 1000 Hz (极其耗时，请耐心等待约 15-20 分钟)...")
        # 降采样是为了减小数据量，1000Hz 对于脊髓诱发电位完全足够，且能加快后续计算
        raw.resample(1000)
        
        print(f"  [3/4] 正在载入内存并执行 1-100 Hz 带通滤波...")
        # 载入内存才能进行滤波计算
        raw.load_data()
        # 滤除 1Hz 以下的基线漂移，以及 100Hz 以上的高频肌电/环境噪声
        raw.filter(l_freq=1, h_freq=100)
        
        print(f"  [4/4] 正在保存为轻量级 .fif 文件...")
        # 保存为 MNE 专属格式，体积会从 1.7GB 缩小到 170MB，以后秒开
        raw.save(output_path, overwrite=True)
        
        print(f"  ✅ 受试者 {sub} 预处理完成！保存路径: {output_path}")
        
    except Exception as e:
        print(f"  ❌ 处理 {sub} 时发生严重错误: {e}")

print("\n" + "="*50)
print("🎉 所有受试者预处理阶段结束！")