import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess_data


def estimate_thickness_fft(wavenumbers, reflectance, n_approx=3.4, plot_fft=False, title=""):
    """
    使用快速傅里叶变换 (FFT) 从反射光谱中估算薄膜厚度。

    参数:
    - wavenumbers (np.array): 波数数组 (cm^-1)。
    - reflectance (np.array): 反射率数组 (%)。
    - n_approx (float): 材料折射率的近似值 (硅约等于3.4)。
    - plot_fft (bool): 是否绘制FFT频谱图进行诊断。
    - title (str): 绘图的标题。

    返回:
    - float: 估算的薄膜厚度 (μm)。
    """
    # 步骤一: 将数据插值到均匀间隔的波数点上，这是FFT的前提
    min_wn, max_wn = wavenumbers.min(), wavenumbers.max()
    num_points = len(wavenumbers)
    # 创建均匀间隔的波数数组
    wn_uniform = np.linspace(min_wn, max_wn, num_points)
    # 将反射率插值到新的均匀数组上
    refl_uniform = np.interp(wn_uniform, wavenumbers, reflectance)

    # 步骤二: 移除光谱的低频背景/包络线，提取干涉条纹信号
    # 使用多项式拟合来模拟背景是一种简单有效的方法
    poly_order = 5 # 多项式阶数，5-10通常效果不错
    background = np.polyval(np.polyfit(wn_uniform, refl_uniform, poly_order), wn_uniform)
    interferogram = refl_uniform - background

    # 步骤三: 执行FFT
    # FFT结果的横坐标是波数的倒数，即“光学厚度”
    fft_result = np.fft.fft(interferogram)
    # 计算每个FFT分量对应的频率（这里即光学厚度，单位 cm）
    # wn_uniform[1] - wn_uniform[0] 是波数间隔
    fft_freq = np.fft.fftfreq(num_points, d=(wn_uniform[1] - wn_uniform[0]))

    # 步骤四: 在正频率部分寻找主峰，其位置对应光学厚度
    mask = fft_freq > 0
    # 找到峰值的索引
    peak_index = np.argmax(np.abs(fft_result[mask]))
    # 找到峰值对应的光学厚度 (cm)
    optical_thickness_cm = fft_freq[mask][peak_index]

    # 步骤五: 从光学厚度计算物理厚度
    # 光学厚度 = 2 * n * d (因子2是因为光在薄膜中走了一个来回)
    d_cm = optical_thickness_cm / (2 * n_approx)
    d_um = d_cm * 10000  # 转换单位到微米

    # (可选) 绘制FFT频谱图用于检查
    if plot_fft:
        plt.figure(figsize=(10, 6))
        # 我们只关心正频率部分
        plt.plot(fft_freq[mask] * 10000, np.abs(fft_result[mask]))
        # 标记找到的峰值
        plt.plot(optical_thickness_cm * 10000, np.abs(fft_result[mask]).max(), 'ro', label=f'Peak found @ {optical_thickness_cm*10000:.2f} μm')
        plt.xlabel('Optical Thickness (μm)')
        plt.ylabel('FFT Amplitude')
        plt.title(f'FFT Spectrum for {title}')
        plt.grid(True)
        plt.legend()
        plt.show()

    return d_um


# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- 数据加载 ---
    file1 = "附件3.xlsx"
    file2 = "附件4.xlsx"

    print("--- 正在加载和预处理数据 ---")
    try:
        df1 = pd.read_excel(file1) # header=None让pandas先不把第一行当表头
        processed_df1 = preprocess_data(df1)
        wavenumbers1 = processed_df1.iloc[:, 0].values.astype(float)
        reflectance1 = processed_df1.iloc[:, 1].values.astype(float)
        print(f"成功加载文件 '{file1}', 找到 {len(wavenumbers1)} 个数据点。")
        
        df2 = pd.read_excel(file2)
        processed_df2 = preprocess_data(df2)
        wavenumbers2 = processed_df1.iloc[:, 0].values.astype(float)
        reflectance2 = processed_df1.iloc[:, 1].values.astype(float)
        print(f"成功加载文件 '{file2}', 找到 {len(wavenumbers2)} 个数据点。")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
        print("请确保 '附件3.xlsx' 和 '附件4.xlsx' 文件与此脚本在同一目录下。")
        exit() # 找不到文件则退出程序
    
    print("\n--- 开始使用FFT估算初始厚度 ---")
    
    # --- 执行FFT估算并绘图 ---
    estimated_d_10 = estimate_thickness_fft(wavenumbers1, reflectance1, plot_fft=True, title="10-degree Data")
    estimated_d_15 = estimate_thickness_fft(wavenumbers2, reflectance2, plot_fft=True, title="15-degree Data")
    
    avg_estimated_d = (estimated_d_10 + estimated_d_15) / 2

    print("\n" + "="*40)
    print("--- FFT 估算结果 ---")
    print(f"基于 10° 数据的估算厚度: {estimated_d_10:.4f} μm")
    print(f"基于 15° 数据的估算厚度: {estimated_d_15:.4f} μm")
    print("-" * 25)
    print(f"平均估算厚度: {avg_estimated_d:.4f} μm")
    print("="*40)
    print("\n建议：")
    print(f"请在您的主拟合代码中，将 `fixed_d_guess` 的值修改为 {avg_estimated_d:.4f}。")
    print("例如: `fixed_d_guess = " + f"{avg_estimated_d:.4f}`")