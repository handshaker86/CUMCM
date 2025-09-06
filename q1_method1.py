import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from q1_method2 import get_sic_refractive_index


def calculate_theta2_rad(sigma_cm_inv, theta1_rad, n1_func):
    """根据斯涅尔定律计算折射角 theta2 (弧度)"""
    n0 = 1.0  # 空气折射率
    n1 = n1_func(sigma_cm_inv)
    sin_theta2 = (n0 / n1) * np.sin(theta1_rad)
    return np.arcsin(sin_theta2)


def calculate_thickness(file_path, theta1_deg):
    print(f"--- 正在处理文件 (复数模型): {file_path} (入射角: {theta1_deg}°) ---")
    df = pd.read_excel(file_path)
    sigma = df.iloc[:, 0].values
    reflectance = df.iloc[:, 1].values

    # 寻峰逻辑不变
    peak_indices, _ = find_peaks(reflectance, prominence=0.1)
    valley_indices, _ = find_peaks(-reflectance, prominence=0.1)
    extrema_indices = np.sort(np.concatenate([peak_indices, valley_indices]))
    sigma_extrema = sigma[extrema_indices]

    valid_range_mask = (sigma_extrema >= 390) & (sigma_extrema <= 4100)
    extrema_indices = extrema_indices[valid_range_mask]
    sigma_extrema = sigma[extrema_indices]
    print(f"在有效折射率范围内找到 {len(sigma_extrema)} 个极值点。")

    # 相对干涉级次 k 不变
    k_relative = np.arange(len(sigma_extrema))

    # --- 修正的线性模型计算 ---
    theta1_rad = np.deg2rad(theta1_deg)
    sin_sq_theta1 = np.sin(theta1_rad) ** 2

    # 1. 获取每个极值点波数对应的【复数】折射率 ñ₁
    n1_complex = get_sic_refractive_index(sigma_extrema)

    # 2. 计算 ñ_eff = sqrt(ñ₁² - sin²(θ₁))
    n_eff_complex = np.sqrt(n1_complex**2 - sin_sq_theta1)

    # 3. 提取 ñ_eff 的实部作为 x 轴的乘数因子
    x_factor = n_eff_complex.real

    # 4. 构建线性拟合的 x 和 y 轴数据
    # x = σ * Real(ñ_eff)
    x_data = sigma_extrema * x_factor
    y_data = k_relative

    # x_threshold = 3500 
    # reliable_data_mask = (x_data > x_threshold)
    # x_data = x_data[reliable_data_mask]
    # y_data = y_data[reliable_data_mask]
    # print(f"用于线性拟合的有效数据点数量: {len(x_data)} (x > {x_threshold})")
    # --- 线性回归 ---
    # 根据新模型 k = (4d) * x, 斜率 slope = 4d
    lin_result = linregress(x_data, y_data)
    slope = lin_result.slope

    # --- 计算厚度 d ---
    d_cm = slope / 4.0
    d_um = d_cm * 1e4

    # --- 结果输出与可视化 ---
    print(f"线性拟合斜率 (Slope): {slope:.6f}")
    print(f"线性拟合 R-squared: {lin_result.rvalue**2:.6f}")
    print(f"计算得到的外延层厚度 d = {d_um:.4f} μm\n")

    plt.subplot(1, 2, 1)
    plt.plot(sigma, reflectance, label="Original Spectrum")
    plt.plot(
        sigma_extrema,
        reflectance[extrema_indices],
        "x",
        color="red",
        label="Detected Extrema",
    )
    plt.title(f"Spectrum and Extrema degree={theta1_deg}°")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Reflectance (%)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Modified linear fit result
    plt.subplot(1, 2, 2)
    plt.scatter(x_data, y_data, label="Extrema Data Points")
    plt.plot(
        x_data,
        lin_result.intercept + slope * x_data,
        "r",
        label=f"Linear Fit\nR²={lin_result.rvalue**2:.5f}",
    )
    plt.title("Modified Linear Fit of Interference Order")
    # The x-label formula uses standard mathematical notation, which is universal.
    plt.xlabel("σ · Real(sqrt(ñ₁² - sin²(θ₁)))  (cm⁻¹)")
    plt.ylabel("Relative Interference Order k")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 执行计算 ---
    # 假设附件1和附件2在代码运行的目录下
    calculate_thickness("附件1.xlsx", 10)
    calculate_thickness("附件2.xlsx", 15)
    

