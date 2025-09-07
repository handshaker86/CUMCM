import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from q2_method2 import get_sic_refractive_index_LD, fit_optical_params, preprocess_data

def calculate_theta2_rad(sigma_cm_inv, theta1_rad, n1_func):
    n0 = 1.0  # 空气折射率
    n1 = n1_func(sigma_cm_inv)
    sin_theta2 = (n0 / n1) * np.sin(theta1_rad)
    return np.arcsin(sin_theta2)

def calculate_thickness(sigma, reflectance, theta1_deg, params):
    peak_indices, _ = find_peaks(reflectance, prominence=0.1)
    valley_indices, _ = find_peaks(-reflectance, prominence=0.1)
    extrema_indices = np.sort(np.concatenate([peak_indices, valley_indices]))
    sigma_extrema = sigma[extrema_indices]

    sigma_extrema = sigma[extrema_indices]
    print(f"在有效折射率范围内找到 {len(sigma_extrema)} 个极值点。")
    k_relative = np.arange(len(sigma_extrema))
    theta1_rad = np.deg2rad(theta1_deg)
    sin_sq_theta1 = np.sin(theta1_rad) ** 2
    n1_complex = get_sic_refractive_index_LD(sigma_extrema, params)
    n_eff_complex = np.sqrt(n1_complex**2 - sin_sq_theta1)
    x_factor = n_eff_complex.real
    x_data = sigma_extrema * x_factor
    y_data = k_relative
    filter_mask_k = (y_data > 5) 
    x_data = x_data[filter_mask_k]
    y_data = y_data[filter_mask_k]

    lin_result = linregress(x_data, y_data)
    slope = lin_result.slope
    d_cm = slope / 4.0
    d_um = d_cm * 1e4

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

    plt.subplot(1, 2, 2)
    plt.scatter(x_data, y_data, label="Extrema Data Points")
    plt.plot(
        x_data,
        lin_result.intercept + slope * x_data,
        "r",
        label=f"Linear Fit (d={d_um:.3f} μm)\nR²={lin_result.rvalue**2:.3f}",
    )
    plt.title("Modified Linear Fit of Interference Order")
    plt.xlabel("σ · Real(sqrt(ñ₁² - sin²(θ₁)))  (cm⁻¹)")
    plt.ylabel("Relative Interference Order k")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df1 = pd.read_excel("附件1.xlsx")
    df1 = preprocess_data(df1)
    sigma1 = df1.iloc[:, 0].values
    reflectance1 = df1.iloc[:, 1].values
    params1 = fit_optical_params(sigma1, reflectance1, 10, 8)  
    calculate_thickness(sigma1, reflectance1, 10, params1)

    df2 = pd.read_excel("附件2.xlsx")
    df2 = preprocess_data(df2)
    sigma2 = df2.iloc[:, 0].values
    reflectance2 = df2.iloc[:, 1].values
    params2 = fit_optical_params(sigma2, reflectance2, 15, 8)  
    calculate_thickness(sigma2, reflectance2, 15, params2)
