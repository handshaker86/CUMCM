import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

MIN_WAVENUMBER = 600
MAX_WAVENUMBER = 4000
SG_WINDOW = 11
SG_ORDER = 2
NORMALIZATION_RANGE = (500, 1000)

def preprocess_data(
    data_df,
    min_wn=MIN_WAVENUMBER,
    max_wn=MAX_WAVENUMBER,
    sg_window=SG_WINDOW,
    sg_order=SG_ORDER,
    norm_range=NORMALIZATION_RANGE,
):
    df = data_df.copy()
    df.columns = ["wavenumber", "reflectance"]
    reststrahlen_band = df[
        (df["wavenumber"] >= norm_range[0]) & (df["wavenumber"] <= norm_range[1])
    ]
    max_R_exp = reststrahlen_band["reflectance"].max()

    if max_R_exp > 100.0:
        correction_factor = 100.0 / max_R_exp
        print(f"  检测到最大反射率为 {max_R_exp:.2f}% (>100%)。")
        df["reflectance_norm"] = df["reflectance"] * correction_factor
    else:
        print("  最大反射率未超过100%，无需归一化。")
        df["reflectance_norm"] = df["reflectance"]

    df_cut = df[(df["wavenumber"] >= min_wn) & (df["wavenumber"] <= max_wn)].copy()
    df_cut["reflectance_processed"] = savgol_filter(
        df_cut["reflectance_norm"], window_length=sg_window, polyorder=sg_order
    )
    final_df = df_cut[["wavenumber", "reflectance_processed"]].reset_index(drop=True)
    return final_df

def get_sic_refractive_index_LD(wavenumbers_cm, params):
    eps_inf, sig_TO, sig_LO, gamma_phonon, sig_p, gamma_e = params
    sigma = np.asarray(wavenumbers_cm)
    i = 1j
    lorentz_term = (
        eps_inf
        * (sigma**2 - sig_LO**2 + i * gamma_phonon * sigma)
        / (sigma**2 - sig_TO**2 + i * gamma_phonon * sigma)
    )
    sigma_safe = sigma + 1e-9
    drude_term = (sig_p**2) / (sigma_safe * (sigma_safe + i * gamma_e))
    epsilon = lorentz_term - drude_term
    n_complex = np.sqrt(epsilon + 1e-12j)
    return n_complex

def get_sic_refractive_index_substrate(wavenumbers_cm):
    fixed_substrate_params = [6.5, 797.7, 992.1, 8.0, 600, 300]
    return get_sic_refractive_index_LD(wavenumbers_cm, fixed_substrate_params)

def calculate_reflectance_thin_film(
    fit_params, wavenumbers_cm, theta0_deg, is_multibeam=False
):
    d_um = fit_params[0]
    material_params = fit_params[1:]
    N1 = get_sic_refractive_index_LD(wavenumbers_cm, material_params)
    N2 = get_sic_refractive_index_substrate(wavenumbers_cm)
    N0 = 1.0
    d_cm = d_um * 1e-4
    theta0_rad = np.deg2rad(theta0_deg)
    sin_theta1 = N0 * np.sin(theta0_rad) / N1
    cos_theta1 = np.lib.scimath.sqrt(1 - sin_theta1**2)
    sin_theta2 = N1 * sin_theta1 / N2
    cos_theta2 = np.lib.scimath.sqrt(1 - sin_theta2**2)
    r01_s = (N0 * np.cos(theta0_rad) - N1 * cos_theta1) / (
        N0 * np.cos(theta0_rad) + N1 * cos_theta1
    )
    r01_p = (N1 * np.cos(theta0_rad) - N0 * cos_theta1) / (
        N1 * np.cos(theta0_rad) + N0 * cos_theta1
    )
    r12_s = (N1 * cos_theta1 - N2 * cos_theta2) / (N1 * cos_theta1 + N2 * cos_theta2)
    r12_p = (N2 * cos_theta1 - N1 * cos_theta2) / (N2 * cos_theta1 + N1 * cos_theta2)
    t01_s = 2 * N0 * np.cos(theta0_rad) / (N0 * np.cos(theta0_rad) + N1 * cos_theta1)
    t01_p = 2 * N0 * np.cos(theta0_rad) / (N1 * np.cos(theta0_rad) + N0 * cos_theta1)
    t10_s = 2 * N1 * cos_theta1 / (N1 * cos_theta1 + N0 * np.cos(theta0_rad))
    t10_p = 2 * N1 * cos_theta1 / (N0 * cos_theta1 + N1 * np.cos(theta0_rad))
    phi = 4 * np.pi * wavenumbers_cm * N1 * d_cm * cos_theta1
    if is_multibeam:
        r_total_s = (r01_s + r12_s * np.exp(1j * phi)) / (
            1 + r01_s * r12_s * np.exp(1j * phi)
        )
        r_total_p = (r01_p + r12_p * np.exp(1j * phi)) / (
            1 + r01_p * r12_p * np.exp(1j * phi)
        )
    else:
        r_total_s = r01_s + t01_s * t10_s * r12_s * np.exp(1j * phi)
        r_total_p = r01_p + t01_p * t10_p * r12_p * np.exp(1j * phi)
    R_s = np.abs(r_total_s) ** 2
    R_p = np.abs(r_total_p) ** 2
    R_model = (R_s + R_p) / 2
    return np.real(R_model)

def fit_optical_params(exp_wavenumbers, exp_reflectance, angle_deg, fixed_d):
    def objective_for_params(
        material_params, wavenumbers_cm, R_exp, theta0_deg, fixed_d_um
    ):
        fit_params = np.insert(material_params, 0, fixed_d_um)
        R_model = calculate_reflectance_thin_film(
            fit_params, wavenumbers_cm, theta0_deg
        )
        mse = np.mean((R_model * 100 - R_exp) ** 2)
        return mse

    initial_material_params = [6.5, 798, 992, 10, 100, 150]
    bounds_material = [(6, 7), (780, 820), (980, 1020), (1, 20), (0, 500), (0, 500)]

    result = minimize(
        objective_for_params,
        initial_material_params,
        args=(exp_wavenumbers, exp_reflectance, angle_deg, fixed_d),
        method="L-BFGS-B",
        bounds=bounds_material,
        options={"disp": False},  
    )
    if result.success:
        print("光学参数拟合成功。")
        print(f"最终 MSE Loss: {result.fun:.6f}")
        return result.x
    else:
        print("警告: 光学参数拟合未能收敛。")
        return None

def fit_thickness(exp_wavenumbers, exp_reflectance, angle_deg, fixed_material_params):
    def objective_for_d(d_um, wavenumbers_cm, R_exp, theta0_deg, fixed_params):
        fit_params = np.insert(fixed_params, 0, d_um)
        R_model = calculate_reflectance_thin_film(
            fit_params, wavenumbers_cm, theta0_deg
        )
        mse = np.mean((R_model * 100 - R_exp) ** 2)
        return mse

    result = minimize_scalar(
        objective_for_d,
        bounds=(8, 10),
        args=(exp_wavenumbers, exp_reflectance, angle_deg, fixed_material_params),
        method="bounded",
    )
    print(f"最终 MSE Loss: {result.fun:.6f}")
    print(f"厚度拟合完成，d = {result.x:.4f} μm")
    return result.x

if __name__ == "__main__":
    file1 = "附件1.xlsx"
    file2 = "附件2.xlsx"

    # --- 数据加载 ---
    df1 = pd.read_excel(file1)
    processed_df1 = preprocess_data(df1)
    wavenumbers1 = processed_df1.iloc[:, 0].values.astype(float)
    reflectance1 = processed_df1.iloc[:, 1].values.astype(float)

    df2 = pd.read_excel(file2)
    processed_df2 = preprocess_data(df2)
    wavenumbers2 = processed_df2.iloc[:, 0].values.astype(float)
    reflectance2 = processed_df2.iloc[:, 1].values.astype(float)

    fixed_d_guess = 9
    params_10deg = fit_optical_params(wavenumbers1, reflectance1, 10, fixed_d_guess)
    params_15deg = fit_optical_params(wavenumbers2, reflectance2, 15, fixed_d_guess)

    if params_10deg is not None and params_15deg is not None:
        param_names = ["eps_inf", "sig_TO", "sig_LO", "gamma_ph", "sig_p", "gamma_e"]
        print(f"{'参数':<10} | {'10度拟合值':<15} | {'15度拟合值':<15}")
        for i, name in enumerate(param_names):
            print(f"{name:<10} | {params_10deg[i]:<15.3f} | {params_15deg[i]:<15.3f}")

        avg_params = (params_10deg + params_15deg) / 2.0
        print("\n计算出的平均光学常数:")
        for name, param in zip(param_names, avg_params):
            print(f"  {name:<10} = {param:.4f}")
        print("=" * 40)

        final_d_10deg = fit_thickness(wavenumbers1, reflectance1, 10, avg_params)
        final_d_15deg = fit_thickness(wavenumbers2, reflectance2, 15, avg_params)

        final_d = np.mean([final_d_10deg, final_d_15deg])
        print("\n" + "=" * 40)
        print("--- 最终分析结果 ---")
        print(f"基于10°数据计算的最终厚度: {final_d_10deg:.4f} μm")
        print(f"基于15°数据计算的最终厚度: {final_d_15deg:.4f} μm")
        print(f"平均厚度 (最终结果): d = {final_d:.4f} μm")

        final_fit_params = np.insert(avg_params, 0, final_d)
        R_fit1 = (
            calculate_reflectance_thin_film(final_fit_params, wavenumbers1, 10) * 100
        )

        # 绘制10度拟合图
        plt.figure(figsize=(12, 6))
        plt.plot(wavenumbers1, reflectance1, label="Experiment Data (10°)", alpha=0.7)
        plt.plot(
            wavenumbers1,
            R_fit1,
            label=f"Final Model Fit (d={final_d_10deg:.3f} μm)",
            color="red",
            linewidth=2,
        )
        plt.title("Final Fit vs. Experimental Data (10°)")
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Reflectance (%)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 绘制15度拟合图
        R_fit2 = (
            calculate_reflectance_thin_film(final_fit_params, wavenumbers2, 15) * 100
        )
        plt.figure(figsize=(12, 6))
        plt.plot(wavenumbers2, reflectance2, label="Experiment Data (15°)", alpha=0.7)
        plt.plot(
            wavenumbers2,
            R_fit2,
            label=f"Final Model Fit (d={final_d_15deg:.3f} μm)",
            color="green",
            linewidth=2,
        )
        plt.title("Final Fit vs. Experimental Data (15°)")
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Reflectance (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
