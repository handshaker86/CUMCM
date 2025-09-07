import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from scipy.ndimage import uniform_filter1d

# (这是一个辅助函数，仅用于 substrate_refractive_index)
def get_silicon_refractive_index_sellmeier(wavenumbers_cm):
    """使用文献公认的Sellmeier方程计算高纯度硅(Si)的本征折射率。"""
    lambda_um = 10000.0 / (wavenumbers_cm + 1e-9)
    lambda_sq = lambda_um**2
    n_sq = (
        11.6858
        + (0.939816**2 * lambda_sq) / (lambda_sq - 1.1071**2)
        - 0.0044155 * lambda_sq
    )
    return np.sqrt(n_sq)


def get_si_refractive_index_LD(wavenumbers_cm, params):
    eps_inf, sig_TO, sig_LO, gamma_phonon, sig_p, gamma_e = params
    sigma = np.asarray(wavenumbers_cm, dtype=float)
    i = 1j
    lorentz_term = (
        eps_inf
        * (sigma**2 - sig_LO**2 + i * gamma_phonon * sigma)
        / (sigma**2 - sig_TO**2 + i * gamma_phonon * sigma)
    )
    sigma_safe = sigma + 1e-9
    drude_term = (sig_p**2) / (sigma_safe * (sigma_safe + i * gamma_e))
    epsilon = lorentz_term - drude_term
    return np.sqrt(epsilon + 1e-12j)


def get_sic_refractive_index_high(sigma, params_low, drude_scale=0.0):
    eps_inf, sig_TO, sig_LO, gamma_ph, sig_p, gamma_e = params_low
    return get_si_refractive_index_LD(
        sigma, [eps_inf, sig_TO, sig_LO, gamma_ph, sig_p * drude_scale, gamma_e]
    )


def get_sic_substrate_n(sigma):
    fixed_drude = [1000.0, 500.0]
    n_sell = get_silicon_refractive_index_sellmeier(sigma)
    eps_sell = n_sell**2
    sigma_safe = sigma + 1e-9
    drude_term = (fixed_drude[0] ** 2) / (
        sigma_safe * (sigma_safe + 1j * fixed_drude[1])
    )
    return np.sqrt(eps_sell - drude_term + 1e-12j)


# -----------------------------
# 2) 余弦平滑权重
# -----------------------------
def cosine_blend_weights(x, x0, half_width):
    w = np.zeros_like(x, dtype=float)
    left = x0 - half_width
    right = x0 + half_width
    mask_lo = x <= left
    mask_hi = x >= right
    mask_tr = (~mask_lo) & (~mask_hi)
    w[mask_lo] = 2.0
    w[mask_hi] = -0.5
    if np.any(mask_tr):
        t = (right - x[mask_tr]) / (2 * half_width)
        w[mask_tr] = 0.5 * (1 - np.cos(np.pi * t))
    return w


# -----------------------------
# 3) 高频平滑加权
# -----------------------------
def highfreq_weight(sigma, split=1500, tw=120, highweight=5.0):
    w_low = cosine_blend_weights(sigma, split, tw)
    return 1.0 + (highweight - 1.0) * (1 - w_low)


# -----------------------------
# 4) 残差平滑
# -----------------------------
def smooth_residual(residual, window=3):
    return uniform_filter1d(residual, size=window)


# -----------------------------
# 5) 薄膜反射率（混合模型）
# -----------------------------
def calculate_reflectance_hybrid(
    d_um, params_low, sigma, theta_deg, split=1500.0, tw=120.0, drude_scale_high=0.0
):
    sigma = np.asarray(sigma)
    theta_rad = np.deg2rad(theta_deg)
    d_cm = d_um * 1e-4

    N_low = get_si_refractive_index_LD(sigma, params_low)
    N_high = get_sic_refractive_index_high(sigma, params_low, drude_scale_high)
    w_low = cosine_blend_weights(sigma, split, tw)
    N1 = w_low * N_low + (1 - w_low) * N_high
    N2 = get_sic_substrate_n(sigma)
    N0 = 1.0

    sin1 = N0 * np.sin(theta_rad) / N1
    cos1 = np.lib.scimath.sqrt(1 - sin1**2)
    sin2 = N1 * sin1 / N2
    cos2 = np.lib.scimath.sqrt(1 - sin2**2)

    r01_s = (N0 * np.cos(theta_rad) - N1 * cos1) / (N0 * np.cos(theta_rad) + N1 * cos1)
    r01_p = (N1 * np.cos(theta_rad) - N0 * cos1) / (N1 * np.cos(theta_rad) + N0 * cos1)
    r12_s = (N1 * cos1 - N2 * cos2) / (N1 * cos1 + N2 * cos2)
    r12_p = (N2 * cos1 - N1 * cos2) / (N2 * cos1 + N1 * cos2)

    phi = 4 * np.pi * sigma * N1 * d_cm * cos1
    r_s = (r01_s + r12_s * np.exp(1j * phi)) / (1 + r01_s * r12_s * np.exp(1j * phi))
    r_p = (r01_p + r12_p * np.exp(1j * phi)) / (1 + r01_p * r12_p * np.exp(1j * phi))

    return (np.abs(r_s) ** 2 + np.abs(r_p) ** 2) / 2.0


# -----------------------------
# 6) Step 1: 高频拟合厚度 d
# -----------------------------
def fit_thickness_high(sigma, R_exp, theta_deg, split=1500.0, tw=120.0):
    sigma_high = sigma[sigma >= split]
    R_high = R_exp[sigma >= split]
    dummy_params_low = [12.0, 800.0, 950.0, 5.0, 0.0, 100.0]

    def obj_d(d):
        R_model = calculate_reflectance_hybrid(
            d,
            dummy_params_low,
            sigma_high,
            theta_deg,
            split=split,
            tw=tw,
            drude_scale_high=0.0,
        )
        return np.mean((R_model * 100 - R_high) ** 2)

    res = minimize_scalar(obj_d, bounds=(1.0, 10.0), method="bounded")
    return res.x


def refine_highfreq_thickness(
    sigma, R_exp, theta_deg, d_init, params_low, split=1500.0, tw=120.0
):
    sigma_high = sigma[sigma >= split]
    R_high = R_exp[sigma >= split]

    def obj(d):
        R_model = calculate_reflectance_hybrid(
            d, params_low, sigma_high, theta_deg, split=split, tw=tw
        )
        return np.mean((R_model * 100 - R_high) ** 2)

    res = minimize_scalar(obj, bounds=(d_init - 0.5, d_init + 0.5), method="bounded")
    return res.x


# -----------------------------
# 7) Step 2: 低频拟合 LD 参数
# -----------------------------
def fit_lowfreq_params(sigma, R_exp, theta_deg, d_fixed, split=1500.0, tw=120.0):
    def obj_low(params):
        R_model = calculate_reflectance_hybrid(
            d_fixed, params, sigma, theta_deg, split=split, tw=tw
        )
        res = R_model * 100 - R_exp
        res = smooth_residual(res, window=3)
        return np.mean(res**2)

    initial = [10.0, 797.0, 972.0, 5.0, 1.0, 100.0]
    bounds = [
        (5.0, 20.0),
        (60.0, 810.0),
        (800.0, 1000.0),
        (0.5, 100.0),
        (0.0, 200.0),
        (1.0, 1000.0),
    ]
    res = minimize(obj_low, initial, bounds=bounds, method="L-BFGS-B")
    return res.x


# -----------------------------
# 8) Step 3: 全波段微调 + 高频加权
# -----------------------------
def fit_full_hybrid(
    sigma,
    R_exp,
    theta_deg,
    d_init,
    params_low_init,
    split=1500.0,
    tw=120.0,
    highweight=5.0,
):
    def obj_full(fit_params):
        d = fit_params[0]
        params_low = fit_params[1:]
        R_model = calculate_reflectance_hybrid(
            d, params_low, sigma, theta_deg, split=split, tw=tw
        )
        res_low = R_model * 100 - R_exp
        res_low = smooth_residual(res_low, window=3)
        w = highfreq_weight(sigma, split=split, tw=tw, highweight=highweight)
        return np.mean((res_low * w) ** 2)

    initial = [d_init] + list(params_low_init)
    bounds = [(3, 4)] + [
        (5.0, 20.0),
        (60.0, 810.0),
        (800.0, 1000.0),
        (0.5, 100.0),
        (0.0, 200.0),
        (1.0, 1000.0),
    ]
    res = minimize(obj_full, initial, bounds=bounds, method="L-BFGS-B")
    return res.x


# -----------------------------
# 9) 主流程
# -----------------------------
if __name__ == "__main__":
    file1 = "附件3.xlsx"
    file2 = "附件4.xlsx"

    df1 = pd.read_excel(file1)
    proc1 = preprocess_data(df1)
    wn1 = proc1.iloc[:, 0].values.astype(float)
    R1 = proc1.iloc[:, 1].values.astype(float)

    df2 = pd.read_excel(file2)
    proc2 = preprocess_data(df2)
    wn2 = proc2.iloc[:, 0].values.astype(float)
    R2 = proc2.iloc[:, 1].values.astype(float)

    split = 1500.0
    tw = 120.0

    # 10° 数据拟合
    d10 = fit_thickness_high(wn1, R1, 10, split=split, tw=tw)
    params_low10 = fit_lowfreq_params(wn1, R1, 10, d10, split=split, tw=tw)
    d10 = refine_highfreq_thickness(wn1, R1, 10, d10, params_low10, split=split, tw=tw)
    fit10 = fit_full_hybrid(wn1, R1, 10, d10, params_low10, split=split, tw=tw)
    d10_final, params_low10_final = fit10[0], fit10[1:]

    # 15° 数据拟合
    d15 = fit_thickness_high(wn2, R2, 15, split=split, tw=tw)
    params_low15 = fit_lowfreq_params(wn2, R2, 15, d15, split=split, tw=tw)
    d15 = refine_highfreq_thickness(wn2, R2, 15, d15, params_low15, split=split, tw=tw)
    fit15 = fit_full_hybrid(wn2, R2, 15, d15, params_low15, split=split, tw=tw)
    d15_final, params_low15_final = fit15[0], fit15[1:]

    # 平均结果
    final_d = (d10_final + d15_final) / 2.0
    final_params_low = (params_low10_final + params_low15_final) / 2.0
    print(f"最终厚度 d = {final_d:.4f} μm")
    print(f"低频参数 = {np.array2string(final_params_low, precision=3)}")

    # 可视化拟合
    # 绘制反射率拟合曲线，并在 label 中显示厚度 d
    for wn, R_exp, angle, label, color, d in [
        (wn1, R1, 10, "10°", "red", d10_final),
        (wn2, R2, 15, "15°", "green", d15_final),
    ]:
        R_fit = (
            calculate_reflectance_hybrid(
                final_d, final_params_low, wn, angle, split=split, tw=tw
            )
            * 100
        )
        plt.figure(figsize=(10, 6))
        plt.plot(wn, R_exp, label=f"Experiment ({label})", alpha=0.7)
        plt.plot(
            wn,
            R_fit,
            label=f"Fit ({label}, d={d:.3f} μm)",
            color=color,
            linewidth=2,
        )
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Reflectance (%)")
        plt.title(f"Reflectance Fit for {label}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 绘制 n/k 光学常数，并在 label 中显示厚度 d
    sigma_plot = np.linspace(400, 4000, 1000)
    N_complex = get_si_refractive_index_LD(sigma_plot, final_params_low)
    n_fit = N_complex.real
    k_fit = N_complex.imag

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(sigma_plot, n_fit, color="blue", label=f"n (d={final_d:.3f} μm)")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Refractive Index (n)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(
        sigma_plot,
        k_fit,
        color="green",
        linestyle="--",
        label=f"k (d={final_d:.3f} μm)",
    )
    ax2.set_ylabel("Extinction Coefficient (k)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Final Optical Constants (n & k) of Si Film")
    plt.tight_layout()
    plt.show()
