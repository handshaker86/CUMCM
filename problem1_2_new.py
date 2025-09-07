# -*- coding: utf-8 -*-
"""
fit_ld_sensitivity_ci.py

融合方案：
1) 步骤一：固定 d (d_fix)，分别在 10° / 15° 上拟合 LD 六参 p=(eps_inf, sig_TO, sig_LO, gamma_ph, sig_p, gamma_e)。
   - 得到 p10, p15 及其近似协方差 Σ_p10, Σ_p15（由数值雅可比 J_p 和残差方差估计 σ_R^2 得到）。
2) 步骤二：平均光学常数 p_avg = (p10 + p15)/2，协方差也取平均 Σ_p_avg = (Σ_p10 + Σ_p15)/2。
3) 步骤三：固定 p_avg，分别在 10° / 15° 上拟合 d，得到 d̂10, d̂15 及条件 SE|p。
   - 计算 J_d（∂R/∂d）与 J_p（∂R/∂p）并用 Δ-method 把 Σ_p_avg 传到 Var(d) 上，得“总 SE”。
4) 逆方差加权合并两角度厚度，给出条件合并与“总不确定度”合并两套结果。

注意：
- 反射率以“百分数”做损失与灵敏度（和你旧代码一致）。
- 如果本地没有 preprocess.py，会自动使用兜底预处理：取前两列作为 (波数, 反射率%).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
# ======================== 预处理兜底 ========================
try:
    from preprocess import preprocess_data  # 用户自己的清洗流程（若存在）
except Exception:
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """兜底：直接返回前两列 [波数, 反射率%]，并丢 NA。"""
        out = df.iloc[:, :2].copy()
        out = out.dropna()
        return out

# ======================== LD 折射率模型 ========================
def get_sic_refractive_index_LD(wavenumbers_cm, params):
    """
    洛伦兹-德鲁德模型（6参）返回复折射率 n(σ)。
    params = [eps_inf, sig_TO, sig_LO, gamma_phonon, sig_p, gamma_e]
    所有频率相关参数单位与 σ 一致（cm^-1），σ=波数。
    """
    eps_inf, sig_TO, sig_LO, gamma_phonon, sig_p, gamma_e = params
    sigma = np.asarray(wavenumbers_cm, dtype=float)
    i = 1j

    # Lorentz(phonon)
    lorentz_term = (
        eps_inf
        * (sigma**2 - sig_LO**2 + i * gamma_phonon * sigma)
        / (sigma**2 - sig_TO**2 + i * gamma_phonon * sigma)
    )
    # Drude(free carrier)
    sigma_safe = sigma + 1e-12
    drude_term = (sig_p**2) / (sigma_safe * (sigma_safe + i * gamma_e))

    epsilon = lorentz_term - drude_term
    # 保证复开方稳定
    return np.sqrt(epsilon + 0j)

def get_sic_refractive_index_substrate(wavenumbers_cm):
    """
    固定一组“重掺杂 SiC 衬底”的代表性参数，可按需修改。
    """
    fixed_substrate_params = [6.5, 797.7, 992.1, 8.0, 600.0, 300.0]
    return get_sic_refractive_index_LD(wavenumbers_cm, fixed_substrate_params)

# ======================== 三层薄膜反射模型 ========================
def calculate_reflectance_thin_film(fit_params, wavenumbers_cm, theta0_deg, is_multibeam=True):
    """
    fit_params = [d_um] + [eps_inf, sig_TO, sig_LO, gamma_ph, sig_p, gamma_e]
    返回非偏振反射率（0..1）。
    is_multibeam=True 使用 Fabry-Pérot 闭式；False 使用首往返(双光束)近似。
    """
    d_um = float(fit_params[0])
    material_params = np.asarray(fit_params[1:], dtype=float)

    N1 = get_sic_refractive_index_LD(wavenumbers_cm, material_params)   # 外延层
    N2 = get_sic_refractive_index_substrate(wavenumbers_cm)             # 衬底
    N0 = 1.0                                                            # 空气

    wv = np.asarray(wavenumbers_cm, dtype=float)
    d_cm = d_um * 1e-4
    t0 = np.deg2rad(theta0_deg)

    # Snell
    sin1 = N0 * np.sin(t0) / N1
    cos1 = np.lib.scimath.sqrt(1 - sin1**2)
    sin2 = N1 * sin1 / N2
    cos2 = np.lib.scimath.sqrt(1 - sin2**2)

    # Fresnel（界面01、12）
    r01s = (N0*np.cos(t0) - N1*cos1) / (N0*np.cos(t0) + N1*cos1)
    r01p = (N1*np.cos(t0) - N0*cos1) / (N1*np.cos(t0) + N0*cos1)
    r12s = (N1*cos1 - N2*cos2) / (N1*cos1 + N2*cos2)
    r12p = (N2*cos1 - N1*cos2) / (N2*cos1 + N1*cos2)

    # 首末透射系数（仅双光束用）
    t01s = 2*N0*np.cos(t0) / (N0*np.cos(t0) + N1*cos1)
    t01p = 2*N0*np.cos(t0) / (N1*np.cos(t0) + N0*cos1)
    t10s = 2*N1*cos1 / (N1*cos1 + N0*np.cos(t0))
    t10p = 2*N1*cos1 / (N0*cos1 + N1*np.cos(t0))

    # 相位 (2δ 的指数相位)；σ=cm^-1，d=cm
    phi = 4 * np.pi * wv * N1 * d_cm * cos1

    if is_multibeam:
        # Airy 闭式：r = (r01 + r12 e^{iφ})/(1 + r01 r12 e^{iφ})
        rs = (r01s + r12s*np.exp(1j*phi)) / (1 + r01s*r12s*np.exp(1j*phi))
        rp = (r01p + r12p*np.exp(1j*phi)) / (1 + r01p*r12p*np.exp(1j*phi))
    else:
        # 双光束：r ≈ r01 + t01 t10 r12 e^{iφ}
        rs = r01s + t01s * t10s * r12s * np.exp(1j*phi)
        rp = r01p + t01p * t10p * r12p * np.exp(1j*phi)

    R = 0.5*(np.abs(rs)**2 + np.abs(rp)**2)
    return np.real(R)

# ======================== 数值雅可比 ========================
def jacobian_d_numeric(d_um, p, wv, angle_deg, is_multibeam=True, delta_um=1e-3):
    Rp = calculate_reflectance_thin_film(np.insert(p, 0, d_um + delta_um), wv, angle_deg, is_multibeam) * 100.0
    Rm = calculate_reflectance_thin_film(np.insert(p, 0, d_um - delta_um), wv, angle_deg, is_multibeam) * 100.0
    return np.real((Rp - Rm) / (2.0 * delta_um))

def jacobian_params_numeric(d_um, p, wv, angle_deg, is_multibeam=True, eps_vec=None):
    p = np.asarray(p, float)
    K = len(p); N = len(wv)
    J = np.zeros((N, K), dtype=float)
    if eps_vec is None:
        eps_vec = np.maximum(1e-3 * np.maximum(1.0, np.abs(p)), 1e-3)

    for k in range(K):
        dp = np.zeros_like(p); dp[k] = eps_vec[k]
        Rp = calculate_reflectance_thin_film(np.insert(p + dp, 0, d_um), wv, angle_deg, is_multibeam) * 100.0
        Rm = calculate_reflectance_thin_film(np.insert(p - dp, 0, d_um), wv, angle_deg, is_multibeam) * 100.0
        J[:, k] = np.real((Rp - Rm) / (2.0 * eps_vec[k]))
    return J

# ======================== 拟合：步骤一（固定 d 拟合 p） ========================
def fit_optical_params(exp_wavenumbers, exp_reflectance_percent, angle_deg, fixed_d_um, is_multibeam=True):
    """
    返回 (p_hat, Sigma_p, sigma_R_sq_hat, report_dict)
    Σ_p ≈ σ^2 (J_p^T J_p)^{-1} （高斯-牛顿近似）
    """
    y = np.asarray(exp_reflectance_percent, float)
    wv = np.asarray(exp_wavenumbers, float)

    def loss_p(p):
        mod = calculate_reflectance_thin_film(np.insert(p, 0, fixed_d_um), wv, angle_deg, is_multibeam) * 100.0
        return np.mean((mod - y) ** 2)

    # 初值与边界（可按需微调）
    p0 = [6.5, 798.0, 992.0, 10.0, 100.0, 150.0]
    bounds = [(6.0, 7.2), (760, 830), (970, 1030), (1, 30), (0, 1000), (0, 1000)]

    res = minimize(loss_p, p0, method="L-BFGS-B", bounds=bounds, options={"disp": False})
    p_hat = res.x

    R_hat = calculate_reflectance_thin_film(np.insert(p_hat, 0, fixed_d_um), wv, angle_deg, is_multibeam) * 100.0
    residuals = y - R_hat
    N = len(y)
    dof = max(N - len(p_hat), 1)
    sigma_R_sq_hat = float(np.sum(residuals**2) / dof)

    # 数值雅可比 J_p (N×6)，百分数单位
    Jp = jacobian_params_numeric(fixed_d_um, p_hat, wv, angle_deg, is_multibeam, eps_vec=None)
    JTJ = Jp.T @ Jp
    try:
        Sigma_p = sigma_R_sq_hat * np.linalg.inv(JTJ + 1e-12*np.eye(JTJ.shape[0]))
    except np.linalg.LinAlgError:
        Sigma_p = np.full((len(p_hat), len(p_hat)), np.nan)

    report = dict(N=N, mse=float(np.mean((R_hat - y)**2)), sigma_R_sq=sigma_R_sq_hat)
    return p_hat, Sigma_p, sigma_R_sq_hat, report

# ======================== 拟合：步骤三（固定 p 拟合 d） ========================
def fit_thickness_and_uncertainty(wv, R_percent, angle_deg, p_fixed, is_multibeam=True,
                                  bounds_um=(8.0, 10.0), delta_um=1e-3,
                                  Sigma_p_for_propagation=None, sigma_R_sq_from_fit=None):
    """
    返回 d_hat 及报告：
      mse, sigma_R_sq, sumJ2, Fisher, SE_cond, CI95_cond, （可选）SE_total, CI95_total
    """
    y = np.asarray(R_percent, float)
    wv = np.asarray(wv, float)

    def loss_d(d_um):
        mod = calculate_reflectance_thin_film(np.insert(p_fixed, 0, d_um), wv, angle_deg, is_multibeam) * 100.0
        return np.mean((mod - y) ** 2)

    res = minimize_scalar(loss_d, bounds=bounds_um, method="bounded")
    d_hat = float(res.x)

    R_hat = calculate_reflectance_thin_film(np.insert(p_fixed, 0, d_hat), wv, angle_deg, is_multibeam) * 100.0
    residuals = y - R_hat
    N = len(y)
    dof = max(N - 1, 1)
    sigma_R_sq_hat = float(np.sum(residuals**2) / dof) if sigma_R_sq_from_fit is None else float(sigma_R_sq_from_fit)

    # 条件方差：SE(d|p)
    Jd = jacobian_d_numeric(d_hat, p_fixed, wv, angle_deg, is_multibeam, delta_um)  # N
    sumJ2 = float(np.sum(Jd**2))
    if sumJ2 <= 0 or not np.isfinite(sumJ2):
        SE_cond = np.inf; Fisher = 0.0
    else:
        Fisher = sumJ2 / sigma_R_sq_hat
        SE_cond = float(np.sqrt(sigma_R_sq_hat / sumJ2))

    z = 1.96
    CI95_cond = (d_hat - z*SE_cond, d_hat + z*SE_cond)

    # Δ-method 把 Σ_p 传到 Var(d)
    SE_total = None; CI95_total = None
    if (Sigma_p_for_propagation is not None) and np.all(np.isfinite(Sigma_p_for_propagation)) and np.isfinite(sumJ2) and (sumJ2 > 0):
        Jp = jacobian_params_numeric(d_hat, p_fixed, wv, angle_deg, is_multibeam, eps_vec=None)  # N×6
        A = sumJ2
        B = (Jd[None, :] @ Jp)  # 1×6
        middle = B @ Sigma_p_for_propagation @ B.T  # 1×1
        var_total = sigma_R_sq_hat / A + float(middle) / (A**2 + 1e-18)
        SE_total = float(np.sqrt(max(var_total, 0.0)))
        CI95_total = (d_hat - z*SE_total, d_hat + z*SE_total)

    report = dict(
        N=N,
        mse=float(np.mean((R_hat - y)**2)),
        sigma_R_sq=sigma_R_sq_hat,
        sumJ2=sumJ2,
        Fisher=Fisher,
        SE_cond=SE_cond,
        CI95_cond=CI95_cond,
        SE_total=SE_total,
        CI95_total=CI95_total,
    )
    return d_hat, report

# ======================== 打印工具 ========================
def print_block_report(title, path, angle_deg, rpt, d_hat):
    print("\n================ 拟合与灵敏度报告 ================")
    print(f"文件: {path}   入射角: {angle_deg}°")
    print(f"N = {rpt['N']}")
    print(f"d̂ = {d_hat:.6f} μm")
    print(f"MSE = {rpt['mse']:.6f}  (%^2)")
    print(f"σ_R^2(无偏) = {rpt['sigma_R_sq']:.6f}  (%^2)")
    print(f"∑ J_i^2 = {rpt['sumJ2']:.6f}  ((%/μm)^2)")
    print(f"Fisher 信息 = {rpt['Fisher']:.6f}  (1/μm^2)")
    print(f"SE(d̂) = {rpt['SE_cond']:.6f} μm")
    ciL, ciH = rpt['CI95_cond']
    print(f"95% CI = [{ciL:.6f}, {ciH:.6f}] μm")
    if rpt.get('SE_total') is not None:
        ciLt, ciHt = rpt['CI95_total']
        print(f"[含参数不确定度] SE_total = {rpt['SE_total']:.6f} μm")
        print(f"[含参数不确定度] 95% CI_total = [{ciLt:.6f}, {ciHt:.6f}] μm")
    print("==================================================")
BANDS = [
    ("2.5–5 μm", 2000.0, 4100.0),
    ("5–17 μm", 590.0, 2000.0),
    ("17–25 μm", 390.0, 590.0),
]

def _band_shares(wv: np.ndarray, Jd: np.ndarray, bands=BANDS):
    """按波段计算 ∑J^2 占比，返回 shares(list) 和 labels(list)。"""
    J2 = np.real(Jd)**2
    total = float(np.sum(J2)) if np.sum(J2) > 0 else 1.0
    shares, labels = [], []
    for label, lo, hi in bands:
        m = (wv >= lo) & (wv < hi)
        shares.append(float(np.sum(J2[m]) / total))
        labels.append(label)
    return shares, labels

def plot_reflectance_overlay(
    wv10, R10, wv15, R15,
    R10_fit=None, R15_fit=None,
    title="Reflectance (10° & 15°)",
    savepath=None
):
    """一张图上叠加 10° 与 15° 的反射率曲线（可选叠加模型拟合曲线）。"""
    plt.figure(figsize=(12, 4.8))
    # 实验
    plt.plot(wv10, R10, lw=1.4, label="Exp 10°")
    plt.plot(wv15, R15, lw=1.4, label="Exp 15°")
    # 模型（可选）
    if R10_fit is not None:
        plt.plot(wv10, R10_fit, lw=2.0, alpha=0.9, label="Model 10°")
    if R15_fit is not None:
        plt.plot(wv15, R15_fit, lw=2.0, alpha=0.9, label="Model 15°")

    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Reflectance (%)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_band_contrib_grouped(
    wv10, J10, wv15, J15, bands=BANDS,
    title="Information Share by Band",
    savepath=None
):
    """分组柱状图：两个入射角在每个波段的 ∑J^2 占比。"""
    s10, labels = _band_shares(wv10, J10, bands=bands)
    s15, _      = _band_shares(wv15, J15, bands=bands)

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 4.6))
    plt.bar(x - width/2, s10, width, label="10°")
    plt.bar(x + width/2, s15, width, label="15°")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    for i, v in enumerate(s10):
        plt.text(i - width/2, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=10)
    for i, v in enumerate(s15):
        plt.text(i + width/2, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.title(title)
    plt.ylabel(r"Share of $\sum J^2$")
    plt.grid(axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_sensitivity_overlay(
    wv10, J10, wv15, J15,
    title=r"Sensitivity $J_d=\partial R/\partial d$ (no cumulative)",
    savepath=None
):
    """一张图上叠加 10° 与 15° 的灵敏度谱（不画累计信息）。"""
    # 为了视觉整洁，按波数升序画
    o10 = np.argsort(wv10); w10 = wv10[o10]; j10 = np.real(J10[o10])
    o15 = np.argsort(wv15); w15 = wv15[o15]; j15 = np.real(J15[o15])

    plt.figure(figsize=(12, 4.8))
    plt.plot(w10, j10, lw=1.6, label="10°")
    plt.plot(w15, j15, lw=1.6, label="15°")
    plt.axhline(0, color="k", lw=0.8, alpha=0.5)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Jacobian  (% per μm)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

BANDS = [
    ("2.5–5 μm", 2000.0, 4100.0),
    ("5–17 μm", 590.0, 2000.0),
    ("17–25 μm", 390.0, 590.0),
]
def _band_shares(wv: np.ndarray, Jd: np.ndarray, bands=BANDS):
    """按波段计算 ∑J^2 占比，返回 shares(list) 和 labels(list)。"""
    J2 = np.real(Jd)**2
    total = float(np.sum(J2)) if np.sum(J2) > 0 else 1.0
    shares, labels = [], []
    for label, lo, hi in bands:
        m = (wv >= lo) & (wv < hi)
        shares.append(float(np.sum(J2[m]) / total))
        labels.append(label)
    return shares, labels

def plot_reflectance_overlay(
    wv10, R10, wv15, R15,
    R10_fit=None, R15_fit=None,
    title="Reflectance (10° & 15°)",
    savepath=None
):
    """一张图上叠加 10° 与 15° 的反射率曲线（可选叠加模型拟合曲线）。"""
    plt.figure(figsize=(12, 4.8))
    # 实验
    plt.plot(wv10, R10, lw=1.4, label="Exp 10°")
    plt.plot(wv15, R15, lw=1.4, label="Exp 15°")
    # 模型（可选）
    if R10_fit is not None:
        plt.plot(wv10, R10_fit, lw=2.0, alpha=0.9, label="Model 10°")
    if R15_fit is not None:
        plt.plot(wv15, R15_fit, lw=2.0, alpha=0.9, label="Model 15°")

    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Reflectance (%)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_band_contrib_grouped(
    wv10, J10, wv15, J15, bands=BANDS,
    title="Information Share by Band",
    savepath=None
):
    """分组柱状图：两个入射角在每个波段的 ∑J^2 占比。"""
    s10, labels = _band_shares(wv10, J10, bands=bands)
    s15, _      = _band_shares(wv15, J15, bands=bands)

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 4.6))
    plt.bar(x - width/2, s10, width, label="10°")
    plt.bar(x + width/2, s15, width, label="15°")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    for i, v in enumerate(s10):
        plt.text(i - width/2, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=10)
    for i, v in enumerate(s15):
        plt.text(i + width/2, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.title(title)
    plt.ylabel(r"Share of $\sum J^2$")
    plt.grid(axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_sensitivity_overlay(
    wv10, J10, wv15, J15,
    title=r"Sensitivity $J_d=\partial R/\partial d$ (no cumulative)",
    savepath=None
):
    """一张图上叠加 10° 与 15° 的灵敏度谱（不画累计信息）。"""
    # 为了视觉整洁，按波数升序画
    o10 = np.argsort(wv10); w10 = wv10[o10]; j10 = np.real(J10[o10])
    o15 = np.argsort(wv15); w15 = wv15[o15]; j15 = np.real(J15[o15])

    plt.figure(figsize=(12, 4.8))
    plt.plot(w10, j10, lw=1.6, label="10°")
    plt.plot(w15, j15, lw=1.6, label="15°")
    plt.axhline(0, color="k", lw=0.8, alpha=0.5)
    plt.title(title)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Jacobian  (% per μm)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
# ======================== 主流程 ========================
def main():
    # 数据文件（10°、15°）
    file1, ang1 = "附件1.xlsx", 10
    file2, ang2 = "附件2.xlsx", 15

    # 读取与预处理
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    df1p = preprocess_data(df1)
    df2p = preprocess_data(df2)
    wv1, R1 = df1p.iloc[:, 0].values.astype(float), df1p.iloc[:, 1].values.astype(float)
    wv2, R2 = df2p.iloc[:, 0].values.astype(float), df2p.iloc[:, 1].values.astype(float)

    # 配置
    IS_MULTIBEAM = True     # True=Airy 多光束；False=双光束
    D_FIX = 9.000           # 步骤一固定 d 的值（可按经验改）
    BOUNDS_UM = (8.0, 10.0)
    DELTA_UM = 1e-3

    # 步骤一：固定 d，拟合 p
    print(f"\n=== 步骤一：固定 d={D_FIX:.3f} μm，拟合 LD 六参 ===")
    p10, Sigma_p10, sigmaR10, rep10 = fit_optical_params(wv1, R1, ang1, D_FIX, is_multibeam=IS_MULTIBEAM)
    p15, Sigma_p15, sigmaR15, rep15 = fit_optical_params(wv2, R2, ang2, D_FIX, is_multibeam=IS_MULTIBEAM)
    print(f"角度 {ang1}° 光学参数：{np.round(p10, 4)}")
    print(f"角度 {ang2}° 光学参数：{np.round(p15, 4)}")

    # 步骤二：参数取平均
    p_avg = 0.5 * (p10 + p15)
    Sigma_p_avg = None
    if np.all(np.isfinite(Sigma_p10)) and np.all(np.isfinite(Sigma_p15)):
        Sigma_p_avg = 0.5 * (Sigma_p10 + Sigma_p15)
    sigmaR_avg = 0.5 * (sigmaR10 + sigmaR15)
    print(f"平均光学参数 p_avg：{np.round(p_avg, 4)}")

    # 步骤三：固定 p_avg，分别拟合 d，并评估不确定度
    d10, rpt10 = fit_thickness_and_uncertainty(
        wv1, R1, ang1, p_avg, is_multibeam=IS_MULTIBEAM,
        bounds_um=BOUNDS_UM, delta_um=DELTA_UM,
        Sigma_p_for_propagation=Sigma_p_avg, sigma_R_sq_from_fit=sigmaR_avg
    )
    d15, rpt15 = fit_thickness_and_uncertainty(
        wv2, R2, ang2, p_avg, is_multibeam=IS_MULTIBEAM,
        bounds_um=BOUNDS_UM, delta_um=DELTA_UM,
        Sigma_p_for_propagation=Sigma_p_avg, sigma_R_sq_from_fit=sigmaR_avg
    )

    # 打印成你要的块状报告
    print_block_report("report", file1, ang1, rpt10, d10)
    print_block_report("report", file2, ang2, rpt15, d15)

    # 步骤四：逆方差加权（条件 / 总不确定度）
    print("\n=== 多角度合并（逆方差加权） ===")
    z = 1.96

    # 条件版
    se10_c, se15_c = rpt10['SE_cond'], rpt15['SE_cond']
    w10c = 0.0 if not np.isfinite(se10_c) else 1.0/(se10_c**2 + 1e-18)
    w15c = 0.0 if not np.isfinite(se15_c) else 1.0/(se15_c**2 + 1e-18)
    if (w10c + w15c) > 0:
        d_cond = (w10c*d10 + w15c*d15)/(w10c + w15c)
        SEc = np.sqrt(1.0/(w10c + w15c))
        print(f"[条件] d̂ = {d_cond:.6f} μm | 95%CI|p = [{d_cond - z*SEc:.6f}, {d_cond + z*SEc:.6f}] μm")
    else:
        print("[条件] 权重为 0，无法合并。")

    # 总不确定度版
        # 总不确定度版
    if (rpt10.get('SE_total') is not None) and (rpt15.get('SE_total') is not None):
        se10_t, se15_t = rpt10['SE_total'], rpt15['SE_total']
        if np.isfinite(se10_t) and np.isfinite(se15_t):
            w10t = 1.0/(se10_t**2 + 1e-18); w15t = 1.0/(se15_t**2 + 1e-18)
            d_tot = (w10t*d10 + w15t*d15)/(w10t + w15t)
            SEt = np.sqrt(1.0/(w10t + w15t))
            print(f"[总计] d̂ = {d_tot:.6f} μm | 95%CI_total = [{d_tot - z*SEt:.6f}, {d_tot + z*SEt:.6f}] μm")
        else:
            print("[总计] SE_total 非有限，未合并。")
    else:
        print("[总计] 缺少 Σ_p 或 SE_total，未执行总不确定度合并。")

    # ========= 下面三段【无条件】执行的可视化 =========
    # 生成模型拟合曲线（用于叠加，可选）
    Rfit10 = calculate_reflectance_thin_film(np.insert(p_avg, 0, d10), wv1, ang1, is_multibeam=IS_MULTIBEAM) * 100.0
    Rfit15 = calculate_reflectance_thin_film(np.insert(p_avg, 0, d15), wv2, ang2, is_multibeam=IS_MULTIBEAM) * 100.0

    # 1) 一张图叠加 10° & 15° 的反射率（含模型曲线，可选）
    plot_reflectance_overlay(
        wv1, R1, wv2, R2,
        R10_fit=Rfit10, R15_fit=Rfit15,
        title="Reflectance @ 10° & 15° (Exp + Model)",
        savepath="overlay_reflectance.png"   # 或 None 直接显示
    )
    import os; print("Saved:", os.path.abspath("overlay_reflectance.png"))

    # 2) 计算灵敏度谱（整段波数，用于下面两张图）
    Jd10 = jacobian_d_numeric(d10, p_avg, wv1, ang1, is_multibeam=IS_MULTIBEAM, delta_um=DELTA_UM)
    Jd15 = jacobian_d_numeric(d15, p_avg, wv2, ang2, is_multibeam=IS_MULTIBEAM, delta_um=DELTA_UM)

    # 2a) 入射角不同波段信息占比（分组柱状图）
    plot_band_contrib_grouped(
        wv1, Jd10, wv2, Jd15,
        bands=BANDS,
        title="Information Share by Band (10° vs 15°)",
        savepath="band_share_grouped.png"
    )
    print("Saved:", os.path.abspath("band_share_grouped.png"))

    # 2b) 入射角灵敏度谱（不画累计信息；两条线叠加）
    plot_sensitivity_overlay(
        wv1, Jd10, wv2, Jd15,
        title=r"Sensitivity Spectrum $J_d$ (10° vs 15°)",
        savepath="sensitivity_overlay.png"
    )
    print("Saved:", os.path.abspath("sensitivity_overlay.png"))
if __name__ == "__main__":
    main()