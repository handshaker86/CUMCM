# -*- coding: utf-8 -*-
"""
sic_rhoa_dense.py  —  SiC 多光束必要性 (|rho_A|) 致密计算 & 连续超阈值区间标注

你要的要点：
- 直接用物理定义在完整波段上致密计算 |rho_A|（不做峰谷配对）
- 400–4000 cm^-1 全覆盖（实际使用文件中的波数列；若范围更宽也自动适配）
- 绿色阴影标出“连续超阈值”区间（|rho_A| > THRESHOLD）
- 两张图分别保存为：
    rho_dense_10_1.png  (附件1.xlsx, 10°)
    rho_dense_15_2.png  (附件2.xlsx, 15°)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- 全局配置 --------------------
LANG = "en"   # "zh" 或 "en"
THRESHOLD = 0.10          # 判定阈值：|rho_A| > THRESHOLD → 需要 TMM
OUT_FIG_DPI = 220         # 输出分辨率
YMAX_CLIP = 0.9           # y 轴上限（自动根据数据裁剪到此值以内，避免过高）

def L(zh, en):
    return zh if LANG == "zh" else en

if LANG == "zh":
    # 兼容中文字体（可选）
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Noto Sans CJK SC","Microsoft YaHei","PingFang SC","Heiti SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

# -------------------- 兜底预处理 --------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """默认：取前两列作为 [波数, 反射率]，仅用第一列（波数）。"""
    out = df.iloc[:, :2].copy()
    out = out.dropna()
    return out

# -------------------- SiC 折射率模型（Lorentz-Drude 6参） --------------------
def n_sic_LD(wavenumbers_cm, params):
    """
    params = [eps_inf, sig_TO, sig_LO, gamma_ph, sig_p, gamma_e]  (单位：cm^-1)
    返回复折射率 n1(σ) （外延层）
    """
    eps_inf, sig_TO, sig_LO, gamma_ph, sig_p, gamma_e = params
    sigma = np.asarray(wavenumbers_cm, dtype=float)
    i = 1j

    # Lorentz(phonon)
    lorentz = (
        eps_inf
        * (sigma**2 - sig_LO**2 + i*gamma_ph*sigma)
        / (sigma**2 - sig_TO**2 + i*gamma_ph*sigma)
    )
    # Drude(free carrier)
    s = sigma + 1e-12
    drude = (sig_p**2) / (s * (s + i*gamma_e))

    eps = lorentz - drude
    return np.sqrt(eps + 0j)

def n_substrate(wavenumbers_cm):
    """重掺杂 SiC 衬底的一组代表值（可按需调整）"""
    p_sub = [6.5, 797.7, 992.1, 8.0, 600.0, 300.0]
    return n_sic_LD(wavenumbers_cm, p_sub)

# -------------------- 几何与菲涅耳 --------------------
def snell_cos_theta(n0, n1, theta0_rad):
    """返回 cos(theta1)；允许 n1 为复数。"""
    s1 = n0*np.sin(theta0_rad)/n1
    return np.lib.scimath.sqrt(1 - s1**2)

def fresnel_r01_rs_rp(n0, n1, cos0, cos1):
    r_s = (n0*cos0 - n1*cos1) / (n0*cos0 + n1*cos1)
    r_p = (n1*cos0 - n0*cos1) / (n1*cos0 + n0*cos1)
    return r_s, r_p

def fresnel_r12_rs_rp(n1, n2, cos1, cos2):
    r_s = (n1*cos1 - n2*cos2) / (n1*cos1 + n2*cos2)
    r_p = (n2*cos1 - n1*cos2) / (n2*cos1 + n1*cos2)
    return r_s, r_p

# -------------------- |rho_A| 致密计算 --------------------
def rhoA_dense(sigma, theta0_deg, d_um, p_layer):
    """
    物理定义：
    rho_A(σ) = |r10 * r12|_unpol * exp[- 4π σ κ1 d cosθ1]
    其中 |r10*r12|_unpol = sqrt( (|r_s|^2 + |r_p|^2)/2 ) 逐界面乘起来。
    """
    n0 = 1.0
    n1 = n_sic_LD(sigma, p_layer)   # 外延层
    n2 = n_substrate(sigma)         # 衬底

    d_cm = d_um * 1e-4
    t0 = np.deg2rad(theta0_deg)

    # 入/出角余弦
    cos0 = np.cos(t0) + 0j
    cos1 = snell_cos_theta(n0, n1, t0)
    cos2 = snell_cos_theta(n1, n2, np.arcsin(n0*np.sin(t0)/n1))

    # 两个界面的 r_s, r_p
    r01s, r01p = fresnel_r01_rs_rp(n0, n1, cos0, cos1)
    r12s, r12p = fresnel_r12_rs_rp(n1, n2, cos1, cos2)

    # 非偏振：强度均方根
    R01 = np.sqrt( (np.abs(r01s)**2 + np.abs(r01p)**2)/2 )
    R12 = np.sqrt( (np.abs(r12s)**2 + np.abs(r12p)**2)/2 )

    # 吸收衰减：kappa = Im(n1)
    kappa = np.imag(n1)
    attenuation = np.exp(-4*np.pi*sigma*kappa*d_cm*np.real(cos1))

    rho = (R01 * R12) * attenuation
    return np.real(rho)

# -------------------- 连续超阈值区间 --------------------
def contiguous_bands(x, y, thr, merge_tol_cm=60.0):
    """
    x 升序，y=|rho_A|；返回 [(L,R), ...] 满足 y>thr 的合并区间
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = y > thr
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    bands = []
    L = x[idx[0]]; R = x[idx[0]]
    for i_prev, i_cur in zip(idx[:-1], idx[1:]):
        if x[i_cur] - x[i_prev] <= merge_tol_cm:  # 近邻/相邻，合并
            R = x[i_cur]
        else:
            bands.append((L, R))
            L, R = x[i_cur], x[i_cur]
    bands.append((L, R))
    return bands

# -------------------- 绘图 --------------------
def plot_rho_with_bands(sigma, rho, bands, angle_deg, savepath):
    ymax = min(max(rho)*1.10, YMAX_CLIP)
    plt.figure(figsize=(12.6, 4.4))
    for Lb, Rb in bands:
        plt.axvspan(Lb, Rb, color="#2ca02c", alpha=0.15)
    plt.plot(sigma, rho, lw=2.0, color="#f28e2b", label=L(r"$|\rho_A|$（致密）", r"$|\rho_A|$ (dense)"))
    plt.axhline(THRESHOLD, ls="--", color="#b22222", lw=1.5, label=L(f"阈值 {THRESHOLD:.2f}", f"Threshold {THRESHOLD:.2f}"))
    plt.xlabel(L("Wavenumber (cm$^{-1}$)", "Wavenumber (cm$^{-1}$)"))
    plt.ylabel(L(r"$|\rho_A|$", r"$|\rho_A|$"))
    plt.ylim(0, max(0.11, ymax))
    plt.xlim(float(sigma.min()), float(sigma.max()))
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.title(L(f"{angle_deg}° 多光束必要性", f"{angle_deg}° Multi-beam Necessity"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=OUT_FIG_DPI, bbox_inches="tight")
    plt.close()
    print("Saved:", os.path.abspath(savepath))

# -------------------- 主流程 --------------------
def run_one(xlsx_path: str, angle_deg: float, out_png: str,
            d_um_est: float,
            p_layer=None):
    """
    - xlsx_path: 仅用其第一列（波数）来确定采样与范围
    - angle_deg: 入射角
    - d_um_est: 估计厚度（用于吸收项），用你前面拟合出来的厚度
    - p_layer: SiC 外延层 LD 参数；如果 None 使用一组稳健初值
    """
    if p_layer is None:
        # 一套与前面拟合相近、能稳定生成结论的默认参数
        p_layer = [6.6, 800.0, 995.0, 10.0, 120.0, 160.0]

    df = pd.read_excel(xlsx_path)
    dfp = preprocess_data(df)
    sigma_raw = np.asarray(dfp.iloc[:, 0].values, float)

    # 用文件本身的采样（去重 & 升序）；若范围超过 [400,4000] 不做截断
    sigma = np.unique(np.sort(sigma_raw))
    # 若数据过稀，可插值加密到 ~2000 点
    if sigma.size < 1200:
        sigma = np.linspace(sigma.min(), sigma.max(), 2000)

    rho = rhoA_dense(sigma, angle_deg, d_um_est, p_layer)
    bands = contiguous_bands(sigma, rho, THRESHOLD, merge_tol_cm=60.0)
    plot_rho_with_bands(sigma, rho, bands, angle_deg, out_png)

def main():
    # —— 你前面得到的厚度估计（吸收项里需要 d）；没有精确值时用 9.7 μm 也能给出保守结论
    d_est_um_for_absorption = 9.7

    # —— 按你的要求输出这两个文件名 ——
    run_one("附件1.xlsx", 10, "rho_dense_10_1.png", d_um_est=d_est_um_for_absorption)
    run_one("附件2.xlsx", 15, "rho_dense_15_2.png", d_um_est=d_est_um_for_absorption)

if __name__ == "__main__":
    main()