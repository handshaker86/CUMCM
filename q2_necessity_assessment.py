import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from q1_method2 import get_sic_refractive_index_LD, get_sic_refractive_index_substrate

LANG = "en"   
THRESHOLD = 0.10          
OUT_FIG_DPI = 220         
YMAX_CLIP = 0.9           

def L(zh, en):
    return zh if LANG == "zh" else en

if LANG == "zh":
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Noto Sans CJK SC","Microsoft YaHei","PingFang SC","Heiti SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[:, :2].copy()
    out = out.dropna()
    return out

def snell_cos_theta(n0, n1, theta0_rad):
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

def rhoA_dense(sigma, theta0_deg, d_um, p_layer):
    n0 = 1.0
    n1 = get_sic_refractive_index_LD(sigma, p_layer)   
    n2 = get_sic_refractive_index_substrate(sigma)         

    d_cm = d_um * 1e-4
    t0 = np.deg2rad(theta0_deg)

    cos0 = np.cos(t0) + 0j
    cos1 = snell_cos_theta(n0, n1, t0)
    cos2 = snell_cos_theta(n1, n2, np.arcsin(n0*np.sin(t0)/n1))

    r01s, r01p = fresnel_r01_rs_rp(n0, n1, cos0, cos1)
    r12s, r12p = fresnel_r12_rs_rp(n1, n2, cos1, cos2)

    R01 = np.sqrt( (np.abs(r01s)**2 + np.abs(r01p)**2)/2 )
    R12 = np.sqrt( (np.abs(r12s)**2 + np.abs(r12p)**2)/2 )

    kappa = np.imag(n1)
    attenuation = np.exp(-4*np.pi*sigma*kappa*d_cm*np.real(cos1))

    rho = (R01 * R12) * attenuation
    return np.real(rho)

def contiguous_bands(x, y, thr, merge_tol_cm=60.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = y > thr
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    bands = []
    L = x[idx[0]]; R = x[idx[0]]
    for i_prev, i_cur in zip(idx[:-1], idx[1:]):
        if x[i_cur] - x[i_prev] <= merge_tol_cm:  
            R = x[i_cur]
        else:
            bands.append((L, R))
            L, R = x[i_cur], x[i_cur]
    bands.append((L, R))
    return bands

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

def run_one(xlsx_path: str, angle_deg: float, out_png: str,
            d_um_est: float,
            p_layer=None):
    if p_layer is None:
        p_layer = [6.6, 800.0, 995.0, 10.0, 120.0, 160.0]

    df = pd.read_excel(xlsx_path)
    dfp = preprocess_data(df)
    sigma_raw = np.asarray(dfp.iloc[:, 0].values, float)
    sigma = np.unique(np.sort(sigma_raw))
    if sigma.size < 1200:
        sigma = np.linspace(sigma.min(), sigma.max(), 2000)

    rho = rhoA_dense(sigma, angle_deg, d_um_est, p_layer)
    bands = contiguous_bands(sigma, rho, THRESHOLD, merge_tol_cm=60.0)
    plot_rho_with_bands(sigma, rho, bands, angle_deg, out_png)

if __name__ == "__main__":
    d_est_um_for_absorption = 9.7
    run_one("附件1.xlsx", 10, "rho_dense_10_1.png", d_um_est=d_est_um_for_absorption)
    run_one("附件2.xlsx", 15, "rho_dense_15_2.png", d_um_est=d_est_um_for_absorption)