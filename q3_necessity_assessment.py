import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q1_method2 import preprocess_data  
import matplotlib

# --- 用户配置区 ---
LANG = "en"            # "zh" 或 "en"
THRESHOLD = 0.10       # |rho_A| 判定阈值（保守阈值）
SIGMA_R = 0.5          # 绝对反射率噪声（单位：百分比点）；若原始数据为0–1标度会自动换算
PEAK_PROM = 0.10       # 峰/谷显著性（百分比点；0–1标度会自动×100）
PEAK_DIST = 4          # 峰/谷最小间距（采样点）
DETREND_WIN = 51       # 去趋势窗口（奇数）
SMOOTH_WIN = 3         # 稀疏 |rho_A| 的移动平均窗口(点数)
INTERP_GRID = 800      # 插值后的采样点数（pairs模式用）
BAND_MERGE = 60        # 区间合并容差(cm^-1)
BAND_EXPAND_FSR = 1.0  # 用本地FSR扩宽单点区间的倍数（pairs模式用）
CONSERVATIVE = True    # True: 用 (rhoA - sigma_rho) 与阈值比较（更保守）
WIN_CM   = 60.0        # 窗口半宽（±cm^-1）
STEP_CM  = 5.0         # 计算步长（cm^-1）
USE_QUANTILE = True    # True 用分位数95%/5%，False 用 max/min

def _use_chinese_font():
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Noto Sans CJK SC","Microsoft YaHei","PingFang SC","Heiti SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

def L(zh, en):
    return zh if LANG == "zh" else en

if LANG == "zh":
    _use_chinese_font()

def normalize_R(R_raw, sigma_r=SIGMA_R, peak_prom=PEAK_PROM):
    R_raw = np.asarray(R_raw, float)
    if np.nanmax(R_raw) > 1.5:
        return R_raw, sigma_r, peak_prom
    else:
        return R_raw * 100.0, sigma_r, peak_prom

def rhoA_visibility_sliding(sigma, R_exp, win_cm=60.0, step_cm=5.0, sigma_R_abs=0.5, use_quantile=True):
    sigma = np.asarray(sigma, float)
    R = np.asarray(R_exp, float)
    smin, smax = float(sigma.min()), float(sigma.max())
    xs = np.arange(smin, smax + step_cm/2, step_cm)

    out_x, out_V, out_sV, out_rho, out_sr = [], [], [], [], []
    for xc in xs:
        m = (sigma >= xc - win_cm) & (sigma <= xc + win_cm)
        if m.sum() < 10:
            continue
        if use_quantile:
            rmax = np.quantile(R[m], 0.95)
            rmin = np.quantile(R[m], 0.05)
        else:
            rmax = np.max(R[m]); rmin = np.min(R[m])

        if (rmax + rmin) <= 1e-9:
            continue
        V = (rmax - rmin) / (rmax + rmin)
        # σ_V
        denom2 = (rmax + rmin)**2 + 1e-24
        sV = (2.0/denom2) * np.sqrt((rmin**2 + rmax**2) * (sigma_R_abs**2))

        Vc = np.clip(V, 1e-6, 0.999999)
        rhoA = (1 - np.sqrt(1 - Vc**2)) / Vc
        gain = abs(rhoA) / (Vc * np.sqrt(1 - Vc**2))
        srho = gain * sV

        out_x.append(xc); out_V.append(V); out_sV.append(sV); out_rho.append(rhoA); out_sr.append(srho)

    return {
        "sigma": np.array(out_x),
        "V": np.array(out_V),
        "sigma_V": np.array(out_sV),
        "rhoA": np.array(out_rho),
        "sigma_rho": np.array(out_sr),
    }

def bands_from_dense(sigma, rhoA, sigma_rho, thr=0.10, merge_tol=60.0, conservative=True):
    if len(sigma) == 0:
        return []
    score = (rhoA - sigma_rho) if (conservative and sigma_rho is not None) else rhoA
    over = score > thr
    if not np.any(over):
        return []
    x = np.asarray(sigma, float)
    idx = np.where(over)[0]
    bands = []
    start = idx[0]
    prev = idx[0]
    for j in idx[1:]:
        if (x[j] - x[prev]) <= merge_tol:
            prev = j
        else:
            bands.append((float(x[start]), float(x[prev])))
            start, prev = j, j
    bands.append((float(x[start]), float(x[prev])))
    return bands

def plot_dense_curve_and_bands(tag, sigma, rhoA, thr, bands=None, save=None):
    plt.figure(figsize=(11,4.2))
    if len(sigma):
        plt.plot(sigma, rhoA, lw=1.4, color="tab:orange", label="|ρ_A| (dense)")
    plt.axhline(thr, ls="--", color="tab:red", label=L(f"阈值 {thr:.2f}", f"Threshold {thr:.2f}"))
    if bands:
        for (Lcm, Rcm) in bands:
            plt.axvspan(Lcm, Rcm, color="tab:green", alpha=0.18)
    plt.xlabel(L("波数 (cm$^{-1}$)", "Wavenumber (cm$^{-1}$)"))
    plt.ylabel("|ρ_A|")
    plt.title(L(f"{tag} 多光束必要性（方法A-2: 滑动窗口）", f"{tag} Multi-beam Necessity "))
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    if save: plt.savefig(save, dpi=200, bbox_inches="tight"); plt.close()
    else: plt.show()

def main():
    file1 = "附件3.xlsx"   # 假设 10°
    file2 = "附件4.xlsx"   # 假设 15°

    # 读取与兜底预处理
    df1 = preprocess_data(pd.read_excel(file1))
    df2 = preprocess_data(pd.read_excel(file2))
    wn1, R1 = df1.iloc[:,0].values.astype(float), df1.iloc[:,1].values.astype(float)
    wn2, R2 = df2.iloc[:,0].values.astype(float), df2.iloc[:,1].values.astype(float)

    # 统一为“百分数标度”
    R1p, sigR1, prom1 = normalize_R(R1, sigma_r=SIGMA_R, peak_prom=PEAK_PROM)
    R2p, sigR2, prom2 = normalize_R(R2, sigma_r=SIGMA_R, peak_prom=PEAK_PROM)

    print(L("方法A：由条纹可见度判定多光束必要性", 
            "Method A: Interference visibility to decide multi-beam necessity"))
    print(L(f"阈值 |ρ_A| = {THRESHOLD:.2f}（可按精度需求调整）",
            f"Threshold |ρ_A| = {THRESHOLD:.2f} (adjust per accuracy target)"))
    
    res1 = rhoA_visibility_sliding(wn1, R1p, win_cm=WIN_CM, step_cm=STEP_CM, sigma_R_abs=sigR1, use_quantile=USE_QUANTILE)
    res2 = rhoA_visibility_sliding(wn2, R2p, win_cm=WIN_CM, step_cm=STEP_CM, sigma_R_abs=sigR2, use_quantile=USE_QUANTILE)

    bands1 = bands_from_dense(res1["sigma"], res1["rhoA"], res1["sigma_rho"],
                                thr=THRESHOLD, merge_tol=BAND_MERGE, conservative=CONSERVATIVE)
    bands2 = bands_from_dense(res2["sigma"], res2["rhoA"], res2["sigma_rho"],
                                thr=THRESHOLD, merge_tol=BAND_MERGE, conservative=CONSERVATIVE)

    def _print_dense(tag, res, bands):
        if res["rhoA"].size == 0:
            print(f"[{tag}] " + L("没有足够的窗口点。", "Not enough windowed samples."))
            return
        score = (res["rhoA"] - res["sigma_rho"]) if CONSERVATIVE else res["rhoA"]
        over = score > THRESHOLD
        ratio = float(over.mean())
        print(f"[{tag}] " + L(f"（保守）超阈值占比: {ratio*100:.1f}%",
                                f"(conservative) over-threshold fraction: {ratio*100:.1f}%"))
        if bands:
            seg = "; ".join([f"{b[0]:.0f}–{b[1]:.0f}" for b in bands])
            print(f"[{tag}] " + L(f"连续超阈值波段 (cm^-1): {seg}",
                                    f"Continuous over-threshold bands (cm^-1): {seg}"))
            print(f"[{tag}] " + L("结论：这些波段需用多光束/TMM；其余波段用双光束即可。",
                                    "Conclusion: Use multi-beam/TMM in those bands; two-beam elsewhere."))

    _print_dense("10°", res1, bands1)
    _print_dense("15°", res2, bands2)

    # 绘图：连续曲线 + 阈值 + 阴影区间
    plot_dense_curve_and_bands("10°", res1["sigma"], res1["rhoA"], THRESHOLD, bands=bands1, save="rho_dense_10.png")
    plot_dense_curve_and_bands("15°", res2["sigma"], res2["rhoA"], THRESHOLD, bands=bands2, save="rho_dense_15.png")
    print("Saved:", os.path.abspath("rho_dense_10.png"))
    print("Saved:", os.path.abspath("rho_dense_15.png"))

if __name__ == "__main__":
    main()
