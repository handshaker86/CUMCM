# -*- coding: utf-8 -*-
"""
visibility_multibeam_A.py

判定多光束/TMM 必要性的 方法A（基于条纹可见度）：
- 方法A-1（"pairs"）：去趋势找峰/谷，按横坐标就近配对；由峰谷求 V，再反演 |rho_A|。
  由于每对极值只产生一个采样点，天然稀疏；本脚本将其插值并合并为连续区间。
- 方法A-2（"sliding"）：滑动窗口（±win_cm）在局部用 rmax/rmin（分位数95%/5%）求 V，几乎连续。

输出：
- 终端：保守超阈值占比、连续超阈值区间（cm^-1）
- 图像：|rho_A| 离散点/连续曲线 + 阈值、区间阴影；可选 V 诊断图

依赖：numpy, pandas, scipy, matplotlib
可选：本地 preprocess.py（若没有则兜底使用前两列）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# =======================
# 0) 配置
# =======================
LANG = "en"            # "zh" 或 "en"
METHOD = "sliding"     # "pairs"（极值配对）或 "sliding"（滑动窗口）
THRESHOLD = 0.10       # |rho_A| 判定阈值（保守阈值）
SIGMA_R = 0.5          # 绝对反射率噪声（单位：百分比点）；若原始数据为0–1标度会自动换算
# —— 极值配对法的峰谷检测参数（若使用 METHOD="pairs"）
PEAK_PROM = 0.10       # 峰/谷显著性（百分比点；0–1标度会自动×100）
PEAK_DIST = 4          # 峰/谷最小间距（采样点）
DETREND_WIN = 51       # 去趋势窗口（奇数）
# —— 插值与区间合并（两种方法都会用到）
SMOOTH_WIN = 3         # 稀疏 |rho_A| 的移动平均窗口(点数)
INTERP_GRID = 800      # 插值后的采样点数（pairs模式用）
BAND_MERGE = 60        # 区间合并容差(cm^-1)
BAND_EXPAND_FSR = 1.0  # 用本地FSR扩宽单点区间的倍数（pairs模式用）
CONSERVATIVE = True    # True: 用 (rhoA - sigma_rho) 与阈值比较（更保守）

# —— 滑动窗口法的参数（METHOD="sliding" 时生效）
WIN_CM   = 60.0        # 窗口半宽（±cm^-1）
STEP_CM  = 5.0         # 计算步长（cm^-1）
USE_QUANTILE = True    # True 用分位数95%/5%，False 用 max/min

# =======================
# 字体与多语言
# =======================
def _use_chinese_font():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei","Noto Sans CJK SC","Microsoft YaHei","PingFang SC","Heiti SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

def L(zh, en):
    return zh if LANG == "zh" else en

if LANG == "zh":
    _use_chinese_font()

# =======================
# 1) 基础工具
# =======================
try:
    from preprocess import preprocess_data  # 如果你有自定义清洗流程
except Exception:
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        out = df.iloc[:, :2].copy()
        out = out.dropna()
        return out

def normalize_R(R_raw, sigma_r=SIGMA_R, peak_prom=PEAK_PROM):
    """
    自动识别反射率标度：
    - 若 max(R) > 1.5，认为已是“百分数”（0–100）
    - 否则把 0–1 标度 ×100，再用同样的噪声σ（单位：百分比点）
    """
    R_raw = np.asarray(R_raw, float)
    if np.nanmax(R_raw) > 1.5:
        return R_raw, sigma_r, peak_prom
    else:
        return R_raw * 100.0, sigma_r, peak_prom

def local_detrend(y, width=51):
    """轻度去趋势：滑动平均作为基线"""
    win = int(width)
    if win % 2 == 0:
        win += 1
    base = uniform_filter1d(y, size=win, mode='nearest')
    return y - base, base

# =======================
# 2) 方法A-1：极值配对法（稀疏点 → 插值 → 区间）
# =======================
def pair_peaks_valleys_nearest(x, y_detr, prominence, distance):
    """
    在去趋势信号 y_detr 上找极值后，按横坐标就近的一一配对（相邻且类型相反）。
    返回 [(x_peak, idx_peak, x_valley, idx_valley), ...]
    """
    peaks, _ = find_peaks(y_detr, prominence=prominence, distance=distance)
    valleys, _ = find_peaks(-y_detr, prominence=prominence, distance=distance)
    if len(peaks) == 0 or len(valleys) == 0:
        return []

    tags = np.r_[np.c_[peaks, np.ones_like(peaks)], np.c_[valleys, -np.ones_like(valleys)]]
    tags = tags[np.argsort(tags[:,0])]
    idx_sorted = tags[:,0].astype(int)
    t_sorted = tags[:,1].astype(int)  # 1=peak, -1=valley

    pairs = []
    i = 0
    while i < len(idx_sorted)-1:
        if t_sorted[i] + t_sorted[i+1] == 0:  # 相邻且一峰一谷
            ip, iv = (idx_sorted[i], idx_sorted[i+1]) if t_sorted[i]==1 else (idx_sorted[i+1], idx_sorted[i])
            pairs.append((x[ip], ip, x[iv], iv))
            i += 2
        else:
            i += 1
    return pairs

def visibility_and_rho_from_pairs(pairs, R_exp, sigma_R_abs):
    """
    输入：pairs: [(x_peak, idx_peak, x_valley, idx_valley), ...]
         R_exp: 原始反射率（百分数单位）
    输出：数组 [sigma_mid, V, rhoA, sigma_V, sigma_rho]
    """
    rows = []
    for xpk, ip, xvl, iv in pairs:
        Rmax, Rmin = float(R_exp[ip]), float(R_exp[iv])
        if (Rmax + Rmin) <= 1e-12:
            continue
        V = (Rmax - Rmin) / (Rmax + Rmin + 1e-12)
        # σ_V 传播
        denom2 = (Rmax + Rmin)**2 + 1e-24
        sigV = (2.0/denom2) * np.sqrt((Rmin**2 + Rmax**2) * (sigma_R_abs**2))
        # 反演 |rho_A|
        Vc = np.clip(V, 1e-6, 0.999999)
        rhoA = (1 - np.sqrt(1 - Vc**2)) / Vc
        gain = abs(rhoA) / (Vc*np.sqrt(1 - Vc**2))
        sig_rho = gain * sigV
        rows.append((0.5*(xpk + xvl), V, rhoA, sigV, sig_rho))
    if not rows:
        return np.zeros((0,5))
    return np.array(rows, float)

def moving_average(y, win=3):
    if win <= 1 or len(y) < 2:
        return np.asarray(y, float)
    y = np.asarray(y, float)
    k = max(1, int(win))
    k = k + (k+1)%2  # 奇数
    pad = k//2
    ypad = np.pad(y, (pad,pad), mode="edge")
    c = np.ones(k)/k
    return np.convolve(ypad, c, mode="valid")

def interpolate_rhoA(sigma_mid, rhoA, sigma_rho=None, smooth_win=3, grid=800):
    order = np.argsort(sigma_mid)
    x = np.asarray(sigma_mid)[order]
    y = np.asarray(rhoA)[order]
    y_s = moving_average(y, win=smooth_win)
    xi = np.linspace(x.min(), x.max(), grid)
    yi = np.interp(xi, x, y_s)
    if sigma_rho is not None:
        s = np.asarray(sigma_rho)[order]
        s_s = moving_average(s, win=smooth_win)
        si = np.interp(xi, x, s_s)
    else:
        si = None
    return xi, yi, si

def estimate_local_fsr(sigma_mid):
    x = np.asarray(sigma_mid, float)
    x = x[np.argsort(x)]
    if len(x) < 2:
        return np.zeros_like(x)
    left  = np.r_[x[1]-x[0], np.diff(x)]
    right = np.r_[np.diff(x), x[-1]-x[-2]]
    fsr = 0.5*(left + right)
    # 回到原索引
    order = np.argsort(sigma_mid)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    fsr_unsorted = np.zeros_like(sigma_mid, float)
    fsr_unsorted[order] = fsr
    return fsr_unsorted

def bands_from_points(sigma_mid, rhoA, sigma_rho, thr=0.10,
                      merge_tol=60.0, expand_fsr=1.0, conservative=True):
    """把离散点扩成区间并合并。"""
    sigma_mid = np.asarray(sigma_mid, float)
    rhoA = np.asarray(rhoA, float)
    if len(sigma_mid) == 0:
        return []
    score = (rhoA - sigma_rho) if (conservative and sigma_rho is not None) else rhoA
    over = score > thr
    if not np.any(over):
        return []
    x = sigma_mid
    fsr = estimate_local_fsr(x)
    Ls = x[over] - 0.5*expand_fsr*fsr[over]
    Rs = x[over] + 0.5*expand_fsr*fsr[over]
    idx = np.argsort(Ls)
    Ls, Rs = Ls[idx], Rs[idx]

    bands = []
    L, R = Ls[0], Rs[0]
    for l, r in zip(Ls[1:], Rs[1:]):
        if l - R <= merge_tol:
            R = max(R, r)
        else:
            bands.append((float(L), float(R)))
            L, R = l, r
    bands.append((float(L), float(R)))
    return bands

# =======================
# 3) 方法A-2：滑动窗口法（准连续）
# =======================
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

# =======================
# 4) 可视化
# =======================
def plot_points_with_threshold(tag, sigma_mid, rhoA, sigma_rho, thr, save=None):
    plt.figure(figsize=(11,4.2))
    if len(sigma_mid):
        plt.errorbar(sigma_mid, rhoA, yerr=sigma_rho, fmt='o', ms=3.5,
                     elinewidth=1, capsize=2, color="tab:blue", label=L("|ρ_A|（可见度反演）","|ρ_A| (from visibility)"))
    plt.axhline(thr, ls="--", color="tab:red", label=L(f"阈值 {thr:.2f}", f"Threshold {thr:.2f}"))
    plt.xlabel(L("波数 (cm$^{-1}$)", "Wavenumber (cm$^{-1}$)"))
    plt.ylabel("|ρ_A|")
    plt.title(L(f"{tag} 多光束必要性（方法A-1: 极值配对）", f"{tag} Multi-beam Necessity "))
    plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
    if save: plt.savefig(save, dpi=200, bbox_inches="tight"); plt.close()
    else: plt.show()

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

# =======================
# 5) 主流程
# =======================
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

    if METHOD.lower() == "pairs":
        # ---------- 方法A-1：极值配对 ----------
        # 10°
        y_detr1, _ = local_detrend(R1p, width=DETREND_WIN)
        pv1 = pair_peaks_valleys_nearest(wn1, y_detr1, prominence=prom1, distance=PEAK_DIST)
        arr1 = visibility_and_rho_from_pairs(pv1, R1p, sigR1)
        res1 = {"sigma_mid": arr1[:,0], "V": arr1[:,1], "rhoA": arr1[:,2], "sigma_V": arr1[:,3], "sigma_rho": arr1[:,4]} if arr1.size else {"sigma_mid":np.array([]),"V":np.array([]),"rhoA":np.array([]),"sigma_V":np.array([]),"sigma_rho":np.array([])}

        # 15°
        y_detr2, _ = local_detrend(R2p, width=DETREND_WIN)
        pv2 = pair_peaks_valleys_nearest(wn2, y_detr2, prominence=prom2, distance=PEAK_DIST)
        arr2 = visibility_and_rho_from_pairs(pv2, R2p, sigR2)
        res2 = {"sigma_mid": arr2[:,0], "V": arr2[:,1], "rhoA": arr2[:,2], "sigma_V": arr2[:,3], "sigma_rho": arr2[:,4]} if arr2.size else {"sigma_mid":np.array([]),"V":np.array([]),"rhoA":np.array([]),"sigma_V":np.array([]),"sigma_rho":np.array([])}

        # 插值为连续曲线（仅用于展示；区间仍由离散点+FSR 扩展再合并）
        xi1, yi1, si1 = interpolate_rhoA(res1["sigma_mid"], res1["rhoA"], res1["sigma_rho"], smooth_win=SMOOTH_WIN, grid=INTERP_GRID) if res1["rhoA"].size else (np.array([]), np.array([]), np.array([]))
        xi2, yi2, si2 = interpolate_rhoA(res2["sigma_mid"], res2["rhoA"], res2["sigma_rho"], smooth_win=SMOOTH_WIN, grid=INTERP_GRID) if res2["rhoA"].size else (np.array([]), np.array([]), np.array([]))

        # 由离散点生成连续超阈值区间
        bands1 = bands_from_points(res1["sigma_mid"], res1["rhoA"], res1["sigma_rho"],
                                   thr=THRESHOLD, merge_tol=BAND_MERGE,
                                   expand_fsr=BAND_EXPAND_FSR, conservative=CONSERVATIVE)
        bands2 = bands_from_points(res2["sigma_mid"], res2["rhoA"], res2["sigma_rho"],
                                   thr=THRESHOLD, merge_tol=BAND_MERGE,
                                   expand_fsr=BAND_EXPAND_FSR, conservative=CONSERVATIVE)

        # 终端打印
        def _print_result(tag, res, bands):
            if res["rhoA"].size == 0:
                print(f"[{tag}] " + L("未得到有效峰谷对。", "No valid peak-valley pairs."))
                return
            over = (res["rhoA"] - res["sigma_rho"]) > THRESHOLD if CONSERVATIVE else (res["rhoA"] > THRESHOLD)
            ratio = float(over.mean())
            print(f"[{tag}] " + L(f"（保守）超阈值占比: {ratio*100:.1f}%", f"(conservative) over-threshold fraction: {ratio*100:.1f}%"))
            if bands:
                seg = "; ".join([f"{b[0]:.0f}–{b[1]:.0f}" for b in bands])
                print(f"[{tag}] " + L(f"连续超阈值波段 (cm^-1): {seg}",
                                       f"Continuous over-threshold bands (cm^-1): {seg}"))
                print(f"[{tag}] " + L("结论：这些波段需用多光束/TMM；其余波段用双光束即可。",
                                       "Conclusion: Use multi-beam/TMM in those bands; two-beam elsewhere."))

        _print_result("10°", res1, bands1)
        _print_result("15°", res2, bands2)

        # 绘图：离散点 + 阈值
        plot_points_with_threshold("10°", res1["sigma_mid"], res1["rhoA"], res1["sigma_rho"], THRESHOLD, save="rho_points_10.png")
        plot_points_with_threshold("15°", res2["sigma_mid"], res2["rhoA"], res2["sigma_rho"], THRESHOLD, save="rho_points_15.png")
        print("Saved:", os.path.abspath("rho_points_10.png"))
        print("Saved:", os.path.abspath("rho_points_15.png"))

        # 绘图：连续曲线（插值）+ 阈值 + 区间阴影（用 bands_from_points 的结果）
        plot_dense_curve_and_bands("10°", xi1, yi1, THRESHOLD, bands=bands1, save="rho_dense_10.png")
        plot_dense_curve_and_bands("15°", xi2, yi2, THRESHOLD, bands=bands2, save="rho_dense_15.png")
        print("Saved:", os.path.abspath("rho_dense_10.png"))
        print("Saved:", os.path.abspath("rho_dense_15.png"))

    else:
        # ---------- 方法A-2：滑动窗口 ----------
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

        # 可选：V 诊断散点
        for tag, res, color in [("10°", res1, "tab:blue"), ("15°", res2, "tab:orange")]:
            if res["sigma"].size == 0: 
                continue
            plt.figure(figsize=(11,4.2))
            plt.scatter(res["sigma"], res["V"], s=10, color=color, label="Visibility V")
            plt.xlabel(L("波数 (cm$^{-1}$)", "Wavenumber (cm$^{-1}$)"))
            plt.ylabel("V")
            plt.title(L(f"{tag} 条纹可见度 V（诊断）", f"{tag} Fringe Visibility V (diagnostic)"))
            plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout()
            fn = f"visibility_{'10' if tag=='10°' else '15'}.png"
            plt.savefig(fn, dpi=200, bbox_inches="tight"); plt.close()
            print("Saved:", os.path.abspath(fn))

if __name__ == "__main__":
    main()
