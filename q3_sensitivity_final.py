import numpy as np
from numpy.linalg import inv
from scipy.ndimage import uniform_filter1d
from q3_test4 import calculate_reflectance_hybrid
import pandas as pd
from preprocess import preprocess_data
from q3_test4 import run_fit


df1 = pd.read_excel("附件3.xlsx")
proc1 = preprocess_data(df1)
wn1 = proc1.iloc[:, 0].values.astype(float)
R1 = proc1.iloc[:, 1].values.astype(float)

df2 = pd.read_excel("附件4.xlsx")
proc2 = preprocess_data(df2)
wn2 = proc2.iloc[:, 0].values.astype(float)
R2 = proc2.iloc[:, 1].values.astype(float)
# -----------------------------
# A) 基础工具：模型、残差、数值雅可比
# -----------------------------
def model_reflectance_percent(sigma, theta_deg, d_um, params_low,
                              split=1500.0, tw=120.0):
    """返回 R_model(%)，不做任何与实验相同的预处理之外的缩放。"""
    R_model = calculate_reflectance_hybrid(
        d_um, params_low, sigma, theta_deg, split=split, tw=tw
    )
    return 100.0 * np.asarray(R_model, dtype=float)

def residual_vector(R_model_pct, R_exp_pct, smooth_window=None):
    """计算残差，支持可选的均值平滑（与你的目标函数一致时可设为3）。"""
    res = R_model_pct - R_exp_pct
    if smooth_window is not None and smooth_window > 1:
        res = uniform_filter1d(res, size=smooth_window)
    return res

def finite_diff_column(func, base_params, idx, h):
    """
    对参数向量 base_params 的第 idx 维做中心差分，返回一列导数（长度=N）。
    func: 传入参数向量 p，返回长度=N 的模型向量（例如 R_model%）
    """
    p = np.array(base_params, dtype=float)
    p[idx] += h
    f_plus = func(p)
    p[idx] -= 2*h
    f_minus = func(p)
    return (f_plus - f_minus) / (2*h)

# -----------------------------
# B) Fisher / CI 计算核心
# -----------------------------
def compute_sensitivity_and_ci(
    sigma, R_exp, theta_deg, d_hat, params_low_hat,
    split=1500.0, tw=120.0,
    smooth_window_for_obj=3,
    p_total_dim=7,
    fd_steps=None
):
    """
    对单个数据集，给出:
      - N, MSE, sigma_R2_unbiased
      - Sum J_d^2 (仅对 d 的灵敏度)
      - Fisher_cond, SE_cond, 95% CI_cond
      - Fisher_total (p×p), SE_total, 95% CI_total
    说明：
      * p_total_dim 为全参数数量：d + 6 个 LD 参数 = 7
      * smooth_window_for_obj 与你最终的微调目标函数一致（=3）
    """
    sigma = np.asarray(sigma, dtype=float)
    R_exp = np.asarray(R_exp, dtype=float)

    # 1) 基线模型与残差
    R_model = model_reflectance_percent(sigma, theta_deg, d_hat, params_low_hat,
                                        split=split, tw=tw)
    res = residual_vector(R_model, R_exp, smooth_window=smooth_window_for_obj)
    N = res.size
    SSE = np.sum(res**2)
    MSE = SSE / N
    # 全参数一共 7 个，自由度 = N - 7
    dof = max(1, N - p_total_dim)
    sigma_R2_unbiased = SSE / dof

    # 2) 条件灵敏度：仅对 d 的导数列
    # 数值步长（相对安全的中心差分）
    if fd_steps is None:
        fd_steps = {
            "d_um": 1e-4,       # 厚度的差分步长（um）
            "eps_inf": 1e-3,
            "sig_TO": 1e-2,
            "sig_LO": 1e-2,
            "gamma_ph": 1e-2,
            "sig_p": 1e-2,
            "gamma_e": 1e-2,
        }

    def model_with_param_vector(pvec):
        d = pvec[0]
        pars = pvec[1:]
        return model_reflectance_percent(
            sigma, theta_deg, d, pars, split=split, tw=tw
        )

    p_hat = np.concatenate(([d_hat], np.array(params_low_hat, dtype=float)))

    # d 的导数列（长度=N）
    Jd = finite_diff_column(model_with_param_vector, p_hat, idx=0, h=fd_steps["d_um"])
    # 用与目标函数一致的残差平滑方式
    if smooth_window_for_obj and smooth_window_for_obj > 1:
        Jd = uniform_filter1d(Jd, size=smooth_window_for_obj)

    SumJ2 = float(np.sum(Jd**2))
    Fisher_cond = SumJ2 / sigma_R2_unbiased
    SE_cond = np.sqrt(1.0 / max(Fisher_cond, 1e-30))
    z975 = 1.959963984540054  # 近似 1.96
    CI_cond = (d_hat - z975 * SE_cond, d_hat + z975 * SE_cond)

    # 3) 全参数 Fisher：J (N×p)，p=7
    # 依次构造每一列
    p = p_hat.size
    J = np.zeros((N, p), dtype=float)

    # 各参数步长
    steps = [
        fd_steps["d_um"],      # d
        fd_steps["eps_inf"],   # eps_inf
        fd_steps["sig_TO"],    # sig_TO
        fd_steps["sig_LO"],    # sig_LO
        fd_steps["gamma_ph"],  # gamma_ph
        fd_steps["sig_p"],     # sig_p
        fd_steps["gamma_e"],   # gamma_e
    ]

    for j in range(p):
        Jcol = finite_diff_column(model_with_param_vector, p_hat, idx=j, h=steps[j])
        if smooth_window_for_obj and smooth_window_for_obj > 1:
            Jcol = uniform_filter1d(Jcol, size=smooth_window_for_obj)
        J[:, j] = Jcol

    # 高斯-牛顿近似下：Fisher_total ≈ (1/σ^2) * J^T J
    JTJ = J.T @ J
    Fisher_total = JTJ / sigma_R2_unbiased

    # 可能出现病态；用逆而非伪逆，失败则退回到伪逆
    try:
        Cov_total = inv(Fisher_total)
    except np.linalg.LinAlgError:
        Cov_total = np.linalg.pinv(Fisher_total)

    SE_total = float(np.sqrt(max(Cov_total[0, 0], 0.0)))
    CI_total = (d_hat - z975 * SE_total, d_hat + z975 * SE_total)

    out = {
        "N": int(N),
        "MSE": float(MSE),
        "sigma_R2_unbiased": float(sigma_R2_unbiased),
        "SumJ2": float(SumJ2),
        "Fisher_cond": float(Fisher_cond),
        "SE_cond": float(SE_cond),
        "CI_cond": (float(CI_cond[0]), float(CI_cond[1])),
        "SE_total": float(SE_total),
        "CI_total": (float(CI_total[0]), float(CI_total[1])),
    }
    return out

# =============================
# SENSITIVITY & CI (PRINT-ONLY)
# =============================
import numpy as np
from numpy.linalg import inv
from scipy.ndimage import uniform_filter1d

def model_reflectance_percent(sigma, theta_deg, d_um, params_low,
                              split=1500.0, tw=120.0):
    R_model = calculate_reflectance_hybrid(
        d_um, params_low, sigma, theta_deg, split=split, tw=tw
    )
    return 100.0 * np.asarray(R_model, dtype=float)

def residual_vector(R_model_pct, R_exp_pct, smooth_window=None):
    res = R_model_pct - R_exp_pct
    if smooth_window is not None and smooth_window > 1:
        res = uniform_filter1d(res, size=smooth_window)
    return res

def finite_diff_column(func, base_params, idx, h):
    p = np.array(base_params, dtype=float)
    p[idx] += h
    f_plus = func(p)
    p[idx] -= 2*h
    f_minus = func(p)
    return (f_plus - f_minus) / (2*h)

def compute_sensitivity_and_ci(
    sigma, R_exp, theta_deg, d_hat, params_low_hat,
    split=1500.0, tw=120.0,
    smooth_window_for_obj=3,
    p_total_dim=7,
    fd_steps=None
):
    sigma = np.asarray(sigma, dtype=float)
    R_exp = np.asarray(R_exp, dtype=float)

    # 1) 基线模型与残差
    R_model = model_reflectance_percent(sigma, theta_deg, d_hat, params_low_hat,
                                        split=split, tw=tw)
    res = residual_vector(R_model, R_exp, smooth_window=smooth_window_for_obj)
    N = res.size
    SSE = np.sum(res**2)
    MSE = SSE / N
    dof = max(1, N - p_total_dim)  # p=7: d + 6个LD参数
    sigma_R2_unbiased = SSE / dof

    # 2) 数值步长
    if fd_steps is None:
        fd_steps = {
            "d_um": 1e-4,
            "eps_inf": 1e-3,
            "sig_TO": 1e-2,
            "sig_LO": 1e-2,
            "gamma_ph": 1e-2,
            "sig_p": 1e-2,
            "gamma_e": 1e-2,
        }

    def model_with_param_vector(pvec):
        d = pvec[0]
        pars = pvec[1:]
        return model_reflectance_percent(
            sigma, theta_deg, d, pars, split=split, tw=tw
        )

    p_hat = np.concatenate(([d_hat], np.array(params_low_hat, dtype=float)))

    # d 的导数列
    Jd = finite_diff_column(model_with_param_vector, p_hat, idx=0, h=fd_steps["d_um"])
    if smooth_window_for_obj and smooth_window_for_obj > 1:
        Jd = uniform_filter1d(Jd, size=smooth_window_for_obj)

    SumJ2 = float(np.sum(Jd**2))
    Fisher_cond = SumJ2 / sigma_R2_unbiased
    SE_cond = np.sqrt(1.0 / max(Fisher_cond, 1e-30))
    z975 = 1.959963984540054
    CI_cond = (d_hat - z975 * SE_cond, d_hat + z975 * SE_cond)

    # 全参数 Fisher
    steps = [
        fd_steps["d_um"],      # d
        fd_steps["eps_inf"],   # eps_inf
        fd_steps["sig_TO"],    # sig_TO
        fd_steps["sig_LO"],    # sig_LO
        fd_steps["gamma_ph"],  # gamma_ph
        fd_steps["sig_p"],     # sig_p
        fd_steps["gamma_e"],   # gamma_e
    ]
    p = p_hat.size
    J = np.zeros((N, p), dtype=float)
    for j in range(p):
        col = finite_diff_column(model_with_param_vector, p_hat, idx=j, h=steps[j])
        if smooth_window_for_obj and smooth_window_for_obj > 1:
            col = uniform_filter1d(col, size=smooth_window_for_obj)
        J[:, j] = col

    JTJ = J.T @ J
    Fisher_total = JTJ / sigma_R2_unbiased
    try:
        Cov_total = inv(Fisher_total)
    except np.linalg.LinAlgError:
        Cov_total = np.linalg.pinv(Fisher_total)

    SE_total = float(np.sqrt(max(Cov_total[0, 0], 0.0)))
    CI_total = (d_hat - z975 * SE_total, d_hat + z975 * SE_total)

    out = {
        "N": int(N),
        "MSE": float(MSE),
        "sigma_R2_unbiased": float(sigma_R2_unbiased),
        "SumJ2": float(SumJ2),
        "Fisher_cond": float(Fisher_cond),
        "SE_cond": float(SE_cond),
        "CI_cond": (float(CI_cond[0]), float(CI_cond[1])),
        "SE_total": float(SE_total),
        "CI_total": (float(CI_total[0]), float(CI_total[1])),
    }
    return out

def format_ci(ci_tuple):
    return f"[{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

def print_dataset_report(tag, theta_deg, d_hat, stats):
    # 按你示例表头字段打印（不输出LaTeX）
    print(f"{tag} | θ0={theta_deg}° | N={stats['N']} | "
          f"d̂={d_hat:.3f} μm | "
          f"MSE={stats['MSE']:.3f} (%^2) | "
          f"σ_R^2(无偏)={stats['sigma_R2_unbiased']:.3f} (%^2) | "
          f"∑J_i^2={stats['SumJ2']:.3f} ((%/μm)^2) | "
          f"Fisher={stats['Fisher_cond']:.3f} (1/μm^2) | "
          f"SE_cond={stats['SE_cond']:.3f} μm | "
          f"95% CI_cond={format_ci(stats['CI_cond'])} μm | "
          f"95% CI_total={format_ci(stats['CI_total'])} μm")

def run_sensitivity_print_only(
    wn3, R3, d10_final, params_low10_final,  # 附件3（10°）
    wn4, R4, d15_final, params_low15_final, # 附件4（15°）
    split=1500.0, tw=120.0, smooth_window_for_obj=3
):
    stats3 = compute_sensitivity_and_ci(
        wn3, R3, 10, d10_final, params_low10_final,
        split=split, tw=tw, smooth_window_for_obj=smooth_window_for_obj
    )
    print_dataset_report("附件3.xlsx", 10, d10_final, stats3)

    stats4 = compute_sensitivity_and_ci(
        wn4, R4, 15, d15_final, params_low15_final,
        split=split, tw=tw, smooth_window_for_obj=smooth_window_for_obj
    )
    print_dataset_report("附件4.xlsx", 15, d15_final, stats4)

    return stats3, stats4

# 在你的 __main__ 末尾（可视化之后）调用：
if __name__ == "__main__":
    # 已有：wn1/R1 <- 附件3.xlsx(10°)，wn2/R2 <- 附件4.xlsx(15°)
    # 已有：d10_final/params_low10_final, d15_final/params_low15_final
    (
    d10_final, params_low10_final, d15_final, params_low15_final,
    wn1, R1, wn2, R2,
    final_d, final_params_low, split, tw
    ) = run_fit()
    stats3, stats4 = run_sensitivity_print_only(
        wn1, R1, d10_final, params_low10_final,
        wn2, R2, d15_final, params_low15_final,
        split=split, tw=tw, smooth_window_for_obj=3
    )
# =========================================
# Sensitivity and Band-wise Information Share (English-only)
# =========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# If results are not in scope, pull once from q3_test4.run_fit()
try:
    _ = d10_final
except NameError:
    from q3_test4 import run_fit
    (
        d10_final, params_low10_final, d15_final, params_low15_final,
        wn1, R1, wn2, R2,
        final_d, final_params_low, split, tw
    ) = run_fit()

def model_reflectance_percent(sigma, theta_deg, d_um, params_low,
                              split=1500.0, tw=120.0):
    R_model = calculate_reflectance_hybrid(
        d_um, params_low, sigma, theta_deg, split=split, tw=tw
    )
    return 100.0 * np.asarray(R_model, dtype=float)

def finite_diff_J_sigma(sigma, theta_deg, d_um, params_low,
                        h_d=1e-4, smooth_window=3, split=1500.0, tw=120.0):
    """Return |J(σ)| = |∂R/∂d| in %/μm, with optional smoothing to match objective."""
    sigma = np.asarray(sigma, dtype=float)
    Rp = model_reflectance_percent(sigma, theta_deg, d_um + h_d, params_low, split, tw)
    Rm = model_reflectance_percent(sigma, theta_deg, d_um - h_d, params_low, split, tw)
    J = (Rp - Rm) / (2.0 * h_d)
    if smooth_window and smooth_window > 1:
        J = uniform_filter1d(J, size=smooth_window)
    return np.abs(J)

# Sensitivity curves
J10 = finite_diff_J_sigma(wn1, 10, d10_final, params_low10_final, h_d=1e-4,
                          smooth_window=3, split=split, tw=tw)
J15 = finite_diff_J_sigma(wn2, 15, d15_final, params_low15_final, h_d=1e-4,
                          smooth_window=3, split=split, tw=tw)

# --------------------------
# Figure 1: Sensitivity vs Wavenumber
# --------------------------
plt.figure(figsize=(11, 5.5))
plt.plot(wn1, J10, label="10°  |J(σ)|", linewidth=2)
plt.plot(wn2, J15, label="15°  |J(σ)|", linewidth=2, alpha=0.9)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("|∂R/∂d| (% / μm)")
plt.title("Sensitivity vs. Wavenumber (10° vs 15°)")
plt.xlim(400, 4000)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Figure 2: Band-wise Information Share (ΣJ²)
# --------------------------
# Fixed bands per your paper: 2.5–5, 5–17, 17–25 μm
# in wavenumber: (2000, 4000), (590, 2000), (400, 590)
bands = [(400, 590), (590, 2000), (2000, 4000)]
band_labels = [
    "400–590 (≈25–17 μm)",
    "590–2000 (≈17–5 μm)",
    "2000–4000 (≈5–2.5 μm)",
]

def band_contrib_percent(wavenumbers, J_abs, bands):
    J2 = J_abs**2
    contrib = []
    for a, b in bands:
        mask = (wavenumbers >= a) & (wavenumbers < b)
        contrib.append(J2[mask].sum())
    total = np.sum(contrib) + 1e-30
    return np.array(contrib) / total * 100.0, np.array(contrib)

p10, raw10 = band_contrib_percent(wn1, J10, bands)
p15, raw15 = band_contrib_percent(wn2, J15, bands)

# Print shares for writing
print("\nBand-wise information share based on ΣJ(σ)^2 (percent):")
for lbl, v10, v15 in zip(band_labels, p10, p15):
    print(f"{lbl:>22s}  |  10°: {v10:6.2f}%   15°: {v15:6.2f}%")

# Grouped bar chart
x = np.arange(len(bands))
width = 0.36
plt.figure(figsize=(10.5, 5.2))
bars10 = plt.bar(x - width/2, p10, width, label="10°")
bars15 = plt.bar(x + width/2, p15, width, label="15°")
plt.xticks(x, band_labels, rotation=0)
plt.ylabel("Information share (%)  based on ΣJ(σ)^2")
plt.title("Band-wise Contribution to Thickness Information (10° vs 15°)")
plt.ylim(0, max(p10.max(), p15.max()) * 1.18)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.legend()
# annotate
for bar in list(bars10) + list(bars15):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%",
             ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()