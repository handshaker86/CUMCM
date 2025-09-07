import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.ndimage import uniform_filter1d
from q3_method import calculate_reflectance_hybrid, fit_thickness_high, fit_lowfreq_params, refine_highfreq_thickness, fit_full_hybrid
from q2_method2 import preprocess_data

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

    R_model = model_reflectance_percent(sigma, theta_deg, d_hat, params_low_hat,
                                        split=split, tw=tw)
    res = residual_vector(R_model, R_exp, smooth_window=smooth_window_for_obj)
    N = res.size
    SSE = np.sum(res**2)
    MSE = SSE / N
    dof = max(1, N - p_total_dim)
    sigma_R2_unbiased = SSE / dof

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
    Jd = finite_diff_column(model_with_param_vector, p_hat, idx=0, h=fd_steps["d_um"])
    if smooth_window_for_obj and smooth_window_for_obj > 1:
        Jd = uniform_filter1d(Jd, size=smooth_window_for_obj)

    SumJ2 = float(np.sum(Jd**2))
    Fisher_cond = SumJ2 / sigma_R2_unbiased
    SE_cond = np.sqrt(1.0 / max(Fisher_cond, 1e-30))
    z975 = 1.959963984540054  
    CI_cond = (d_hat - z975 * SE_cond, d_hat + z975 * SE_cond)
    p = p_hat.size
    J = np.zeros((N, p), dtype=float)

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
   
    stats3, stats4 = run_sensitivity_print_only(
        wn1, R1, d10_final, params_low10_final,
        wn2, R2, d15_final, params_low15_final,
        split=split, tw=tw, smooth_window_for_obj=3
    )

    J10 = finite_diff_J_sigma(wn1, 10, d10_final, params_low10_final, h_d=1e-4,
                            smooth_window=3, split=split, tw=tw)
    J15 = finite_diff_J_sigma(wn2, 15, d15_final, params_low15_final, h_d=1e-4,
                            smooth_window=3, split=split, tw=tw)

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