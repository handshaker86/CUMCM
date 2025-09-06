import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# --- 物理模型与参数定义 ---

# 假设的材料光学常数 (折射率)
# 注意：在真实竞赛中，应查找并使用波数相关的色散模型以获得更高精度。
N0 = 1.0  # 空气折射率
# N1 = 2.55  # SiC外延层折射率 (近似为常数)
# N2 = 2.8  # SiC衬底的复折射率 (近似为常数, 包含吸收)


def create_refractive_index_interpolator(
    csv_path: str,
    wavelength_col: str = "wavelength",
    n_col: str = "n",
    kind: str = "cubic",
):
    """

    Args:
        csv_path (str): CSV檔案的路徑。
        wavenumber_col (str): CSV中包含波数的欄位名稱。
        n_col (str): CSV中包含折射率的欄位名稱。
        kind (str): 插值的類型 ('linear', 'cubic', etc.)。推薦使用 'cubic'。

    Returns:
        function: 一個插值函數。輸入一個波数(或numpy陣列)，返回對應的折射率。
    """
    try:
        # 步驟 1: 讀取數據
        df = pd.read_csv(csv_path)

        # 步驟 2: 準備已知數據點
        wavelength_data = df[wavelength_col].values.astype(float)
        wavenumber = 1e4 / wavelength_data
        n = df[n_col].values

        # interp1d 要求x轴数据单调递增
        sort_indices = np.argsort(wavenumber)
        wavenumber = wavenumber[sort_indices]
        n = n[sort_indices]

        print(f"成功從 '{csv_path}' 讀取數據並建立 '{kind}' 插值器。")

        # 步驟 3: 創建並返回插值函數
        # fill_value="extrapolate" 允許對稍微超出範圍的點進行外插
        interpolator_function = interp1d(
            wavenumber,
            n,
            kind=kind,
            bounds_error=False,  # 當超出範圍時不報錯
            fill_value="extrapolate",  # 而是進行外插
        )
        return interpolator_function

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{csv_path}'")
        return None
    except KeyError as e:
        print(f"錯誤：CSV檔案中找不到欄位 {e}。請檢查欄位名稱是否正確。")
        return None


def get_sic_refractive_index(wavenumbers_cm):
    wavenumbers_cm = np.asarray(wavenumbers_cm)
    lambda_um = 1e4 / wavenumbers_cm
    lambda_um_sq = lambda_um**2

    sigma = wavenumbers_cm
    sigma_sq = sigma**2

    # 17-25
    cond1 = (sigma >= 390) & (sigma < 590)
    # 5-17
    cond2 = (sigma >= 590) & (sigma < 2000)
    # 2.5-5
    cond3 = (sigma >= 2000) & (sigma <= 4100)

    # cond1 (390 ~ 590 cm⁻¹)
    n_sq_1 = 1 + 6.055 + (2.669 * lambda_um_sq) / (lambda_um_sq - 167.8)
    n_1 = np.sqrt(np.maximum(0, n_sq_1))

    # cond2 (590 ~ 2000 cm⁻¹)
    numerator = sigma_sq - (992.1**2) + 1j * 10 * sigma
    denominator = sigma_sq - (797.7**2) + 1j * 10 * sigma
    epsilon = 6.5 * (numerator / denominator)
    n_complex = np.sqrt(epsilon)
    n_2 = n_complex

    # cond3 (2000 ~ 4000 cm⁻¹)
    n_sq_3 = (
        1
        + (0.20075 * lambda_um_sq) / (lambda_um_sq + 12.07224)
        + (5.54861 * lambda_um_sq) / (lambda_um_sq - 0.02641)
        + (35.65066 * lambda_um_sq) / (lambda_um_sq - 1268.24708)
    )
    n_3 = np.sqrt(np.maximum(0, n_sq_3))

    conditions = [cond1, cond2, cond3]
    choices = [n_1, n_2, n_3]

    n_result = np.select(conditions, choices, default=0.0)

    uncovered_mask = (n_result == 0.0) & (wavenumbers_cm != 0)
    if np.any(uncovered_mask):
        problematic_wavenumbers = wavenumbers_cm[uncovered_mask]
        print(f"警告: 折射率計算結果中存在0值，請檢查波数範圍是否完全覆蓋。")
        print(f"出問題的波数點 (前5個): {problematic_wavenumbers[:5]}")

    return n_result


def calculate_reflectance(d_um, wavenumbers_cm, theta0_deg, interpolator):
    """
    根据双光束干涉和菲涅尔方程计算理论反射率。

    参数:
    d_um (float): 外延层厚度，单位：微米 (μm)
    wavenumbers_cm (np.array): 波数数组，单位：cm^-1
    theta0_deg (float): 入射角，单位：度

    返回:
    np.array: 对应每个波数的理论总反射率 R_model
    """
    N1 = get_sic_refractive_index(wavenumbers_cm)
    N2 = interpolator(wavenumbers_cm)
    # --- 单位转换 ---
    d_cm = d_um * 1e-4  # 将厚度从 μm 转换为 cm
    theta0_rad = np.deg2rad(theta0_deg)  # 入射角转为弧度

    # --- 计算各层角度 (斯涅尔定律) ---
    # 使用 np.lib.scimath.sqrt 以处理复数开方
    sin_theta1 = N0 * np.sin(theta0_rad) / N1
    cos_theta1 = np.lib.scimath.sqrt(1 - sin_theta1**2)

    sin_theta2 = N1 * sin_theta1 / N2
    cos_theta2 = np.lib.scimath.sqrt(1 - sin_theta2**2)

    # --- 计算菲涅尔系数 ---
    # 界面 0 -> 1 (空气 -> 外延层)
    r01_s = (N0 * np.cos(theta0_rad) - N1 * cos_theta1) / (
        N0 * np.cos(theta0_rad) + N1 * cos_theta1
    )
    r01_p = (N1 * np.cos(theta0_rad) - N0 * cos_theta1) / (
        N1 * np.cos(theta0_rad) + N0 * cos_theta1
    )
    t01_s = 2 * N0 * np.cos(theta0_rad) / (N0 * np.cos(theta0_rad) + N1 * cos_theta1)
    t01_p = 2 * N0 * np.cos(theta0_rad) / (N1 * np.cos(theta0_rad) + N0 * cos_theta1)

    # 界面 1 -> 0 (外延层 -> 空气)
    # 注意：t10 != t01。根据斯托克斯关系 t01*t10 - r01*r10 = 1 (r10=-r01)
    t10_s = 2 * N1 * cos_theta1 / (N1 * cos_theta1 + N0 * np.cos(theta0_rad))
    t10_p = 2 * N1 * cos_theta1 / (N0 * cos_theta1 + N1 * np.cos(theta0_rad))

    # 界面 1 -> 2 (外延层 -> 衬底)
    r12_s = (N1 * cos_theta1 - N2 * cos_theta2) / (N1 * cos_theta1 + N2 * cos_theta2)
    r12_p = (N2 * cos_theta1 - N1 * cos_theta2) / (N2 * cos_theta1 + N1 * cos_theta2)

    # --- 计算相位差 ---
    # φ = 4 * pi * σ * n1 * d * cos(θ1)
    # σ (wavenumbers_cm) 单位是 cm^-1, d_cm 单位是 cm
    phi = 4 * np.pi * wavenumbers_cm * N1 * d_cm * cos_theta1

    # --- 计算总反射振幅 (复数) ---
    # r_total = r_01 + t_01 * r_12 * t_10 * exp(i * φ)
    # 使用 1j 表示虚数单位 i
    r_total_s = r01_s + t01_s * r12_s * t10_s * np.exp(1j * phi)
    r_total_p = r01_p + t01_p * r12_p * t10_p * np.exp(1j * phi)

    # --- 计算反射率 (振幅的模的平方) ---
    R_s = np.abs(r_total_s) ** 2
    R_p = np.abs(r_total_p) ** 2

    # 非偏振光的总反射率
    R_model = (R_s + R_p) / 2

    # 结果是复数时取实部（由于数值精度问题可能产生极小的虚部）
    return np.real(R_model)


def mse_loss(d_um, wavenumbers_cm, R_exp, theta0_deg, interpolator):
    """
    计算理论与实验反射率的均方误差(MSE)。
    """
    R_model = calculate_reflectance(d_um, wavenumbers_cm, theta0_deg, interpolator)
    mse = np.mean((R_model * 100 - R_exp) ** 2)  # R_exp单位是%，所以R_model*100
    return mse


def process_file(filepath, angle_deg, interpolator):
    """
    读取数据文件，执行优化，并报告结果。
    """
    print(f"\n--- 正在处理文件: {filepath} (入射角: {angle_deg}°) ---")

    # --- 读取数据 ---
    try:
        df = pd.read_excel(filepath)
        # 假设列名为 '波数(cm-1)' 和 '反射率(%)' 或类似的名称
        # 为确保代码健壮性，我们直接使用列的索引
        wavenumbers = df.iloc[:, 0].values
        reflectance_exp = df.iloc[:, 1].values
        print(f"数据加载成功，共 {len(wavenumbers)} 个数据点。")
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。请确保文件在正确的路径下。")
        return None, None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None, None

    # --- 优化求解 ---
    # 定义厚度d的搜索范围 (单位: μm)
    bounds = (8, 10)
    print(f"开始在范围 {bounds} μm 内优化求解厚度 d ...")

    # 使用 minimize_scalar 进行优化
    result = minimize_scalar(
        mse_loss,
        bounds=bounds,
        args=(wavenumbers, reflectance_exp, angle_deg, interpolator),
        method="bounded",  # 使用'bounded'方法确保在指定范围内搜索
    )

    best_d = result.x
    min_mse = result.fun

    print(f"优化完成！")
    print(f"计算出的最佳外延层厚度 d = {best_d:.4f} μm")
    print(f"对应的最小均方误差(MSE) = {min_mse:.4f}")

    # --- 结果可视化 ---
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, reflectance_exp, label="Experiment data", alpha=0.7)

    # 使用计算出的最佳厚度生成理论曲线
    # 使用計算出的最佳厚度生成理論曲線
    R_fit = calculate_reflectance(best_d, wavenumbers, angle_deg, interpolator) * 100

    # 繪製模型擬合曲線
    plt.plot(
        wavenumbers,
        R_fit,
        label=f"Model Fit (d={best_d:.2f} μm)",
        color="red",
        linewidth=2,
    )

    plt.title(f"Spectral Data vs. Model Fit (Incidence Angle: {angle_deg}°)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Reflectance (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_d, min_mse


# --- 主程序 ---
if __name__ == "__main__":
    # 假设附件文件与脚本在同一目录下
    file1 = "附件1.xlsx"
    file2 = "附件2.xlsx"
    refractive_csv = "Si_n.csv"

    # 读取并创建折射率插值函数 (如果需要)
    refractive_index_interpolator = create_refractive_index_interpolator(
        refractive_csv, wavelength_col="wavelength", n_col="n", kind="cubic"
    )

    # 处理第一个文件 (10°)
    d1, mse1 = process_file(
        file1, angle_deg=10, interpolator=refractive_index_interpolator
    )

    # 处理第二个文件 (15°)
    d2, mse2 = process_file(
        file2, angle_deg=15, interpolator=refractive_index_interpolator
    )

    # --- 结果可靠性分析 ---
    if d1 is not None and d2 is not None:
        print("\n--- 问题2：结果可靠性分析 ---")
        print(f"入射角 10° 时计算的厚度: d1 = {d1:.4f} μm")
        print(f"入射角 15° 时计算的厚度: d2 = {d2:.4f} μm")

        # 计算两次结果的差异
        difference = np.abs(d1 - d2)
        relative_difference = difference / np.mean([d1, d2]) * 100

        print(f"两次计算结果的绝对差异: {difference:.4f} μm")
        print(f"两次计算结果的相对差异: {relative_difference:.2f}%")

        if relative_difference < 5:  # 设置一个阈值，例如5%
            print(
                "\n结论：两次测量计算出的厚度非常接近，表明该模型和算法具有较高的可靠性。"
            )
        else:
            print("\n结论：两次测量计算出的厚度差异较大，可能原因包括：")
            print("1. 所假设的光学常数(n1, n2)与实际材料不符，或其色散效应不可忽略。")
            print("2. 实验数据中存在噪声或其他未考虑的物理效应（如多光束干涉）。")
            print("3. 模型本身（双光束干涉）的近似不够精确。")
