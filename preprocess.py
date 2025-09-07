import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1. 用户配置区 ---
# 在这里修改您的文件路径和处理参数

# 您的数据文件名
FILE_PATH = "附件1.xlsx"

# 步骤一：设定您认为信号可靠的波数范围 (单位: cm⁻¹)
# 仪器在两端的信号通常最差，通过设定范围可以有效去除低信噪比区域。
# 这是一个典型的中红外光谱仪的可靠范围，请根据您的仪器进行调整。
MIN_WAVENUMBER = 600
MAX_WAVENUMBER = 4000

# 步骤二：设定Savitzky-Golay (SG)滤波器参数
# window_length: 窗口大小，必须是奇数。值越大，平滑效果越强，但可能丢失细节。
# polyorder: 多项式阶数，必须小于窗口大小。阶数越高，对原始形状的保留越好。
SG_WINDOW = 11  # 推荐从 5, 7, 9, 11... 尝试
SG_ORDER = 2  # 对于大多数光谱，2或3是很好的选择
NORMALIZATION_RANGE = (800, 1000)


# --- 2. 数据加载与预处理函数 ---


def load_and_normalize_data(filepath):
    """从Excel加载数据，并将反射率从百分比转换为0-1的小数。"""
    print(f"正在从 '{filepath}' 加载数据...")
    df = pd.read_excel(filepath)  # 假设没有表头
    wavenumbers = df.iloc[:, 0].values.astype(float)
    reflectance = df.iloc[:, 1].values.astype(float)

    # 自动检测并转换百分比数据
    if np.max(reflectance) > 1.1:
        print("检测到反射率大于1，自动从百分比转换为小数...")
        reflectance = reflectance / 100.0

    # 将数据放入DataFrame方便处理
    data_df = pd.DataFrame({"wavenumber": wavenumbers, "reflectance": reflectance})
    # 按波数排序，确保数据是连续的
    data_df = data_df.sort_values(by="wavenumber").reset_index(drop=True)

    return data_df


def preprocess_data(
    data_df,
    min_wn=MIN_WAVENUMBER,
    max_wn=MAX_WAVENUMBER,
    sg_window=SG_WINDOW,
    sg_order=SG_ORDER,
    norm_range=NORMALIZATION_RANGE,
):
    """
    对DataFrame中的光谱数据进行预处理。
    流程: 1. 光谱归一化 -> 2. 按波数范围切除 -> 3. Savitzky-Golay平滑

    参数:
        data_df (pd.DataFrame): 包含 'wavenumber' 和 'reflectance' 列的原始数据。
        min_wn (float): 可靠波数范围的下限。
        max_wn (float): 可靠波数范围的上限。
        sg_window (int): Savitzky-Golay 滤波器的窗口大小 (必须是奇数)。
        sg_order (int): Savitzky-Golay 滤波器的多项式阶数。
        norm_range (tuple): 用于寻找最大反射率以进行归一化的波数范围 (例如 (800, 1000))。

    返回:
        pd.DataFrame: 包含 'wavenumber' 和 'reflectance_processed' 两列的已处理数据。
    """
    # --- 为确保函数健壮，先假设列名 ---
    # 这假设输入的DataFrame第一列是波数，第二列是反射率
    df = data_df.copy()
    df.columns = ["wavenumber", "reflectance"]

    # --- 步骤一: 光谱归一化 (处理 >100% 的问题) ---
    print("步骤一：执行光谱归一化...")

    # 选取在Reststrahlen带内的数据
    reststrahlen_band = df[
        (df["wavenumber"] >= norm_range[0]) & (df["wavenumber"] <= norm_range[1])
    ]

    # 创建一个新的 'reflectance_norm' 列
    if not reststrahlen_band.empty:
        max_R_exp = reststrahlen_band["reflectance"].max()
        if max_R_exp > 100.0:
            correction_factor = 100.0 / max_R_exp
            print(f"  检测到最大反射率为 {max_R_exp:.2f}% (>100%)。")
            print(f"  计算出校正因子为: {correction_factor:.4f}")
            df["reflectance_norm"] = df["reflectance"] * correction_factor
        else:
            print("  最大反射率未超过100%，无需归一化。")
            df["reflectance_norm"] = df["reflectance"]
    else:
        print("警告: 在指定的归一化范围内未找到数据，跳过归一化。")
        df["reflectance_norm"] = df["reflectance"]

    # --- 步骤二：移除低信噪比区域 (通过波数范围切片) ---
    print(f"步骤二：移除波数范围 [{min_wn}, {max_wn}]之外的数据...")
    df_cut = df[(df["wavenumber"] >= min_wn) & (df["wavenumber"] <= max_wn)].copy()

    # --- 步骤三：应用Savitzky-Golay平滑滤波器 ---
    print(f"步骤三：应用Savitzky-Golay平滑 (窗口={sg_window}, 阶数={sg_order})...")

    # 【BUG修复】确保平滑操作作用于'reflectance_norm'列
    df_cut["reflectance_processed"] = savgol_filter(
        df_cut["reflectance_norm"], window_length=sg_window, polyorder=sg_order
    )

    # --- 步骤四：返回简洁的最终结果 ---
    # 只保留后续拟合所需的列
    final_df = df_cut[["wavenumber", "reflectance_processed"]].reset_index(drop=True)

    return final_df


# --- 3. 主程序：执行并可视化 ---

if __name__ == "__main__":
    # 加载原始数据
    raw_data_df = load_and_normalize_data(FILE_PATH)
    print(f"加载了 {len(raw_data_df)} 个原始数据点。")

    # 执行预处理
    processed_data_df = preprocess_data(
        raw_data_df, MIN_WAVENUMBER, MAX_WAVENUMBER, SG_WINDOW, SG_ORDER
    )
    print(f"预处理后剩下 {len(processed_data_df)} 个数据点。")

    # 可视化对比结果
    plt.figure(figsize=(14, 7))

    # 绘制原始数据
    plt.plot(
        raw_data_df["wavenumber"],
        raw_data_df["reflectance"] * 100,
        color="gray",
        linestyle="-",
        alpha=0.5,
        label="Raw Experimental Data",
    )

    # 绘制预处理后的数据
    plt.plot(
        processed_data_df["wavenumber"],
        processed_data_df["reflectance_smooth"] * 100,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Preprocessed Data (Cut & Smoothed)",
    )

    # 用灰色区域标出被移除的低信噪比区域
    full_range = [raw_data_df["wavenumber"].min(), raw_data_df["wavenumber"].max()]
    plt.axvspan(
        full_range[0],
        MIN_WAVENUMBER,
        color="gray",
        alpha=0.2,
        label="Low SNR Region (Removed)",
    )
    plt.axvspan(MAX_WAVENUMBER, full_range[1], color="gray", alpha=0.2)

    # 图表美化
    plt.title("Comparison of Raw vs. Preprocessed Reflectivity Data")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Reflectivity (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(full_range)  # 显示完整的波数范围

    # 添加第二坐标轴（波长）
    ax1 = plt.gca()
    ax2 = ax1.secondary_xaxis(
        "top", functions=(lambda x: 10000 / (x + 1e-9), lambda x: 10000 / (x + 1e-9))
    )
    ax2.set_xlabel("Wavelength (μm)")

    plt.show()

    # 您现在可以将在 'processed_data_df' 中的数据用于后续的拟合
    # 例如:
    # final_wavenumbers = processed_data_df['wavenumber'].values
    # final_reflectance = processed_data_df['reflectance_smooth'].values
    # best_params = fit_reflectivity_model(final_wavenumbers, final_reflectance, ...)
