"""第一步：检查 ETTh1 数据与 Informer 的窗口定义。

本文件不训练模型，只复现 Informer 论文实现中最重要的数据约定：
    - 多变量输入/多变量预测（M）
    - seq_len = 96
    - label_len = 48
    - pred_len = 24
    - 训练/验证/测试按 12 / 4 / 4 个月的小时数划分
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Keep Chinese diagnostic text readable when launched from a Windows terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------
# 1. 实验配置
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ETT" / "ETTh1.csv"

SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 24

# 论文官方代码采用 30 天/月的近似划分。
HOURS_PER_MONTH = 30 * 24
TRAIN_END = 12 * HOURS_PER_MONTH
VAL_END = TRAIN_END + 4 * HOURS_PER_MONTH
TEST_END = VAL_END + 4 * HOURS_PER_MONTH


def build_time_mark(dates: pd.Series) -> np.ndarray:
    """构造最小时间标记：月、日、星期、小时。

    Informer 后续会把这些时间标记嵌入为时间特征。本阶段只确认
    时间标记的形状为 (sequence_length, 4)。
    """
    dates = pd.to_datetime(dates)

    return np.stack(
        [
            dates.dt.month.to_numpy(),
            dates.dt.day.to_numpy(),
            dates.dt.dayofweek.to_numpy(),
            dates.dt.hour.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)


def get_one_window(
    scaled_values: np.ndarray,
    time_marks: np.ndarray,
    start_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按官方实现的索引逻辑获取一个 Informer 样本。"""
    s_begin = start_index
    s_end = s_begin + SEQ_LEN

    # Decoder 读取最后 LABEL_LEN 个已知点，再接 PRED_LEN 个预测位置。
    r_begin = s_end - LABEL_LEN
    r_end = r_begin + LABEL_LEN + PRED_LEN

    seq_x = scaled_values[s_begin:s_end]
    seq_y = scaled_values[r_begin:r_end]

    seq_x_mark = time_marks[s_begin:s_end]
    seq_y_mark = time_marks[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark


def print_split_summary(name: str, start: int, end: int, dates: pd.Series) -> None:
    """打印一个数据划分的全局边界。"""
    print(
        f"{name:>5}: index [{start:5d}, {end:5d}), "
        f"{dates.iloc[start]} 至 {dates.iloc[end - 1]}，"
        f"共 {end - start} 个时间点"
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据集：{DATA_PATH}\n"
            "请将 ETTh1.csv 放到 data/ETT/ 目录。"
        )

    # --------------------------------------------------------------
    # 2. 读取与基础检查
    # --------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    expected_columns = [
        "date", "HUFL", "HULL", "MUFL",
        "MULL", "LUFL", "LULL", "OT",
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            "ETTh1 列名与预期不一致。\n"
            f"期望：{expected_columns}\n"
            f"实际：{list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    feature_columns = [column for column in df.columns if column != "date"]

    print("===== ETTh1 原始数据 =====")
    print("数据路径：", DATA_PATH)
    print("行数：", len(df))
    print("特征列：", feature_columns)
    print("时间范围：", df["date"].min(), "至", df["date"].max())

    print("\n===== 数据质量检查 =====")
    print("缺失值：")
    print(df.isna().sum())
    print("重复时间点：", df["date"].duplicated().sum())
    print("是否时间升序：", df["date"].is_monotonic_increasing)

    if len(df) < TEST_END:
        raise ValueError(
            f"数据长度为 {len(df)}，小于论文划分所需的 {TEST_END}。"
        )

    # --------------------------------------------------------------
    # 3. 论文源码的时间划分
    # --------------------------------------------------------------
    print("\n===== 论文风格的时间划分 =====")
    print_split_summary("train", 0, TRAIN_END, df["date"])
    print_split_summary("val", TRAIN_END, VAL_END, df["date"])
    print_split_summary("test", VAL_END, TEST_END, df["date"])

    # --------------------------------------------------------------
    # 4. 只使用训练区间拟合标准化统计量
    # --------------------------------------------------------------
    values = df[feature_columns].to_numpy(dtype=np.float32)

    train_values = values[:TRAIN_END]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)

    if np.any(std == 0):
        raise ValueError("训练集存在标准差为 0 的特征，无法标准化。")

    scaled_values = (values - mean) / std
    time_marks = build_time_mark(df["date"])

    print("\n===== 训练集标准化检查 =====")
    print("训练集特征均值（应接近 0）：")
    print(np.round(scaled_values[:TRAIN_END].mean(axis=0), 4))
    print("训练集特征标准差（应接近 1）：")
    print(np.round(scaled_values[:TRAIN_END].std(axis=0), 4))

    # --------------------------------------------------------------
    # 5. 查看 train / val / test 的第一个窗口
    # --------------------------------------------------------------
    # 验证与测试窗口从边界前 SEQ_LEN 开始，以保留可用历史。
    split_window_starts = {
        "train": 0,
        "val": TRAIN_END - SEQ_LEN,
        "test": VAL_END - SEQ_LEN,
    }

    print("\n===== 一个 Informer 样本的张量形状 =====")
    for split_name, start_index in split_window_starts.items():
        seq_x, seq_y, seq_x_mark, seq_y_mark = get_one_window(
            scaled_values,
            time_marks,
            start_index,
        )

        print(f"\n[{split_name}]")
        print("seq_x.shape:     ", seq_x.shape)
        print("seq_y.shape:     ", seq_y.shape)
        print("seq_x_mark.shape:", seq_x_mark.shape)
        print("seq_y_mark.shape:", seq_y_mark.shape)

        x_start_time = df["date"].iloc[start_index]
        x_end_time = df["date"].iloc[start_index + SEQ_LEN - 1]
        y_start_time = df["date"].iloc[
            start_index + SEQ_LEN - LABEL_LEN
        ]
        y_end_time = df["date"].iloc[
            start_index + SEQ_LEN - LABEL_LEN + LABEL_LEN + PRED_LEN - 1
        ]

        print("Encoder 历史范围：", x_start_time, "至", x_end_time)
        print("Decoder 范围：    ", y_start_time, "至", y_end_time)

    print("\n===== 当前阶段结论 =====")
    print("ETTh1 的一个样本将送入 Informer：")
    print(f"Encoder 输入：({SEQ_LEN}, {len(feature_columns)})")
    print(f"Decoder 输入/标签：({LABEL_LEN + PRED_LEN}, {len(feature_columns)})")
    print("时间标记包含：month、day、day_of_week、hour。")


if __name__ == "__main__":
    main()
