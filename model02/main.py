import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
# ============================================================
# 0. 配置
# ============================================================
DATA_PATH = "daily_sales.csv"
GENERATE_DEMO_DATA = True


# ============================================================
# 1. 生成模拟数据集
# ============================================================
def generate_daily_sales_csv(output_path):
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        start="2024-01-01",
        end="2025-12-30",
        freq="D"
    )

    n_days = len(dates)
    day_index = np.arange(n_days)

    df = pd.DataFrame({
        "date": dates
    })

    # 价格：带有月度周期和随机波动
    price = (
        10
        + 0.55 * np.sin(2 * np.pi * day_index / 30)
        + rng.normal(0, 0.16, n_days)
    )

    # 促销：随机促销 + 每月固定短促销
    promotion = rng.binomial(n=1, p=0.20, size=n_days)

    for start in range(15, n_days, 30):
        end = min(start + 3, n_days)
        promotion[start:end] = 1

    # 简化节假日
    holiday = np.zeros(n_days, dtype=int)

    holiday_ranges = [
        ("2024-01-01", "2024-01-03"),
        ("2024-05-01", "2024-05-05"),
        ("2024-10-01", "2024-10-07"),
        ("2025-01-01", "2025-01-03"),
        ("2025-05-01", "2025-05-05"),
        ("2025-10-01", "2025-10-07")
    ]

    for start_date, end_date in holiday_ranges:
        mask = (
            (df["date"] >= start_date)
            & (df["date"] <= end_date)
        )
        holiday[mask] = 1

    # 销量的构成：趋势 + 季节性 + 周期性 + 促销 + 价格 + 噪声
    day_of_week = df["date"].dt.dayofweek.to_numpy()

    weekend_effect = np.where(day_of_week >= 5, 5.0, 0.0)
    yearly_seasonality = 6.0 * np.sin(2 * np.pi * day_index / 365)
    weekly_seasonality = 3.0 * np.sin(2 * np.pi * day_index / 7)
    trend = 0.035 * day_index
    noise = rng.normal(0, 5.0, n_days)

    sales = (
        100
        + trend
        + yearly_seasonality
        + weekly_seasonality
        + weekend_effect
        - 4.0 * (price - 10)
        + 15.0 * promotion
        - 6.0 * holiday
        + noise
    )

    df["sales"] = sales.round(2)
    df["price"] = price.round(2)
    df["promotion"] = promotion
    df["holiday"] = holiday

    df = df[[
        "date",
        "sales",
        "price",
        "promotion",
        "holiday"
    ]]

    # --------------------------------------------------------
    # 故意加入少量数据质量问题，供后续自动检查与清洗
    # --------------------------------------------------------

    # 3 条目标值 sales 缺失
    df.loc[[73, 261, 518], "sales"] = np.nan

    # 2 条特征 price 缺失
    df.loc[[146, 402], "price"] = np.nan

    # 3 个重复日期，但销量略有差异
    duplicate_indices = [119, 377, 645]
    duplicates = df.loc[duplicate_indices].copy()

    duplicates["sales"] = (
        duplicates["sales"]
        + rng.normal(0, 2.0, len(duplicates))
    ).round(2)

    df_raw = pd.concat(
        [df, duplicates],
        ignore_index=True
    )

    # 打乱顺序，模拟真实 CSV 未按时间排序
    df_raw = df_raw.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    df_raw.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"已生成模拟数据集：{output_path}")
    print(f"原始行数：{len(df_raw)}")


# ============================================================
# 2. Data Profile：列类型识别
# ============================================================
def is_probably_datetime_column(column_name, series):
    name_hint = bool(re.search(
        r"date|time|day|month|year|日期|时间",
        str(column_name),
        re.IGNORECASE
    ))

    if pd.api.types.is_datetime64_any_dtype(series):
        return True, pd.to_datetime(series)

    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")

        non_missing_count = series.notna().sum()
        parsed_count = parsed.notna().sum()

        parse_ratio = (
            parsed_count / non_missing_count
            if non_missing_count > 0 else 0
        )

        if name_hint and parse_ratio >= 0.8:
            return True, parsed

    return False, None


def infer_column_type(column_name, series):
    is_datetime, parsed_date = is_probably_datetime_column(
        column_name,
        series
    )

    if is_datetime:
        return "datetime", parsed_date

    non_missing = series.dropna()
    n_unique = non_missing.nunique()

    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(non_missing.unique().tolist())

        if (
            len(unique_values) <= 2
            and set(unique_values).issubset({0, 1})
        ):
            return "binary", None

        id_hint = bool(re.search(
            r"id|index|编号|序号",
            str(column_name),
            re.IGNORECASE
        ))

        if (
            id_hint
            and n_unique / max(len(non_missing), 1) > 0.95
        ):
            return "identifier", None

        return "numeric", None

    id_hint = bool(re.search(
        r"id|index|编号|序号",
        str(column_name),
        re.IGNORECASE
    ))

    if n_unique / max(len(non_missing), 1) > 0.95:
        if id_hint:
            return "identifier", None

        return "text_or_high_cardinality", None

    return "categorical", None


# ============================================================
# 3. 构建 Data Profile
# ============================================================
def build_data_profile(df):
    profile = {
        "dataset": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": list(df.columns)
        },
        "columns": {}
    }

    for column_name in df.columns:
        series = df[column_name]

        detected_type, parsed_date = infer_column_type(
            column_name,
            series
        )

        column_profile = {
            "detected_type": detected_type,
            "original_dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True))
        }

        if detected_type == "datetime":
            valid_dates = parsed_date.dropna()

            column_profile["min"] = (
                str(valid_dates.min().date())
                if not valid_dates.empty else None
            )

            column_profile["max"] = (
                str(valid_dates.max().date())
                if not valid_dates.empty else None
            )

            column_profile["is_time_candidate"] = True

            if len(valid_dates) >= 2:
                sorted_dates = valid_dates.sort_values()
                date_diffs = sorted_dates.diff().dt.days.dropna()

                column_profile["most_common_interval_days"] = (
                    int(date_diffs.mode().iloc[0])
                    if not date_diffs.empty else None
                )

                column_profile["is_daily_like"] = bool(
                    (date_diffs == 1).mean() >= 0.8
                )

        elif detected_type in {"numeric", "binary"}:
            valid_values = series.dropna()

            if not valid_values.empty:
                column_profile["min"] = round(
                    float(valid_values.min()), 4
                )
                column_profile["max"] = round(
                    float(valid_values.max()), 4
                )
                column_profile["mean"] = round(
                    float(valid_values.mean()), 4
                )
                column_profile["std"] = round(
                    float(valid_values.std()), 4
                )

            if detected_type == "binary":
                column_profile["unique_values"] = sorted(
                    valid_values.unique().tolist()
                )

        elif detected_type == "categorical":
            value_counts = series.value_counts(
                dropna=True
            ).head(10)

            column_profile["top_values"] = {
                str(key): int(value)
                for key, value in value_counts.items()
            }

        else:
            column_profile["examples"] = (
                series.dropna()
                .astype(str)
                .head(3)
                .tolist()
            )

        profile["columns"][column_name] = column_profile

    # 找到日期候选列后，额外检查重复时间点
    time_columns = [
        name
        for name, info in profile["columns"].items()
        if info.get("is_time_candidate", False)
    ]

    if time_columns:
        time_column = time_columns[0]

        parsed_time = pd.to_datetime(
            df[time_column],
            errors="coerce"
        )

        profile["dataset"]["time_column_candidate"] = time_column
        profile["dataset"]["duplicate_time_points"] = int(
            parsed_time.duplicated().sum()
        )

    return profile


# ============================================================
# 4. 具体任务配置：预测下一天的销量
# ============================================================
task_config = {
    "task_type": "time_series_forecasting",

    "target_column": "sales",
    "time_column": "date",
    "forecast_horizon": 1,

    # 这些是原始 CSV 中真实存在的历史列
    "historical_features": [
        "sales",
        "price",
        "promotion",
        "holiday"
    ],

    # 这些变量在预测目标日假设可以提前获得
    "future_known_features": [
        "holiday"
    ],

    # 这些不是原始列，而是后续根据 date 自动构造
    "derived_future_features": [
        "day_of_week"
    ],

    "validation_strategy": "time_series_split",
    "evaluation_metric": "MAE"
}


# ============================================================
# 5. Task Config 合法性检查
# ============================================================
def validate_task_config(df, task_config):
    errors = []
    warnings = []

    required_keys = [
        "task_type",
        "target_column",
        "time_column",
        "forecast_horizon",
        "historical_features",
        "future_known_features",
        "validation_strategy",
        "evaluation_metric"
    ]

    for key in required_keys:
        if key not in task_config:
            errors.append(f"Task Config 缺少必要字段：{key}")

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }

    if task_config["task_type"] != "time_series_forecasting":
        errors.append(
            "当前验证器仅支持 time_series_forecasting。"
        )

    target_column = task_config["target_column"]
    time_column = task_config["time_column"]
    historical_features = task_config["historical_features"]
    future_known_features = task_config["future_known_features"]

    # 检查目标列与时间列
    if target_column not in df.columns:
        errors.append(f"目标列不存在：{target_column}")

    if time_column not in df.columns:
        errors.append(f"时间列不存在：{time_column}")

    # 检查各输入列是否真实存在
    for feature in historical_features:
        if feature not in df.columns:
            errors.append(f"历史特征不存在：{feature}")

    for feature in future_known_features:
        if feature not in df.columns:
            errors.append(f"未来已知特征不存在：{feature}")

    # 最明显的数据泄漏检查
    if target_column in future_known_features:
        errors.append(
            f"数据泄漏：目标列 {target_column} "
            "不能放入 future_known_features。"
        )

    # 预测步长检查
    forecast_horizon = task_config["forecast_horizon"]

    if (
        not isinstance(forecast_horizon, int)
        or forecast_horizon <= 0
    ):
        errors.append("forecast_horizon 必须是正整数。")

    # 时间列检查
    if time_column in df.columns:
        parsed_time = pd.to_datetime(
            df[time_column],
            errors="coerce"
        )

        invalid_time_count = int(parsed_time.isna().sum())

        if invalid_time_count > 0:
            errors.append(
                f"时间列 {time_column} 有 "
                f"{invalid_time_count} 个无法解析的值。"
            )

        duplicate_time_count = int(parsed_time.duplicated().sum())

        if duplicate_time_count > 0:
            warnings.append(
                f"时间列 {time_column} 有 "
                f"{duplicate_time_count} 个重复时间点；"
                "训练前需要聚合或去重。"
            )

        if (
            parsed_time.notna().all()
            and not parsed_time.is_monotonic_increasing
        ):
            warnings.append(
                f"时间列 {time_column} 未按升序排列；"
                "训练前需要排序。"
            )

    # 目标列检查
    if target_column in df.columns:
        target_series = df[target_column]

        if not pd.api.types.is_numeric_dtype(target_series):
            errors.append(
                f"目标列 {target_column} 应为数值型。"
            )

        target_missing_count = int(target_series.isna().sum())

        if target_missing_count > 0:
            warnings.append(
                f"目标列 {target_column} 有 "
                f"{target_missing_count} 个缺失值；"
                "不能直接将其作为监督标签。"
            )

    # 特征列缺失值检查
    all_raw_features = set(
        historical_features + future_known_features
    )

    for feature in all_raw_features:
        if feature in df.columns and feature != target_column:
            missing_count = int(df[feature].isna().sum())

            if missing_count > 0:
                warnings.append(
                    f"特征列 {feature} 有 {missing_count} 个缺失值；"
                    "后续需要在时间划分后进行因果填补。"
                )

    # 时间序列验证方式检查
    if task_config["validation_strategy"] != "time_series_split":
        warnings.append(
            "时间序列通常应使用 time_series_split，"
            "不要随机划分。"
        )

    valid_metrics = {"MAE", "MSE", "RMSE"}

    if task_config["evaluation_metric"] not in valid_metrics:
        warnings.append(
            "回归任务常用指标为 MAE、MSE 或 RMSE。"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


# ============================================================
# 6. 结构性清洗
# ============================================================
def clean_time_series_structure(df, time_column):
    cleaned_df = df.copy()

    # 先将日期真正转换为 datetime
    cleaned_df[time_column] = pd.to_datetime(
        cleaned_df[time_column],
        errors="coerce"
    )

    # 无法解析日期的数据不能安全用于时间序列任务
    cleaned_df = cleaned_df.dropna(
        subset=[time_column]
    ).copy()

    # 同一天多条记录的聚合规则
    aggregation_rules = {
        "sales": "mean",
        "price": "mean",
        "promotion": "max",
        "holiday": "max"
    }

    cleaned_df = (
        cleaned_df
        .groupby(time_column, as_index=False)
        .agg(aggregation_rules)
    )

    # 保证时间升序
    cleaned_df = (
        cleaned_df
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    # 目标是否原本存在：后续训练时可以跳过缺失 target 的样本
    cleaned_df["sales_observed"] = (
        cleaned_df["sales"].notna().astype(int)
    )

    return cleaned_df


# ============================================================
# 7. 主流程
# ============================================================
if __name__ == "__main__":

    # 生成数据集；之后使用真实数据时改为 False
    if GENERATE_DEMO_DATA:
        generate_daily_sales_csv(DATA_PATH)

    # --------------------------------------------------------
    # A. CSV -> DataFrame
    # --------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print("\n===== 原始 DataFrame 前 5 行 =====")
    print(df.head())

    print("\n===== 原始数据缺失值 =====")
    print(df.isna().sum())

    # --------------------------------------------------------
    # B. 清洗前 Data Profile
    # --------------------------------------------------------
    raw_profile = build_data_profile(df)

    print("\n===== 清洗前 Data Profile =====")
    print(json.dumps(
        raw_profile,
        ensure_ascii=False,
        indent=2
    ))

    # --------------------------------------------------------
    # C. 清洗前 Task Config 检查
    # --------------------------------------------------------
    raw_check = validate_task_config(df, task_config)

    print("\n===== 清洗前 Task Config 检查 =====")
    print("配置是否可继续：", raw_check["valid"])

    print("\nErrors：")
    for error in raw_check["errors"]:
        print("-", error)

    print("\nWarnings：")
    for warning in raw_check["warnings"]:
        print("-", warning)

    # --------------------------------------------------------
    # D. 结构性清洗
    # --------------------------------------------------------
    clean_df = clean_time_series_structure(
        df,
        time_column=task_config["time_column"]
    )

    print("\n===== 清洗后数据基本信息 =====")
    print("清洗后行数：", len(clean_df))
    print("重复日期数量：", clean_df.duplicated(
        subset=["date"]
    ).sum())

    print("\n清洗后缺失值：")
    print(clean_df.isna().sum())

    print("\n===== 清洗后前 5 行 =====")
    print(clean_df.head())

    # --------------------------------------------------------
    # E. 清洗后再次检查
    # --------------------------------------------------------
    clean_check = validate_task_config(
        clean_df,
        task_config
    )

    print("\n===== 清洗后 Task Config 检查 =====")
    print("配置是否可继续：", clean_check["valid"])

    print("\nErrors：")
    for error in clean_check["errors"]:
        print("-", error)

    print("\nWarnings：")
    for warning in clean_check["warnings"]:
        print("-", warning)


# ============================================================
# 7. 按时间顺序划分训练、验证、测试集
# ============================================================
def split_time_series(
    df,
    train_ratio=0.70,
    val_ratio=0.15
):
    """
    输入 df 必须已经：
    1. 按日期升序排列；
    2. 每个日期只有一条记录。
    """
    n_samples = len(df)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ============================================================
# 8. 输出各集合的时间范围
# ============================================================
def print_split_summary(train_df, val_df, test_df, time_column):
    print("\n===== 时间序列数据划分 =====")

    for name, split_df in [
        ("训练集", train_df),
        ("验证集", val_df),
        ("测试集", test_df)
    ]:
        print(
            f"{name}：{len(split_df)} 条，"
            f"{split_df[time_column].min().date()} 至 "
            f"{split_df[time_column].max().date()}"
        )

# --------------------------------------------------------
# F. 时间序列划分：必须在填补与标准化之前完成
# --------------------------------------------------------
train_df, val_df, test_df = split_time_series(
    clean_df,
    train_ratio=0.70,
    val_ratio=0.15
)

print_split_summary(
    train_df,
    val_df,
    test_df,
    time_column=task_config["time_column"]
)

# ============================================================
# 9. 因果填补：只能从过去向未来填补
# ============================================================
def causal_impute_splits(train_df, val_df, test_df):
    """
    保留原始 target（sales）不变；
    创建供模型输入使用的 sales_input 与 price_input。

    ffill 的方向是过去 -> 未来。
    因此验证集可以使用训练集最后一个已知值，
    测试集可以使用训练集、验证集及测试集此前已知值。
    """

    train_part = train_df.copy()
    val_part = val_df.copy()
    test_part = test_df.copy()

    train_part["split"] = "train"
    val_part["split"] = "val"
    test_part["split"] = "test"

    # 保证按完整时间线前向填补，
    # 但不会使用任何未来日期的数据。
    full_df = pd.concat(
        [train_part, val_part, test_part],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    # 原始 sales 列保持不变，只新增输入列
    full_df["sales_input"] = full_df["sales"].ffill()

    # 原始 price 列保持不变，只新增输入列
    full_df["price_input"] = full_df["price"].ffill()

    # 如果第一天恰好缺失，ffill 无法填补；
    # 这种情况不能用未来值 bfill，应保留 NaN 并在构造窗口时跳过。
    initial_sales_missing = int(
        full_df["sales_input"].isna().sum()
    )

    initial_price_missing = int(
        full_df["price_input"].isna().sum()
    )

    if initial_sales_missing > 0:
        print(
            "警告：sales_input 开头仍有缺失值，"
            "后续需要跳过无法构造历史窗口的样本。"
        )

    if initial_price_missing > 0:
        print(
            "警告：price_input 开头仍有缺失值，"
            "后续需要跳过无法构造历史窗口的样本。"
        )

    # 重新拆回三个集合
    train_ready = full_df[
        full_df["split"] == "train"
    ].copy()

    val_ready = full_df[
        full_df["split"] == "val"
    ].copy()

    test_ready = full_df[
        full_df["split"] == "test"
    ].copy()

    # 之后不再需要 split 辅助列
    for split_df in [train_ready, val_ready, test_ready]:
        split_df.drop(columns=["split"], inplace=True)

    return train_ready, val_ready, test_ready
# --------------------------------------------------------
# G. 因果填补：创建模型输入列
# --------------------------------------------------------
train_ready, val_ready, test_ready = causal_impute_splits(
    train_df,
    val_df,
    test_df
)

print("\n===== 因果填补后缺失值 =====")
print("\n训练集：")
print(train_ready.isna().sum())

print("\n验证集：")
print(val_ready.isna().sum())

print("\n测试集：")
print(test_ready.isna().sum())

print("\n===== 原 sales 缺失时的处理方式 =====")
print(
    train_ready.loc[
        train_ready["sales"].isna(),
        ["date", "sales", "sales_input"]
    ]
)

# ============================================================
# 10. 训练集拟合标准化统计量
# ============================================================
def fit_scaler_from_train(train_df):
    """
    只读取训练集统计量。
    sales 的统计量来自原始、非缺失目标值，
    后续既用于 sales_input，也用于 sales 标签的标准化。
    """

    sales_train = train_df["sales"].dropna()
    price_train = train_df["price"].dropna()

    sales_mean = float(sales_train.mean())
    sales_std = float(sales_train.std())

    price_mean = float(price_train.mean())
    price_std = float(price_train.std())

    if sales_std == 0:
        raise ValueError("训练集 sales 标准差为 0，无法标准化。")

    if price_std == 0:
        raise ValueError("训练集 price 标准差为 0，无法标准化。")

    return {
        "sales_mean": sales_mean,
        "sales_std": sales_std,
        "price_mean": price_mean,
        "price_std": price_std
    }


# ============================================================
# 11. 应用标准化
# ============================================================
def apply_scaler(df, scaler):
    scaled_df = df.copy()

    # 历史输入：供模型读取
    scaled_df["sales_input_scaled"] = (
        scaled_df["sales_input"] - scaler["sales_mean"]
    ) / scaler["sales_std"]

    scaled_df["price_input_scaled"] = (
        scaled_df["price_input"] - scaler["price_mean"]
    ) / scaler["price_std"]

    # 训练标签：sales 缺失时结果仍是 NaN，这是正确的
    scaled_df["sales_target_scaled"] = (
        scaled_df["sales"] - scaler["sales_mean"]
    ) / scaler["sales_std"]

    # promotion / holiday 是 0/1 二值变量，不做标准化
    return scaled_df


# ============================================================
# 12. 预测值还原为原始销量尺度
# ============================================================
def inverse_sales_scale(scaled_values, scaler):
    """
    可接收 numpy 数组、Tensor 或普通数值。
    """
    return (
        scaled_values * scaler["sales_std"]
        + scaler["sales_mean"]
    )

# --------------------------------------------------------
# H. 只基于训练集拟合标准化统计量
# --------------------------------------------------------
scaler = fit_scaler_from_train(train_ready)

print("\n===== 训练集标准化统计量 =====")
print(f"sales_mean = {scaler['sales_mean']:.4f}")
print(f"sales_std  = {scaler['sales_std']:.4f}")
print(f"price_mean = {scaler['price_mean']:.4f}")
print(f"price_std  = {scaler['price_std']:.4f}")

# --------------------------------------------------------
# I. 使用同一套统计量变换三个数据集
# --------------------------------------------------------
train_scaled = apply_scaler(train_ready, scaler)
val_scaled = apply_scaler(val_ready, scaler)
test_scaled = apply_scaler(test_ready, scaler)

print("\n===== 标准化后训练集检查 =====")
print(
    "sales_target_scaled mean:",
    round(
        train_scaled["sales_target_scaled"].mean(),
        4
    )
)

print(
    "sales_target_scaled std: ",
    round(
        train_scaled["sales_target_scaled"].std(),
        4
    )
)

print(
    "price_input_scaled mean:",
    round(
        train_scaled["price_input_scaled"].mean(),
        4
    )
)

print(
    "price_input_scaled std: ",
    round(
        train_scaled["price_input_scaled"].std(),
        4
    )
)

# ============================================================
# 13. 日期特征工程
# ============================================================
def add_calendar_features(df, time_column="date"):
    result_df = df.copy()

    # 确保日期列为 datetime 类型
    result_df[time_column] = pd.to_datetime(
        result_df[time_column]
    )

    # 0=周一，...，6=周日
    result_df["day_of_week"] = (
        result_df[time_column].dt.dayofweek
    )

    # 固定创建 7 列，保证 train / val / test 特征列完全一致
    for day in range(7):
        result_df[f"day_of_week_{day}"] = (
            result_df["day_of_week"] == day
        ).astype(np.float32)

    return result_df

# --------------------------------------------------------
# J. 从日期构造未来已知日历特征
# --------------------------------------------------------
train_feature_df = add_calendar_features(train_scaled)
val_feature_df = add_calendar_features(val_scaled)
test_feature_df = add_calendar_features(test_scaled)

print("\n===== 日期特征示例 =====")
print(
    train_feature_df[
        [
            "date",
            "day_of_week",
            "day_of_week_0",
            "day_of_week_1",
            "day_of_week_2",
            "day_of_week_3",
            "day_of_week_4",
            "day_of_week_5",
            "day_of_week_6"
        ]
    ].head(10)
)

# ============================================================
# 14. 构造带有未来已知变量的时间序列窗口
# ============================================================
def build_time_series_windows(
    full_df,
    history_feature_columns,
    future_known_feature_columns,
    target_column,
    start_target_index,
    end_target_index,
    input_length=14,
    forecast_horizon=1
):
    """
    full_df：
        已按时间升序排列的完整特征表。

    start_target_index / end_target_index：
        规定“预测目标 y”属于哪个数据集。
        例如验证集目标虽然从验证区间开始，
        但历史窗口 X 可以使用训练集末尾的日期。

    返回：
        X: (N, input_length, 历史特征数)
        Z: (N, forecast_horizon, 未来已知特征数)
        y: (N, forecast_horizon, 1)
        target_dates: 每个样本对应的目标日期
    """

    x_list = []
    future_list = []
    y_list = []
    target_dates = []

    # target_index 指的是第一个预测目标所在的位置
    for target_index in range(
        start_target_index,
        end_target_index - forecast_horizon + 1
    ):
        history_start = target_index - input_length
        history_end = target_index

        # 若历史不足 input_length 天，则跳过
        if history_start < 0:
            continue

        # X：过去 input_length 天
        x_window = full_df.iloc[
            history_start:history_end
        ][history_feature_columns]

        # Z：预测日及后续 forecast_horizon 天已知的信息
        future_window = full_df.iloc[
            target_index:target_index + forecast_horizon
        ][future_known_feature_columns]

        # y：预测日及后续 forecast_horizon 天的目标值
        y_window = full_df.iloc[
            target_index:target_index + forecast_horizon
        ][target_column]

        # 任何模型输入或训练标签存在缺失，就不构造该样本
        if (
            x_window.isna().any().any()
            or future_window.isna().any().any()
            or y_window.isna().any().any()
        ):
            continue

        x_list.append(
            x_window.to_numpy(dtype=np.float32)
        )

        future_list.append(
            future_window.to_numpy(dtype=np.float32)
        )

        y_list.append(
            y_window.to_numpy(dtype=np.float32)
        )

        target_dates.append(
            full_df.iloc[target_index]["date"]
        )

    n_history_features = len(history_feature_columns)
    n_future_features = len(future_known_feature_columns)

    # 防止没有有效样本时 torch.tensor([]) 形状混乱
    if len(x_list) == 0:
        empty_x = torch.empty(
            (0, input_length, n_history_features),
            dtype=torch.float32
        )

        empty_future = torch.empty(
            (0, forecast_horizon, n_future_features),
            dtype=torch.float32
        )

        empty_y = torch.empty(
            (0, forecast_horizon, 1),
            dtype=torch.float32
        )

        return empty_x, empty_future, empty_y, target_dates

    x_tensor = torch.tensor(
        np.array(x_list),
        dtype=torch.float32
    )

    future_tensor = torch.tensor(
        np.array(future_list),
        dtype=torch.float32
    )

    # 原来形状：(N, forecast_horizon)
    # 增加最后一维，表示只有一个预测目标 sales
    y_tensor = torch.tensor(
        np.array(y_list),
        dtype=torch.float32
    ).unsqueeze(-1)

    return x_tensor, future_tensor, y_tensor, target_dates

# --------------------------------------------------------
# K. 指定后续模型实际读取的特征列
# --------------------------------------------------------
history_feature_columns = [
    "sales_input_scaled",
    "price_input_scaled",
    "promotion",
    "holiday",
    "day_of_week_0",
    "day_of_week_1",
    "day_of_week_2",
    "day_of_week_3",
    "day_of_week_4",
    "day_of_week_5",
    "day_of_week_6"
]

future_known_feature_columns = [
    "holiday",
    "day_of_week_0",
    "day_of_week_1",
    "day_of_week_2",
    "day_of_week_3",
    "day_of_week_4",
    "day_of_week_5",
    "day_of_week_6"
]

target_column = "sales_target_scaled"

input_length = 14
forecast_horizon = task_config["forecast_horizon"]

# --------------------------------------------------------
# L. 拼回完整时间线，以便验证集和测试集窗口能使用此前历史
# --------------------------------------------------------
full_feature_df = pd.concat(
    [train_feature_df, val_feature_df, test_feature_df],
    ignore_index=True
).sort_values("date").reset_index(drop=True)

train_end_index = len(train_feature_df)
val_end_index = train_end_index + len(val_feature_df)
test_end_index = len(full_feature_df)

# --------------------------------------------------------
# M. 构造三个集合的窗口
# --------------------------------------------------------
train_x, train_future, train_y, train_dates = (
    build_time_series_windows(
        full_df=full_feature_df,
        history_feature_columns=history_feature_columns,
        future_known_feature_columns=future_known_feature_columns,
        target_column=target_column,
        start_target_index=0,
        end_target_index=train_end_index,
        input_length=input_length,
        forecast_horizon=forecast_horizon
    )
)

val_x, val_future, val_y, val_dates = (
    build_time_series_windows(
        full_df=full_feature_df,
        history_feature_columns=history_feature_columns,
        future_known_feature_columns=future_known_feature_columns,
        target_column=target_column,
        start_target_index=train_end_index,
        end_target_index=val_end_index,
        input_length=input_length,
        forecast_horizon=forecast_horizon
    )
)

test_x, test_future, test_y, test_dates = (
    build_time_series_windows(
        full_df=full_feature_df,
        history_feature_columns=history_feature_columns,
        future_known_feature_columns=future_known_feature_columns,
        target_column=target_column,
        start_target_index=val_end_index,
        end_target_index=test_end_index,
        input_length=input_length,
        forecast_horizon=forecast_horizon
    )
)

print("\n===== 滑动窗口形状 =====")
print("历史特征数：", len(history_feature_columns))
print("未来已知特征数：", len(future_known_feature_columns))

print("train_x:     ", train_x.shape)
print("train_future:", train_future.shape)
print("train_y:     ", train_y.shape)

print("val_x:       ", val_x.shape)
print("val_future:  ", val_future.shape)
print("val_y:       ", val_y.shape)

print("test_x:      ", test_x.shape)
print("test_future: ", test_future.shape)
print("test_y:      ", test_y.shape)

print("\n===== 第一个训练样本 =====")
print("预测目标日期：", train_dates[0].date())
print("X 的形状：", train_x[0].shape)
print("Z 的形状：", train_future[0].shape)
print("y 的形状：", train_y[0].shape)

# ============================================================
# 15. 为传统机器学习模型准备输入
# ============================================================
def flatten_window_features(x_tensor, future_tensor):
    """
    将时序窗口展平给传统模型使用。

    原始：
    x_tensor      : (N, 14, 11)
    future_tensor : (N, 1, 8)

    展平并拼接后：
    X_flat        : (N, 14 * 11 + 1 * 8)
                  = (N, 162)
    """
    x_array = x_tensor.cpu().numpy()
    future_array = future_tensor.cpu().numpy()

    x_flat = x_array.reshape(x_array.shape[0], -1)
    future_flat = future_array.reshape(
        future_array.shape[0],
        -1
    )

    return np.concatenate([x_flat, future_flat], axis=1)


# ============================================================
# 16. 将标准化销量还原为原始销量
# ============================================================
def inverse_sales_numpy(values_scaled, scaler):
    return (
        values_scaled * scaler["sales_std"]
        + scaler["sales_mean"]
    )


# ============================================================
# 17. 计算原始销量尺度上的回归指标
# ============================================================
def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


# ============================================================
# 18. 自动模型选择：只使用验证集决定排名
# ============================================================
def run_model_selection(
    train_x,
    train_future,
    train_y,
    val_x,
    val_future,
    val_y,
    scaler
):
    # 给传统模型的展平输入
    train_X_flat = flatten_window_features(
        train_x,
        train_future
    )

    val_X_flat = flatten_window_features(
        val_x,
        val_future
    )

    # y 原来是 (N, 1, 1)，压平为 (N,)
    train_y_scaled = train_y.cpu().numpy().reshape(-1)
    val_y_scaled = val_y.cpu().numpy().reshape(-1)

    # 还原到真实销量尺度，用于最终 MAE / MSE / RMSE
    val_y_original = inverse_sales_numpy(
        val_y_scaled,
        scaler
    )

    results = []
    trained_models = {}

    # --------------------------------------------------------
    # 候选 1：最后一个销量值 Baseline
    # train_x[:, -1, 0]：
    # 最后一个历史时间点的第 0 个特征，即 sales_input_scaled
    # --------------------------------------------------------
    baseline_pred_scaled = (
        val_x[:, -1, 0]
        .cpu()
        .numpy()
    )

    baseline_pred_original = inverse_sales_numpy(
        baseline_pred_scaled,
        scaler
    )

    baseline_metrics = calculate_regression_metrics(
        val_y_original,
        baseline_pred_original
    )

    results.append({
        "model": "Last Value Baseline",
        **baseline_metrics
    })

    # --------------------------------------------------------
    # 候选 2~4：传统回归模型
    # --------------------------------------------------------
    candidate_models = {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, model in candidate_models.items():
        # 只在训练集拟合
        model.fit(train_X_flat, train_y_scaled)

        # 在验证集预测
        val_pred_scaled = model.predict(val_X_flat)

        # 将预测还原为原始销量尺度
        val_pred_original = inverse_sales_numpy(
            val_pred_scaled,
            scaler
        )

        metrics = calculate_regression_metrics(
            val_y_original,
            val_pred_original
        )

        results.append({
            "model": model_name,
            **metrics
        })

        trained_models[model_name] = model

    leaderboard = (
        pd.DataFrame(results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    return leaderboard, trained_models


# ============================================================
# 19. 运行候选模型并输出验证集排行榜
# ============================================================
leaderboard, trained_models = run_model_selection(
    train_x=train_x,
    train_future=train_future,
    train_y=train_y,
    val_x=val_x,
    val_future=val_future,
    val_y=val_y,
    scaler=scaler
)

print("\n===== 验证集模型排行榜 =====")
print(leaderboard.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

best_model_name = leaderboard.iloc[0]["model"]

print("\n验证集最优候选模型：", best_model_name)