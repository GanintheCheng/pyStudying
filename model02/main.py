import json
import re

import numpy as np
import pandas as pd


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