import json
import re

import numpy as np
import pandas as pd


def is_probably_datetime_column(column_name, series):
    """
    判断一列是否可能是日期列。
    返回：
    - True / False
    - 若成功解析，则返回解析后的 datetime Series；否则为 None
    """
    name_hint = bool(re.search(
        r"date|time|day|month|year|日期|时间",
        str(column_name),
        re.IGNORECASE
    ))

    # 本来就是 datetime 类型
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, pd.to_datetime(series)

    # 对 object/string 类型尝试解析日期
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")

        non_missing = series.notna().sum()
        parsed_count = parsed.notna().sum()

        # 大部分非空值都能解析成日期，才认为是日期列
        parse_ratio = parsed_count / non_missing if non_missing > 0                                                                                                                                                      else 0

        if name_hint and parse_ratio >= 0.8:
            return True, parsed

    return False, None


def infer_column_type(column_name, series):
    """
    推断列的基础数据类型：
    datetime / binary / numeric / categorical / identifier
    """
    is_datetime, parsed_date = is_probably_datetime_column(column_name, series)

    if is_datetime:
        return "datetime", parsed_date

    non_missing = series.dropna()
    n_unique = non_missing.nunique()

    # 数值列
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(non_missing.unique().tolist())

        # 只有 0/1，则视为二值变量
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
            return "binary", None

        # 若几乎每行都唯一，列名也像 ID，则标记为标识符
        name_hint = bool(re.search(
            r"id|index|编号|序号",
            str(column_name),
            re.IGNORECASE
        ))

        if name_hint and n_unique / max(len(non_missing), 1) > 0.95:
            return "identifier", None

        return "numeric", None

    # 非数值文本列
    name_hint = bool(re.search(
        r"id|index|编号|序号",
        str(column_name),
        re.IGNORECASE
    ))

    # 文本型且几乎每行唯一，通常是 ID、标题或自由文本
    if n_unique / max(len(non_missing), 1) > 0.95:
        if name_hint:
            return "identifier", None
        return "text_or_high_cardinality", None

    return "categorical", None


def build_data_profile(df):
    """
    输入：原始或初步清洗后的 DataFrame
    输出：适合后续交给规则模块或 LLM 阅读的字典
    """
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
        detected_type, parsed_date = infer_column_type(column_name, series)

        column_profile = {
            "detected_type": detected_type,
            "original_dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True))
        }

        # 日期列信息
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

            # 排序后检查相邻日期的间隔
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

        # 数值列信息
        elif detected_type in {"numeric", "binary"}:
            valid_values = series.dropna()

            if not valid_values.empty:
                column_profile["min"] = round(float(valid_values.min()), 4)
                column_profile["max"] = round(float(valid_values.max()), 4)
                column_profile["mean"] = round(float(valid_values.mean()), 4)
                column_profile["std"] = round(float(valid_values.std()), 4)

            if detected_type == "binary":
                column_profile["unique_values"] = sorted(
                    valid_values.unique().tolist()
                )

        # 类别列信息
        elif detected_type == "categorical":
            value_counts = series.value_counts(dropna=True).head(10)

            column_profile["top_values"] = {
                str(key): int(value)
                for key, value in value_counts.items()
            }

        # 标识符或高基数文本
        else:
            examples = series.dropna().astype(str).head(3).tolist()
            column_profile["examples"] = examples

        profile["columns"][column_name] = column_profile

    return profile


# ============================================================
# 使用示例
# ============================================================
df = pd.read_csv("daily_sales.csv")

profile = build_data_profile(df)

print("===== Data Profile =====")
print(json.dumps(profile, ensure_ascii=False, indent=2))