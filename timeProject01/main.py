import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 读取原始 CSV
# ============================================================
CSV_PATH = "simulated_daily_sales.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

print("===== 原始数据 =====")
print("原始行数：", len(df))
print(df.head())


# ============================================================
# 2. 时间列转换与时间排序
# ============================================================
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("\n===== 排序后时间范围 =====")
print("开始日期：", df["date"].min())
print("结束日期：", df["date"].max())


# ============================================================
# 3. 检查原始缺失值与重复日期
# ============================================================
print("\n===== 原始缺失值 =====")
print(df.isnull().sum())

print("\n===== 原始重复日期数量 =====")
print(df["date"].duplicated().sum())

print("\n===== 原始重复日期记录 =====")
duplicate_rows = df[df["date"].duplicated(keep=False)].sort_values("date")
print(duplicate_rows)


# ============================================================
# 4. 合并重复日期
#    同一天 sales 取平均；其他一致的特征取第一条
# ============================================================
df = (
    df.groupby("date", as_index=False)
    .agg(
        sales=("sales", "mean"),
        price=("price", "first"),
        promotion=("promotion", "first"),
        holiday=("holiday", "first"),
    )
)

print("\n===== 合并重复日期后 =====")
print("处理后行数：", len(df))
print("重复日期数量：", df["date"].duplicated().sum())


# ============================================================
# 5. 处理缺失值
#    注意：
#    - sales 保持原始 NaN，因为它是训练标签，不能伪造；
#    - sales_input 是给历史输入用的填充值；
#    - price 用前一天价格填补，只使用过去信息。
# ============================================================
df["sales_missing"] = df["sales"].isna().astype(int)

df["price"] = df["price"].ffill()
df["sales_input"] = df["sales"].ffill()

# 本数据的第一个日期没有缺失；若首日有缺失，前向填充无法处理，应单独决定删除或补值策略。
if df["price"].isna().any() or df["sales_input"].isna().any():
    raise ValueError("首日存在无法以前向填充处理的缺失值，请单独处理。")

print("\n===== 清洗后缺失值 =====")
print(df[["sales", "sales_input", "price"]].isnull().sum())

print("\n===== 原 sales 缺失的记录 =====")
print(df.loc[
    df["sales_missing"] == 1,
    ["date", "sales", "sales_input", "price", "promotion", "holiday"]
])


# ============================================================
# 6. 检查日期是否连续
# ============================================================
date_diff = df["date"].diff().dt.days

print("\n===== 时间间隔统计 =====")
print(date_diff.value_counts().sort_index())

abnormal_gap = df.loc[
    date_diff.dropna().index[date_diff.dropna() != 1],
    ["date"]
]

print("\n===== 非 1 天的时间间隔 =====")
print(abnormal_gap)


# ============================================================
# 7. 查看最终可用于建模的数据
# ============================================================
print("\n===== 最终数据前 5 行 =====")
print(df.head())

print("\n===== 最终数据后 5 行 =====")
print(df.tail())

print("\n===== 最终列名 =====")
print(df.columns.tolist())


# ============================================================
# 8. 可视化原始序列
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# 原始 sales 有 3 个缺失标签，因此图上可能有小断点，这是正常的
axes[0].plot(df["date"], df["sales"], label="Sales", color="tab:blue")
axes[0].set_title("Daily Sales")
axes[0].set_ylabel("Sales")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(df["date"], df["price"], label="Price", color="tab:orange")
axes[1].set_title("Daily Price")
axes[1].set_ylabel("Price")
axes[1].grid(alpha=0.3)
axes[1].legend()

axes[2].step(
    df["date"],
    df["promotion"],
    where="mid",
    label="Promotion",
    color="tab:green",
)
axes[2].set_title("Promotion Indicator")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Promotion (0/1)")
axes[2].set_yticks([0, 1])
axes[2].grid(alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()

# ============================================================
# 9. 特征与销量的初步关系
# ============================================================

print("\n===== 促销与销量 =====")
promotion_stats = (
    df.groupby("promotion")["sales"]
    .agg(["count", "mean", "std", "min", "max"])
    .rename(index={0: "No promotion", 1: "Promotion"})
)
print(promotion_stats)


print("\n===== 节假日与销量 =====")
holiday_stats = (
    df.groupby("holiday")["sales"]
    .agg(["count", "mean", "std"])
    .rename(index={0: "Non-holiday", 1: "Holiday"})
)
print(holiday_stats)


print("\n===== 数值列相关系数 =====")
correlation = df[["sales", "price", "promotion", "holiday"]].corr()
print(correlation)


print("\n===== 按价格区间统计销量 =====")
df["price_group"] = pd.qcut(
    df["price"],
    q=4,
    duplicates="drop",
)

price_stats = (
    df.groupby("price_group", observed=True)["sales"]
    .agg(["count", "mean", "std"])
)
print(price_stats)

# ============================================================
# 10. 添加日历特征
# ============================================================

# Monday=0, Tuesday=1, ..., Sunday=6
df["day_of_week"] = df["date"].dt.dayofweek

print("\n===== 星期几分布 =====")
print(df["day_of_week"].value_counts().sort_index())


# ============================================================
# 11. 按时间顺序划分训练、验证、测试区间
# ============================================================

n = len(df)

train_end = int(n * 0.70)       # 511
val_end = int(n * 0.85)         # 620

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\n===== 数据集划分 =====")
print(
    f"训练集：{len(train_df)} 条，"
    f"{train_df['date'].min().date()} 至 {train_df['date'].max().date()}"
)
print(
    f"验证集：{len(val_df)} 条，"
    f"{val_df['date'].min().date()} 至 {val_df['date'].max().date()}"
)
print(
    f"测试集：{len(test_df)} 条，"
    f"{test_df['date'].min().date()} 至 {test_df['date'].max().date()}"
)

# ============================================================
# 12. 用训练集统计量进行标准化
# ============================================================

# sales 的统计量只由训练集真实标签计算；pandas 会自动跳过 NaN
sales_mean = train_df["sales"].mean()
sales_std = train_df["sales"].std()

price_mean = train_df["price"].mean()
price_std = train_df["price"].std()

# 统计量只在训练集上“学习”，但可用于变换全部时间段
df["sales_input_scaled"] = (
    df["sales_input"] - sales_mean
) / sales_std

df["sales_scaled"] = (
    df["sales"] - sales_mean
) / sales_std

df["price_scaled"] = (
    df["price"] - price_mean
) / price_std

# 加入新列后，重新切分出含标准化列的数据
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\n===== 训练集标准化统计量 =====")
print(f"sales_mean = {sales_mean:.4f}")
print(f"sales_std  = {sales_std:.4f}")
print(f"price_mean = {price_mean:.4f}")
print(f"price_std  = {price_std:.4f}")

print("\n===== 标准化后训练集均值与标准差 =====")
print("sales_input_scaled mean:", train_df["sales_input_scaled"].mean())
print("sales_input_scaled std: ", train_df["sales_input_scaled"].std())
print("price_scaled mean:      ", train_df["price_scaled"].mean())
print("price_scaled std:       ", train_df["price_scaled"].std())

# ============================================================
# 13. 星期几独热编码
# ============================================================

weekday_columns = []

for day in range(7):
    column_name = f"weekday_{day}"
    df[column_name] = (df["day_of_week"] == day).astype(float)
    weekday_columns.append(column_name)

print("\n===== 星期几独热编码示例 =====")
print(
    df[
        ["date", "day_of_week"] + weekday_columns
    ].head(8)
)

import numpy as np
import torch


# ============================================================
# 14. 定义模型输入与窗口构造函数
# ============================================================

input_length = 14
prediction_length = 1

# 每个历史日期都可观测的变量
historical_columns = [
    "sales_input_scaled",
    "price_scaled",
    "promotion",
    "holiday",
] + weekday_columns

# 预测目标日提前已知的日历变量
future_known_columns = [
    "holiday",
] + weekday_columns


def create_windows(
    data_frame,
    target_start,
    target_end,
    input_length,
    historical_columns,
    future_known_columns,
):
    """
    target_start: 第一个预测目标所在的行号，包含
    target_end:   最后一个预测目标之后的行号，不包含
    """

    x_list = []
    future_list = []
    y_list = []
    target_date_list = []

    for target_index in range(target_start, target_end):
        # 原始销量标签缺失时，不能用于监督训练或评估
        if pd.isna(data_frame.loc[target_index, "sales_scaled"]):
            continue

        # X：目标日前 input_length 天的历史信息
        x = data_frame.loc[
            target_index - input_length: target_index - 1,
            historical_columns,
        ].to_numpy(dtype=np.float32)

        # Z：预测目标日已经知道的日历信息
        future_known = data_frame.loc[
            target_index,
            future_known_columns,
        ].to_numpy(dtype=np.float32)

        # y：目标日销量
        y = np.float32(data_frame.loc[target_index, "sales_scaled"])

        x_list.append(x)
        future_list.append(future_known)
        y_list.append(y)
        target_date_list.append(data_frame.loc[target_index, "date"])

    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)

    # (样本数, 未来步数=1, 未来已知特征数)
    future_tensor = torch.tensor(
        np.array(future_list),
        dtype=torch.float32,
    ).unsqueeze(1)

    # (样本数, 预测步数=1, 目标特征数=1)
    y_tensor = torch.tensor(
        np.array(y_list),
        dtype=torch.float32,
    ).view(-1, 1, 1)

    return x_tensor, future_tensor, y_tensor, target_date_list


# ============================================================
# 15. 按“目标日”范围构造训练、验证、测试窗口
# ============================================================

train_x, train_future, train_y, train_dates = create_windows(
    df,
    target_start=input_length,
    target_end=train_end,
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns,
)

val_x, val_future, val_y, val_dates = create_windows(
    df,
    target_start=train_end,
    target_end=val_end,
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns,
)

test_x, test_future, test_y, test_dates = create_windows(
    df,
    target_start=val_end,
    target_end=len(df),
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns,
)

print("\n===== 窗口形状 =====")
print("历史特征数：", len(historical_columns))
print("未来已知特征数：", len(future_known_columns))

print("train_x:     ", train_x.shape)
print("train_future:", train_future.shape)
print("train_y:     ", train_y.shape)

print("val_x:       ", val_x.shape)
print("val_future:  ", val_future.shape)
print("val_y:       ", val_y.shape)

print("test_x:      ", test_x.shape)
print("test_future: ", test_future.shape)
print("test_y:      ", test_y.shape)

print("\n===== 第一个训练窗口 =====")
print("历史结束日期：", df.loc[input_length - 1, "date"].date())
print("预测目标日期：", train_dates[0].date())
print("X 的形状：", train_x[0].shape)
print("未来已知特征 Z 的形状：", train_future[0].shape)
print("y：", train_y[0])

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 16. 创建 Dataset 与 DataLoader
# ============================================================

batch_size = 32

train_dataset = TensorDataset(
    train_x,
    train_future,
    train_y,
)

val_dataset = TensorDataset(
    val_x,
    val_future,
    val_y,
)

test_dataset = TensorDataset(
    test_x,
    test_future,
    test_y,
)

# 训练窗口可以随机打乱顺序；
# 每个窗口内部的 14 天顺序不会改变，因此不会破坏时间信息。
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# 验证、测试保持时间顺序，便于之后画预测曲线。
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

batch_x, batch_future, batch_y = next(iter(train_loader))

print("\n===== 一个训练 Batch 的形状 =====")
print("batch_x:     ", batch_x.shape)
print("batch_future:", batch_future.shape)
print("batch_y:     ", batch_y.shape)

import copy
import torch
import torch.nn as nn
# ============================================================
# 19. 训练配置
# ============================================================

class LSTMWithFutureKnown(nn.Module):
    def __init__(
        self,
        historical_feature_size,
        future_feature_size,
        hidden_size=32,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=historical_feature_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(
            hidden_size + future_feature_size,
            1,
        )

    def forward(self, x, future_known):
        lstm_output, _ = self.lstm(x)

        # (batch_size, hidden_size)
        h_last = lstm_output[:, -1, :]
        h_last = self.dropout(h_last)

        # (batch_size, 1, 8) → (batch_size, 8)
        future_vector = future_known.squeeze(1)

        # (batch_size, hidden_size + 8)
        combined = torch.cat(
            [h_last, future_vector],
            dim=1,
        )

        # (batch_size, 1) → (batch_size, 1, 1)
        prediction = self.output_layer(combined)
        return prediction.unsqueeze(1)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LSTMWithFutureKnown(
    historical_feature_size=len(historical_columns),
    future_feature_size=len(future_known_columns),
    hidden_size=32,
    dropout=0.2,
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
)

max_epochs = 100
early_stopping_patience = 15

best_val_loss = float("inf")
best_model_state = None
wait = 0


# ============================================================
# 20. 验证函数
# ============================================================

def evaluate_loss(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_future, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_future = batch_future.to(device)
            batch_y = batch_y.to(device)

            prediction = model(batch_x, batch_future)
            loss = criterion(prediction, batch_y)

            # loss 是当前 batch 的平均值，因此乘 batch_size 累加
            batch_size_now = batch_y.size(0)
            total_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

    return total_loss / total_samples


# ============================================================
# 21. 训练与验证
# ============================================================

for epoch in range(max_epochs):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_future, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_future = batch_future.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        prediction = model(batch_x, batch_future)

        # prediction 和 batch_y 都是 (batch_size, 1, 1)
        loss = criterion(prediction, batch_y)

        loss.backward()
        optimizer.step()

        batch_size_now = batch_y.size(0)
        total_train_loss += loss.item() * batch_size_now
        total_train_samples += batch_size_now

    train_loss = total_train_loss / total_train_samples
    val_loss = evaluate_loss(
        model,
        val_loader,
        criterion,
        device,
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0 or epoch == max_epochs - 1:
        print(
            f"epoch={epoch:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"lr={current_lr:.6f}, "
            f"wait={wait}"
        )

    if wait >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch}")
        break


# ============================================================
# 22. 恢复验证集最优模型
# ============================================================

model.load_state_dict(best_model_state)

print("\n===== 训练完成 =====")
print(f"最佳验证集 MSE（标准化尺度）: {best_val_loss:.6f}")
print(f"实际训练轮数：{epoch + 1}")

import math


# ============================================================
# 23. 测试集预测
# ============================================================

model.eval()

prediction_scaled_list = []
target_scaled_list = []

with torch.no_grad():
    for batch_x, batch_future, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_future = batch_future.to(device)

        prediction = model(batch_x, batch_future)

        prediction_scaled_list.append(prediction.cpu())
        target_scaled_list.append(batch_y)

prediction_scaled = torch.cat(prediction_scaled_list, dim=0)
target_scaled = torch.cat(target_scaled_list, dim=0)

# (样本数, 1, 1) → 原始销量尺度
prediction_original = (
    prediction_scaled * sales_std + sales_mean
)

target_original = (
    target_scaled * sales_std + sales_mean
)


# ============================================================
# 24. LSTM 指标：原始销量尺度
# ============================================================

mae = torch.mean(
    torch.abs(prediction_original - target_original)
).item()

mse = torch.mean(
    (prediction_original - target_original) ** 2
).item()

rmse = math.sqrt(mse)

print("\n===== LSTM 测试结果 =====")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# 25. Last Value Baseline
#    用窗口最后一天的历史销量，预测下一天销量
# ============================================================

# test_x[:, -1, 0]：
# 最后一个历史日的第 0 个特征，即 sales_input_scaled
baseline_scaled = test_x[:, -1, 0].view(-1, 1, 1)

baseline_original = (
    baseline_scaled * sales_std + sales_mean
)

baseline_mae = torch.mean(
    torch.abs(baseline_original - target_original)
).item()

baseline_mse = torch.mean(
    (baseline_original - target_original) ** 2
).item()

baseline_rmse = math.sqrt(baseline_mse)

print("\n===== Last Value Baseline 测试结果 =====")
print(f"MAE : {baseline_mae:.4f}")
print(f"MSE : {baseline_mse:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")


# ============================================================
# 26. 测试集预测曲线
# ============================================================

test_dates_for_plot = pd.to_datetime(test_dates)

plt.figure(figsize=(14, 5))

plt.plot(
    test_dates_for_plot,
    target_original.squeeze(-1).squeeze(-1).numpy(),
    label="Actual Sales",
    color="tab:blue",
)

plt.plot(
    test_dates_for_plot,
    prediction_original.squeeze(-1).squeeze(-1).numpy(),
    label="LSTM Prediction",
    color="tab:orange",
)

plt.plot(
    test_dates_for_plot,
    baseline_original.squeeze(-1).squeeze(-1).numpy(),
    label="Last Value Baseline",
    color="tab:green",
    linestyle="--",
)

plt.title("Test Set Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 27. 测试集误差分析
# ============================================================

result_df = pd.DataFrame(
    {
        "date": pd.to_datetime(test_dates),
        "actual_sales": target_original.squeeze().numpy(),
        "lstm_prediction": prediction_original.squeeze().numpy(),
        "baseline_prediction": baseline_original.squeeze().numpy(),
    }
)

result_df["lstm_abs_error"] = (
    result_df["actual_sales"] - result_df["lstm_prediction"]
).abs()

result_df["baseline_abs_error"] = (
    result_df["actual_sales"] - result_df["baseline_prediction"]
).abs()

# 合并预测当天的业务特征，方便分析误差原因
result_df = result_df.merge(
    df[["date", "price", "promotion", "holiday", "day_of_week"]],
    on="date",
    how="left",
)

print("\n===== LSTM 误差最大的 10 个测试样本 =====")
print(
    result_df
    .sort_values("lstm_abs_error", ascending=False)
    .head(10)
    .round(2)
)

print("\n===== 不同促销状态下的 LSTM 平均绝对误差 =====")
print(
    result_df
    .groupby("promotion")["lstm_abs_error"]
    .agg(["count", "mean", "max"])
    .rename(index={0: "No promotion", 1: "Promotion"})
    .round(2)
)

print("\n===== 不同星期下的 LSTM 平均绝对误差 =====")
print(
    result_df
    .groupby("day_of_week")["lstm_abs_error"]
    .agg(["count", "mean", "max"])
    .round(2)
)


# ============================================================
# 28. 实验 B：加入预测当天已知的 price 和 promotion
# ============================================================

future_known_columns_plan = [
    "price_scaled",
    "promotion",
    "holiday",
] + weekday_columns

train_x_plan, train_future_plan, train_y_plan, train_dates_plan = create_windows(
    df,
    target_start=input_length,
    target_end=train_end,
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns_plan,
)

val_x_plan, val_future_plan, val_y_plan, val_dates_plan = create_windows(
    df,
    target_start=train_end,
    target_end=val_end,
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns_plan,
)

test_x_plan, test_future_plan, test_y_plan, test_dates_plan = create_windows(
    df,
    target_start=val_end,
    target_end=len(df),
    input_length=input_length,
    historical_columns=historical_columns,
    future_known_columns=future_known_columns_plan,
)

print("\n===== 实验 B 窗口形状 =====")
print("train_future:", train_future_plan.shape)
print("val_future:  ", val_future_plan.shape)
print("test_future: ", test_future_plan.shape)


# ============================================================
# 29. 实验 B 的 DataLoader
# ============================================================

train_loader_plan = DataLoader(
    TensorDataset(train_x_plan, train_future_plan, train_y_plan),
    batch_size=32,
    shuffle=True,
)

val_loader_plan = DataLoader(
    TensorDataset(val_x_plan, val_future_plan, val_y_plan),
    batch_size=32,
    shuffle=False,
)

test_loader_plan = DataLoader(
    TensorDataset(test_x_plan, test_future_plan, test_y_plan),
    batch_size=32,
    shuffle=False,
)


# ============================================================
# 30. 训练实验 B 的新模型
# ============================================================

torch.manual_seed(42)

model_plan = LSTMWithFutureKnown(
    historical_feature_size=len(historical_columns),
    future_feature_size=len(future_known_columns_plan),
    hidden_size=32,
    dropout=0.2,
).to(device)

criterion_plan = nn.MSELoss()

optimizer_plan = torch.optim.AdamW(
    model_plan.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

scheduler_plan = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_plan,
    mode="min",
    factor=0.5,
    patience=8,
)

max_epochs = 100
early_stopping_patience = 15

best_plan_val_loss = float("inf")
best_plan_state = None
wait = 0

for epoch in range(max_epochs):
    model_plan.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_future, batch_y in train_loader_plan:
        batch_x = batch_x.to(device)
        batch_future = batch_future.to(device)
        batch_y = batch_y.to(device)

        optimizer_plan.zero_grad()

        prediction = model_plan(batch_x, batch_future)
        loss = criterion_plan(prediction, batch_y)

        loss.backward()
        optimizer_plan.step()

        batch_size_now = batch_y.size(0)
        total_train_loss += loss.item() * batch_size_now
        total_train_samples += batch_size_now

    train_loss = total_train_loss / total_train_samples

    val_loss = evaluate_loss(
        model_plan,
        val_loader_plan,
        criterion_plan,
        device,
    )

    scheduler_plan.step(val_loss)

    if val_loss < best_plan_val_loss:
        best_plan_val_loss = val_loss
        best_plan_state = copy.deepcopy(model_plan.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0 or epoch == max_epochs - 1:
        current_lr = optimizer_plan.param_groups[0]["lr"]

        print(
            f"experiment_B epoch={epoch:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"lr={current_lr:.6f}, "
            f"wait={wait}"
        )

    if wait >= early_stopping_patience:
        print(f"实验 B 提前停止于 epoch {epoch}")
        break

model_plan.load_state_dict(best_plan_state)

print("\n===== 实验 B 训练完成 =====")
print(f"最佳验证集 MSE（标准化尺度）: {best_plan_val_loss:.6f}")
print(f"实际训练轮数：{epoch + 1}")


# ============================================================
# 31. 实验 B 测试集指标
# ============================================================

model_plan.eval()

plan_prediction_list = []
plan_target_list = []

with torch.no_grad():
    for batch_x, batch_future, batch_y in test_loader_plan:
        prediction = model_plan(
            batch_x.to(device),
            batch_future.to(device),
        )

        plan_prediction_list.append(prediction.cpu())
        plan_target_list.append(batch_y)

plan_prediction_scaled = torch.cat(plan_prediction_list, dim=0)
plan_target_scaled = torch.cat(plan_target_list, dim=0)

plan_prediction_original = (
    plan_prediction_scaled * sales_std + sales_mean
)

plan_target_original = (
    plan_target_scaled * sales_std + sales_mean
)

plan_mae = torch.mean(
    torch.abs(plan_prediction_original - plan_target_original)
).item()

plan_mse = torch.mean(
    (plan_prediction_original - plan_target_original) ** 2
).item()

plan_rmse = math.sqrt(plan_mse)

print("\n===== 实验 B：LSTM + 已知当天价格与促销 =====")
print(f"MAE : {plan_mae:.4f}")
print(f"MSE : {plan_mse:.4f}")
print(f"RMSE: {plan_rmse:.4f}")

print("\n===== 与实验 A 对比 =====")
print(f"实验 A MAE : {mae:.4f}")
print(f"实验 B MAE : {plan_mae:.4f}")
print(f"实验 A MSE : {mse:.4f}")
print(f"实验 B MSE : {plan_mse:.4f}")
print(f"实验 A RMSE: {rmse:.4f}")
print(f"实验 B RMSE: {plan_rmse:.4f}")

# ============================================================
# 32. 实验 A / B 按促销状态的误差对比
# ============================================================

plan_result_df = pd.DataFrame(
    {
        "date": pd.to_datetime(test_dates_plan),
        "actual_sales": plan_target_original.squeeze().numpy(),
        "experiment_b_prediction": plan_prediction_original.squeeze().numpy(),
    }
)

plan_result_df["experiment_b_abs_error"] = (
    plan_result_df["actual_sales"]
    - plan_result_df["experiment_b_prediction"]
).abs()

comparison_df = result_df[
    [
        "date",
        "actual_sales",
        "promotion",
        "lstm_abs_error",
    ]
].merge(
    plan_result_df[
        [
            "date",
            "experiment_b_prediction",
            "experiment_b_abs_error",
        ]
    ],
    on="date",
    how="inner",
)

comparison_df = comparison_df.rename(
    columns={
        "lstm_abs_error": "experiment_a_abs_error",
    }
)

print("\n===== 按促销状态比较实验 A 与 B 的 MAE =====")

promotion_comparison = (
    comparison_df
    .groupby("promotion")[
        ["experiment_a_abs_error", "experiment_b_abs_error"]
    ]
    .mean()
    .rename(
        index={0: "No promotion", 1: "Promotion"},
        columns={
            "experiment_a_abs_error": "Experiment A MAE",
            "experiment_b_abs_error": "Experiment B MAE",
        },
    )
)

promotion_comparison["MAE Reduction"] = (
    promotion_comparison["Experiment A MAE"]
    - promotion_comparison["Experiment B MAE"]
)

print(promotion_comparison.round(4))


print("\n===== 促销日中实验 B 改善最大的 10 个样本 =====")

comparison_df["error_reduction"] = (
    comparison_df["experiment_a_abs_error"]
    - comparison_df["experiment_b_abs_error"]
)

print(
    comparison_df[
        comparison_df["promotion"] == 1
    ]
    .sort_values("error_reduction", ascending=False)
    .head(10)
    .round(2)
)

import torch.nn.functional as F


# ============================================================
# 33. TCN 的基础模块
# ============================================================

class Chomp1d(nn.Module):
    """
    删除卷积右侧的 padding，
    使当前时刻只能使用当前及过去信息。
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    一个 TCN Block：
    两层因果膨胀卷积 + ReLU + Dropout + 残差连接
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 若输入、输出通道不同，用 1×1 卷积匹配残差形状
        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.residual = nn.Identity()

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = self.residual(x)

        return self.final_relu(out + residual)


# ============================================================
# 34. TCN 预测模型
# ============================================================

class TCNWithFutureKnown(nn.Module):
    def __init__(
        self,
        historical_feature_size,
        future_feature_size,
        channels=(16, 16, 16),
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        blocks = []
        in_channels = historical_feature_size

        for block_index, out_channels in enumerate(channels):
            dilation = 2 ** block_index

            block = TemporalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )

            blocks.append(block)
            in_channels = out_channels

        self.tcn = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(
            channels[-1] + future_feature_size,
            1,
        )

    def forward(self, x, future_known):
        """
        x:            (batch_size, 14, 11)
        future_known: (batch_size, 1, 10)
        """

        # Conv1d 要求：(batch_size, channels, time_length)
        x = x.transpose(1, 2)
        # (batch_size, 11, 14)

        features = self.tcn(x)
        # (batch_size, 16, 14)

        # 最后一个时间位置汇总过去 14 天信息
        last_feature = features[:, :, -1]
        # (batch_size, 16)

        future_vector = future_known.squeeze(1)
        # (batch_size, 10)

        combined = torch.cat(
            [last_feature, future_vector],
            dim=1,
        )
        # (batch_size, 26)

        prediction = self.output_layer(combined)
        # (batch_size, 1)

        return prediction.unsqueeze(1)
        # (batch_size, 1, 1)


# ============================================================
# 35. 检查 TCN 输出形状
# ============================================================

torch.manual_seed(42)

tcn_model = TCNWithFutureKnown(
    historical_feature_size=len(historical_columns),
    future_feature_size=len(future_known_columns_plan),
    channels=(16, 16, 16),
    kernel_size=3,
    dropout=0.2,
).to(device)

print("\n===== TCN 模型 =====")
print(tcn_model)

with torch.no_grad():
    tcn_demo_prediction = tcn_model(
        batch_x.to(device),
        batch_future.to(device),
    )

print("\n===== TCN 输出形状 =====")
plan_batch_x, plan_batch_future, plan_batch_y = next(iter(train_loader_plan))

tcn_demo_prediction = tcn_model(
    plan_batch_x.to(device),
    plan_batch_future.to(device),
)

print("prediction:", tcn_demo_prediction.shape)
print("target:    ", plan_batch_y.shape)

# ============================================================
# 36. 训练 TCN
# ============================================================

torch.manual_seed(42)

tcn_model = TCNWithFutureKnown(
    historical_feature_size=len(historical_columns),
    future_feature_size=len(future_known_columns_plan),
    channels=(16, 16, 16),
    kernel_size=3,
    dropout=0.2,
).to(device)

tcn_criterion = nn.MSELoss()

tcn_optimizer = torch.optim.AdamW(
    tcn_model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

tcn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    tcn_optimizer,
    mode="min",
    factor=0.5,
    patience=8,
)

max_epochs = 100
early_stopping_patience = 15

best_tcn_val_loss = float("inf")
best_tcn_state = None
wait = 0

for epoch in range(max_epochs):
    tcn_model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_future, batch_y in train_loader_plan:
        batch_x = batch_x.to(device)
        batch_future = batch_future.to(device)
        batch_y = batch_y.to(device)

        tcn_optimizer.zero_grad()

        prediction = tcn_model(batch_x, batch_future)
        loss = tcn_criterion(prediction, batch_y)

        loss.backward()
        tcn_optimizer.step()

        batch_size_now = batch_y.size(0)
        total_train_loss += loss.item() * batch_size_now
        total_train_samples += batch_size_now

    train_loss = total_train_loss / total_train_samples

    val_loss = evaluate_loss(
        tcn_model,
        val_loader_plan,
        tcn_criterion,
        device,
    )

    tcn_scheduler.step(val_loss)

    if val_loss < best_tcn_val_loss:
        best_tcn_val_loss = val_loss
        best_tcn_state = copy.deepcopy(tcn_model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0 or epoch == max_epochs - 1:
        current_lr = tcn_optimizer.param_groups[0]["lr"]

        print(
            f"TCN epoch={epoch:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"lr={current_lr:.6f}, "
            f"wait={wait}"
        )

    if wait >= early_stopping_patience:
        print(f"TCN 提前停止于 epoch {epoch}")
        break

tcn_model.load_state_dict(best_tcn_state)

print("\n===== TCN 训练完成 =====")
print(f"最佳验证集 MSE（标准化尺度）: {best_tcn_val_loss:.6f}")
print(f"实际训练轮数：{epoch + 1}")


# ============================================================
# 37. TCN 最终测试
# ============================================================

tcn_model.eval()

tcn_prediction_list = []
tcn_target_list = []

with torch.no_grad():
    for batch_x, batch_future, batch_y in test_loader_plan:
        prediction = tcn_model(
            batch_x.to(device),
            batch_future.to(device),
        )

        tcn_prediction_list.append(prediction.cpu())
        tcn_target_list.append(batch_y)

tcn_prediction_scaled = torch.cat(tcn_prediction_list, dim=0)
tcn_target_scaled = torch.cat(tcn_target_list, dim=0)

tcn_prediction_original = (
    tcn_prediction_scaled * sales_std + sales_mean
)

tcn_target_original = (
    tcn_target_scaled * sales_std + sales_mean
)

tcn_mae = torch.mean(
    torch.abs(tcn_prediction_original - tcn_target_original)
).item()

tcn_mse = torch.mean(
    (tcn_prediction_original - tcn_target_original) ** 2
).item()

tcn_rmse = math.sqrt(tcn_mse)

print("\n===== TCN 测试结果 =====")
print(f"MAE : {tcn_mae:.4f}")
print(f"MSE : {tcn_mse:.4f}")
print(f"RMSE: {tcn_rmse:.4f}")


print("\n===== 模型对比 =====")
print(f"Last Value Baseline MAE: {baseline_mae:.4f}")
print(f"LSTM 实验 B MAE:        {plan_mae:.4f}")
print(f"TCN MAE:                {tcn_mae:.4f}")