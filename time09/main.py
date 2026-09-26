import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 基础设置
# ============================================================

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

input_length = 12
prediction_length = 1

batch_size = 8
learning_rate = 0.003
epochs = 500
patience = 30


# ============================================================
# 2. 构造多变量时间序列
#
# 每天的特征：
# [sales, price, promotion]
#
# sales[t] 受当天 price[t]、promotion[t] 影响。
# 因此预测 sales[t] 时，将 price[t]、promotion[t] 作为未来已知变量提供。
# ============================================================

time_steps = torch.arange(120, dtype=torch.float32)

# 0：无促销；1：有促销
promotion = ((time_steps % 10) < 3).float()

# 促销时价格相对较低
price = (
    10
    + 0.6 * torch.sin(2 * torch.pi * time_steps / 20)
    - 1.2 * promotion
)

trend = 0.15 * time_steps
seasonality = 4 * torch.sin(2 * torch.pi * time_steps / 12)
noise = torch.randn(120) * 0.8

sales = (
    50
    + trend
    + seasonality
    - 1.2 * price
    + 5.0 * promotion
    + noise
)

# 每行：[sales, price, promotion]
features = torch.stack(
    [sales, price, promotion],
    dim=1,
)

print("原始特征形状：", features.shape)
print("前 5 天数据：")
print(features[:5])


# ============================================================
# 3. 可视化原始数据
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(sales, label="Sales")
axes[0].set_title("Sales")
axes[0].set_ylabel("Sales")
axes[0].grid()

axes[1].plot(price, color="orange", label="Price")
axes[1].set_title("Price")
axes[1].set_ylabel("Price")
axes[1].grid()

axes[2].step(
    time_steps.numpy(),
    promotion.numpy(),
    where="mid",
    color="green",
    label="Promotion",
)
axes[2].set_title("Promotion")
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("Promotion")
axes[2].set_yticks([0, 1])
axes[2].grid()

plt.tight_layout()
plt.show()


# ============================================================
# 4. 按目标值所在时间点划分数据集
#
# 训练标签：sales[12] 到 sales[71]
# 验证标签：sales[72] 到 sales[95]
# 测试标签：sales[96] 到 sales[119]
#
# 验证/测试阶段可以使用更早的历史数据，
# 因为这些历史销售额在真实预测时已经发生、可以观察到。
# ============================================================

train_target_start = input_length
train_target_end = 71

val_target_start = 72
val_target_end = 95

test_target_start = 96
test_target_end = 119

print("\n===== 目标值时间划分 =====")
print(
    f"训练集目标：第 {train_target_start} 到 "
    f"第 {train_target_end} 个时间点"
)
print(
    f"验证集目标：第 {val_target_start} 到 "
    f"第 {val_target_end} 个时间点"
)
print(
    f"测试集目标：第 {test_target_start} 到 "
    f"第 {test_target_end} 个时间点"
)


# ============================================================
# 5. 只用训练时间段的统计量标准化
# ============================================================

train_raw_features = features[:72]

sales_mean = train_raw_features[:, 0].mean()
sales_std = train_raw_features[:, 0].std()

price_mean = train_raw_features[:, 1].mean()
price_std = train_raw_features[:, 1].std()

print("\n===== 训练集统计量 =====")
print(f"sales_mean = {sales_mean.item():.4f}")
print(f"sales_std  = {sales_std.item():.4f}")
print(f"price_mean = {price_mean.item():.4f}")
print(f"price_std  = {price_std.item():.4f}")


def transform_features(feature_tensor):
    """
    输入：
    (time_length, 3)

    列：
    0 -> sales：标准化
    1 -> price：标准化
    2 -> promotion：保留 0 / 1
    """
    transformed = feature_tensor.clone()

    transformed[:, 0] = (
        feature_tensor[:, 0] - sales_mean
    ) / sales_std

    transformed[:, 1] = (
        feature_tensor[:, 1] - price_mean
    ) / price_std

    transformed[:, 2] = feature_tensor[:, 2]

    return transformed


def inverse_sales(scaled_sales):
    """将标准化后的 sales 恢复到原始尺度。"""
    return scaled_sales * sales_std + sales_mean


all_scaled_features = transform_features(features)


# ============================================================
# 6. 构造“历史输入 + 未来已知变量 + 目标”的窗口
# ============================================================

def create_known_future_samples(
    feature_tensor,
    target_start,
    target_end,
    input_length,
    prediction_length,
):
    """
    对预测时刻 t：

    历史输入 x：
    [t-input_length, ..., t-1] 的 sales、price、promotion

    未来已知变量 future_known：
    [t, ..., t+prediction_length-1] 的 price、promotion

    标签 y：
    [t, ..., t+prediction_length-1] 的 sales
    """
    x_list = []
    future_known_list = []
    y_list = []

    for target_index in range(target_start, target_end + 1):
        future_end = target_index + prediction_length

        # 保证未来标签不越界
        if future_end > len(feature_tensor):
            break

        # 历史窗口：不包含预测日
        x = feature_tensor[
            target_index - input_length:target_index
        ]

        # 预测日及后续预测日已知的 price、promotion
        future_known = feature_tensor[
            target_index:future_end,
            1:3,
        ]

        # 预测日及后续预测日的 sales
        y = feature_tensor[
            target_index:future_end,
            0,
        ]

        x_list.append(x)
        future_known_list.append(future_known)
        y_list.append(y)

    x_tensor = torch.stack(x_list)

    # (samples, prediction_length, 2)
    future_known_tensor = torch.stack(future_known_list)

    # (samples, prediction_length) → (samples, prediction_length, 1)
    y_tensor = torch.stack(y_list).unsqueeze(-1)

    return x_tensor, future_known_tensor, y_tensor


train_x, train_future, train_y = create_known_future_samples(
    all_scaled_features,
    train_target_start,
    train_target_end,
    input_length,
    prediction_length,
)

val_x, val_future, val_y = create_known_future_samples(
    all_scaled_features,
    val_target_start,
    val_target_end,
    input_length,
    prediction_length,
)

test_x, test_future, test_y = create_known_future_samples(
    all_scaled_features,
    test_target_start,
    test_target_end,
    input_length,
    prediction_length,
)

print("\n===== 窗口形状 =====")
print("train_x:     ", train_x.shape)
print("train_future:", train_future.shape)
print("train_y:     ", train_y.shape)

print("val_x:       ", val_x.shape)
print("val_future:  ", val_future.shape)
print("val_y:       ", val_y.shape)

print("test_x:      ", test_x.shape)
print("test_future: ", test_future.shape)
print("test_y:      ", test_y.shape)


# ============================================================
# 7. Dataset 和 DataLoader
# ============================================================

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

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

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


# ============================================================
# 8. LSTM + 未来已知变量模型
# ============================================================

class LSTMWithKnownFuture(nn.Module):
    def __init__(
        self,
        input_size,
        future_feature_size,
        hidden_size,
        prediction_length,
        dropout=0.2,
    ):
        super().__init__()

        self.prediction_length = prediction_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # LSTM 最后隐藏状态
        # + 未来 prediction_length 天的已知特征
        self.output_layer = nn.Linear(
            hidden_size + future_feature_size * prediction_length,
            prediction_length,
        )

    def forward(self, x, future_known):
        """
        x:
        (batch, input_length, 3)

        future_known:
        (batch, prediction_length, 2)
        """
        lstm_output, _ = self.lstm(x)

        # (batch, input_length, hidden_size)
        # → (batch, hidden_size)
        last_output = lstm_output[:, -1, :]
        last_output = self.dropout(last_output)

        # (batch, prediction_length, 2)
        # → (batch, prediction_length * 2)
        future_flat = future_known.reshape(
            future_known.size(0),
            -1,
        )

        # 拼接历史摘要与未来已知变量
        combined = torch.cat(
            [last_output, future_flat],
            dim=1,
        )

        # (batch, hidden_size + future_feature_size * prediction_length)
        # → (batch, prediction_length)
        prediction = self.output_layer(combined)

        # → (batch, prediction_length, 1)
        return prediction.unsqueeze(-1)


model = LSTMWithKnownFuture(
    input_size=3,
    future_feature_size=2,
    hidden_size=16,
    prediction_length=prediction_length,
    dropout=0.2,
).to(device)

print("\n===== 模型 =====")
print(model)


# ============================================================
# 9. 损失函数、优化器、评估函数
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)


def evaluate(model, data_loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_future, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_future = batch_future.to(device)
            batch_y = batch_y.to(device)

            prediction = model(batch_x, batch_future)
            loss = criterion(prediction, batch_y)

            current_batch_size = batch_y.size(0)

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

            all_predictions.append(prediction.cpu())
            all_targets.append(batch_y.cpu())

    average_loss = total_loss / total_samples

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return average_loss, predictions, targets


# ============================================================
# 10. 训练、验证集选最优模型、Early Stopping
# ============================================================

best_val_loss = float("inf")
best_model_state = None

wait = 0

for epoch in range(epochs):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_future, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_future = batch_future.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        prediction = model(batch_x, batch_future)
        loss = criterion(prediction, batch_y)

        loss.backward()
        optimizer.step()

        current_batch_size = batch_y.size(0)

        total_train_loss += loss.item() * current_batch_size
        total_train_samples += current_batch_size

    train_loss = total_train_loss / total_train_samples

    val_loss, _, _ = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 20 == 0:
        print(
            f"epoch={epoch}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"wait={wait}"
        )

    if wait >= patience:
        print(f"Early stopping at epoch {epoch}")
        break


# ============================================================
# 11. 测试集评估
# ============================================================

model.load_state_dict(best_model_state)

test_loss, test_prediction_scaled, test_target_scaled = evaluate(
    model,
    test_loader,
)

test_prediction = inverse_sales(test_prediction_scaled)
test_target = inverse_sales(test_target_scaled)

print("\n===== 测试结果 =====")
print(f"test_loss（标准化尺度）: {test_loss:.6f}")


# ============================================================
# 12. MAE、MSE、RMSE
# ============================================================

def calculate_metrics(prediction, target):
    error = prediction - target

    mae = torch.mean(torch.abs(error))
    mse = torch.mean(error ** 2)
    rmse = torch.sqrt(mse)

    return {
        "MAE": mae.item(),
        "MSE": mse.item(),
        "RMSE": rmse.item(),
    }


lstm_metrics = calculate_metrics(
    test_prediction,
    test_target,
)

print("\n===== LSTM + 已知未来变量 =====")
print(f"MAE : {lstm_metrics['MAE']:.4f}")
print(f"MSE : {lstm_metrics['MSE']:.4f}")
print(f"RMSE: {lstm_metrics['RMSE']:.4f}")


# ============================================================
# 13. Last Value Baseline
# ============================================================

def last_value_baseline(x, prediction_length):
    """
    x: (batch, input_length, 3)

    只取最后一个历史时间点的 sales，
    不利用 price、promotion。
    """
    last_sales = x[:, -1:, 0:1]

    return last_sales.repeat(1, prediction_length, 1)


baseline_prediction_scaled = last_value_baseline(
    test_x,
    prediction_length,
)

baseline_prediction = inverse_sales(
    baseline_prediction_scaled,
)

baseline_metrics = calculate_metrics(
    baseline_prediction,
    test_target,
)

print("\n===== Last Value Baseline =====")
print(f"MAE : {baseline_metrics['MAE']:.4f}")
print(f"MSE : {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")


# ============================================================
# 14. 可视化测试集预测
# ============================================================

actual_values = test_target.squeeze(-1).squeeze(-1).numpy()
lstm_values = test_prediction.squeeze(-1).squeeze(-1).numpy()
baseline_values = baseline_prediction.squeeze(-1).squeeze(-1).numpy()

sample_index = range(1, len(actual_values) + 1)

plt.figure(figsize=(12, 5))

plt.plot(
    sample_index,
    actual_values,
    marker="o",
    label="Actual Sales",
)

plt.plot(
    sample_index,
    lstm_values,
    marker="s",
    linestyle="--",
    label="LSTM with Known Future Features",
)

plt.plot(
    sample_index,
    baseline_values,
    marker="^",
    linestyle=":",
    label="Last Value Baseline",
)

plt.title("Forecast Comparison with Known Future Variables")
plt.xlabel("Test Sample Index")
plt.ylabel("Sales")
plt.grid()
plt.legend()
plt.show()