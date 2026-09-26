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
prediction_length = 3

batch_size = 8
learning_rate = 0.005
epochs = 500
patience = 30


# ============================================================
# 2. 构造单变量时间序列：趋势 + 周期 + 噪声
# ============================================================

time_steps = torch.arange(120, dtype=torch.float32)

trend = 0.15 * time_steps
seasonality = 5 * torch.sin(2 * torch.pi * time_steps / 12)
noise = torch.randn(120) * 0.8

series = 50 + trend + seasonality + noise

plt.figure(figsize=(12, 4))
plt.plot(series.numpy(), marker="o", markersize=3)
plt.title("Original Time Series")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.grid()
plt.show()


# ============================================================
# 3. 只使用训练时间段计算标准化统计量
# ============================================================

# 训练时间范围：0 到 71
train_raw_series = series[:72]

train_mean = train_raw_series.mean()
train_std = train_raw_series.std()

print(f"训练集均值：{train_mean.item():.4f}")
print(f"训练集标准差：{train_std.item():.4f}")


def standardize(data):
    return (data - train_mean) / train_std


def inverse_standardize(data):
    return data * train_std + train_mean


scaled_series = standardize(series)


# ============================================================
# 4. 构造直接多步预测样本
#
# 输入：target_index 前 input_length 步
# 标签：target_index 开始的 prediction_length 步
# ============================================================

def create_direct_forecast_samples(
    scaled_series,
    target_start,
    target_end,
    input_length,
    prediction_length,
):
    x_list = []
    y_list = []

    for target_index in range(target_start, target_end + 1):
        future_end = target_index + prediction_length

        if future_end > len(scaled_series):
            break

        # 历史输入：例如 x1 到 x12
        x = scaled_series[
            target_index - input_length:target_index
        ]

        # 未来标签：例如 x13、x14、x15
        y = scaled_series[
            target_index:future_end
        ]

        x_list.append(x)
        y_list.append(y)

    # 单变量输入和目标，补上特征维度 1
    x_tensor = torch.stack(x_list).unsqueeze(-1)
    y_tensor = torch.stack(y_list).unsqueeze(-1)

    return x_tensor, y_tensor


# 训练标签覆盖时间点 12 到 71
train_x, train_y = create_direct_forecast_samples(
    scaled_series,
    target_start=input_length,
    target_end=69,
    input_length=input_length,
    prediction_length=prediction_length,
)

# 验证标签覆盖时间点 72 到 95
val_x, val_y = create_direct_forecast_samples(
    scaled_series,
    target_start=72,
    target_end=93,
    input_length=input_length,
    prediction_length=prediction_length,
)

# 测试标签覆盖时间点 96 到 119
test_x, test_y = create_direct_forecast_samples(
    scaled_series,
    target_start=96,
    target_end=117,
    input_length=input_length,
    prediction_length=prediction_length,
)

print("\n===== 数据形状 =====")
print("train_x:", train_x.shape)
print("train_y:", train_y.shape)
print("val_x:", val_x.shape)
print("val_y:", val_y.shape)
print("test_x:", test_x.shape)
print("test_y:", test_y.shape)


# ============================================================
# 5. Dataset 和 DataLoader
# ============================================================

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)
test_dataset = TensorDataset(test_x, test_y)

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
# 6. LSTM 直接多步预测模型
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        prediction_length,
        dropout=0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # 一次输出未来 prediction_length 个预测值
        self.output_layer = nn.Linear(
            hidden_size,
            prediction_length,
        )

    def forward(self, x):
        # x: (batch, 12, 1)
        lstm_output, _ = self.lstm(x)

        # (batch, 12, hidden_size)
        # → (batch, hidden_size)
        last_output = lstm_output[:, -1, :]

        last_output = self.dropout(last_output)

        # (batch, hidden_size)
        # → (batch, 3)
        prediction = self.output_layer(last_output)

        # (batch, 3)
        # → (batch, 3, 1)
        return prediction.unsqueeze(-1)


model = LSTMForecaster(
    input_size=1,
    hidden_size=32,
    prediction_length=prediction_length,
    dropout=0.1,
).to(device)

print("\n===== 模型 =====")
print(model)


# ============================================================
# 7. 损失函数、优化器、评估函数
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
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            prediction = model(batch_x)
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
# 8. 训练、验证集选最优模型、Early Stopping
# ============================================================

best_val_loss = float("inf")
best_model_state = None
wait = 0

for epoch in range(epochs):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        prediction = model(batch_x)
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
# 9. 测试集评估
# ============================================================

model.load_state_dict(best_model_state)

test_loss, test_prediction_scaled, test_target_scaled = evaluate(
    model,
    test_loader,
)

test_prediction = inverse_standardize(test_prediction_scaled)
test_target = inverse_standardize(test_target_scaled)

print("\n===== 测试结果 =====")
print(f"test_loss（标准化尺度）: {test_loss:.6f}")


# ============================================================
# 10. MAE、MSE、RMSE
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


overall_metrics = calculate_metrics(
    test_prediction,
    test_target,
)

print("\n===== 总体指标：未来 3 步一起计算 =====")
print(f"MAE : {overall_metrics['MAE']:.4f}")
print(f"MSE : {overall_metrics['MSE']:.4f}")
print(f"RMSE: {overall_metrics['RMSE']:.4f}")


# ============================================================
# 11. 分别计算第 1、2、3 步预测指标
# ============================================================

for step in range(prediction_length):
    step_prediction = test_prediction[:, step, :]
    step_target = test_target[:, step, :]

    step_metrics = calculate_metrics(
        step_prediction,
        step_target,
    )

    print(f"\n===== 未来第 {step + 1} 步预测指标 =====")
    print(f"MAE : {step_metrics['MAE']:.4f}")
    print(f"MSE : {step_metrics['MSE']:.4f}")
    print(f"RMSE: {step_metrics['RMSE']:.4f}")


# ============================================================
# 12. Last Value Baseline
#
# 每个未来预测值都等于最后一个历史值
# ============================================================

def last_value_baseline(x, prediction_length):
    # x: (batch, input_length, 1)
    last_value = x[:, -1:, :]

    # (batch, 1, 1) → (batch, prediction_length, 1)
    return last_value.repeat(1, prediction_length, 1)


baseline_prediction_scaled = last_value_baseline(
    test_x,
    prediction_length,
)

baseline_prediction = inverse_standardize(
    baseline_prediction_scaled,
)

baseline_metrics = calculate_metrics(
    baseline_prediction,
    test_target,
)

print("\n===== Last Value Baseline：总体指标 =====")
print(f"MAE : {baseline_metrics['MAE']:.4f}")
print(f"MSE : {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")


# ============================================================
# 13. 展示一个测试样本的历史、真实未来和预测未来
# ============================================================

sample_index = 0

history_values = inverse_standardize(
    test_x[sample_index]
).squeeze(-1).numpy()

actual_future = test_target[
    sample_index
].squeeze(-1).numpy()

predicted_future = test_prediction[
    sample_index
].squeeze(-1).numpy()

baseline_future = baseline_prediction[
    sample_index
].squeeze(-1).numpy()

history_steps = range(1, input_length + 1)
future_steps = range(
    input_length + 1,
    input_length + prediction_length + 1,
)

plt.figure(figsize=(12, 5))

plt.plot(
    history_steps,
    history_values,
    marker="o",
    label="Historical Sales",
)

plt.plot(
    future_steps,
    actual_future,
    marker="o",
    linewidth=2,
    label="Actual Future Sales",
)

plt.plot(
    future_steps,
    predicted_future,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Direct LSTM Prediction",
)

plt.plot(
    future_steps,
    baseline_future,
    marker="^",
    linestyle=":",
    linewidth=2,
    label="Last Value Baseline",
)

plt.axvline(
    input_length + 0.5,
    color="gray",
    linestyle="--",
)

plt.title("One Test Sample: Past 12 Steps to Future 3 Steps")
plt.xlabel("Time Step in the Window")
plt.ylabel("Sales")
plt.grid()
plt.legend()
plt.show()