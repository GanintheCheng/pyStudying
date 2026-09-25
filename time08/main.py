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
learning_rate = 0.001
epochs = 500


# ============================================================
# 2. 构造带趋势、周期性和噪声的单变量时间序列
# ============================================================

time_steps = torch.arange(120, dtype=torch.float32)

trend = 0.15 * time_steps
seasonality = 5 * torch.sin(2 * torch.pi * time_steps / 12)
noise = torch.randn(120) * 0.8

series_tensor = 50 + trend + seasonality + noise
series = series_tensor.tolist()

plt.figure(figsize=(10, 4))
plt.plot(series, marker="o", markersize=3)
plt.title("Original Time Series")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.grid()
plt.show()


# ============================================================
# 3. 按时间顺序划分数据集
# ============================================================

train_series = series[:72]
val_series = series[72:96]
test_series = series[96:]

print("训练集长度：", len(train_series))
print("验证集长度：", len(val_series))
print("测试集长度：", len(test_series))


# ============================================================
# 4. 只使用训练集统计量进行标准化
# ============================================================

train_tensor = torch.tensor(train_series, dtype=torch.float32)

train_mean = train_tensor.mean()
train_std = train_tensor.std()

print(f"训练集均值：{train_mean.item():.4f}")
print(f"训练集标准差：{train_std.item():.4f}")


def standardize(data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return (data_tensor - train_mean) / train_std


def inverse_standardize(data):
    return data * train_std + train_mean


train_scaled = standardize(train_series)
val_scaled = standardize(val_series)
test_scaled = standardize(test_series)


# ============================================================
# 5. 创建滑动窗口
# ============================================================

def create_windows(series_data, input_length, prediction_length):
    x_list = []
    y_list = []

    sample_count = len(series_data) - input_length - prediction_length + 1

    for start in range(sample_count):
        x = series_data[start:start + input_length]

        y = series_data[
            start + input_length:
            start + input_length + prediction_length
        ]

        x_list.append(x)
        y_list.append(y)

    # x: (sample_count, input_length, 1)
    # y: (sample_count, prediction_length, 1)
    x_tensor = torch.tensor(x_list, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1)

    return x_tensor, y_tensor


train_x, train_y = create_windows(
    train_scaled.tolist(),
    input_length,
    prediction_length,
)

val_x, val_y = create_windows(
    val_scaled.tolist(),
    input_length,
    prediction_length,
)

test_x, test_y = create_windows(
    test_scaled.tolist(),
    input_length,
    prediction_length,
)

print("\n===== 窗口数据形状 =====")
print("train_x:", train_x.shape)
print("train_y:", train_y.shape)
print("val_x:", val_x.shape)
print("val_y:", val_y.shape)
print("test_x:", test_x.shape)
print("test_y:", test_y.shape)


# ============================================================
# 6. Dataset 和 DataLoader
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
# 7. 时间序列 Transformer 模型
# ============================================================

class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        prediction_length,
        max_length,
        dropout=0.1,
    ):
        super().__init__()

        # 原始数值特征 → Transformer 维度
        self.value_embedding = nn.Linear(
            input_size,
            d_model,
        )

        # 可学习的位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
        )

        # 单个 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        # 堆叠多个 Encoder Layer
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 最后一个历史位置的特征 → 未来预测值
        self.output_layer = nn.Linear(
            d_model,
            prediction_length,
        )

    def forward(self, x):
        # x: (batch_size, input_length, input_size)
        _, time_length, _ = x.shape

        # 数值嵌入
        # (batch, 12, 1) → (batch, 12, d_model)
        x = self.value_embedding(x)

        # 加入位置编码
        x = x + self.position_embedding[:, :time_length, :]

        # Transformer Encoder
        # (batch, 12, d_model) → (batch, 12, d_model)
        features = self.encoder(x)

        # 取最后一个历史时间点的特征
        # (batch, 12, d_model) → (batch, d_model)
        last_feature = features[:, -1, :]

        # 预测未来值
        # (batch, d_model) → (batch, prediction_length)
        prediction = self.output_layer(last_feature)

        # (batch, prediction_length) → (batch, prediction_length, 1)
        return prediction.unsqueeze(-1)


model = TransformerForecaster(
    input_size=1,
    d_model=16,
    nhead=2,
    num_layers=1,
    prediction_length=1,
    max_length=input_length,
    dropout=0.1,
).to(device)

print("\n===== Transformer 模型 =====")
print(model)


# ============================================================
# 8. 损失函数和优化器
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# 9. 验证与测试函数
# ============================================================

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
# 10. 训练，并在验证集上选择最佳模型
# ============================================================

best_val_loss = float("inf")
best_model_state = None

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

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(
            f"epoch={epoch}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )


# ============================================================
# 11. 加载最佳验证集模型，并进行测试
# ============================================================

model.load_state_dict(best_model_state)

test_loss, test_prediction_scaled, test_target_scaled = evaluate(
    model,
    test_loader,
)

# 恢复为原始 sales 尺度
test_prediction = inverse_standardize(test_prediction_scaled)
test_target = inverse_standardize(test_target_scaled)

print("\n===== Transformer 测试损失 =====")
print(f"test_loss（标准化尺度）: {test_loss:.6f}")


# ============================================================
# 12. 指标：在原始 sales 尺度下计算
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


transformer_metrics = calculate_metrics(
    test_prediction,
    test_target,
)

print("\n===== Transformer 模型指标（原始 sales 尺度） =====")
print(f"MAE : {transformer_metrics['MAE']:.4f}")
print(f"MSE : {transformer_metrics['MSE']:.4f}")
print(f"RMSE: {transformer_metrics['RMSE']:.4f}")


# ============================================================
# 13. Last Value Baseline
# ============================================================

def last_value_baseline(x, prediction_length):
    # x: (batch_size, input_length, 1)
    last_value = x[:, -1:, :]

    # 每个未来时间点都预测为最后一个历史值
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

print("\n===== Last Value Baseline 指标（原始 sales 尺度） =====")
print(f"MAE : {baseline_metrics['MAE']:.4f}")
print(f"MSE : {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")


# ============================================================
# 14. 绘制测试集预测结果
# ============================================================

actual_values = test_target.squeeze(-1).squeeze(-1).numpy()
transformer_values = test_prediction.squeeze(-1).squeeze(-1).numpy()
baseline_values = baseline_prediction.squeeze(-1).squeeze(-1).numpy()

window_index = range(1, len(actual_values) + 1)

plt.figure(figsize=(12, 5))

plt.plot(
    window_index,
    actual_values,
    marker="o",
    label="Actual Value",
)

plt.plot(
    window_index,
    transformer_values,
    marker="s",
    linestyle="--",
    label="Transformer Model",
)

plt.plot(
    window_index,
    baseline_values,
    marker="^",
    linestyle=":",
    label="Last Value Baseline",
)

plt.title("Test Set Forecast Comparison")
plt.xlabel("Test Window Index")
plt.ylabel("Sales")
plt.grid()
plt.legend()
plt.show()