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
learning_rate = 0.005
epochs = 500


# ============================================================
# 2. 构造一个带趋势、季节性和噪声的单变量时间序列
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
# 3. 按时间顺序划分训练集、验证集和测试集
# ============================================================

train_series = series[:72]
val_series = series[72:96]
test_series = series[96:]

print("训练集长度：", len(train_series))
print("验证集长度：", len(val_series))
print("测试集长度：", len(test_series))


# ============================================================
# 4. 只用训练集的统计量进行标准化
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
# 5. 滑动窗口：过去 input_length 个点 → 未来 prediction_length 个点
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
# 7. TCN 模型
# ============================================================

class Chomp1d(nn.Module):
    """
    删除卷积右侧因 padding 产生的额外时间点，
    从而保证因果卷积不会看到未来数据。
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
    一个 TCN 模块：
    扩张因果卷积 → ReLU → 扩张因果卷积 → ReLU → 残差连接
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        dilation,
        dropout=0.1,
    ):
        super().__init__()

        # 这个 padding 配合 Chomp1d 实现因果卷积
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 若输入通道数和输出通道数不同，需要转换后才能相加
        if input_channels != output_channels:
            self.residual = nn.Conv1d(
                input_channels,
                output_channels,
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


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_channels,
        prediction_length,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        self.tcn = nn.Sequential(
            # dilation=1：重点提取相邻时间点的局部模式
            TemporalBlock(
                input_channels=input_size,
                output_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
            ),

            # dilation=2：可以看到更远的历史点
            TemporalBlock(
                input_channels=hidden_channels,
                output_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=2,
                dropout=dropout,
            ),

            # dilation=4：进一步扩大感受野
            TemporalBlock(
                input_channels=hidden_channels,
                output_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=4,
                dropout=dropout,
            ),
        )

        self.output_layer = nn.Linear(
            hidden_channels,
            prediction_length,
        )

    def forward(self, x):
        # 输入 x: (batch_size, time_length, feature_count)
        # Conv1d 要求: (batch_size, channel_count, time_length)
        x = x.transpose(1, 2)

        # (batch_size, 1, 12) → (batch_size, hidden_channels, 12)
        features = self.tcn(x)

        # 取最后一个历史时刻的特征
        last_features = features[:, :, -1]

        # (batch_size, hidden_channels) → (batch_size, prediction_length)
        prediction = self.output_layer(last_features)

        # (batch_size, prediction_length) → (batch_size, prediction_length, 1)
        return prediction.unsqueeze(-1)


model = TCNForecaster(
    input_size=1,
    hidden_channels=32,
    prediction_length=prediction_length,
    kernel_size=3,
    dropout=0.1,
).to(device)

print("\n===== TCN 模型 =====")
print(model)


# ============================================================
# 8. 损失函数与优化器
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# 9. 验证 / 测试函数
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
# 10. 训练：按验证集 loss 保存最佳模型参数
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
# 11. 加载验证集上最优模型，在测试集评估
# ============================================================

model.load_state_dict(best_model_state)

test_loss, test_prediction_scaled, test_target_scaled = evaluate(
    model,
    test_loader,
)

# 从标准化后的数值恢复到原始 sales 尺度
test_prediction = inverse_standardize(test_prediction_scaled)
test_target = inverse_standardize(test_target_scaled)

print("\n===== TCN 测试集损失 =====")
print(f"test_loss（标准化尺度）: {test_loss:.6f}")


# ============================================================
# 12. 计算原始尺度下的 MAE / MSE / RMSE
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


tcn_metrics = calculate_metrics(
    test_prediction,
    test_target,
)

print("\n===== TCN 模型指标（原始 sales 尺度） =====")
print(f"MAE : {tcn_metrics['MAE']:.4f}")
print(f"MSE : {tcn_metrics['MSE']:.4f}")
print(f"RMSE: {tcn_metrics['RMSE']:.4f}")


# ============================================================
# 13. Last Value Baseline
# ============================================================

def last_value_baseline(x, prediction_length):
    # x: (batch_size, input_length, 1)
    last_value = x[:, -1:, :]

    # 未来每一步都预测成最后一个历史值
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
# 14. 可视化：TCN 与 Last Value Baseline
# ============================================================

actual_values = test_target.squeeze(-1).squeeze(-1).numpy()
tcn_values = test_prediction.squeeze(-1).squeeze(-1).numpy()
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
    tcn_values,
    marker="s",
    linestyle="--",
    label="TCN Model",
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