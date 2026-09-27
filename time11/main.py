import copy
import statistics

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
hidden_size = 16
learning_rate = 0.005

epochs = 300
patience = 30

# ============================================================
# 2. 构造单变量时间序列
# ============================================================

time_steps = torch.arange(120, dtype=torch.float32)

trend = 0.15 * time_steps
seasonality = 5 * torch.sin(2 * torch.pi * time_steps / 12)
noise = torch.randn(120) * 0.8

series = 50 + trend + seasonality + noise

plt.figure(figsize=(12, 4))
plt.plot(series.numpy(), marker="o", markersize=3)
plt.axvline(96, color="red", linestyle="--", label="Final Test Starts")
plt.title("Original Time Series")
plt.xlabel("Time Step")
plt.ylabel("Sales")
plt.grid()
plt.legend()
plt.show()


# ============================================================
# 3. LSTM 模型
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size,
            dilation,
            dropout=0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

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

        return self.final_relu(out + self.residual(x))


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
            TemporalBlock(
                input_channels=input_size,
                output_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
            ),
            TemporalBlock(
                input_channels=hidden_channels,
                output_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=2,
                dropout=dropout,
            ),
        )

        self.output_layer = nn.Linear(
            hidden_channels,
            prediction_length,
        )

    def forward(self, x):
        # (batch, time, feature)
        # → (batch, feature/channel, time)
        x = x.transpose(1, 2)

        features = self.tcn(x)

        # (batch, hidden_channels, time)
        # → (batch, hidden_channels)
        last_feature = features[:, :, -1]

        prediction = self.output_layer(last_feature)

        return prediction.unsqueeze(-1)


# ============================================================
# 4. 构造指定目标时间范围内的滑动窗口
# ============================================================

def create_windows_for_target_range(
        scaled_series,
        target_start,
        target_end,
        input_length,
        prediction_length,
):
    """
    target_start、target_end 均包含在范围内。

    例如 target_index=48：
    输入：第 36 到 47 个时间点
    标签：第 48 个时间点
    """
    x_list = []
    y_list = []

    for target_index in range(target_start, target_end + 1):
        future_end = target_index + prediction_length

        if future_end > len(scaled_series):
            break

        x = scaled_series[
            target_index - input_length:target_index
        ]

        y = scaled_series[
            target_index:future_end
        ]

        x_list.append(x)
        y_list.append(y)

    x_tensor = torch.stack(x_list).unsqueeze(-1)
    y_tensor = torch.stack(y_list).unsqueeze(-1)

    return x_tensor, y_tensor


# ============================================================
# 5. 评估函数
# ============================================================

criterion = nn.MSELoss()


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


def calculate_mae(prediction, target):
    return torch.mean(torch.abs(prediction - target)).item()


# ============================================================
# 6. 单个滚动验证折的训练与评估
# ============================================================

def run_one_fold(
        raw_series,
        train_end,
        val_end,
        fold_index,
):
    """
    train_end、val_end 都是“右开区间”边界。

    train: [0, train_end)
    val:   [train_end, val_end)
    """
    best_epoch = 0
    print("\n" + "=" * 60)
    print(
        f"Fold {fold_index}: "
        f"训练 [0, {train_end - 1}]，"
        f"验证 [{train_end}, {val_end - 1}]"
    )

    # --------------------------------------------------------
    # 1. 只使用当前 Fold 的训练期计算标准化统计量
    # --------------------------------------------------------

    train_raw = raw_series[:train_end]

    fold_mean = train_raw.mean()
    fold_std = train_raw.std()

    scaled_series = (raw_series - fold_mean) / fold_std

    # --------------------------------------------------------
    # 2. 构造训练样本和验证样本
    # --------------------------------------------------------

    train_x, train_y = create_windows_for_target_range(
        scaled_series=scaled_series,
        target_start=input_length,
        target_end=train_end - 1,
        input_length=input_length,
        prediction_length=prediction_length,
    )

    val_x, val_y = create_windows_for_target_range(
        scaled_series=scaled_series,
        target_start=train_end,
        target_end=val_end - 1,
        input_length=input_length,
        prediction_length=prediction_length,
    )

    print("train_x:", train_x.shape)
    print("val_x:  ", val_x.shape)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # 3. 每个 Fold 都从随机初始化开始训练新模型
    # --------------------------------------------------------

    torch.manual_seed(100 + fold_index)

    model = TCNForecaster(
        input_size=1,
        hidden_channels=16,
        prediction_length=prediction_length,
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_val_mae = float("inf")
    best_model_state = None
    wait = 0

    # --------------------------------------------------------
    # 4. 当前 Fold 的训练和 Early Stopping
    # --------------------------------------------------------

    for epoch in range(epochs):
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()

        # 当前验证集结果
        _, val_prediction_scaled, val_target_scaled = evaluate(
            model,
            val_loader,
        )

        # 恢复原始 sales 尺度后计算 MAE
        val_prediction = (
                val_prediction_scaled * fold_std + fold_mean
        )

        val_target = (
                val_target_scaled * fold_std + fold_mean
        )

        val_mae = calculate_mae(
            val_prediction,
            val_target,
        )

        # 按验证 MAE 选择最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch % 50 == 0:
            print(
                f"epoch={epoch}, "
                f"val_mae={val_mae:.4f}, "
                f"wait={wait}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # --------------------------------------------------------
    # 5. 恢复当前 Fold 最优模型，再计算验证 MAE
    # --------------------------------------------------------

    model.load_state_dict(best_model_state)

    _, val_prediction_scaled, val_target_scaled = evaluate(
        model,
        val_loader,
    )

    val_prediction = (
            val_prediction_scaled * fold_std + fold_mean
    )

    val_target = (
            val_target_scaled * fold_std + fold_mean
    )

    final_val_mae = calculate_mae(
        val_prediction,
        val_target,
    )

    print(f"Fold {fold_index} 最佳验证 MAE: {final_val_mae:.4f}")
    f"最佳 epoch: {best_epoch}"

    return final_val_mae,best_epoch


# ============================================================
# 7. 定义扩展窗口滚动验证折
#
# 最后 [96, 119] 不参与滚动验证，留作最终测试集
# ============================================================

folds = [
    (48, 60),
    (60, 72),
    (72, 84),
    (84, 96),
]

all_fold_mae = []
all_best_epochs = []
for fold_index, (train_end, val_end) in enumerate(
    folds,
    start=1,
):
    fold_mae, best_epoch = run_one_fold(
        raw_series=series,
        train_end=train_end,
        val_end=val_end,
        fold_index=fold_index,
    )

    all_fold_mae.append(fold_mae)
    all_best_epochs.append(best_epoch)
    final_epochs = round(statistics.median(all_best_epochs))

    print("各 Fold 最佳 epoch：", all_best_epochs)
    print("最终训练 epoch：", final_epochs)
# ============================================================
# 8. 输出滚动验证结果
# ============================================================

print("\n" + "=" * 60)
print("===== 滚动验证汇总 =====")

for fold_index, fold_mae in enumerate(all_fold_mae, start=1):
    print(f"Fold {fold_index} 验证 MAE: {fold_mae:.4f}")

mean_mae = statistics.mean(all_fold_mae)
std_mae = statistics.stdev(all_fold_mae)

print(f"\n平均验证 MAE: {mean_mae:.4f}")
print(f"验证 MAE 标准差: {std_mae:.4f}")

print(
    "\n最后 24 个时间点 [96, 119] 没有参与模型选择，"
    "后续将用于最终测试。"
)
# ============================================================
# 9. 最终训练：使用全部开发数据 [0, 95]
#    测试集 [96, 119] 此前从未参与模型选择
# ============================================================

print("\n" + "=" * 60)
print("===== 最终训练与测试 =====")

final_train_end = 96

# 只用全部开发数据计算最终标准化统计量
final_train_raw = series[:final_train_end]

final_mean = final_train_raw.mean()
final_std = final_train_raw.std()

final_scaled_series = (
    series - final_mean
) / final_std


# 开发数据中的训练样本：
# 输入在过去，标签位于 [12, 95]
final_train_x, final_train_y = create_windows_for_target_range(
    scaled_series=final_scaled_series,
    target_start=input_length,
    target_end=final_train_end - 1,
    input_length=input_length,
    prediction_length=prediction_length,
)

# 最终测试样本：
# 标签位于 [96, 119]
final_test_x, final_test_y = create_windows_for_target_range(
    scaled_series=final_scaled_series,
    target_start=final_train_end,
    target_end=len(series) - 1,
    input_length=input_length,
    prediction_length=prediction_length,
)

print("最终训练集 x：", final_train_x.shape)
print("最终测试集 x：", final_test_x.shape)

final_train_loader = DataLoader(
    TensorDataset(final_train_x, final_train_y),
    batch_size=batch_size,
    shuffle=True,
)

final_test_loader = DataLoader(
    TensorDataset(final_test_x, final_test_y),
    batch_size=batch_size,
    shuffle=False,
)


# ============================================================
# 10. 创建全新 TCN，并固定训练 18 个 epoch
# ============================================================

torch.manual_seed(999)

final_model = TCNForecaster(
    input_size=1,
    hidden_channels=16,
    prediction_length=prediction_length,
    kernel_size=3,
    dropout=0.1,
).to(device)

final_optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)

for epoch in range(final_epochs):
    final_model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in final_train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        final_optimizer.zero_grad()

        prediction = final_model(batch_x)
        loss = criterion(prediction, batch_y)

        loss.backward()
        final_optimizer.step()

        current_batch_size = batch_y.size(0)

        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    train_loss = total_loss / total_samples

    print(
        f"final epoch={epoch + 1}/{final_epochs}, "
        f"train_loss={train_loss:.6f}"
    )


# ============================================================
# 11. 最终测试：只执行一次
# ============================================================

test_loss, test_prediction_scaled, test_target_scaled = evaluate(
    final_model,
    final_test_loader,
)

# 恢复原始 sales 尺度
test_prediction = (
    test_prediction_scaled * final_std + final_mean
)

test_target = (
    test_target_scaled * final_std + final_mean
)


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


final_tcn_metrics = calculate_metrics(
    test_prediction,
    test_target,
)

print("\n===== 最终 TCN 测试结果 =====")
print(f"test_loss（标准化尺度）: {test_loss:.6f}")
print(f"MAE : {final_tcn_metrics['MAE']:.4f}")
print(f"MSE : {final_tcn_metrics['MSE']:.4f}")
print(f"RMSE: {final_tcn_metrics['RMSE']:.4f}")


# ============================================================
# 12. Last Value Baseline 最终测试
# ============================================================

def last_value_baseline(x, prediction_length):
    last_value = x[:, -1:, :]
    return last_value.repeat(1, prediction_length, 1)


baseline_prediction_scaled = last_value_baseline(
    final_test_x,
    prediction_length,
)

baseline_prediction = (
    baseline_prediction_scaled * final_std + final_mean
)

baseline_metrics = calculate_metrics(
    baseline_prediction,
    test_target,
)

print("\n===== Last Value Baseline 最终测试结果 =====")
print(f"MAE : {baseline_metrics['MAE']:.4f}")
print(f"MSE : {baseline_metrics['MSE']:.4f}")
print(f"RMSE: {baseline_metrics['RMSE']:.4f}")


# ============================================================
# 13. 最终测试集预测图
# ============================================================

actual_values = test_target.squeeze(-1).squeeze(-1).numpy()
tcn_values = test_prediction.squeeze(-1).squeeze(-1).numpy()
baseline_values = baseline_prediction.squeeze(-1).squeeze(-1).numpy()

test_index = range(1, len(actual_values) + 1)

plt.figure(figsize=(12, 5))

plt.plot(
    test_index,
    actual_values,
    marker="o",
    label="Actual Value",
)

plt.plot(
    test_index,
    tcn_values,
    marker="s",
    linestyle="--",
    label="Final TCN",
)

plt.plot(
    test_index,
    baseline_values,
    marker="^",
    linestyle=":",
    label="Last Value Baseline",
)

plt.title("Final Test Set Forecast Comparison")
plt.xlabel("Test Time Index")
plt.ylabel("Sales")
plt.grid()
plt.legend()
plt.show()