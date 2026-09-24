import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_windows(series, input_length, prediction_length):
    x_list = []
    y_list = []

    sample_count = len(series) - input_length - prediction_length + 1

    if sample_count <= 0:
        raise ValueError("序列长度不足，无法创建窗口样本。")

    for start in range(sample_count):
        x = series[start:start + input_length]

        y = series[
            start + input_length:
            start + input_length + prediction_length
        ]

        x_list.append(x)
        y_list.append(y)

    x_tensor = torch.tensor(
        x_list,
        dtype=torch.float32,
    ).unsqueeze(-1)

    y_tensor = torch.tensor(
        y_list,
        dtype=torch.float32,
    ).unsqueeze(-1)

    return x_tensor, y_tensor


def standardize(series, mean, std):
    series_tensor = torch.tensor(
        series,
        dtype=torch.float32,
    )

    return (series_tensor - mean) / std


def inverse_standardize(series, mean, std):
    return series * std + mean


class LSTMForecaster(nn.Module):
    """
    历史时间序列
    → LSTM 逐步读取
    → 取最后一个时间点的输出
    → Linear 输出未来预测值
    """

    def __init__(self, input_size, hidden_size, prediction_length):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            prediction_length,
        )

    def forward(self, x):
        # x.shape:
        # (batch_size, input_length, input_size)
        #
        # 当前为：
        # (batch_size, 12, 1)

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        # lstm_output.shape:
        # (batch_size, input_length, hidden_size)
        #
        # 取最后一个时间点的输出：
        # (batch_size, hidden_size)
        last_output = lstm_output[:, -1, :]

        # (batch_size, hidden_size)
        # → (batch_size, prediction_length)
        prediction = self.output_layer(last_output)

        # → (batch_size, prediction_length, 1)
        return prediction.unsqueeze(-1)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            total_loss += loss.item() * batch_y.size(0)
            total_samples += batch_y.size(0)

    return total_loss / total_samples


def calculate_metrics(prediction, target):
    error = prediction - target

    mae = torch.mean(torch.abs(error))
    mse = torch.mean(error ** 2)
    rmse = torch.sqrt(mse)

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
    }


def last_value_baseline(x, prediction_length):
    """
    基线模型：重复历史窗口的最后一个值。
    """
    last_value = x[:, -1:, :]

    return last_value.repeat(
        1,
        prediction_length,
        1,
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    # ==================================================
    # 1. 构造模拟销量序列
    # ==================================================
    time_steps = torch.arange(
        120,
        dtype=torch.float32,
    )

    trend = 0.15 * time_steps

    seasonality = 5 * torch.sin(
        2 * torch.pi * time_steps / 12
    )

    noise = torch.randn(120) * 0.8

    series_tensor = 50 + trend + seasonality + noise
    series = series_tensor.tolist()

    print("===== 原始时间序列前 20 个值 =====")
    print(series[:20])

    plt.figure(figsize=(10, 4.5))

    plt.plot(
        time_steps.numpy(),
        series,
        linewidth=1.5,
        label="Sales",
    )

    plt.title("Synthetic Sales Time Series")
    plt.xlabel("Time Step")
    plt.ylabel("Sales")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 2. 时间窗口参数
    # ==================================================
    input_length = 12
    prediction_length = 1

    # ==================================================
    # 3. 按时间顺序划分数据
    # ==================================================
    train_series = series[:72]
    val_series = series[72:96]
    test_series = series[96:]

    print("\n===== 原始数据划分 =====")
    print("训练集长度：", len(train_series))
    print("验证集长度：", len(val_series))
    print("测试集长度：", len(test_series))

    # ==================================================
    # 4. 使用训练集统计量标准化
    # ==================================================
    train_tensor = torch.tensor(
        train_series,
        dtype=torch.float32,
    )

    train_mean = train_tensor.mean()
    train_std = train_tensor.std(unbiased=False)

    train_series_scaled = standardize(
        train_series,
        train_mean,
        train_std,
    ).tolist()

    val_series_scaled = standardize(
        val_series,
        train_mean,
        train_std,
    ).tolist()

    test_series_scaled = standardize(
        test_series,
        train_mean,
        train_std,
    ).tolist()

    print("\n===== 标准化信息 =====")
    print(f"训练集 mean: {train_mean.item():.4f}")
    print(f"训练集 std: {train_std.item():.4f}")

    # ==================================================
    # 5. 创建滑动窗口
    # ==================================================
    X_train, y_train = create_windows(
        train_series_scaled,
        input_length,
        prediction_length,
    )

    X_val, y_val = create_windows(
        val_series_scaled,
        input_length,
        prediction_length,
    )

    X_test, y_test = create_windows(
        test_series_scaled,
        input_length,
        prediction_length,
    )

    print("\n===== 窗口形状 =====")
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)

    print("X_val.shape:", X_val.shape)
    print("y_val.shape:", y_val.shape)

    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)

    # ==================================================
    # 6. 创建 DataLoader
    # ==================================================
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    # ==================================================
    # 7. 创建 LSTM 模型
    # ==================================================
    torch.manual_seed(42)

    model = LSTMForecaster(
        input_size=1,
        hidden_size=32,
        prediction_length=prediction_length,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
    )

    # ==================================================
    # 8. 训练与验证
    # ==================================================
    best_val_loss = float("inf")
    best_model_state = None

    total_epochs = 500

    for epoch in range(total_epochs):
        model.train()

        train_total_loss = 0.0
        train_total_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * batch_y.size(0)
            train_total_samples += batch_y.size(0)

        train_loss = train_total_loss / train_total_samples

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 50 == 0:
            print(
                f"epoch={epoch}, "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}"
            )

    # ==================================================
    # 9. 加载最佳模型并测试
    # ==================================================
    model.load_state_dict(best_model_state)

    test_loss = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n===== 最终测试结果 =====")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"test_loss（标准化尺度）={test_loss:.6f}")

    # ==================================================
    # 10. LSTM 模型与 Last Value Baseline 预测
    # ==================================================
    model.eval()

    with torch.no_grad():
        LSTM_prediction_scaled = model(X_test)

        baseline_prediction_scaled = last_value_baseline(
            X_test,
            prediction_length,
        )

    # ==================================================
    # 11. 反标准化
    # ==================================================
    LSTM_prediction_original = inverse_standardize(
        LSTM_prediction_scaled,
        train_mean,
        train_std,
    )

    baseline_prediction_original = inverse_standardize(
        baseline_prediction_scaled,
        train_mean,
        train_std,
    )

    target_original = inverse_standardize(
        y_test,
        train_mean,
        train_std,
    )

    # ==================================================
    # 12. 计算指标
    # ==================================================
    LSTM_metrics = calculate_metrics(
        LSTM_prediction_original,
        target_original,
    )

    baseline_metrics = calculate_metrics(
        baseline_prediction_original,
        target_original,
    )

    print("\n===== 测试集指标（原始销量尺度）=====")

    print("\nLSTM 模型：")
    print(f"MAE : {LSTM_metrics['mae']:.4f}")
    print(f"MSE : {LSTM_metrics['mse']:.4f}")
    print(f"RMSE: {LSTM_metrics['rmse']:.4f}")

    print("\nLast Value Baseline：")
    print(f"MAE : {baseline_metrics['mae']:.4f}")
    print(f"MSE : {baseline_metrics['mse']:.4f}")
    print(f"RMSE: {baseline_metrics['rmse']:.4f}")

    # ==================================================
    # 13. 输出部分预测结果
    # ==================================================
    print("\n===== 前 5 个测试窗口预测结果 =====")

    show_count = min(5, len(X_test))

    for index in range(show_count):
        history_original = inverse_standardize(
            X_test[index].squeeze(-1),
            train_mean,
            train_std,
        )

        print(f"\n测试样本 {index + 1}")
        print("历史输入：", history_original.tolist())
        print("真实未来：", target_original[index].squeeze(-1).tolist())

        print(
            "LSTM 模型预测：",
            LSTM_prediction_original[index].squeeze(-1).tolist(),
        )

        print(
            "基线模型预测：",
            baseline_prediction_original[index].squeeze(-1).tolist(),
        )

    # ==================================================
    # 14. 可视化预测效果
    # ==================================================
    true_values = target_original.reshape(-1).numpy()
    LSTM_values = LSTM_prediction_original.reshape(-1).numpy()
    baseline_values = baseline_prediction_original.reshape(-1).numpy()

    test_window_index = list(range(1, len(true_values) + 1))

    plt.figure(figsize=(10, 4.5))

    plt.plot(
        test_window_index,
        true_values,
        marker="o",
        linewidth=2,
        label="Actual Value",
    )

    plt.plot(
        test_window_index,
        LSTM_values,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="LSTM Model",
    )

    plt.plot(
        test_window_index,
        baseline_values,
        marker="^",
        linestyle=":",
        linewidth=2,
        label="Last Value Baseline",
    )

    plt.title("Test Set Forecast Comparison")
    plt.xlabel("Test Window Index")
    plt.ylabel("Sales")
    plt.xticks(test_window_index)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "LSTM_forecast_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()