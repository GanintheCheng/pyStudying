import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def create_windows(series, input_length, prediction_length):
    """
    将一条时间序列切成多个滑动窗口样本。

    例如：
    input_length = 3
    prediction_length = 1

    [100, 101, 102, 103]
    → [100, 101, 102] 预测 [103]
    """
    x_list = []
    y_list = []

    sample_count = len(series) - input_length - prediction_length + 1

    if sample_count <= 0:
        raise ValueError("序列长度不足，无法创建窗口样本。")

    for start in range(sample_count):
        # 历史输入窗口
        x = series[start:start + input_length]

        # 紧接着的未来真实值
        y = series[
            start + input_length:
            start + input_length + prediction_length
        ]

        x_list.append(x)
        y_list.append(y)

    # 单变量序列：最后一维 1 表示每个时间点只有一个特征
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
    """
    使用训练集的 mean 和 std 标准化序列。
    """
    series_tensor = torch.tensor(
        series,
        dtype=torch.float32,
    )

    return (series_tensor - mean) / std


def inverse_standardize(series, mean, std):
    """
    将标准化后的数据还原为原始数值。
    """
    return series * std + mean


class LinearForecaster(nn.Module):
    """
    最简单的时间序列预测模型：
    过去 input_length 个值 → 未来 prediction_length 个值
    """

    def __init__(self, input_length, prediction_length):
        super().__init__()

        self.linear = nn.Linear(
            input_length,
            prediction_length,
        )

    def forward(self, x):
        # 输入形状：(batch_size, input_length, 1)
        # 单变量序列，去掉最后的特征维度
        x = x.squeeze(-1)

        # (batch_size, input_length)
        # → (batch_size, prediction_length)
        prediction = self.linear(x)

        # 恢复最后的单变量特征维度
        # → (batch_size, prediction_length, 1)
        return prediction.unsqueeze(-1)


def evaluate(model, loader, criterion):
    """
    在验证集或测试集上计算平均 MSE Loss。
    只预测，不计算梯度，不更新参数。
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            # loss 是当前 batch 的平均值
            total_loss += loss.item() * batch_y.size(0)
            total_samples += batch_y.size(0)

    return total_loss / total_samples


if __name__ == "__main__":
    # ==================================================
    # 1. 原始时间序列
    # ==================================================
    # 模拟连续 30 天的销量
    series = list(range(100, 130))

    print("===== 原始时间序列 =====")
    print(series)

    # ==================================================
    # 2. 设置时间序列窗口参数
    # ==================================================
    input_length = 3
    prediction_length = 1

    # ==================================================
    # 3. 按时间顺序划分原始数据
    # ==================================================
    train_series = series[:18]      # 第 1 天到第 18 天
    val_series = series[18:24]      # 第 19 天到第 24 天
    test_series = series[24:]       # 第 25 天到第 30 天

    print("\n===== 原始数据划分 =====")
    print("训练集：", train_series)
    print("验证集：", val_series)
    print("测试集：", test_series)

    # ==================================================
    # 4. 只从训练集计算标准化参数
    # ==================================================
    train_tensor = torch.tensor(
        train_series,
        dtype=torch.float32,
    )

    train_mean = train_tensor.mean()
    train_std = train_tensor.std(unbiased=False)

    # 三个集合都使用训练集 mean 和 std
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

    print("\n标准化后的训练集前 5 个值：")
    print(train_series_scaled[:5])

    print("\n标准化后的验证集：")
    print(val_series_scaled)

    print("\n标准化后的测试集：")
    print(test_series_scaled)

    # ==================================================
    # 5. 在标准化数据上创建滑动窗口
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
    # 6. 创建 Dataset 和 DataLoader
    # ==================================================
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
    )

    print("\n===== DataLoader 信息 =====")
    print("训练集窗口数：", len(train_dataset))
    print("验证集窗口数：", len(val_dataset))
    print("测试集窗口数：", len(test_dataset))

    # ==================================================
    # 7. 创建模型、损失函数和优化器
    # ==================================================
    torch.manual_seed(42)

    model = LinearForecaster(
        input_length=input_length,
        prediction_length=prediction_length,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    # ==================================================
    # 8. 训练与验证
    # ==================================================
    best_val_loss = float("inf")
    best_model_state = None

    total_epochs = 300

    for epoch in range(total_epochs):
        model.train()

        train_total_loss = 0.0
        train_total_samples = 0

        # 训练集：反向传播并更新参数
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * batch_y.size(0)
            train_total_samples += batch_y.size(0)

        train_loss = train_total_loss / train_total_samples

        # 验证集：只计算效果，不更新参数
        val_loss = evaluate(
            model,
            val_loader,
            criterion,
        )

        # 保存验证集表现最好的模型
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
    # 9. 加载最佳模型，并进行最终测试
    # ==================================================
    model.load_state_dict(best_model_state)

    test_loss = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n===== 最终测试结果 =====")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"test_loss={test_loss:.6f}")

    # ==================================================
    # 10. 预测测试集，并还原为原始销量尺度
    # ==================================================
    model.eval()

    with torch.no_grad():
        prediction_scaled = model(X_test)

    prediction_original = inverse_standardize(
        prediction_scaled,
        train_mean,
        train_std,
    )

    target_original = inverse_standardize(
        y_test,
        train_mean,
        train_std,
    )

    print("\n===== 测试集预测结果（原始销量尺度）=====")

    for index in range(len(X_test)):
        # 测试历史输入：从标准化尺度还原为原始销量
        history_original = inverse_standardize(
            X_test[index].squeeze(-1),
            train_mean,
            train_std,
        )

        print(f"\n测试样本 {index + 1}")
        print("历史输入：", history_original.tolist())
        print("真实未来：", target_original[index].squeeze(-1).tolist())
        print("模型预测：", prediction_original[index].squeeze(-1).tolist())