import torch

def create_windows(series, input_length, prediction_length):
    x_list = []
    y_list = []

    # 能构造出的窗口数量
    sample_count = len(series) - input_length - prediction_length + 1

    for start in range(sample_count):
        # 历史输入
        x = series[start:start + input_length]

        # 紧接着的未来真实值
        y = series[
            start + input_length:
            start + input_length + prediction_length
        ]

        x_list.append(x)
        y_list.append(y)

    # 转为 float Tensor，并补上“特征数”这一维
    x_tensor = torch.tensor(x_list, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).unsqueeze(-1)

    return x_tensor, y_tensor


series = [10, 12, 13, 15, 14, 16, 18, 19]

input_length = 3
prediction_length = 2

X, y = create_windows(
    series,
    input_length,
    prediction_length,
)

from torch.utils.data import TensorDataset, DataLoader


# 一个样本 = 一段历史输入 X + 对应的未来目标 y
dataset = TensorDataset(X, y)

# 每次取两个窗口样本
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
)

import torch.nn as nn


class LinearForecaster(nn.Module):
    def __init__(self, input_length, prediction_length):
        super().__init__()

        # 输入 3 个历史值，输出 2 个未来预测值
        self.linear = nn.Linear(
            input_length,
            prediction_length,
        )

    def forward(self, x):
        # 输入 x 原本是：(batch_size, input_length, 1)
        # 这里是单变量序列，因此去掉最后的特征维度
        x = x.squeeze(-1)

        # (batch_size, input_length)
        # → (batch_size, prediction_length)
        prediction = self.linear(x)

        # 补回“特征数 = 1”这一维，
        # 使预测结果与 batch_y 的形状一致
        prediction = prediction.unsqueeze(-1)

        return prediction


model = LinearForecaster(
    input_length=input_length,
    prediction_length=prediction_length,
)

batch_x, batch_y = next(iter(loader))

prediction = model(batch_x)





torch.manual_seed(42)

# 重新创建模型，使随机初始化可复现
model = LinearForecaster(
    input_length=input_length,
    prediction_length=prediction_length,
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

model.train()

for epoch in range(5000):
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:
        # 1. 清空上一轮 batch 留下的梯度
        optimizer.zero_grad()

        # 2. 前向预测
        prediction = model(batch_x)

        # 3. 计算预测值和真实未来值之间的均方误差
        loss = criterion(prediction, batch_y)

        # 4. 反向传播
        loss.backward()

        # 5. 更新模型的权重和偏置
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    average_loss = total_loss / total_samples

    if epoch % 50 == 0:
        print(f"epoch={epoch}, train_loss={average_loss:.6f}")
model.eval()

with torch.no_grad():
    predictions = model(X)

print("\n===== 训练后的结果 =====")

for index in range(len(X)):
    history = X[index].squeeze(-1).tolist()
    target = y[index].squeeze(-1).tolist()
    prediction = predictions[index].squeeze(-1).tolist()

    print(f"\n样本 {index + 1}")
    print("历史输入：", history)
    print("真实未来：", target)
    print("模型预测：", prediction)
