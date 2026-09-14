import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 固定随机种子：每次运行时初始 weight、bias 一致，便于复现
torch.manual_seed(42)

# 4 个样本，每个样本只有 1 个输入特征
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

# 每个样本对应的真实标签
target = torch.tensor([
    [1.0],
    [4.0],
    [7.0],
    [10.0],
])

# 创建模型、损失函数、优化器
model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(2000):
    optimizer.zero_grad()

    prediction = model(x)
    loss = criterion(prediction, target)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

# 查看学到的参数
print("weight:", model.linear.weight.item())
print("bias:", model.linear.bias.item())

model.eval()

with torch.no_grad():
    prediction = model(
        torch.tensor([[5.0], [10.0]])
    )
    print(prediction.squeeze().tolist())
