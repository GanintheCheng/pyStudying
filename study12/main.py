import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.network(x)


torch.manual_seed(42)

# XOR 数据
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

# 两个输入相同 → 0；不同 → 1
y = torch.tensor([0, 1, 1, 0])

loader = DataLoader(
    TensorDataset(x, y),
    batch_size=4,
    shuffle=True,
)

model = MLPClassifier()
criterion = nn.CrossEntropyLoss()

# Adam 是另一种优化器；训练循环接口与 SGD 完全相同
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(1000):
    model.train()

    for batch_x, batch_y in loader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        predicted = logits.argmax(dim=1)
        accuracy = (predicted == batch_y).float().mean().item()

        print(
            f"epoch={epoch}, "
            f"loss={loss.item():.4f}, "
            f"accuracy={accuracy:.2%}"
        )

model.eval()

with torch.no_grad():
    logits = model(x)
    predicted = logits.argmax(dim=1)

print("真实标签：", y.tolist())
print("预测标签：", predicted.tolist())