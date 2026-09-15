import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(42)

# 训练集：8 条数据，每条有 2 个特征
train_x = torch.tensor([
    [-2.0, -1.0],
    [-1.0, -2.0],
    [-2.0, -2.0],
    [-1.5, -1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 2.0],
    [1.5, 1.0],
])

# 前 4 条属于类别 0，后 4 条属于类别 1
train_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

# 验证集：模型训练时不使用它更新参数
val_x = torch.tensor([
    [-2.0, -1.5],
    [-1.5, -2.5],
    [1.5, 2.0],
    [2.5, 1.5],
])

val_y = torch.tensor([0, 0, 1, 1])

train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=2,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(val_x, val_y),
    batch_size=2,
    shuffle=False,
)

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def evaluate(model, loader):
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            predicted = logits.argmax(dim=1)

            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

    return total_correct / total_samples


for epoch in range(100):
    model.train()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        train_accuracy = evaluate(model, train_loader)
        val_accuracy = evaluate(model, val_loader)

        print(
            f"epoch={epoch}, "
            f"loss={loss.item():.4f}, "
            f"train_acc={train_accuracy:.2%}, "
            f"val_acc={val_accuracy:.2%}"
        )

test_x = torch.tensor([
    [-3.0, -2.0],
    [-2.0, -3.0],
    [2.0, 3.0],
    [3.0, 2.0],
])

test_y = torch.tensor([0, 0, 1, 1])

test_loader = DataLoader(
    TensorDataset(test_x, test_y),
    batch_size=2,
    shuffle=False,
)

test_accuracy = evaluate(model, test_loader)

print(f"test_acc={test_accuracy:.2%}")

new_x = torch.tensor([
    [-2.0, -1.0],
    [2.0, 1.0],
])

model.eval()

with torch.no_grad():
    logits = model(new_x)
    predicted = logits.argmax(dim=1)

print("类别得分：", logits)
print("预测类别：", predicted)