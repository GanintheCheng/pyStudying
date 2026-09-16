import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.network(x)


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


torch.manual_seed(42)

# 1. 读取 CSV
df = pd.read_csv("students.csv", encoding="gb18030")

# 2. 选择特征 X 与标签 y
features = ["study_hours", "attendance"]

X = df[features]
y = df["passed"]

# 3. 划分训练集 60%、验证集 20%、测试集 20%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# 4. 只在训练集上学习标准化规则
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5. 转为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 6. 创建 Dataset 与 DataLoader
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=4,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=4,
    shuffle=False,
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=4,
    shuffle=False,
)

# 7. 创建模型、损失函数与优化器
model = StudentClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# 8. 训练
for epoch in range(200):
    model.train()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        print(
            f"epoch={epoch}, "
            f"loss={loss.item():.4f}, "
            f"train_acc={train_acc:.2%}, "
            f"val_acc={val_acc:.2%}"
        )

# 9. 最终测试集评估
test_acc = evaluate(model, test_loader)
print(f"\n最终 test_acc={test_acc:.2%}")

# 10. 推理新学生数据
new_students = pd.DataFrame([
    {"study_hours": 2.0, "attendance": 55},
    {"study_hours": 6.0, "attendance": 85},
])

new_students_scaled = scaler.transform(new_students)
new_x = torch.tensor(new_students_scaled, dtype=torch.float32)

model.eval()

with torch.no_grad():
    logits = model(new_x)
    predicted = logits.argmax(dim=1)

for student, result in zip(new_students.to_dict("records"), predicted.tolist()):
    status = "通过" if result == 1 else "未通过"
    print(f"{student} → 预测：{status}")