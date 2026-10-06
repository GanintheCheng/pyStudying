import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# 1. 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)


# ============================================================
# 2. 数据预处理
# ============================================================
# 训练集：加入轻微旋转、平移，作为数据增强
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 验证集和测试集：不能做随机增强
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 两个训练数据集读取同一批 MNIST 图片，
# 但一个使用训练增强，另一个使用纯评估预处理。
train_dataset_with_augment = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

train_dataset_for_eval = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=eval_transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=eval_transform
)

# 从 60,000 张官方训练图像中划分：
# 54,000 训练，6,000 验证
all_indices = np.arange(len(train_dataset_with_augment))
np.random.shuffle(all_indices)

train_indices = all_indices[:54000]
val_indices = all_indices[54000:]

train_dataset = Subset(train_dataset_with_augment, train_indices)
val_dataset = Subset(train_dataset_for_eval, val_indices)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\n===== 数据集大小 =====")
print("训练集：", len(train_dataset))
print("验证集：", len(val_dataset))
print("测试集：", len(test_dataset))


# ============================================================
# 3. 残差块
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：F(x) + x
        out = out + identity
        out = F.relu(out)

        return out


# ============================================================
# 4. Mini-ResNet MNIST 模型
# ============================================================
class MiniResNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        # (B, 1, 28, 28) -> (B, 16, 28, 28)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # (B, 16, 28, 28) -> (B, 16, 28, 28)
        self.res_block = ResidualBlock(channels=16)

        # (B, 16, 28, 28) -> (B, 16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # (B, 16, 14, 14) -> (B, 16, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 正则化：仅训练时随机失活部分特征
        self.dropout = nn.Dropout(p=0.3)

        # (B, 16) -> (B, 10)
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block(x)
        x = self.pool(x)
        x = self.global_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits


model = MiniResNetMNIST().to(device)

print("\n===== 模型结构 =====")
print(model)


# ============================================================
# 5. 训练与评估函数
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_labels, all_predictions


# ============================================================
# 6. 训练：按验证损失保存最佳模型
# ============================================================
num_epochs = 15
best_val_loss = float("inf")
best_model_state = None

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"epoch={epoch + 1:02d}/{num_epochs}, "
        f"train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2%}, "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print("  保存当前最佳模型")

# 恢复验证集损失最低的模型
model.load_state_dict(best_model_state)


# ============================================================
# 7. 测试集评估
# ============================================================
test_loss, test_acc, test_labels, test_predictions = evaluate(
    model, test_loader, criterion, device
)

print("\n===== 测试结果 =====")
print(f"test_loss={test_loss:.4f}")
print(f"test_acc={test_acc:.2%}")

print("\n===== 混淆矩阵 =====")
cm = confusion_matrix(test_labels, test_predictions)
print(cm)

print("\n===== 分类报告 =====")
print(classification_report(
    test_labels,
    test_predictions,
    digits=4
))


# ============================================================
# 8. 绘制训练曲线
# ============================================================
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()