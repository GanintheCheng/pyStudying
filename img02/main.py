import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================================================
# 1. 基础设置
# =========================================================
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 2. 图像预处理
# =========================================================

# -------------------- 重点 1：数据增强 --------------------
# 只给训练集使用随机增强。
# RandomAffine 会对同一张训练图片随机进行轻微旋转与平移。
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,          # 随机旋转范围：-10° ~ 10°
        translate=(0.1, 0.1) # 水平、垂直最多平移图片大小的 10%
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])

# 验证集和测试集不能使用随机增强。
# 它们只做固定、确定性的转换，保证评估稳定公平。
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])


# =========================================================
# 3. 数据集与 DataLoader
# =========================================================

# 注意：
# 两个数据集都对应 MNIST 的训练部分，但 transform 不同。
# 这样训练集会增强，验证集不会增强。
train_dataset_all = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

val_dataset_all = datasets.MNIST(
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

# 固定随机种子，保证每次训练集/验证集划分一致
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    len(train_dataset_all),
    generator=generator
)

train_size = 50000
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 同样的索引，但使用不同 transform 的 Dataset
train_dataset = Subset(train_dataset_all, train_indices)
val_dataset = Subset(val_dataset_all, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n===== 数据集大小 =====")
print("训练集：", len(train_dataset))
print("验证集：", len(val_dataset))
print("测试集：", len(test_dataset))

images, labels = next(iter(train_loader))
print("\n===== 一个训练 Batch 的形状 =====")
print("images.shape:", images.shape)  # (64, 1, 28, 28)
print("labels.shape:", labels.shape)  # (64,)


# =========================================================
# 4. CNN 模型
# =========================================================
class BetterMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 原始 MNIST 是 1 通道，
        # 先转换到 16 通道，才能进入后面的残差块。
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # -------- 残差主分支 --------
        # 输入、输出都是 16 通道，因此可与 identity 直接相加。
        self.conv1 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()

        # 残差块结束后再池化
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # (B, 16, 14, 14) → (B, 16, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        # 输入：(B, 1, 28, 28)
        x = self.stem(x)
        # x: (B, 16, 28, 28)

        # 保存进入残差块前的 x
        identity = x

        # Conv → BN → ReLU → Conv → BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：F(x) + x
        out = out + identity
        out = self.relu(out)

        # 残差块完成后再池化
        out = self.pool(out)
        # out: (B, 16, 14, 14)

        out = self.global_pool(out)
        # out: (B, 16, 1, 1)

        out = torch.flatten(out, start_dim=1)
        # out: (B, 16)

        out = self.dropout(out)
        logits = self.classifier(out)
        # logits: (B, 10)

        return logits


model = BetterMNISTCNN().to(device)

print("\n===== 模型结构 =====")
print(model)


# =========================================================
# 5. 训练与评估函数
# =========================================================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_predictions)

    return avg_loss, acc, all_labels, all_predictions


# =========================================================
# 6. 训练、验证、早停与最佳模型保存
# =========================================================
best_val_loss = float("inf")
best_model_state = None
wait = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()

    train_loss = total_train_loss / total_train_samples
    train_acc = train_correct / total_train_samples

    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"epoch={epoch:02d}, "
        f"train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2%}, "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
        print("  保存当前最佳模型")
    else:
        wait += 1

    if wait >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break


# =========================================================
# 7. 加载最佳模型并测试
# =========================================================
model.load_state_dict(best_model_state)

test_loss, test_acc, y_true, y_pred = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print("\n===== 测试结果 =====")
print(f"test_loss={test_loss:.4f}")
print(f"test_acc={test_acc:.2%}")

cm = confusion_matrix(y_true, y_pred)

print("\n===== 混淆矩阵 =====")
print(cm)

print("\n===== 分类报告 =====")
print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)


# =========================================================
# 8. 训练曲线
# =========================================================
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 9. 混淆矩阵可视化
# =========================================================
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MNIST Confusion Matrix")

for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.tight_layout()
plt.show()