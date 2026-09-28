import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 图像预处理
# ============================================================

transform = transforms.Compose([
    # PIL 图像 → PyTorch Tensor
    # (H, W, C) → (C, H, W)
    # 像素值 0~255 → 0.0~1.0
    transforms.ToTensor(),

    # 使用 MNIST 的常用均值、标准差做标准化
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,),
    ),
])


# ============================================================
# 2. 下载并读取 MNIST
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))


# ============================================================
# 3. 创建 DataLoader
# ============================================================

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# ============================================================
# 4. 查看一个 Batch
# ============================================================

images, labels = next(iter(train_loader))

print("\n===== 一个 Batch =====")
print("images.shape:", images.shape)
print("labels.shape:", labels.shape)

print("\n前 10 个标签：")
print(labels[:10])

import matplotlib.pyplot as plt
# ============================================================
# 5. 可视化一个 Batch 中的前 12 张图片
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(8, 6))

for index, ax in enumerate(axes.flat):
    # images[index] 的形状是 (1, 28, 28)
    image = images[index]

    # 还原标准化：
    # standardized = (original - mean) / std
    # original = standardized * std + mean
    image = image * 0.3081 + 0.1307

    # (1, 28, 28) → (28, 28)
    image = image.squeeze(0)

    ax.imshow(
        image.numpy(),
        cmap="gray",
    )

    ax.set_title(f"Label: {labels[index].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

import torch.nn as nn


# ============================================================
# 6. 定义 CNN
# ============================================================

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 输入：(B, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # 输入：(B, 16, 14, 14)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Linear(
            in_features=32,
            out_features=10,
        )

    def forward(self, x):
        x = self.features(x)
        # (B, 1, 28, 28)
        # → (B, 32, 7, 7)

        x = self.global_pool(x)
        # (B, 32, 7, 7)
        # → (B, 32, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # (B, 32, 1, 1)
        # → (B, 32)

        logits = self.classifier(x)
        # (B, 32)
        # → (B, 10)

        return logits


# ============================================================
# 7. 检查输出形状
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MNISTCNN().to(device)

print(model)

with torch.no_grad():
    logits = model(images.to(device))

print("\n===== CNN 输出 =====")
print("logits.shape:", logits.shape)
print("labels.shape:", labels.shape)

from torch.utils.data import DataLoader, random_split


# ============================================================
# 8. 从原始训练集划分训练集与验证集
# ============================================================

train_size = 50_000
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(
    train_dataset,
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 64

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

print("\n===== 数据划分 =====")
print("训练集：", len(train_subset))
print("验证集：", len(val_subset))
print("测试集：", len(test_dataset))

import copy


# ============================================================
# 9. 创建模型、损失函数与优化器
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MNISTCNN().to(device)

# 多分类任务：logits 为 (B, 10)，标签为 (B,)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5,
)


# ============================================================
# 10. 验证函数
# ============================================================

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size_now = labels.size(0)

            # 当前 batch 的平均 loss × 当前 batch 样本数
            total_loss += loss.item() * batch_size_now

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size_now

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# 11. 训练与验证
# ============================================================

max_epochs = 10

best_val_loss = float("inf")
best_model_state = None

for epoch in range(max_epochs):
    model.train()

    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        # logits: (B, 10)
        # labels: (B,)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size_now = labels.size(0)

        total_train_loss += loss.item() * batch_size_now

        predictions = logits.argmax(dim=1)

        total_train_correct += (
            predictions == labels
        ).sum().item()

        total_train_samples += batch_size_now

    train_loss = total_train_loss / total_train_samples
    train_accuracy = total_train_correct / total_train_samples

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"epoch={epoch}, "
        f"train_loss={train_loss:.4f}, "
        f"train_acc={train_accuracy:.2%}, "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_accuracy:.2%}, "
        f"lr={current_lr:.6f}"
    )

    # 保存验证集 loss 最低的参数
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())

        print("保存当前最佳模型参数")


# ============================================================
# 12. 恢复验证集最佳模型
# ============================================================

model.load_state_dict(best_model_state)

print("\n===== 训练结束 =====")
print(f"最佳验证集 loss：{best_val_loss:.4f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ============================================================
# 13. 测试集最终评估
# ============================================================

test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device,
)

print("\n===== 测试集结果 =====")
print(f"test_loss={test_loss:.4f}")
print(f"test_acc={test_acc:.2%}")


# ============================================================
# 14. 收集测试集预测结果
# ============================================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ============================================================
# 15. 混淆矩阵与分类报告
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("\n===== 混淆矩阵 =====")
print(cm)

print("\n===== 分类报告 =====")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4,
    )
)