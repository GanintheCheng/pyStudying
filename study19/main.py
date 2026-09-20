import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# =========================
# 1. 准备数据
# =========================

texts = [
    "这门课程讲得很清楚",
    "老师讲解得非常细致",
    "内容实用，收获很多",
    "这节课很有帮助",
    "讲得很好，容易理解",
    "课程安排合理，体验不错",

    "内容太难了，听不懂",
    "老师讲得很混乱",
    "这节课没有什么帮助",
    "课程安排非常糟糕",
    "解释不清楚，不推荐",
    "学习体验很差",
]

# 1：好评，0：差评
labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
]


from sklearn.model_selection import train_test_split

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=4,
    random_state=42,
    stratify=labels,
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=2,
    random_state=42,
    stratify=temp_labels,
)

# =========================
# 2. Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese"
)

def encode_texts(texts, labels):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    label_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return encoded, label_tensor


train_encoded, train_label_tensor = encode_texts(
    train_texts,
    train_labels,
)

val_encoded, val_label_tensor = encode_texts(
    val_texts,
    val_labels,
)

test_encoded, test_label_tensor = encode_texts(
    test_texts,
    test_labels,
)


# =========================
# 3. Dataset / DataLoader
# =========================

train_dataset = TensorDataset(
    train_encoded["input_ids"],
    train_encoded["attention_mask"],
    train_label_tensor,
)

val_dataset = TensorDataset(
    val_encoded["input_ids"],
    val_encoded["attention_mask"],
    val_label_tensor,
)

test_dataset = TensorDataset(
    test_encoded["input_ids"],
    test_encoded["attention_mask"],
    test_label_tensor,
)

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


# =========================
# 4. 模型、优化器、学习率调度器
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5,
)


# =========================
# 5. 验证函数
# =========================

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in loader:
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
            )

            predictions = outputs.logits.argmax(dim=1)

            batch_size = batch_labels.size(0)

            total_loss += outputs.loss.item() * batch_size
            total_correct += (
                predictions == batch_labels
            ).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# =========================
# 6. 训练
# =========================

best_val_loss = float("inf")

for epoch in range(8):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = batch_labels.size(0)
        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

    scheduler.step()

    train_loss = total_train_loss / total_train_samples
    val_loss, val_acc = evaluate(model, val_loader)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"epoch={epoch}, "
        f"train_loss={train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2%}, "
        f"lr={current_lr:.8f}"
    )

    # 保存验证集上最好的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pt",
        )


# =========================
# 7. 加载最佳模型并测试
# =========================

model.load_state_dict(
    torch.load("best_model.pt")
)

test_loss, test_acc = evaluate(
    model,
    test_loader,
)

print()
print(f"test_loss={test_loss:.4f}")
print(f"test_acc={test_acc:.2%}")