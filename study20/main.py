import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 1. 读取并检查 CSV：数据是什么？任务是什么？
# ============================================================

CSV_PATH = "c_course_feedback_300.csv"

df = pd.read_csv(
    CSV_PATH,
    encoding="gb18030",
)

required_columns = {"text", "label"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"CSV 缺少字段：{missing_columns}")

df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print("\n===== 数据说明 =====")
print("数据来源：", CSV_PATH)
print("任务：根据 C 语言课程反馈文本，判断其属于好评还是差评")
print("样本数量：", len(df))
print("类别数量：", df["label"].nunique())
print("标签含义：0=差评，1=好评")
print("类别分布：")
print(df["label"].value_counts().sort_index())

texts = df["text"].tolist()
labels = df["label"].tolist()

# ============================================================
# 2. 分层划分数据集
# ============================================================

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels,
)

print("\n===== 数据划分 =====")
print("训练集：", len(train_texts), "条，类别分布：", pd.Series(train_labels).value_counts().to_dict())
print("验证集：", len(val_texts), "条，类别分布：", pd.Series(val_labels).value_counts().to_dict())
print("测试集：", len(test_texts), "条，类别分布：", pd.Series(test_labels).value_counts().to_dict())

# ============================================================
# 3. 设备与 Tokenizer
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n当前设备：", device)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


# ============================================================
# 4. Dataset：每条样本返回文本编码和标签
# ============================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=32,
        )
        item["labels"] = self.labels[index]
        return item


train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=data_collator,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator,
)

# ============================================================
# 5. BERT 分类模型、优化器和学习率调度器
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5,
)


# ============================================================
# 6. 评估函数：loss、Accuracy、Precision、Recall、F1、混淆矩阵
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=1)
            batch_size = batch["labels"].size(0)

            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size
            all_labels.extend(batch["labels"].cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_samples
    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
        "labels": all_labels,
        "predictions": all_predictions,
    }
    return metrics


# ============================================================
# 7. 训练：每轮训练后验证，保存验证集 loss 最低的模型
# ============================================================

best_val_loss = float("inf")
total_epochs = 3

for epoch in range(total_epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    for batch in train_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        predictions = outputs.logits.argmax(dim=1)
        batch_size = batch["labels"].size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (predictions == batch["labels"]).sum().item()
        train_samples += batch_size

    scheduler.step()

    train_loss = train_loss_sum / train_samples
    train_acc = train_correct / train_samples
    val_metrics = evaluate(model, val_loader, device)

    print(
        f"epoch={epoch}, train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2%}, "
        f"val_loss={val_metrics['loss']:.4f}, "
        f"val_acc={val_metrics['accuracy']:.2%}, "
        f"lr={optimizer.param_groups[0]['lr']:.8f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        torch.save(model.state_dict(), "best_model.pt")
        print("保存当前最佳模型")

# ============================================================
# 8. 测试最佳模型：模型表现如何？
# ============================================================

model.load_state_dict(torch.load("best_model.pt", map_location=device))
test_metrics = evaluate(model, test_loader, device)

print("\n===== 测试结果 =====")
print(f"test_loss={test_metrics['loss']:.4f}")
print(f"accuracy={test_metrics['accuracy']:.2%}")
print(f"precision={test_metrics['precision']:.4f}")
print(f"recall={test_metrics['recall']:.4f}")
print(f"f1={test_metrics['f1']:.4f}")
print("confusion matrix:")
print(test_metrics["confusion_matrix"])
print("classification report:")
print(
    classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=["差评", "好评"],
        zero_division=0,
    )
)

# ============================================================
# 9. 错误分析：错误来自哪里？
# ============================================================

print("===== 测试集错误样本 =====")
for text, true_label, predicted_label in zip(
        test_texts,
        test_labels,
        test_metrics["predictions"],
):
    if true_label != predicted_label:
        print(
            f"文本：{text} | "
            f"真实标签：{true_label} | "
            f"预测标签：{predicted_label}"
        )


# ============================================================
# 10. 新文本预测
# ============================================================

def predict_text(model, tokenizer, text, device):
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    predicted_id = outputs.logits.argmax(dim=1).item()
    return predicted_id, outputs.logits


label_names = {0: "差评", 1: "好评"}
new_texts = [
    "老师讲得非常好，我学到了很多",
    "这节课毫无条理，听不懂",
]

print("\n===== 新文本预测 =====")
for text in new_texts:
    predicted_id, logits = predict_text(
        model,
        tokenizer,
        text,
        device,
    )
    print(f"文本：{text}")
    print(f"logits：{logits}")
    print(f"预测结果：{label_names[predicted_id]}")
