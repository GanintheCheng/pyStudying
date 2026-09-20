import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

#原始数据集#
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

labels = [
    1, 1, 1, 1, 1, 1,  # 好评
    0, 0, 0, 0, 0, 0,  # 差评
]
# 6 2 2 分组
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
# 定义三组的张量
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
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
# dataLodaer的定义
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

            logits = outputs.logits
            loss = outputs.loss

            predicted = logits.argmax(dim=1)

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)

for epoch in range(15):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        predicted = logits.argmax(dim=1)

        train_loss_sum += loss.item() * batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()
        train_samples += batch_labels.size(0)

    train_loss = train_loss_sum / train_samples
    train_acc = train_correct / train_samples

    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"epoch={epoch}, "
        f"train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2%}, "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2%}"
    )

test_loss, test_acc = evaluate(model, test_loader)

print(
    f"test_loss={test_loss:.4f}, "
    f"test_acc={test_acc:.2%}"
)