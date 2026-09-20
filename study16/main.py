import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1. 原始文本与标签
# 1：好评，0：差评
texts = [
    "这门课程讲得很清楚，收获很多",
    "老师讲解细致，我很喜欢",
    "内容太难理解了，体验很差",
    "课程安排混乱，不推荐",
    "课程时间太早，不推荐",
    "老师讲的不好，不喜欢",
    "有女生长得好看，爱上这个课",
]
labels = [1, 1, 0, 0, 0, 0, 1]


# 2. 加载 tokenizer 与预训练 BERT 分类模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2,
)


# 3. 全部文本编码成张量
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)

label_tensor = torch.tensor(labels, dtype=torch.long)


# 4. Dataset 与 DataLoader
dataset = TensorDataset(
    encoded["input_ids"],
    encoded["attention_mask"],
    label_tensor,
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)


# 5. 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)


# 6. 训练
model.train()

for epoch in range(50):
    total_loss = 0.0

    for batch_input_ids, batch_attention_mask, batch_labels in loader:
        # 清空上一 batch 的梯度
        optimizer.zero_grad()

        # 前向计算：内部得到 logits 和 loss
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )

        loss = outputs.loss

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch={epoch}, loss={total_loss:.4f}")


# 7. 用新文本推理
model.eval()

new_texts = [
    "老师讲得非常好，我学到了很多",
    "这节课毫无条理，听不懂",
]

new_encoded = tokenizer(
    new_texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(
        input_ids=new_encoded["input_ids"],
        attention_mask=new_encoded["attention_mask"],
    )

    logits = outputs.logits
    predicted = logits.argmax(dim=1)

print("logits：")
print(logits)

print("预测类别：")
print(predicted)

for text, label in zip(new_texts, predicted.tolist()):
    result = "好评" if label == 1 else "差评"
    print(f"{text} → {result}")
