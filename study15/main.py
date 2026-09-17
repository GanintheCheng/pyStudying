from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts = [
    "这门课程讲得很清楚",
    "我喜欢深度学习",
]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=16,
    return_tensors="pt",
)

print("input_ids:")
print(encoded["input_ids"])

print("\nattention_mask:")
print(encoded["attention_mask"])

print("\n第一个文本的 token：")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))