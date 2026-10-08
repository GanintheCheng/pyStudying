"""Inspect the key calculation-saving idea of ProbSparse Attention."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.attention import ProbSparseAttention  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)

BATCH_SIZE = 2
SEQ_LEN = 96
D_MODEL = 64
N_HEADS = 4

x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
attention_layer = ProbSparseAttention(
    d_model=D_MODEL,
    n_heads=N_HEADS,
    factor=5,
    dropout=0.0,
)
output, info = attention_layer(x, return_info=True)

sample_k = int(info["sample_k"])
n_top = int(info["n_top"])
top_indices = info["top_query_indices"]

full_score_count = SEQ_LEN * SEQ_LEN
probsparse_score_count = SEQ_LEN * sample_k + n_top * SEQ_LEN

print("===== ProbSparse Attention =====")
print("输入 x：", x.shape)
print("输出：  ", output.shape)
print("\n序列长度 L：", SEQ_LEN)
print("每个 Query 用于快速评估的采样 Key 数 sample_k：", sample_k)
print("每个头进行完整 Attention 的重要 Query 数 n_top：", n_top)
print("Top-u Query 索引形状：", top_indices.shape, "= (B, heads, n_top)")
print("第 0 个样本、第 0 个头挑出的 Query 位置：", top_indices[0, 0].tolist())

print("\n===== 单个样本、单个头的 QK 分数计算量对比 =====")
print(f"Full Attention：{SEQ_LEN} × {SEQ_LEN} = {full_score_count}")
print(
    "ProbSparse："
    f"{SEQ_LEN} × {sample_k}（采样评估） + "
    f"{n_top} × {SEQ_LEN}（Top-u 完整计算） = {probsparse_score_count}"
)
print(f"当前长度下约为 Full Attention 的 {probsparse_score_count / full_score_count:.1%}")

print("\n注意：L=96 时节省有限；序列越长，O(L log L) 相对 O(L²) 的优势越明显。")
