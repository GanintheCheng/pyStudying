"""Run this file to inspect Informer's DataEmbedding tensor flow."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.embedding import DataEmbedding  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)

BATCH_SIZE = 32
SEQ_LEN = 96
FEATURES = 7
D_MODEL = 64

# A real DataLoader will later produce these two tensors.
x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES)
x_mark = torch.stack(
    [
        torch.randint(1, 13, (BATCH_SIZE, SEQ_LEN)),
        torch.randint(1, 32, (BATCH_SIZE, SEQ_LEN)),
        torch.randint(0, 7, (BATCH_SIZE, SEQ_LEN)),
        torch.randint(0, 24, (BATCH_SIZE, SEQ_LEN)),
    ],
    dim=-1,
)

embedding = DataEmbedding(c_in=FEATURES, d_model=D_MODEL, dropout=0.1)
embedding.train()
output = embedding(x, x_mark)

print("===== Informer DataEmbedding =====")
print("原始数值 x:       ", x.shape, "= (B, L, 7)")
print("时间标记 x_mark:  ", x_mark.shape, "= (B, L, 4)")
print("ValueEmbedding:    ", embedding.value_embedding(x).shape)
print("PositionEmbedding: ", embedding.position_embedding(x).shape)
print("TemporalEmbedding: ", embedding.temporal_embedding(x_mark).shape)
print("最终输出:          ", output.shape, "= (B, L, d_model)")
print("\n每个时间点的 7 个原始变量，现已变成 64 维向量；")
print("下一层 ProbSparse self-attention 将处理这个 (B, 96, 64) 张量。")
