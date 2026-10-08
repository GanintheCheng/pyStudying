"""Inspect tensor flow through standard multi-head Full Attention."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.attention import FullAttention  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)

BATCH_SIZE = 2
SEQ_LEN = 96
D_MODEL = 64
N_HEADS = 4

x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
attention_layer = FullAttention(d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0)
output, weights = attention_layer(x, return_attention=True)

print("===== Full Attention：Encoder Self-Attention =====")
print("输入 x:             ", x.shape)
print("Q / K / V 投影后:    ", (BATCH_SIZE, N_HEADS, SEQ_LEN, D_MODEL // N_HEADS))
print("注意力权重 weights:  ", weights.shape)
print("输出 output:         ", output.shape)

# Softmax is applied across all keys.  Each query row must sum to 1.
row_sums = weights.sum(dim=-1)
print("\n每个 Query 的注意力权重和是否为 1：", torch.allclose(row_sums, torch.ones_like(row_sums)))
print("第 0 个样本、第 0 个头、第 0 个 Query 的前 8 个权重：")
print(torch.round(weights[0, 0, 0, :8] * 10000) / 10000)

# Decoder needs a causal mask, unlike the encoder history window.
decoder_x = torch.randn(BATCH_SIZE, 72, D_MODEL)
_, masked_weights = attention_layer(
    decoder_x,
    causal_mask=True,
    return_attention=True,
)
print("\n===== Causal Mask 检查（Decoder 使用） =====")
print("Decoder 权重形状：", masked_weights.shape)
print("第 10 个位置对未来位置 11..71 的最大权重：",
      masked_weights[0, 0, 10, 11:].max().item())
