"""A compact, trainable Informer-style forecasting model."""

import torch
from torch import nn

from src.attention import FullAttention, ProbSparseAttention
from src.embedding import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DistillLayer(nn.Module):
    """Informer distilling: Conv1d -> ELU -> MaxPool halves sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="circular")
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pool(self.activation(self.batch_norm(self.conv(x))))
        return x.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer], distill: bool, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.distill_layers = nn.ModuleList([DistillLayer(d_model) for _ in range(len(layers) - 1)] if distill else [])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.distill_layers):
                x = self.distill_layers[index](x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder attention, then FFN."""

    def __init__(self, self_attention: FullAttention, cross_attention: FullAttention, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, causal_mask=True)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory)))
        return self.norm3(x + self.dropout(self.ffn(x)))


class InformerForecaster(nn.Module):
    """Minimal Informer for direct multivariate prediction of the next pred_len steps."""

    def __init__(self, enc_in: int, dec_in: int, c_out: int, pred_len: int, d_model: int = 64, n_heads: int = 4, e_layers: int = 2, d_layers: int = 1, d_ff: int = 128, factor: int = 5, dropout: float = 0.1, distill: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        encoder_layers = [EncoderLayer(ProbSparseAttention(d_model, n_heads, factor=factor, dropout=dropout), d_model, d_ff, dropout) for _ in range(e_layers)]
        self.encoder = Encoder(encoder_layers, distill=distill, d_model=d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(FullAttention(d_model, n_heads, dropout), FullAttention(d_model, n_heads, dropout), d_model, d_ff, dropout) for _ in range(d_layers)])
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, enc_x: torch.Tensor, enc_mark: torch.Tensor, dec_x: torch.Tensor, dec_mark: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.enc_embedding(enc_x, enc_mark))
        x = self.dec_embedding(dec_x, dec_mark)
        for layer in self.decoder_layers:
            x = layer(x, memory)
        return self.projection(x[:, -self.pred_len :, :])
