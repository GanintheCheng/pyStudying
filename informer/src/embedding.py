"""Informer input embedding modules.

The encoder does not directly consume raw sensor values.  For each time point
we add a value representation, a position representation and calendar features.
"""

import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positions with shape (1, max_len, d_model)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is used only to obtain its sequence length.
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Map input features (for ETTh1: 7) to d_model with a 1D convolution."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, c_in) -> (B, c_in, L) -> Conv1d -> (B, L, d_model)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class TemporalEmbedding(nn.Module):
    """Learned embeddings for [month, day, day_of_week, hour]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.month_embed = nn.Embedding(13, d_model)  # months use 1..12
        self.day_embed = nn.Embedding(32, d_model)    # days use 1..31
        self.weekday_embed = nn.Embedding(7, d_model) # Monday=0 .. Sunday=6
        self.hour_embed = nn.Embedding(24, d_model)   # 0..23

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        # x_mark: (B, L, 4), columns [month, day, day_of_week, hour]
        x_mark = x_mark.long()
        month, day, weekday, hour = x_mark.unbind(dim=-1)
        return (
            self.month_embed(month)
            + self.day_embed(day)
            + self.weekday_embed(weekday)
            + self.hour_embed(hour)
        )


class DataEmbedding(nn.Module):
    """The complete Informer embedding: value + position + calendar + dropout."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        # All three terms have shape (B, L, d_model), so addition is elementwise.
        embedded = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(embedded)
