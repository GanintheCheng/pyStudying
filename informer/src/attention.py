"""Attention layers used by the from-scratch Informer implementation."""

import math

import torch
from torch import nn


class FullAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    This is our correctness baseline.  It computes every Query-Key pair, so
    its attention score tensor has shape (B, n_heads, L, L).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, n_heads, L, d_head)."""
        batch_size, length, _ = x.shape
        x = x.reshape(batch_size, length, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, L, d_head) -> (B, L, d_model)."""
        batch_size, _, length, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, length, self.d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        causal_mask: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.

        For encoder self-attention, call with only ``queries``.  Then Q=K=V.
        ``causal_mask=True`` is mainly for decoder self-attention: position t
        cannot attend to positions after t.
        """
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._split_heads(self.q_projection(queries))
        k = self._split_heads(self.k_projection(keys))
        v = self._split_heads(self.v_projection(values))

        # (B, H, L_q, d_head) @ (B, H, d_head, L_k) -> (B, H, L_q, L_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal_mask:
            query_length = scores.size(-2)
            key_length = scores.size(-1)
            mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=scores.device,
            ).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = attention @ v

        output = self.out_projection(self._merge_heads(context))
        if return_attention:
            return output, attention
        return output


class ProbSparseAttention(nn.Module):
    """A readable implementation of Informer's ProbSparse self-attention.

    For every query we sample a few keys and estimate its sparsity score
    ``max(sampled QK) - mean(sampled QK)``.  Only the top-u queries receive
    full attention over all keys.  The remaining queries use an inexpensive
    initial context (the mean of values, or a cumulative sum in causal mode).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if factor < 1:
            raise ValueError("factor must be at least 1.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        x = x.reshape(batch_size, length, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(batch_size, length, self.d_model)

    def _sample_sizes(self, query_length: int, key_length: int) -> tuple[int, int]:
        """u = c * ceil(log L), capped so short sequences remain valid."""
        sample_k = min(key_length, self.factor * math.ceil(math.log(key_length)))
        n_top = min(query_length, self.factor * math.ceil(math.log(query_length)))
        return sample_k, n_top

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        causal_mask: bool = False,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._split_heads(self.q_projection(queries))
        k = self._split_heads(self.k_projection(keys))
        v = self._split_heads(self.v_projection(values))
        batch_size, _, query_length, _ = q.shape
        key_length = k.size(2)
        sample_k, n_top = self._sample_sizes(query_length, key_length)

        # Step 1: estimate each Query's importance from sampled keys only.
        # sampled_keys: (B, H, L_q, sample_k, d_head)
        sampled_key_indices = torch.randint(
            key_length,
            (query_length, sample_k),
            device=q.device,
        )
        sampled_keys = k[:, :, sampled_key_indices, :]
        sampled_scores = (q.unsqueeze(-2) * sampled_keys).sum(dim=-1)
        sparsity_score = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)

        # Step 2: select the top-u queries separately for every sample and head.
        top_query_indices = torch.topk(sparsity_score, k=n_top, dim=-1).indices
        gather_index = top_query_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
        top_queries = torch.gather(q, dim=2, index=gather_index)

        # Step 3: only selected queries obtain full QK scores against all keys.
        top_scores = (top_queries @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal_mask:
            if query_length != key_length:
                raise ValueError("Causal self-attention requires matching Q/K lengths here.")
            key_positions = torch.arange(key_length, device=q.device)
            future_mask = key_positions.view(1, 1, 1, key_length) > top_query_indices.unsqueeze(-1)
            top_scores = top_scores.masked_fill(future_mask, float("-inf"))

        top_attention = self.dropout(torch.softmax(top_scores, dim=-1))
        top_context = top_attention @ v

        # Step 4: initialize ordinary-query contexts cheaply, then replace Top-u.
        if causal_mask:
            context = v.cumsum(dim=2)
        else:
            context = v.mean(dim=2, keepdim=True).expand(-1, -1, query_length, -1).clone()
        context.scatter_(dim=2, index=gather_index, src=top_context)

        output = self.out_projection(self._merge_heads(context))
        if return_info:
            info: dict[str, torch.Tensor | int] = {
                "sample_k": sample_k,
                "n_top": n_top,
                "top_query_indices": top_query_indices,
            }
            return output, info
        return output
