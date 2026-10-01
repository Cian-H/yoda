"""Multihead Pooled Attention (MPA) implementation for semantic representation aggregation.

Aggregates variable length input token sequences into fixed-dimensional latent
representations via multihead cross-attention with learnable query tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn

logger = logging.getLogger(__name__)


class MultiheadPooledAttention(nn.Module):
    """Aggregates variable length inputs to fixed tokens via multihead cross-attention."""

    ACTIVATION_REGISTRY: ClassVar[dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
        "softmax": nn.Softmax(dim=-1),
        "time-series": nn.Softmax(dim=-1),
        "linear": nn.Identity(),
        "relational": nn.Identity(),
    }

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 4,
        attn_activation: str | Callable[[torch.Tensor], torch.Tensor] = "softmax",
        num_heads: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes MultiheadPooledAttention.

        Args:
            embed_dim: The number of features in the input sequence embedding.
            num_queries: The number of learnable latent vectors used to compress the sequence.
            attn_activation: The activation function applied to attention logits.
            num_heads: Number of attention heads for the cross-attention mechanism.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_queries
        self.activation = attn_activation

        if num_heads is None:
            num_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)
        self.num_heads = num_heads

        self.query_token = nn.Parameter(
            torch.randn(1, num_queries, embed_dim, device=device, dtype=dtype)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, device=device, dtype=dtype
        )
        self.attn_matrix_hookpoint = nn.Identity()
        self.proj = nn.Linear(num_queries * embed_dim, embed_dim, device=device, dtype=dtype)

        logger.debug(
            "architecture.attention.init",
            extra={
                "embed_dim": embed_dim,
                "num_queries": num_queries,
                "num_heads": num_heads,
                "activation": attn_activation if isinstance(attn_activation, str) else "callable",
            },
        )

    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Resolves string keys to activations, passing through callables directly."""
        if isinstance(self.activation, str):
            return self.ACTIVATION_REGISTRY.get(self.activation.lower(), nn.Identity())
        return self.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuses input sequence into a single context representation.

        Args:
            x: Input tensor of shape `(batch_size, seq_len, embed_dim)`.

        Returns:
            Aggregated tensor of shape `(batch_size, embed_dim)`.
        """
        batch_size = x.size(0)
        q = self.query_token.expand(batch_size, -1, -1).contiguous()
        x = x.contiguous()

        b, t, d = q.shape
        _, s, _ = x.shape
        h = self.attn.num_heads
        d_h = d // h

        # Extract projection weights from the fused in_proj_weight
        in_w = self.attn.in_proj_weight
        in_b = self.attn.in_proj_bias

        # Project Q, K, V
        q_proj = nn.functional.linear(q, in_w[:d], in_b[:d] if in_b is not None else None)
        k_proj = nn.functional.linear(
            x, in_w[d : 2 * d], in_b[d : 2 * d] if in_b is not None else None
        )
        v_proj = nn.functional.linear(x, in_w[2 * d :], in_b[2 * d :] if in_b is not None else None)

        # Reshape for multi-head computation: [B, H, T/S, D_h]
        q_proj = q_proj.view(b, t, h, d_h).transpose(1, 2)
        k_proj = k_proj.view(b, s, h, d_h).transpose(1, 2)
        v_proj = v_proj.view(b, s, h, d_h).transpose(1, 2)

        # Scaled dot-product attention logits: [B, H, T, S]
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (d_h**0.5)

        # Apply activation
        activation_fn = self._get_activation_fn()
        attn_weights = activation_fn(attn_weights)

        # Route the attention weight matrix through hookpoint for explainability/auditing
        self.attn_matrix_hookpoint(attn_weights)

        # Context fusion
        out = torch.matmul(attn_weights, v_proj)  # [B, H, T, D_h]
        out = out.transpose(1, 2).reshape(b, t, d)
        out = self.attn.out_proj(out)

        out = out.reshape(batch_size, self.num_tokens * self.embed_dim)
        return self.proj(out)
