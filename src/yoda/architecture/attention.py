"""Multihead Pooled Attention (MPA) implementation for semantic representation aggregation.

Aggregates variable length input token sequences into fixed-dimensional latent
representations via multihead cross-attention with learnable query tokens.
"""

import logging
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import nn

from yoda.architecture.activations import SoftExp
from yoda.architecture.belnap_transformer import BelnapAttention, BelnapState

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "BelnapMultiheadPooledAttention",
    "MultiheadPooledAttention",
]


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

        # Replace monolithic nn.MultiheadAttention with explicit projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

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
        h = self.num_heads
        d_h = d // h

        # Project Q, K, V
        q_proj = self.q_proj(q).view(b, t, h, d_h).transpose(1, 2)
        k_proj = self.k_proj(x).view(b, s, h, d_h).transpose(1, 2)
        v_proj = self.v_proj(x).view(b, s, h, d_h).transpose(1, 2)

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
        out = self.out_proj(out)

        out = out.reshape(batch_size, self.num_tokens * self.embed_dim)
        return self.proj(out)


class BelnapMultiheadPooledAttention(nn.Module):
    """Belnap Bilattice Multihead Pooled Attention.

    Pools variable-length sequences into fixed-dimensional latent representations using
    epistemic query probes and Belnap four-valued cross-attention. Preserves both positive
    and negative evidence, suppressing uninformative inputs to Neither (k -> 0) and
    preserving contradiction (Both: e+ >> 0, e- >> 0) without softmax attention sinks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 4,
        num_heads: int | None = None,
        init_bias: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes BelnapMultiheadPooledAttention.

        Args:
            embed_dim: Dimensionality of token embeddings.
            num_queries: Number of epistemic query probe tokens.
            num_heads: Number of attention heads. Defaults to 4, 2, or 1 based on embed_dim.
            init_bias: Initial bias for BelnapAttention output projections. Defaults to 0.0.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        if num_heads is None:
            num_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Epistemic learnable query probes for positive and negative evidence
        self.raw_q_pos = nn.Parameter(
            torch.empty(1, num_queries, embed_dim, device=device, dtype=dtype)
        )
        self.raw_q_neg = nn.Parameter(
            torch.empty(1, num_queries, embed_dim, device=device, dtype=dtype)
        )
        nn.init.uniform_(self.raw_q_pos, -0.5, 0.5)
        nn.init.uniform_(self.raw_q_neg, -0.5, 0.5)

        # Input adapter projection for raw continuous tensors
        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim, device=device, dtype=dtype)
        self.softexp = SoftExp(in_features=2 * embed_dim, device=device, dtype=dtype)

        # Belnap bipolar cross-attention
        self.belnap_attn = BelnapAttention(
            d_model=embed_dim,
            n_heads=num_heads,
            bias=True,
            init_bias=init_bias,
        )

        # Output projection from concatenated dual evidence features to latent vector
        self.proj = nn.Linear(
            num_queries * 2 * embed_dim,
            embed_dim,
            device=device,
            dtype=dtype,
        )

        logger.debug(
            "architecture.belnap_mpa.init",
            extra={
                "embed_dim": embed_dim,
                "num_queries": num_queries,
                "num_heads": num_heads,
                "init_bias": init_bias,
            },
        )

    def forward(
        self,
        x: BelnapState | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BelnapState]:
        """Executes Belnap epistemic pooled attention over input sequences.

        Args:
            x: Input sequence as either a `BelnapState` or a continuous `torch.Tensor`
                of shape `(batch_size, seq_len, embed_dim)`.
            mask: Optional attention mask of shape `(batch_size, 1, num_queries, seq_len)`.

        Returns:
            A tuple of `(latent, pooled_state)`:
            - `latent`: Projected fixed-dimensional representation `(batch_size, embed_dim)`.
            - `pooled_state`: Aggregated `BelnapState` of shape
              `(batch_size, num_queries, embed_dim)`.
        """
        if isinstance(x, BelnapState):
            kv_state = x
            batch_size = x.e_pos.size(0)
        else:
            batch_size = x.size(0)
            proj = self.softexp(self.input_proj(x))
            pos_raw, neg_raw = torch.sigmoid(proj).chunk(2, dim=-1)
            kv_state = BelnapState(e_pos=pos_raw, e_neg=neg_raw)

        # Epistemic query state bounded strictly in [0, 1]
        q_pos = torch.sigmoid(self.raw_q_pos).expand(batch_size, -1, -1).contiguous()
        q_neg = torch.sigmoid(self.raw_q_neg).expand(batch_size, -1, -1).contiguous()
        q_state = BelnapState(e_pos=q_pos, e_neg=q_neg)

        # Bipolar cross-attention pooling
        pooled_state = self.belnap_attn(q=q_state, kv=kv_state, mask=mask)

        # Flatten dual evidence and project into latent space
        features = torch.cat([pooled_state.e_pos, pooled_state.e_neg], dim=-1).flatten(1)
        latent = self.proj(features)

        logger.debug(
            "architecture.belnap_mpa.forward",
            extra={
                "batch_size": batch_size,
                "input_type": "BelnapState" if isinstance(x, BelnapState) else "Tensor",
                "num_queries": self.num_queries,
            },
        )

        return latent, pooled_state
