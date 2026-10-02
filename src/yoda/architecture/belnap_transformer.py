"""Belnap Bilattice Native Transformer architecture components.

Implements four-valued paraconsistent logic (Belnap bilattice B4 = {T, F, N, B})
directly within neural tensor transformations. Representations consist of dual evidence
pairs (e_pos, e_neg) in [0, 1]^d x [0, 1]^d, enabling principled handling of missing
information (Neither) and contradictory evidence (Both) without softmax attention sinks
or false compromise vector collapses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class BelnapState:
    """Continuous 4-valued Belnap bilattice state container.

    Tracks dual continuous evidence components:
    - `e_pos`: Positive support / evidence for feature assertion in [0, 1].
    - `e_neg`: Negative support / refutation against feature assertion in [0, 1].
    """

    e_pos: torch.Tensor
    e_neg: torch.Tensor

    @property
    def truth(self) -> torch.Tensor:
        """Truth axis projection: t = (e_pos - e_neg + 1) / 2 in [0, 1]."""
        return 0.5 * (self.e_pos - self.e_neg + 1.0)

    @property
    def knowledge(self) -> torch.Tensor:
        """Knowledge axis projection: k = (e_pos + e_neg) / 2 in [0, 1]."""
        return 0.5 * (self.e_pos + self.e_neg)

    def negate(self) -> BelnapState:
        """Logical negation (~): swaps evidence polarities, inverting truth."""
        return BelnapState(e_pos=self.e_neg, e_neg=self.e_pos)

    def conflate(self, weight: float = 0.1) -> BelnapState:
        """Applies conflation operator to prune cross-polarity contradiction."""
        res_pos = torch.relu(self.e_pos - weight * self.e_neg)
        res_neg = torch.relu(self.e_neg - weight * self.e_pos)
        return BelnapState(e_pos=res_pos, e_neg=res_neg)

    @classmethod
    def from_tk(
        cls,
        truth: torch.Tensor,
        knowledge: torch.Tensor,
    ) -> BelnapState:
        """Reconstructs evidence state from truth and knowledge coordinates.

        Inverse mapping:
            e_pos = clamp(k + (2t - 1) / 2, 0, 1)
            e_neg = clamp(k - (2t - 1) / 2, 0, 1)
        """
        bias = 0.5 * (2.0 * truth - 1.0)
        e_pos = torch.clamp(knowledge + bias, min=0.0, max=1.0)
        e_neg = torch.clamp(knowledge - bias, min=0.0, max=1.0)
        return cls(e_pos=e_pos, e_neg=e_neg)


class BelnapAttention(nn.Module):
    """Bipolar Support/Refutation Multi-Head Attention with source knowledge gating."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        bias: bool = True,
        init_bias: float = 0.0,
    ) -> None:
        """Initializes BelnapAttention.

        Args:
            d_model: Dimensionality of input evidence representations.
            n_heads: Number of attention heads.
            bias: Whether linear projection layers include bias.
            init_bias: Initial bias for output evidence projections. Defaults to 0.0.
        """
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Dual evidence projections
        self.w_q_pos = nn.Linear(d_model, d_model, bias=bias)
        self.w_q_neg = nn.Linear(d_model, d_model, bias=bias)
        self.w_k_pos = nn.Linear(d_model, d_model, bias=bias)
        self.w_k_neg = nn.Linear(d_model, d_model, bias=bias)
        self.w_v_pos = nn.Linear(d_model, d_model, bias=bias)
        self.w_v_neg = nn.Linear(d_model, d_model, bias=bias)

        # Output projections
        self.w_out_pos = nn.Linear(d_model, d_model, bias=bias)
        self.w_out_neg = nn.Linear(d_model, d_model, bias=bias)

        # Initialize output evidence biases negative so uninformative input suppresses to Neither
        if self.w_out_pos.bias is not None:
            nn.init.constant_(self.w_out_pos.bias, init_bias)
        if self.w_out_neg.bias is not None:
            nn.init.constant_(self.w_out_neg.bias, init_bias)

        logger.debug(
            "architecture.belnap_attention.init",
            extra={"d_model": d_model, "n_heads": n_heads, "d_k": self.d_k},
        )

    def forward(
        self,
        q: BelnapState,
        kv: BelnapState | None = None,
        mask: torch.Tensor | None = None,
    ) -> BelnapState:
        """Executes bipolar attention over dual evidence states.

        Args:
            q: Query BelnapState with shapes `(batch_size, seq_len_q, d_model)`.
            kv: Key-Value BelnapState with shapes `(batch_size, seq_len_kv, d_model)`.
                Defaults to query state for self-attention.
            mask: Optional attention mask of shape `(batch_size, 1, seq_len_q, seq_len_kv)`.

        Returns:
            Aggregated output BelnapState bounded strictly in `[0, 1]`.
        """
        if kv is None:
            kv = q

        batch_size, l_q, _ = q.e_pos.shape
        _, l_kv, _ = kv.e_pos.shape

        # Step 1: Dual Projections
        q_pos = self.w_q_pos(q.e_pos).view(batch_size, l_q, self.n_heads, self.d_k).transpose(1, 2)
        q_neg = self.w_q_neg(q.e_neg).view(batch_size, l_q, self.n_heads, self.d_k).transpose(1, 2)

        k_pos = (
            self.w_k_pos(kv.e_pos).view(batch_size, l_kv, self.n_heads, self.d_k).transpose(1, 2)
        )
        k_neg = (
            self.w_k_neg(kv.e_neg).view(batch_size, l_kv, self.n_heads, self.d_k).transpose(1, 2)
        )

        v_pos = (
            self.w_v_pos(kv.e_pos).view(batch_size, l_kv, self.n_heads, self.d_k).transpose(1, 2)
        )
        v_neg = (
            self.w_v_neg(kv.e_neg).view(batch_size, l_kv, self.n_heads, self.d_k).transpose(1, 2)
        )

        # Step 2: Affinity Decomposition
        scale = math.sqrt(self.d_k)
        # Support Affinity: Coherent alignment (Q+ K+^T + Q- K-^T) / sqrt(d_k)
        s_pos = (
            torch.matmul(q_pos, k_pos.transpose(-2, -1))
            + torch.matmul(q_neg, k_neg.transpose(-2, -1))
        ) / scale

        # Refutation Affinity: Opposing polarity collision (Q+ K-^T + Q- K+^T) / sqrt(d_k)
        s_neg = (
            torch.matmul(q_pos, k_neg.transpose(-2, -1))
            + torch.matmul(q_neg, k_pos.transpose(-2, -1))
        ) / scale

        if mask is not None:
            if mask.dtype == torch.bool:
                s_pos = s_pos.masked_fill(~mask, -1e9)
                s_neg = s_neg.masked_fill(~mask, -1e9)
            else:
                s_pos = s_pos + mask
                s_neg = s_neg + mask

        # Step 3: Knowledge Gating by source token intrinsic knowledge
        # Intrinsic knowledge mass per source token: k_src in [0, 1]
        k_src = kv.knowledge.mean(dim=-1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_kv)
        k_gate = 0.5 + 0.5 * k_src
        a_pos = torch.sigmoid(s_pos) * k_gate
        a_neg = torch.sigmoid(s_neg) * k_gate

        # Step 4: Evidence Aggregation
        agg_pos = torch.matmul(a_pos, v_pos)  # (B, H, L_q, d_k)
        agg_neg = torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)

        agg_pos = agg_pos.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)
        agg_neg = agg_neg.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)

        # Output bounds in [0, 1]
        out_pos = torch.sigmoid(self.w_out_pos(agg_pos))
        out_neg = torch.sigmoid(self.w_out_neg(agg_neg))

        return BelnapState(e_pos=out_pos, e_neg=out_neg)


class BelnapFFN(nn.Module):
    """Bilattice Feed-Forward Network with conflation operator."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        conflation_weight: float = 0.1,
    ) -> None:
        """Initializes BelnapFFN.

        Args:
            d_model: Dimensionality of evidence representation.
            d_hidden: Hidden layer expansion dimensionality. Defaults to `4 * d_model`.
            conflation_weight: Weight lambda for contradictory evidence pruning.
        """
        super().__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.conflation_weight = conflation_weight
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_model, d_hidden)
        self.w_o1 = nn.Linear(d_hidden, d_model)
        self.w_o2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: BelnapState) -> BelnapState:
        """Transforms evidence through conflation and non-linear feature projection."""
        # 1. Conflation operator resolves conflicting evidence
        x_res = x.conflate(self.conflation_weight)

        # 2. Hidden expansion with smooth Mish activation
        h_mid = nn.functional.mish(self.w_1(x_res.e_pos) + self.w_2(x_res.e_neg))

        # 3. Output projections bounded in [0, 1]
        e_ffn_pos = torch.sigmoid(self.w_o1(h_mid))
        e_ffn_neg = torch.sigmoid(self.w_o2(h_mid))

        return BelnapState(e_pos=e_ffn_pos, e_neg=e_ffn_neg)


class BelnapTransformerBlock(nn.Module):
    """Unified Belnap Transformer Block combining self/cross attention and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_hidden: int | None = None,
        conflation_weight: float = 0.1,
        residual_weight: float = 0.5,
    ) -> None:
        """Initializes BelnapTransformerBlock.

        Args:
            d_model: Evidence representation dimension.
            n_heads: Number of attention heads.
            d_hidden: Hidden dimensionality of FFN.
            conflation_weight: Weight lambda for evidence conflation.
            residual_weight: Interpolation weight alpha for convex combination residual join.
        """
        super().__init__()
        self.residual_weight = residual_weight
        self.self_attn = BelnapAttention(d_model=d_model, n_heads=n_heads)
        self.cross_attn = BelnapAttention(d_model=d_model, n_heads=n_heads)
        self.ffn = BelnapFFN(
            d_model=d_model,
            d_hidden=d_hidden,
            conflation_weight=conflation_weight,
        )

    @staticmethod
    def _residual_join(
        base: BelnapState,
        delta: BelnapState,
        alpha: float = 0.5,
    ) -> BelnapState:
        """Applies convex combination residual join preserving [0, 1] evidence invariants.

        Formula: res = (1 - alpha) * base + alpha * delta
        Guarantees strict boundedness in [0, 1] without saturation clipping or dead zones.
        """
        res_pos = (1.0 - alpha) * base.e_pos + alpha * delta.e_pos
        res_neg = (1.0 - alpha) * base.e_neg + alpha * delta.e_neg
        return BelnapState(e_pos=res_pos, e_neg=res_neg)

    def forward(
        self,
        x: BelnapState,
        context: BelnapState | None = None,
    ) -> BelnapState:
        """Processes input state through self-attention, cross-attention, and FFN."""
        # Self-attention + residual join
        x_attn = self.self_attn(x, x)
        x = self._residual_join(x, x_attn, self.residual_weight)

        # Cross-attention (if context is supplied) + residual join
        if context is not None:
            x_cross = self.cross_attn(x, context)
            x = self._residual_join(x, x_cross, self.residual_weight)

        # Belnap FFN + residual join
        x_ffn = self.ffn(x)
        return self._residual_join(x, x_ffn, self.residual_weight)


class BelnapDecisionTransformer(nn.Module):
    """End-to-end multi-branch Belnap Decision Transformer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        num_choices: int = 5,
        d_hidden: int | None = None,
        conflation_weight: float = 0.1,
        residual_weight: float = 0.5,
    ) -> None:
        """Initializes BelnapDecisionTransformer.

        Args:
            d_model: Evidence representation dimension.
            n_heads: Number of attention heads.
            n_layers: Number of stacked BelnapTransformerBlock layers.
            num_choices: Number of candidate decision choices.
            d_hidden: FFN hidden dimension.
            conflation_weight: Weight for bilattice evidence conflation.
            residual_weight: Interpolation weight alpha for convex combination residual joins.
        """
        super().__init__()
        self.d_model = d_model
        self.num_choices = num_choices

        # Input projections into dual evidence Belnap states
        self.q_proj_pos = nn.Linear(d_model, d_model)
        self.q_proj_neg = nn.Linear(d_model, d_model)

        self.state_proj_pos = nn.Linear(d_model, d_model)
        self.state_proj_neg = nn.Linear(d_model, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                BelnapTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_hidden=d_hidden,
                    conflation_weight=conflation_weight,
                    residual_weight=residual_weight,
                )
                for _ in range(n_layers)
            ]
        )

        # Output choice heads
        self.choice_head_pos = nn.Linear(d_model, num_choices)
        self.choice_head_neg = nn.Linear(d_model, num_choices)

    def forward(
        self,
        q_emb: torch.Tensor,
        state_emb: torch.Tensor,
    ) -> dict[str, Any]:
        """Executes forward decision inference across query and state tokens.

        Args:
            q_emb: Query embedding tensor of shape `(batch_size, d_model)` or
                `(batch_size, 1, d_model)`.
            state_emb: Context state sequence embedding of shape
                `(batch_size, seq_len, d_model)`.

        Returns:
            Dictionary containing:
            - `logits`: Calibrated decision logits of shape `(batch_size, num_choices)`.
            - `truth`: Choice truth coordinates in [0, 1] of shape `(batch_size, num_choices)`.
            - `knowledge`: Choice knowledge coordinates in [0, 1] of shape
                `(batch_size, num_choices)`.
            - `choice`: Selected choice indices of shape `(batch_size,)`.
        """
        if q_emb.dim() == 2:
            q_emb = q_emb.unsqueeze(1)

        if state_emb.dim() == 2:
            state_emb = state_emb.unsqueeze(1)

        # Project into Belnap evidence states
        q_pos = torch.sigmoid(self.q_proj_pos(q_emb))
        q_neg = torch.sigmoid(self.q_proj_neg(q_emb))
        q_state = BelnapState(e_pos=q_pos, e_neg=q_neg)

        s_pos = torch.sigmoid(self.state_proj_pos(state_emb))
        s_neg = torch.sigmoid(self.state_proj_neg(state_emb))
        s_state = BelnapState(e_pos=s_pos, e_neg=s_neg)

        # Propagate through stacked Belnap Transformer blocks
        curr_state = q_state
        for layer in self.layers:
            curr_state = layer(curr_state, context=s_state)

        # Pool sequence tokens for decision readout
        pooled_pos = curr_state.e_pos.mean(dim=1)
        pooled_neg = curr_state.e_neg.mean(dim=1)

        # Compute choice-specific positive and negative evidence
        choice_pos = torch.sigmoid(self.choice_head_pos(pooled_pos))
        choice_neg = torch.sigmoid(self.choice_head_neg(pooled_neg))

        choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
        truth = choice_state.truth
        knowledge = choice_state.knowledge

        # Calibrated logits scaled by truth polarity and epistemic knowledge
        logits = (choice_pos - choice_neg) * knowledge
        choice = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "truth": truth,
            "knowledge": knowledge,
            "choice": choice,
        }
