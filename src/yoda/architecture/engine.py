"""Top-level YodaDecisionEngine and BelnapDecisionHead architecture modules.

Wires together Phase 1 (Encoders), Phase 2 (Belnap MPA Pooling), Phase 3 (Reasoning Core),
and Phase 4 (Decision Head) using the Cascaded Cross-Attention (Sequential Interrogation) flow.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from yoda.architecture.attention import BelnapMultiheadPooledAttention
from yoda.architecture.belnap_transformer import BelnapState, BelnapTransformerBlock
from yoda.architecture.encoders import (
    ConstraintEncoder,
    SymbolicStateEncoder,
    TextEncoder,
)

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "BelnapDecisionHead",
    "YodaDecisionEngine",
]


class BelnapDecisionHead(nn.Module):
    """Projects Belnap evidence state into calibrated decision logits, truth, and knowledge."""

    def __init__(
        self,
        d_model: int,
        num_choices: int = 5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes BelnapDecisionHead.

        Args:
            d_model: Dimensionality of evidence representation.
            num_choices: Number of discrete decision choices.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.d_model = d_model
        self.num_choices = num_choices
        self.w_pos = nn.Linear(d_model, num_choices, device=device, dtype=dtype)
        self.w_neg = nn.Linear(d_model, num_choices, device=device, dtype=dtype)

        # Bilinear candidate evidence scoring projections
        self.cand_query_pos = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_query_neg = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_opt_pos = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_opt_neg = nn.Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: BelnapState,
        candidate_states: BelnapState | None = None,
    ) -> dict[str, torch.Tensor]:
        """Projects dual evidence states into decision logits and coordinates.

        If candidate_states is provided (shape: [batch_size, num_choices, d_model]),
        computes pairwise candidate-level alignment scores to eliminate static index bias.
        Otherwise falls back to linear projection heads over pooled evidence.

        Args:
            x: Input BelnapState of shape `(batch_size, num_queries, d_model)` or
                `(batch_size, d_model)`.
            candidate_states: Optional per-candidate BelnapState of shape
                `(batch_size, num_choices, d_model)`.

        Returns:
            Dictionary containing:
            - `logits`: Calibrated decision logits of shape `(batch_size, num_choices)`.
            - `truth`: Choice truth coordinates in [0, 1] of shape `(batch_size, num_choices)`.
            - `knowledge`: Choice knowledge coordinates in [0, 1] of shape
                `(batch_size, num_choices)`.
            - `choice`: Selected choice indices of shape `(batch_size,)`.
        """
        if x.e_pos.dim() == 3:
            pooled_pos = x.e_pos.mean(dim=1)
            pooled_neg = x.e_neg.mean(dim=1)
        else:
            pooled_pos = x.e_pos
            pooled_neg = x.e_neg

        if candidate_states is not None:
            # Pairwise candidate alignment scoring
            q_p = self.cand_query_pos(pooled_pos).unsqueeze(1)  # [B, 1, D]
            q_n = self.cand_query_neg(pooled_neg).unsqueeze(1)  # [B, 1, D]
            c_p = self.cand_opt_pos(candidate_states.e_pos)     # [B, num_choices, D]
            c_n = self.cand_opt_neg(candidate_states.e_neg)     # [B, num_choices, D]

            scale = (self.d_model ** 0.5)
            pos_affinity = (q_p * c_p + q_n * c_n).sum(dim=-1) / scale
            neg_affinity = (q_p * c_n + q_n * c_p).sum(dim=-1) / scale

            choice_pos = torch.sigmoid(pos_affinity)
            choice_neg = torch.sigmoid(neg_affinity)
        else:
            choice_pos = torch.sigmoid(self.w_pos(pooled_pos))
            choice_neg = torch.sigmoid(self.w_neg(pooled_neg))

        choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
        truth = choice_state.truth
        knowledge = choice_state.knowledge

        logits = (choice_pos - choice_neg) * knowledge
        choice = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "truth": truth,
            "knowledge": knowledge,
            "choice": choice,
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
        }


class YodaDecisionEngine(nn.Module):
    """Top-level neural-symbolic decision engine.

    Integrates encoders, Belnap MPA, and the reasoning core into a unified decision model.
    """

    def __init__(
        self,
        text_model_name: str = "dummy",
        embed_dim: int = 256,
        num_q_probes: int = 4,
        num_c_probes: int = 16,
        num_k_probes: int = 8,
        num_choices: int = 5,
        n_heads: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes YodaDecisionEngine.

        Args:
            text_model_name: Name of text backbone model or "dummy" for fast unit testing.
            embed_dim: Dimension of latent embedding space.
            num_q_probes: Number of query epistemic probe tokens for Belnap MPA.
            num_c_probes: Number of context epistemic probe tokens for Belnap MPA.
            num_k_probes: Number of constraint epistemic probe tokens for Belnap MPA.
            num_choices: Number of decision choices.
            n_heads: Number of attention heads for Belnap attention and transformer blocks.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.text_model_name = text_model_name
        self.embed_dim = embed_dim
        self.num_q_probes = num_q_probes
        self.num_c_probes = num_c_probes
        self.num_k_probes = num_k_probes
        self.num_choices = num_choices
        self.n_heads = n_heads

        # Phase 1: Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            target_dim=embed_dim,
            device=device,
            dtype=dtype,
        )
        self.state_encoder = SymbolicStateEncoder(self.text_encoder)
        self.constraint_encoder = ConstraintEncoder(self.text_encoder)

        # Phase 2: Epistemic Pooling (Belnap MPA)
        self.q_mpa = BelnapMultiheadPooledAttention(
            embed_dim=embed_dim,
            num_queries=num_q_probes,
            num_heads=n_heads,
            device=device,
            dtype=dtype,
        )
        self.c_mpa = BelnapMultiheadPooledAttention(
            embed_dim=embed_dim,
            num_queries=num_c_probes,
            num_heads=n_heads,
            device=device,
            dtype=dtype,
        )
        self.k_mpa = BelnapMultiheadPooledAttention(
            embed_dim=embed_dim,
            num_queries=num_k_probes,
            num_heads=n_heads,
            device=device,
            dtype=dtype,
        )

        # Phase 3: Reasoning Core (Cascaded Cross-Attention)
        self.context_reasoning = BelnapTransformerBlock(
            d_model=embed_dim,
            n_heads=n_heads,
        )
        self.constraint_reasoning = BelnapTransformerBlock(
            d_model=embed_dim,
            n_heads=n_heads,
        )

        # Phase 4: Judgment
        self.decision_head = BelnapDecisionHead(
            d_model=embed_dim,
            num_choices=num_choices,
            device=device,
            dtype=dtype,
        )

        logger.debug(
            "architecture.yoda_decision_engine.init",
            extra={
                "text_model_name": text_model_name,
                "embed_dim": embed_dim,
                "num_q_probes": num_q_probes,
                "num_c_probes": num_c_probes,
                "num_k_probes": num_k_probes,
                "num_choices": num_choices,
                "n_heads": n_heads,
            },
        )

    def forward(
        self,
        queries: list[str],
        states: list[dict[str, Any]],
        constraints: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        """Executes sequential epistemic reasoning over queries, states, and constraints.

        Args:
            queries: Batch of query text strings.
            states: Batch of symbolic state dictionaries.
            constraints: Batch of constraint string lists.

        Returns:
            Dictionary containing:
            - `logits`: Calibrated decision logits of shape `(batch_size, num_choices)`.
            - `truth`: Choice truth coordinates in [0, 1] of shape `(batch_size, num_choices)`.
            - `knowledge`: Choice knowledge coordinates in [0, 1] of shape
                `(batch_size, num_choices)`.
            - `choice`: Selected choice indices of shape `(batch_size,)`.
        """
        # 1. Encoding
        q_emb = self.text_encoder(queries)
        c_emb = self.state_encoder(states)
        k_emb = self.constraint_encoder(constraints)

        # 2. Pooling (Epistemic Sieve)
        _, q_state = self.q_mpa(q_emb)
        _, c_state = self.c_mpa(c_emb)
        _, k_state = self.k_mpa(k_emb)

        # 3. Reasoning (Sequential Interrogation)
        reasoning = self.context_reasoning(x=q_state, context=c_state)
        reasoning = self.constraint_reasoning(x=reasoning, context=k_state)

        # 4. Judgment (Pairwise candidate alignment if constraint probes match num_choices)
        cand_states = k_state if k_state.e_pos.size(1) == self.num_choices else None
        return self.decision_head(reasoning, candidate_states=cand_states)
