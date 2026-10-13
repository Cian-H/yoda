"""Top-level YodaDecisionEngine and BelnapDecisionHead architecture modules.

Wires together Phase 1 (Encoders), Phase 2 (Belnap MPA Pooling), Phase 3 (Reasoning Core),
and Phase 4 (Decision Head) using the Cascaded Cross-Attention (Sequential Interrogation) flow.
"""

import logging
from typing import Any, cast

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
        logit_scale: float = 8.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes BelnapDecisionHead.

        Args:
            d_model: Dimensionality of evidence representation.
            num_choices: Number of discrete decision choices.
            logit_scale: Amplification applied to bilinear candidate-alignment
                logits. The evidence channels keep their paraconsistent
                magnitudes (unmentioned candidates rest at Neither), while the
                ranking logits operate at a trainable scale so grouped
                focal/margin losses receive useful gradients from step one.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.d_model = d_model
        self.num_choices = num_choices
        self.logit_scale = float(logit_scale)
        self.w_pos = nn.Linear(d_model, num_choices, device=device, dtype=dtype)
        self.w_neg = nn.Linear(d_model, num_choices, device=device, dtype=dtype)

        # Bilinear candidate evidence scoring projections
        self.cand_query_pos = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_query_neg = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_opt_pos = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.cand_opt_neg = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # Initialize linear evidence readout with positive weights so positive evidence
        # reliably maps to positive decision support
        scale = 1.0 / (d_model**0.5)
        nn.init.uniform_(self.w_pos.weight, 0.05 * scale, scale)
        nn.init.zeros_(self.w_pos.bias)
        nn.init.uniform_(self.w_neg.weight, 0.05 * scale, scale)
        nn.init.zeros_(self.w_neg.bias)

        # Initialize bilinear candidate projections with identity matrix so initial candidate
        # alignment preserves direct Belnap inner products (ADR-0011)
        nn.init.eye_(self.cand_query_pos.weight)
        nn.init.zeros_(self.cand_query_pos.bias)
        nn.init.eye_(self.cand_query_neg.weight)
        nn.init.zeros_(self.cand_query_neg.bias)
        nn.init.eye_(self.cand_opt_pos.weight)
        nn.init.zeros_(self.cand_opt_pos.bias)
        nn.init.eye_(self.cand_opt_neg.weight)
        nn.init.zeros_(self.cand_opt_neg.bias)

    def forward(
        self,
        x: BelnapState,
        candidate_states: BelnapState | None = None,
    ) -> dict[str, torch.Tensor]:
        """Projects dual evidence states into decision logits and coordinates.

        If candidate_states is provided (shape: [batch_size, num_choices, d_model] or
        [batch_size, d_model]), computes bilinear candidate-level alignment scores on
        centered truth-order evidence coordinates to eliminate static index bias and
        additive context cancellation. Zero-affinity candidates rest at Neither (0, 0)
        evidence, and logits are gated by candidate knowledge. Otherwise falls back to
        linear projection heads over pooled evidence with knowledge-synchronized logits.

        Args:
            x: Input BelnapState of shape `(batch_size, num_queries, d_model)` or
                `(batch_size, d_model)`.
            candidate_states: Optional per-candidate BelnapState of shape
                `(batch_size, num_choices, d_model)` or `(batch_size, d_model)`.

        Returns:
            Dictionary containing:
            - `logits`: Calibrated decision logits of shape `(batch_size, num_choices)`.
            - `truth`: Choice truth coordinates in [0, 1] of shape `(batch_size, num_choices)`.
            - `knowledge`: Choice knowledge coordinates in [0, 1] of shape
                `(batch_size, num_choices)`.
            - `choice`: Selected choice indices of shape `(batch_size,)`.
        """
        pooled_pos = x.e_pos.mean(dim=1) if x.e_pos.dim() == 3 else x.e_pos
        pooled_neg = x.e_neg.mean(dim=1) if x.e_neg.dim() == 3 else x.e_neg

        if candidate_states is not None:
            # Bilinear candidate alignment scoring. The query contributes raw
            # evidence mass (confidence weighting) while the candidate contributes
            # centered truth-order signal eps = 2e - 1 in [-1, 1]: neutral
            # (Neither-ish) candidate evidence ~0.5 maps to ~0, so unmentioned
            # candidates produce zero alignment signal instead of positive
            # magnitude inflation, while decisive candidates (eps -> +-1)
            # transmit full affinity.
            if candidate_states.e_pos.dim() == 2:
                # 2D candidate states: [B, D] (e.g. scalar head evaluation)
                q_p = self.cand_query_pos(pooled_pos)  # [B, D]
                q_n = self.cand_query_neg(pooled_neg)  # [B, D]
                c_p = self.cand_opt_pos(2.0 * candidate_states.e_pos - 1.0)  # [B, D]
                c_n = self.cand_opt_neg(2.0 * candidate_states.e_neg - 1.0)  # [B, D]

                scale = self.d_model**0.5
                pos_affinity = (q_p * c_p + q_n * c_n).sum(dim=-1) / scale  # [B]
                neg_affinity = (q_p * c_n + q_n * c_p).sum(dim=-1) / scale  # [B]
                # Information mass of the candidate evidence: identical to the
                # knowledge coordinate for raw [0, 1] evidence, and equal to the
                # contrast magnitude for group-centered candidate features.
                cand_k = (candidate_states.e_pos.abs() + candidate_states.e_neg.abs()).mean(
                    dim=-1
                ) / 2.0
                raw_diff = pos_affinity - neg_affinity
            else:
                # 3D candidate states: [B, num_choices, D]
                q_p = self.cand_query_pos(pooled_pos).unsqueeze(1)  # [B, 1, D]
                q_n = self.cand_query_neg(pooled_neg).unsqueeze(1)  # [B, 1, D]
                c_p = self.cand_opt_pos(2.0 * candidate_states.e_pos - 1.0)  # [B, num_choices, D]
                c_n = self.cand_opt_neg(2.0 * candidate_states.e_neg - 1.0)  # [B, num_choices, D]

                scale = self.d_model**0.5
                pos_affinity = (q_p * c_p + q_n * c_n).sum(dim=-1) / scale
                neg_affinity = (q_p * c_n + q_n * c_p).sum(dim=-1) / scale
                cand_k = (candidate_states.e_pos.abs() + candidate_states.e_neg.abs()).mean(
                    dim=-1
                ) / 2.0
                raw_diff = pos_affinity - neg_affinity
                if self.num_choices == 1 and raw_diff.size(-1) == 1:
                    pos_affinity = pos_affinity.squeeze(-1)
                    neg_affinity = neg_affinity.squeeze(-1)
                    raw_diff = raw_diff.squeeze(-1)
                    cand_k = cand_k.squeeze(-1)

            # Epistemic gate from candidate knowledge: zero-knowledge candidates
            # cannot produce confident logits.
            cand_gate = (2.0 * cand_k).clamp(max=1.0)
            # Alignment evidence: zero affinity rests at Neither (0, 0) instead of
            # the contradictory Both midpoint, so unmentioned candidates carry
            # k -> 0 without false support or refutation. The sigmoid-weighted
            # tanh magnitude keeps gradients alive on both affinity signs: the
            # dominant channel encodes the polarity, the off-channel stays small.
            choice_pos = torch.sigmoid(pos_affinity) * torch.tanh(pos_affinity).abs() * cand_gate
            choice_neg = torch.sigmoid(neg_affinity) * torch.tanh(neg_affinity).abs() * cand_gate

            choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
            truth = choice_state.truth
            knowledge = choice_state.knowledge
            logits = raw_diff * cand_gate * self.logit_scale
        elif self.num_choices == 1:
            raw_pos = self.w_pos(pooled_pos).squeeze(-1)  # [B]
            raw_neg = self.w_neg(pooled_neg).squeeze(-1)  # [B]
            raw_diff = raw_pos - raw_neg

            state_k = (pooled_pos + pooled_neg).mean(dim=-1) / 2.0  # [B]
            cand_gate = (2.0 * state_k).clamp(max=1.0)
            choice_pos = torch.sigmoid(raw_pos) * cand_gate
            choice_neg = torch.sigmoid(raw_neg) * cand_gate

            choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
            truth = choice_state.truth
            knowledge = choice_state.knowledge
            logits = raw_diff * knowledge
        else:
            raw_pos = self.w_pos(pooled_pos)  # [B, num_choices]
            raw_neg = self.w_neg(pooled_neg)  # [B, num_choices]
            raw_diff = raw_pos - raw_neg

            state_k = (pooled_pos + pooled_neg).mean(dim=-1, keepdim=True) / 2.0  # [B, 1]
            cand_gate = (2.0 * state_k).clamp(max=1.0)
            choice_pos = torch.sigmoid(raw_pos) * cand_gate
            choice_neg = torch.sigmoid(raw_neg) * cand_gate

            choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
            truth = choice_state.truth
            knowledge = choice_state.knowledge
            logits = raw_diff * knowledge

        if self.num_choices == 1 or logits.dim() == 1:
            choice = (logits > 0).long()
        else:
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
        embed_dim: int | None = None,
        num_q_probes: int = 4,
        num_c_probes: int = 16,
        num_k_probes: int = 8,
        num_choices: int = 5,
        n_heads: int = 4,
        num_reasoning_blocks: int = 1,
        conflation_weight: float = 0.1,
        d_hidden_multiplier: float = 2.0,
        residual_weight: float = 0.15,
        dropout: float = 0.0,
        use_candidate_affinity: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes YodaDecisionEngine.

        Args:
            text_model_name: Name of text backbone model or "dummy" for fast unit testing.
            embed_dim: Dimension of latent embedding space. If None, adopts native encoder dim.
            num_q_probes: Number of query epistemic probe tokens for Belnap MPA.
            num_c_probes: Number of context epistemic probe tokens for Belnap MPA.
            num_k_probes: Number of constraint epistemic probe tokens for Belnap MPA.
            num_choices: Number of decision choices.
            n_heads: Number of attention heads for Belnap attention and transformer blocks.
            num_reasoning_blocks: Number of cascaded reasoning transformer blocks.
            conflation_weight: Weight lambda for contradictory evidence pruning.
            d_hidden_multiplier: Multiplier for FFN hidden dimensionality.
            residual_weight: Interpolation weight alpha for convex combination residual joins.
            dropout: Dropout probability for reasoning transformer blocks and FFN.
            use_candidate_affinity: If True, uses dynamic bilateral candidate affinity scoring.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.text_model_name = text_model_name
        self.num_q_probes = num_q_probes
        self.num_c_probes = num_c_probes
        self.num_k_probes = num_k_probes
        self.num_choices = num_choices
        self.n_heads = n_heads
        self.num_reasoning_blocks = num_reasoning_blocks
        self.conflation_weight = conflation_weight
        self.d_hidden_multiplier = d_hidden_multiplier
        self.residual_weight = residual_weight
        self.dropout = dropout
        self.use_candidate_affinity = use_candidate_affinity

        # Phase 1: Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            target_dim=embed_dim,
            device=device,
            dtype=dtype,
        )
        effective_dim = self.text_encoder.dim
        self.embed_dim = effective_dim

        self.state_encoder = SymbolicStateEncoder(self.text_encoder)
        self.constraint_encoder = ConstraintEncoder(self.text_encoder)
        self.cand_proj = nn.Linear(effective_dim, 2 * effective_dim, device=device, dtype=dtype)
        # Project task scalar (-1.0, 0.0, 1.0) into expressive latent space
        self.task_proj = nn.Sequential(
            nn.Linear(1, effective_dim, device=device, dtype=dtype),
            nn.Mish(),
            nn.Linear(effective_dim, effective_dim, device=device, dtype=dtype),
        )

        # Phase 2: Epistemic Pooling (Belnap MPA)
        self.q_mpa = BelnapMultiheadPooledAttention(
            embed_dim=effective_dim,
            num_queries=num_q_probes,
            num_heads=n_heads,
            include_projection=False,
            device=device,
            dtype=dtype,
        )
        self.c_mpa = BelnapMultiheadPooledAttention(
            embed_dim=effective_dim,
            num_queries=num_c_probes,
            num_heads=n_heads,
            include_projection=False,
            device=device,
            dtype=dtype,
        )
        self.k_mpa = BelnapMultiheadPooledAttention(
            embed_dim=effective_dim,
            num_queries=num_k_probes,
            num_heads=n_heads,
            include_projection=False,
            device=device,
            dtype=dtype,
        )

        # Phase 3: Reasoning Core (Cascaded Cross-Attention)
        self.reasoning_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "context": BelnapTransformerBlock(
                            d_model=effective_dim,
                            n_heads=n_heads,
                            d_hidden=int(effective_dim * d_hidden_multiplier),
                            conflation_weight=conflation_weight,
                            residual_weight=residual_weight,
                            dropout=dropout,
                        ),
                        "constraint": BelnapTransformerBlock(
                            d_model=effective_dim,
                            n_heads=n_heads,
                            d_hidden=int(effective_dim * d_hidden_multiplier),
                            conflation_weight=conflation_weight,
                            residual_weight=residual_weight,
                            dropout=dropout,
                        ),
                    }
                )
                for _ in range(num_reasoning_blocks)
            ]
        )

        # Phase 4: Judgment
        self.decision_head = BelnapDecisionHead(
            d_model=effective_dim,
            num_choices=num_choices,
            device=device,
            dtype=dtype,
        )
        if num_choices != 1:
            self.scalar_head = BelnapDecisionHead(
                d_model=effective_dim,
                num_choices=1,
                device=device,
                dtype=dtype,
            )
        else:
            self.scalar_head = self.decision_head

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
                "num_reasoning_blocks": num_reasoning_blocks,
                "conflation_weight": conflation_weight,
                "d_hidden_multiplier": d_hidden_multiplier,
                "residual_weight": residual_weight,
                "dropout": dropout,
            },
        )

    @property
    def context_reasoning(self) -> BelnapTransformerBlock:
        """First context reasoning block (for backward compatibility)."""
        layer = cast(nn.ModuleDict, self.reasoning_layers[0])
        return cast(BelnapTransformerBlock, layer["context"])

    @property
    def constraint_reasoning(self) -> BelnapTransformerBlock:
        """First constraint reasoning block (for backward compatibility)."""
        layer = cast(nn.ModuleDict, self.reasoning_layers[0])
        return cast(BelnapTransformerBlock, layer["constraint"])

    def forward(
        self,
        queries: list[str],
        states: list[dict[str, Any]],
        constraints: list[list[str]] | list[str] | None = None,
        candidates: list[str] | None = None,
        task_scalars: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Executes sequential epistemic reasoning over queries, states, and constraints/candidates.

        Args:
            queries: Batch of query text strings.
            states: Batch of symbolic state dictionaries.
            constraints: Optional batch of constraint lists or flat candidate strings.
            candidates: Optional flat list of candidate strings for independent evaluation.
            task_scalars: Optional batch of task scalars of shape `(batch_size, 1)`.
            active_mask: Optional boolean tensor marking valid active criteria.
            return_diagnostics: If True, attaches intermediate stage trajectory and
                attribution probes into the returned dictionary under `"diagnostics"`.

        Returns:
            Dictionary containing calibrated decision logits, truth, knowledge, etc.
        """
        is_independent = False
        target_cand: list[str] | list[list[str]]
        if candidates is not None:
            target_cand = candidates
            is_independent = True
        elif constraints is not None:
            target_cand = constraints
            if (len(constraints) > 0 and isinstance(constraints[0], str)) or self.num_choices == 1:
                is_independent = True
        else:
            target_cand = []
            is_independent = self.num_choices == 1

        # 1. Encoding
        if hasattr(self.text_encoder, "forward_with_mask"):
            q_emb, q_mask = self.text_encoder.forward_with_mask(queries)
        else:
            q_emb = self.text_encoder(queries)
            q_mask = None

        if task_scalars is not None:
            task_vector = self.task_proj(task_scalars)
            if q_emb.dim() == 3:
                task_vector = task_vector.unsqueeze(1)
            q_emb = q_emb + task_vector

        if hasattr(self.state_encoder, "forward_with_mask"):
            c_emb, c_mask = self.state_encoder.forward_with_mask(states)
        else:
            c_emb = self.state_encoder(states)
            c_mask = None

        if hasattr(self.constraint_encoder, "forward_with_mask"):
            k_emb, k_mask = self.constraint_encoder.forward_with_mask(target_cand)
        else:
            k_emb = self.constraint_encoder(target_cand)
            k_mask = None

        # 2. Pooling (Epistemic Sieve)
        q_state = cast(BelnapState, self.q_mpa(q_emb, mask=q_mask))
        c_state = cast(BelnapState, self.c_mpa(c_emb, mask=c_mask))
        k_state = cast(BelnapState, self.k_mpa(k_emb, mask=k_mask))

        # 3. Cascaded Cross-Attention Reasoning Core
        stage_names = ["post_pooling"]
        stage_states = [q_state]

        curr_state = q_state
        for idx, layer in enumerate(self.reasoning_layers):
            layer_dict = cast(nn.ModuleDict, layer)
            context_block = cast(BelnapTransformerBlock, layer_dict["context"])
            constraint_block = cast(BelnapTransformerBlock, layer_dict["constraint"])
            curr_state = context_block(x=curr_state, context=c_state)
            suffix = f"_{idx}" if len(self.reasoning_layers) > 1 else ""
            stage_names.append(f"post_context{suffix}")
            stage_states.append(curr_state)

            curr_state = constraint_block(x=curr_state, context=k_state)
            stage_names.append(f"post_constraint{suffix}")
            stage_states.append(curr_state)

        final_state = curr_state

        if is_independent:
            head = self.scalar_head if self.num_choices != 1 else self.decision_head
            # Pool per-candidate probe states to [B, D] and score candidates via
            # bilinear query-candidate affinity (R4): eliminates the additive
            # context term that cancels across candidates under linear scoring.
            # Candidate features come from the masked mean of the raw candidate
            # token embeddings (polarity split via a scaled sigmoid pair): the
            # Belnap MPA evidence is reserved for the reasoning core because its
            # sigmoid-bounded channels wash out per-candidate contrast, which
            # the bilinear alignment needs at full strength.
            k_mean_mask = k_mask.unsqueeze(-1).to(k_emb.dtype) if k_mask is not None else 1.0
            k_token_mass = k_mean_mask.sum(dim=1).clamp(min=1.0)
            k_pooled_raw = (k_emb * k_mean_mask).sum(dim=1) / k_token_mass
            pooled_cand = BelnapState(
                e_pos=torch.sigmoid(k_pooled_raw),
                e_neg=torch.sigmoid(-k_pooled_raw),
            )
            # Group-relative candidate scoring: candidates sharing an identical
            # (query, state, task) context form one decision group, and each
            # candidate is scored on its deviation from the group's mean
            # candidate evidence. This removes the candidate-common plausibility
            # direction that otherwise satisfies per-candidate evidence targets
            # without discriminating within the group (the 50% ceiling). Single
            # candidate groups keep their raw evidence to avoid collapse.
            cand_state = self._center_candidate_states(
                pooled_cand,
                queries=queries,
                states=states,
                task_scalars=task_scalars,
            )
            final_out = head(final_state, candidate_states=cand_state)
            if return_diagnostics:
                probe_outs = [head(s, candidate_states=cand_state) for s in stage_states]
                stage_logits = torch.stack([p["logits"] for p in probe_outs], dim=0)
                stage_knowledge = torch.stack([p["knowledge"] for p in probe_outs], dim=0)
                stage_truth = torch.stack([p["truth"] for p in probe_outs], dim=0)

                attributions_list = [stage_logits[0]]
                for i in range(1, len(stage_states)):
                    attributions_list.append(stage_logits[i] - stage_logits[i - 1])
                attributions = torch.stack(attributions_list, dim=0)

                final_out["diagnostics"] = {
                    "stage_names": stage_names,
                    "stage_logits": stage_logits,
                    "stage_knowledge": stage_knowledge,
                    "stage_truth": stage_truth,
                    "attributions": attributions,
                }
            return final_out

        legacy_constraints: list[list[str]] = cast(list[list[str]], target_cand)
        return self._legacy_forward(
            final_state=final_state,
            k_state=k_state,
            legacy_constraints=legacy_constraints,
            active_mask=active_mask,
            return_diagnostics=return_diagnostics,
            stage_states=stage_states,
            stage_names=stage_names,
        )

    def _context_group_keys(
        self,
        queries: list[str],
        states: list[dict[str, Any]],
        task_scalars: torch.Tensor | None,
    ) -> list[tuple[str, str, float]]:
        """Builds per-row keys identifying the shared (query, state, task) context."""
        if task_scalars is not None:
            tasks = [float(t) for t in task_scalars.reshape(-1).tolist()]
        else:
            tasks = [0.0] * len(queries)
        return [
            (q, self.state_encoder.format_state(s), t)
            for q, s, t in zip(queries, states, tasks, strict=False)
        ]

    def _center_candidate_states(
        self,
        cand_state: BelnapState,
        queries: list[str],
        states: list[dict[str, Any]],
        task_scalars: torch.Tensor | None,
    ) -> BelnapState:
        """Centers candidate evidence against the group mean per shared context.

        Candidates evaluated under an identical (query, state, task) context form
        one decision group; each candidate's evidence is replaced by its
        deviation from the group mean so bilinear scoring ranks candidates on
        their distinctive signal. Single-candidate groups keep raw evidence.
        """
        if cand_state.e_pos.size(0) <= 1:
            return cand_state
        keys = self._context_group_keys(queries, states, task_scalars)
        group_members: dict[tuple[str, str, float], list[int]] = {}
        for idx, key in enumerate(keys):
            group_members.setdefault(key, []).append(idx)

        device = cand_state.e_pos.device
        pos = cand_state.e_pos.clone()
        neg = cand_state.e_neg.clone()
        for members in group_members.values():
            if len(members) > 1:
                rows = torch.tensor(members, device=device, dtype=torch.long)
                # The +0.5 evidence midpoint offset restores the [0, 1] evidence
                # range: the head's centered coordinates eps = 2e - 1 cancel the
                # offset exactly (recovering the pure group deviation), while the
                # candidate knowledge gate reads the restored information mass.
                pos[rows] = (
                    cand_state.e_pos[rows] - cand_state.e_pos[rows].mean(dim=0, keepdim=True) + 0.5
                )
                neg[rows] = (
                    cand_state.e_neg[rows] - cand_state.e_neg[rows].mean(dim=0, keepdim=True) + 0.5
                )
        logger.debug(
            "architecture.yoda_decision_engine.group_relative_scoring",
            extra={
                "batch_rows": int(cand_state.e_pos.size(0)),
                "num_groups": len(group_members),
                "multi_candidate_groups": sum(1 for m in group_members.values() if len(m) > 1),
            },
        )
        return BelnapState(e_pos=pos, e_neg=neg)

    def _resolve_cand_states(
        self,
        legacy_constraints: list[list[str]],
        k_state: BelnapState | None,
    ) -> BelnapState | None:
        """Resolves candidate states for choice affinity scoring."""
        if (
            self.use_candidate_affinity
            and len(legacy_constraints) > 0
            and len(legacy_constraints[0]) > 0
        ):
            cand_embs = self.constraint_encoder.encode_candidates(legacy_constraints)
            pos_raw, neg_raw = torch.sigmoid(self.cand_proj(cand_embs)).chunk(2, dim=-1)
            return BelnapState(e_pos=pos_raw, e_neg=neg_raw)
        if k_state is not None and k_state.e_pos.size(1) == self.num_choices:
            return k_state
        return None

    def _resolve_active_mask(
        self,
        legacy_constraints: list[list[str]],
        num_out_choices: int,
        device: torch.device,
        active_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Constructs or formats the boolean mask for active candidate choices."""
        if active_mask is not None:
            return active_mask.to(device=device)
        if not legacy_constraints:
            return None

        mask_list: list[list[bool]] = []
        for sample in legacy_constraints:
            sample_len = len(sample)
            row = [
                (not str(sample[c_idx]).startswith("none:")) if c_idx < sample_len else False
                for c_idx in range(num_out_choices)
            ]
            mask_list.append(row)
        return torch.tensor(mask_list, device=device, dtype=torch.bool)

    def _resolve_stage_states(
        self,
        target_state: BelnapState,
        stage_states: list[BelnapState] | None,
        stage_names: list[str] | None,
        x_0: BelnapState | None,
        x_1: BelnapState | None,
        x_2: BelnapState | None,
    ) -> tuple[list[BelnapState], list[str]]:
        """Resolves intermediate stages and their human-readable labels."""
        if stage_states is not None:
            names = (
                stage_names
                if stage_names is not None
                else ["post_pooling"] + [f"stage_{i}" for i in range(1, len(stage_states))]
            )
            return stage_states, names

        if x_0 is not None and x_1 is not None and x_2 is not None:
            return [x_0, x_1, x_2], ["post_pooling", "post_context", "post_constraint"]

        return [target_state], ["post_constraint"]

    def _build_stage_diagnostics(
        self,
        stage_states: list[BelnapState],
        stage_names: list[str],
        cand_states: BelnapState | None,
        active_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Calculates stage attributions and intermediate probes for diagnostics."""
        probe_outs = [self.decision_head(s, candidate_states=cand_states) for s in stage_states]
        if active_mask is not None and active_mask.numel() > 0:
            for probe in probe_outs:
                probe["logits"] = probe["logits"].masked_fill(~active_mask, -1e9)
                probe["truth"] = probe["truth"].masked_fill(~active_mask, 0.0)
                probe["knowledge"] = probe["knowledge"].masked_fill(~active_mask, 0.0)

        stage_logits = torch.stack([p["logits"] for p in probe_outs], dim=0)
        stage_knowledge = torch.stack([p["knowledge"] for p in probe_outs], dim=0)
        stage_truth = torch.stack([p["truth"] for p in probe_outs], dim=0)

        attributions_list = [stage_logits[0]]
        for i in range(1, len(stage_states)):
            attributions_list.append(stage_logits[i] - stage_logits[i - 1])
        attributions = torch.stack(attributions_list, dim=0)

        return {
            "stage_names": stage_names,
            "stage_logits": stage_logits,
            "stage_knowledge": stage_knowledge,
            "stage_truth": stage_truth,
            "attributions": attributions,
        }

    def _legacy_forward(
        self,
        final_state: BelnapState | None = None,
        k_state: BelnapState | None = None,
        legacy_constraints: list[list[str]] | None = None,
        active_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        stage_states: list[BelnapState] | None = None,
        stage_names: list[str] | None = None,
        x_0: BelnapState | None = None,
        x_1: BelnapState | None = None,
        x_2: BelnapState | None = None,
    ) -> dict[str, Any]:
        """Executes legacy multi-choice judgment and active masking."""
        target_state = final_state if final_state is not None else x_2
        if target_state is None:
            msg = "Either final_state or x_2 must be provided."
            raise ValueError(msg)

        resolved_stages, resolved_names = self._resolve_stage_states(
            target_state=target_state,
            stage_states=stage_states,
            stage_names=stage_names,
            x_0=x_0,
            x_1=x_1,
            x_2=x_2,
        )

        constraints = legacy_constraints or []
        cand_states = self._resolve_cand_states(constraints, k_state)

        final_out = self.decision_head(target_state, candidate_states=cand_states)
        device = final_out["logits"].device
        num_out_choices = final_out["logits"].shape[1]

        mask = self._resolve_active_mask(constraints, num_out_choices, device, active_mask)
        if mask is not None and mask.numel() > 0:
            final_out["logits"] = final_out["logits"].masked_fill(~mask, -1e9)
            final_out["truth"] = final_out["truth"].masked_fill(~mask, 0.0)
            final_out["knowledge"] = final_out["knowledge"].masked_fill(~mask, 0.0)
            final_out["choice_pos"] = final_out["choice_pos"].masked_fill(~mask, 0.0)
            final_out["choice_neg"] = final_out["choice_neg"].masked_fill(~mask, 0.0)
            final_out["choice"] = torch.argmax(final_out["logits"], dim=-1)
            final_out["active_mask"] = mask

        if return_diagnostics:
            final_out["diagnostics"] = self._build_stage_diagnostics(
                stage_states=resolved_stages,
                stage_names=resolved_names,
                cand_states=cand_states,
                active_mask=mask,
            )

        return final_out
