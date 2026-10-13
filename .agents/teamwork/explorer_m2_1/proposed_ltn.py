"""Logic Tensor Network (LTN) semantic constraints for Yoda Decision Engine.

Implements differentiable probabilistic and t-norm logic constraints:
1. Paraconsistent Contradiction Penalty (Both):
   (choice_pos * choice_neg).mean() * current_decisiveness_weight
2. Exactly-One XOR on Positive Evidence Support (Choice Tasks, task_scalar > 0.75):
   -log(sum_i e_i^+ prod_{j != i} (1 - e_j^+) + eps)
   Ensures unchosen candidate options at Neither (0, 0) have (1 - e_j^+) = 1.0,
   incurring 0 constraint penalty.
3. At-Least-One OR on Positive Evidence Support (Multi-Choice Tasks, 0.25 < task_scalar <= 0.75):
   -log(1 - prod_i (1 - e_i^+) + eps)
4. Operational Box Boundedness (Score/Null Tasks, task_scalar <= 0.25):
   max(0, s - b)^2 + max(0, a - s)^2
5. Hierarchical Łukasiewicz Implication (Taxonomy / Graph Tasks):
   mean(max(0, t_child - t_parent)^2)
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "LTNConstraintLoss",
]


class LTNConstraintLoss(nn.Module):
    """Logic Tensor Network (LTN) constraints for Yoda Decision Engine.

    Dynamically routes logical invariants based on the task type (represented
    by continuous task control scalar) and hierarchical graph dependencies.
    Evaluates multi-choice exclusivity constraints (XOR/OR) on positive support (choice_pos)
    to protect unmentioned Neither options from false penalties.
    """

    def __init__(
        self,
        bound_a: float = -3.0,
        bound_b: float = 3.0,
        bound_weight: float = 1.0,
        hierarchy_weight: float = 1.0,
        decisiveness_weight: float = 0.1,
        warmup_epochs: int = 0,
        eps: float = 1e-7,
    ) -> None:
        """Initializes LTNConstraintLoss.

        Args:
            bound_a: Lower operational bound for continuous scoring / null tasks.
            bound_b: Upper operational bound for continuous scoring / null tasks.
            bound_weight: Multiplier weight for operational box boundedness loss.
            hierarchy_weight: Multiplier weight for hierarchical implication loss.
            decisiveness_weight: Target multiplier weight for contradiction penalty.
            warmup_epochs: Number of epochs for linear warmup from 0.0 to decisiveness_weight.
            eps: Epsilon for numerical stability inside log evaluations.
        """
        super().__init__()
        self.bound_a = float(bound_a)
        self.bound_b = float(bound_b)
        self.bound_weight = float(bound_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.decisiveness_weight = float(decisiveness_weight)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.current_epoch: int = 0
        self.current_decisiveness_weight: float = (
            0.0 if self.warmup_epochs > 0 else self.decisiveness_weight
        )
        self.eps = float(eps)

    def step_epoch(self, epoch: int | None = None) -> float:
        """Updates current_decisiveness_weight according to linear warmup schedule.

        Schedule: lambda(e) = lambda_target * min(1.0, e / W) when W > 0.

        Args:
            epoch: Optional epoch index (0-indexed). If None, increments internal counter.

        Returns:
            Updated current_decisiveness_weight.
        """
        if epoch is not None:
            self.current_epoch = max(0, int(epoch))
        else:
            self.current_epoch += 1

        effective_warmup = self.warmup_epochs if self.warmup_epochs > 0 else 5
        progress = min(1.0, float(self.current_epoch) / float(effective_warmup))
        self.current_decisiveness_weight = self.decisiveness_weight * progress

        return self.current_decisiveness_weight

    def _compute_xor(
        self,
        pos_support: torch.Tensor | None = None,
        *,
        truth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluates Exactly-One (XOR) on positive support: P(XOR) = sum_i e_i^+ prod_{j != i} (1 - e_j^+).

        For unchosen candidate options at Neither (0, 0), positive support e_j^+ = 0,
        yielding (1 - e_j^+) = 1.0, which contributes neutral factor 1.0 without false penalty.

        Args:
            pos_support: 1D tensor of positive evidence support values in [0, 1].
            truth: Backward-compatible alias for pos_support.

        Returns:
            Scalar XOR constraint penalty tensor.
        """
        support = pos_support if pos_support is not None else truth
        if support is None:
            raise ValueError("pos_support or truth tensor must be provided.")

        flat = support.view(-1)
        K = flat.numel()
        if K == 0:
            return torch.tensor(0.0, device=support.device, dtype=support.dtype)
        if K == 1:
            # Single option: probability of being chosen is e_0^+
            return -torch.log(flat[0].clamp(min=self.eps, max=1.0))

        p_not = 1.0 - flat
        p_xor = torch.tensor(0.0, device=support.device, dtype=support.dtype)
        for i in range(K):
            term = flat[i]
            for j in range(K):
                if i != j:
                    term = term * p_not[j]
            p_xor = p_xor + term

        # In paraconsistent / open-world Belnap settings, if an evaluated slice consists
        # entirely of unmentioned options (pos=0, neg=0), p_none = prod_i (1 - e_i^+) captures
        # non-assertion / zero-conflict, ensuring unmentioned candidates alone incur 0 loss.
        p_none = torch.prod(p_not)
        total_p = p_xor + p_none

        return -torch.log(total_p.clamp(min=self.eps, max=1.0))

    def _compute_xor_positive_support(self, pos_support: torch.Tensor) -> torch.Tensor:
        """Alias for _compute_xor evaluating Exactly-One on positive evidence support."""
        return self._compute_xor(pos_support)

    def _compute_or(
        self,
        pos_support: torch.Tensor | None = None,
        *,
        truth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluates At-Least-One (OR) on positive support: P(OR) = 1 - prod_i (1 - e_i^+).

        Ensures unmentioned candidate options at Neither do not falsely satisfy OR,
        while active candidate choices fully satisfy OR regardless of Neither presence.

        Args:
            pos_support: 1D tensor of positive evidence support values in [0, 1].
            truth: Backward-compatible alias for pos_support.

        Returns:
            Scalar OR constraint penalty tensor.
        """
        support = pos_support if pos_support is not None else truth
        if support is None:
            raise ValueError("pos_support or truth tensor must be provided.")

        flat = support.view(-1)
        K = flat.numel()
        if K == 0:
            return torch.tensor(0.0, device=support.device, dtype=support.dtype)

        p_not = 1.0 - flat
        prod_not = torch.prod(p_not)
        p_or = 1.0 - prod_not
        return -torch.log(p_or.clamp(min=self.eps, max=1.0))

    def _compute_or_positive_support(self, pos_support: torch.Tensor) -> torch.Tensor:
        """Alias for _compute_or evaluating At-Least-One on positive evidence support."""
        return self._compute_or(pos_support)

    def _compute_boundedness(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluates operational box constraint: max(0, s - b)^2 + max(0, a - s)^2."""
        if values.numel() == 0:
            return torch.tensor(0.0, device=values.device, dtype=values.dtype)
        upper_viol = F.relu(values - self.bound_b)
        lower_viol = F.relu(self.bound_a - values)
        return (upper_viol**2 + lower_viol**2).mean()

    def _compute_hierarchy(
        self,
        truth: torch.Tensor,
        hierarchy_edges: list[tuple[int, int]] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluates Łukasiewicz continuous implication: max(0, t_child - t_parent)^2."""
        if hierarchy_edges is None:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

        violations: list[torch.Tensor] = []
        if isinstance(hierarchy_edges, torch.Tensor):
            edges_list = hierarchy_edges.tolist()
        else:
            edges_list = hierarchy_edges

        flat_truth = truth.view(-1)
        num_items = flat_truth.size(0)

        for edge in edges_list:
            c, p = edge
            if 0 <= c < num_items and 0 <= p < num_items:
                diff = flat_truth[c] - flat_truth[p]
                viol = F.relu(diff) ** 2
                violations.append(viol)

        if not violations:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

        return torch.stack(violations).mean()

    def _compute_grouped_constraints(
        self,
        pos_support: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        task_scalars: torch.Tensor | None = None,
        *,
        truth: torch.Tensor | None = None,
        choice_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Routes task-conditioned logical constraints across candidate groups using positive support."""
        support = (
            pos_support
            if pos_support is not None
            else (choice_pos if choice_pos is not None else truth)
        )
        if support is None or group_ids is None or task_scalars is None:
            raise ValueError(
                "pos_support (or choice_pos/truth), group_ids, and task_scalars must be provided."
            )

        unique_groups = torch.unique(group_ids)
        task_losses: list[torch.Tensor] = []

        for gid in unique_groups:
            mask = group_ids == gid
            if task_scalars.shape[0] == group_ids.shape[0]:
                g_task = task_scalars[mask][0].item()
            else:
                g_task = task_scalars[gid].item()

            g_pos = support[mask]
            g_logits = logits[mask] if logits is not None else None

            # Route by task scalar:
            # 1. Choice Task (scalar > 0.75): Exactly-One (XOR) on positive support
            if g_task > 0.75:
                task_losses.append(self._compute_xor(g_pos))
            # 2. Multi-Choice Task (0.25 < scalar <= 0.75): At-Least-One (OR) on positive support
            elif 0.25 < g_task <= 0.75:
                task_losses.append(self._compute_or(g_pos))
            # 3. Score / Null Task (scalar <= 0.25): Operational Box Boundedness
            else:
                if g_logits is not None:
                    bound_loss = self._compute_boundedness(g_logits) * self.bound_weight
                    task_losses.append(bound_loss)

        if task_losses:
            return torch.stack(task_losses).mean()
        return torch.tensor(0.0, device=support.device, dtype=support.dtype)

    def _compute_legacy_constraints(
        self,
        pos_support: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        task_scalars: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        *,
        truth: torch.Tensor | None = None,
        choice_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Routes task-conditioned logical constraints across legacy batch rows using positive support."""
        support = (
            pos_support
            if pos_support is not None
            else (choice_pos if choice_pos is not None else truth)
        )
        if support is None or task_scalars is None:
            raise ValueError(
                "pos_support (or choice_pos/truth) and task_scalars must be provided."
            )

        batch_size = support.size(0)
        row_losses: list[torch.Tensor] = []

        for b in range(batch_size):
            task_val = task_scalars[b].item()
            row_pos = support[b]
            if active_mask is not None:
                mask_b = active_mask[b]
                row_pos = row_pos[mask_b]
                row_logits = logits[b][mask_b] if logits is not None else None
            else:
                row_logits = logits[b] if logits is not None else None

            # 1. Choice Task (scalar > 0.75): Exactly-One (XOR) on positive support
            if task_val > 0.75:
                row_losses.append(self._compute_xor(row_pos))
            # 2. Multi-Choice Task (0.25 < scalar <= 0.75): At-Least-One (OR) on positive support
            elif 0.25 < task_val <= 0.75:
                row_losses.append(self._compute_or(row_pos))
            # 3. Score / Null Task (scalar <= 0.25): Operational Box Boundedness
            else:
                if row_logits is not None:
                    bound_loss = self._compute_boundedness(row_logits) * self.bound_weight
                    row_losses.append(bound_loss)

        if row_losses:
            return torch.stack(row_losses).mean()
        return torch.tensor(0.0, device=support.device, dtype=support.dtype)

    def _compute_bivalence_penalty(
        self, choice_pos: torch.Tensor, choice_neg: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates classical bivalence complementarity: -log(e^+ (1-e^-) + e^- (1-e^+) + eps)."""
        prob_comp = choice_pos * (1.0 - choice_neg) + choice_neg * (1.0 - choice_pos)
        return -torch.log(prob_comp.clamp(min=self.eps))

    def forward(
        self,
        outputs: dict[str, Any],
        task_scalars: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        hierarchy_edges: list[tuple[int, int]] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the task-conditioned LTN constraint loss.

        Args:
            outputs: Model outputs containing `choice_pos`, `choice_neg`, optionally `truth`,
                and optionally `logits`.
            task_scalars: Optional tensor indicating task category/mode.
            active_mask: Optional boolean tensor for valid criteria masking.
            group_ids: Optional grouping tensor for unrolled independent evaluation.
            hierarchy_edges: Optional list of (child_idx, parent_idx) index tuples.

        Returns:
            Scalar tensor loss combining decisiveness penalty and task constraints.
        """
        choice_pos = outputs["choice_pos"]
        choice_neg = outputs["choice_neg"]
        truth = outputs.get("truth")
        logits = outputs.get("logits")

        # Check hierarchy edges in arguments or outputs dictionary
        h_edges = hierarchy_edges
        if h_edges is None and "hierarchy_edges" in outputs:
            h_edges = outputs["hierarchy_edges"]

        # Evaluate hierarchy implication loss (evaluated on truth coordinate)
        imp_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if h_edges is not None:
            if truth is None:
                truth = 0.5 * (choice_pos - choice_neg + 1.0)
            imp_loss = self._compute_hierarchy(truth, h_edges) * self.hierarchy_weight

        # Grouped candidate evaluation path
        if group_ids is not None:
            decisiveness_penalty = (
                choice_pos * choice_neg
            ).mean() * self.current_decisiveness_weight

            task_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
            if task_scalars is not None:
                task_loss = self._compute_grouped_constraints(
                    pos_support=choice_pos,
                    logits=logits,
                    group_ids=group_ids,
                    task_scalars=task_scalars,
                )
            return decisiveness_penalty + task_loss + imp_loss

        # Legacy fixed-dimension evaluation path
        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None
        if mask_f is not None:
            denom = mask_f.sum().clamp(min=1.0)
            decisiveness_terms = choice_pos * choice_neg
            decisiveness_penalty = (
                (decisiveness_terms * mask_f).sum() / denom
            ) * self.current_decisiveness_weight
        else:
            decisiveness_penalty = (
                choice_pos * choice_neg
            ).mean() * self.current_decisiveness_weight

        task_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            task_loss = self._compute_legacy_constraints(
                pos_support=choice_pos,
                logits=logits,
                task_scalars=task_scalars,
                active_mask=active_mask,
            )

        return decisiveness_penalty + task_loss + imp_loss
