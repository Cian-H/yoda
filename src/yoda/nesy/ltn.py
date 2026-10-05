"""Logic Tensor Network (LTN) semantic constraints for Yoda Decision Engine.

Implements differentiable probabilistic and t-norm logic constraints:
1. Epistemic Regularization (Belnap Bilattice):
   - Gullibility penalty: knowledge.mean() * gullibility_weight
   - Ignorance penalty: ignorance.mean() * ignorance_weight
2. Exactly-One XOR (Choice Tasks, task_scalar > 0.75):
   -log(sum_i t_i prod_{j != i} (1 - t_j) + eps)
3. At-Least-One OR (Multi-Choice Tasks, 0.25 < task_scalar <= 0.75):
   -log(1 - prod_i (1 - t_i) + eps)
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
    """

    def __init__(
        self,
        bound_a: float = -3.0,
        bound_b: float = 3.0,
        bound_weight: float = 1.0,
        hierarchy_weight: float = 1.0,
        gullibility_weight: float = 0.1,
        ignorance_weight: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        """Initializes LTNConstraintLoss.

        Args:
            bound_a: Lower operational bound for continuous scoring / null tasks.
            bound_b: Upper operational bound for continuous scoring / null tasks.
            bound_weight: Multiplier weight for operational box boundedness loss.
            hierarchy_weight: Multiplier weight for hierarchical implication loss.
            gullibility_weight: Multiplier weight for epistemic gullibility penalty.
            ignorance_weight: Multiplier weight for epistemic ignorance penalty.
            eps: Epsilon for numerical stability inside log evaluations.
        """
        super().__init__()
        self.bound_a = float(bound_a)
        self.bound_b = float(bound_b)
        self.bound_weight = float(bound_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.gullibility_weight = float(gullibility_weight)
        self.ignorance_weight = float(ignorance_weight)
        self.eps = float(eps)

    def _compute_xor(self, truth: torch.Tensor) -> torch.Tensor:
        """Evaluates Exactly-One (XOR) probability: P(XOR) = sum_i t_i prod_{j != i} (1 - t_j)."""
        K = truth.numel()
        if K == 0:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)
        if K == 1:
            # Single option: probability of being true is t_0
            return -torch.log(truth[0].clamp(min=self.eps))

        p_not = 1.0 - truth
        p_xor = torch.tensor(0.0, device=truth.device, dtype=truth.dtype)
        for i in range(K):
            term = truth[i]
            for j in range(K):
                if i != j:
                    term = term * p_not[j]
            p_xor = p_xor + term

        return -torch.log(p_xor.clamp(min=self.eps))

    def _compute_or(self, truth: torch.Tensor) -> torch.Tensor:
        """Evaluates At-Least-One (OR) probability: P(OR) = 1 - prod_i (1 - t_i)."""
        K = truth.numel()
        if K == 0:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

        p_not = 1.0 - truth
        prod_not = torch.prod(p_not)
        p_or = 1.0 - prod_not
        return -torch.log(p_or.clamp(min=self.eps))

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
        truth: torch.Tensor,
        logits: torch.Tensor | None,
        group_ids: torch.Tensor,
        task_scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Routes task-conditioned logical constraints across candidate groups."""
        unique_groups = torch.unique(group_ids)
        task_losses: list[torch.Tensor] = []

        for gid in unique_groups:
            mask = group_ids == gid
            if task_scalars.shape[0] == group_ids.shape[0]:
                g_task = task_scalars[mask][0].item()
            else:
                g_task = task_scalars[gid].item()

            g_truth = truth[mask]
            g_logits = logits[mask] if logits is not None else None

            # Route by task scalar:
            # 1. Choice Task (scalar > 0.75): Exactly-One (XOR)
            if g_task > 0.75:
                task_losses.append(self._compute_xor(g_truth))
            # 2. Multi-Choice Task (0.25 < scalar <= 0.75): At-Least-One (OR)
            elif 0.25 < g_task <= 0.75:
                task_losses.append(self._compute_or(g_truth))
            # 3. Score / Null Task (scalar <= 0.25): Operational Box Boundedness
            else:
                if g_logits is not None:
                    bound_loss = self._compute_boundedness(g_logits) * self.bound_weight
                    task_losses.append(bound_loss)

        if task_losses:
            return torch.stack(task_losses).mean()
        return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

    def _compute_legacy_constraints(
        self,
        truth: torch.Tensor,
        logits: torch.Tensor | None,
        task_scalars: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Routes task-conditioned logical constraints across legacy batch rows."""
        batch_size = truth.size(0)
        row_losses: list[torch.Tensor] = []

        for b in range(batch_size):
            task_val = task_scalars[b].item()
            row_truth = truth[b]
            if active_mask is not None:
                mask_b = active_mask[b]
                row_truth = row_truth[mask_b]
                row_logits = logits[b][mask_b] if logits is not None else None
            else:
                row_logits = logits[b] if logits is not None else None

            # 1. Choice Task (scalar > 0.75): Exactly-One (XOR)
            if task_val > 0.75:
                row_losses.append(self._compute_xor(row_truth))
            # 2. Multi-Choice Task (0.25 < scalar <= 0.75): At-Least-One (OR)
            elif 0.25 < task_val <= 0.75:
                row_losses.append(self._compute_or(row_truth))
            # 3. Score / Null Task (scalar <= 0.25): Operational Box Boundedness
            else:
                if row_logits is not None:
                    bound_loss = self._compute_boundedness(row_logits) * self.bound_weight
                    row_losses.append(bound_loss)

        if row_losses:
            return torch.stack(row_losses).mean()
        return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

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
            outputs: Model outputs containing `choice_pos`, `choice_neg`, `truth`,
                and optionally `knowledge` and `logits`.
            task_scalars: Optional tensor indicating task category/mode.
            active_mask: Optional boolean tensor for valid criteria masking.
            group_ids: Optional grouping tensor for unrolled independent evaluation.
            hierarchy_edges: Optional list of (child_idx, parent_idx) index tuples.

        Returns:
            Scalar tensor loss combining epistemic regularization and task constraints.
        """
        choice_pos = outputs["choice_pos"]
        choice_neg = outputs["choice_neg"]
        truth = outputs["truth"]
        logits = outputs.get("logits")

        knowledge = outputs.get("knowledge", choice_pos * choice_neg)
        ignorance = (1.0 - choice_pos) * (1.0 - choice_neg)

        # Check hierarchy edges in arguments or outputs dictionary
        h_edges = hierarchy_edges
        if h_edges is None and "hierarchy_edges" in outputs:
            h_edges = outputs["hierarchy_edges"]

        # Evaluate hierarchy implication loss
        imp_loss = self._compute_hierarchy(truth, h_edges) * self.hierarchy_weight

        # Grouped candidate evaluation path
        if group_ids is not None:
            gullibility_penalty = knowledge.mean() * self.gullibility_weight
            ignorance_penalty = ignorance.mean() * self.ignorance_weight
            epistemic_loss = gullibility_penalty + ignorance_penalty

            task_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
            if task_scalars is not None:
                task_loss = self._compute_grouped_constraints(
                    truth=truth,
                    logits=logits,
                    group_ids=group_ids,
                    task_scalars=task_scalars,
                )
            return epistemic_loss + task_loss + imp_loss

        # Legacy fixed-dimension evaluation path
        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None
        if mask_f is not None:
            denom = mask_f.sum().clamp(min=1.0)
            gullibility_penalty = ((knowledge * mask_f).sum() / denom) * self.gullibility_weight
            ignorance_penalty = ((ignorance * mask_f).sum() / denom) * self.ignorance_weight
        else:
            gullibility_penalty = knowledge.mean() * self.gullibility_weight
            ignorance_penalty = ignorance.mean() * self.ignorance_weight
        epistemic_loss = gullibility_penalty + ignorance_penalty

        task_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            task_loss = self._compute_legacy_constraints(
                truth=truth,
                logits=logits,
                task_scalars=task_scalars,
                active_mask=active_mask,
            )

        return epistemic_loss + task_loss + imp_loss
