import torch
from torch import nn


class LTNConstraintLoss(nn.Module):
    """Logic Tensor Network (LTN) constraints for Yoda Decision Engine."""

    def __init__(self):
        super().__init__()

    def forward(
        self, outputs: dict[str, torch.Tensor], task_scalars: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Computes the LTN constraint loss.

        Args:
            outputs: Dictionary containing 'choice_pos', 'choice_neg', and 'truth'.
            task_scalars: Optional tensor of shape (batch_size, 1) indicating task type.
                          > 0.5 means it's a multiple choice task.

        Returns:
            Scalar tensor containing the total LTN constraint loss.
        """
        choice_pos = outputs["choice_pos"]
        choice_neg = outputs["choice_neg"]

        # Non-Contradiction Loss: shouldn't strongly believe both positive and negative evidence.
        # T(e+, e-) = e+ * e-
        nc_loss = (choice_pos * choice_neg).mean()

        me_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            truth = outputs["truth"]
            # Mutual Exclusivity & Existential Constraint:
            # 1. Truths shouldn't heavily overlap: sum_{i!=j} t_i * t_j
            # 2. At least one choice MUST be true: sum(t_i) should be close to 1.0
            sum_t = truth.sum(dim=-1)
            sum_t_sq = (truth ** 2).sum(dim=-1)
            
            overlap_penalty = sum_t ** 2 - sum_t_sq
            existence_penalty = (sum_t - 1.0) ** 2
            
            me_loss_per_batch = overlap_penalty + existence_penalty

            task_mask = (task_scalars > 0.5).float().squeeze(-1)
            me_loss = (me_loss_per_batch * task_mask).mean()

        return nc_loss + me_loss
