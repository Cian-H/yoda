import torch
from torch import nn


class LTNConstraintLoss(nn.Module):
    """Logic Tensor Network (LTN) constraints for Yoda Decision Engine."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        task_scalars: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the LTN constraint loss.

        Args:
            outputs: Dictionary containing 'choice_pos', 'choice_neg', and 'truth'.
            task_scalars: Optional tensor of shape (batch_size, 1) indicating task type.
                          > 0.5 means it's a multiple choice task.
            active_mask: Optional boolean tensor of shape (batch_size, num_choices)
                         indicating valid active candidates.

        Returns:
            Scalar tensor containing the total LTN constraint loss.
        """
        choice_pos = outputs["choice_pos"]
        choice_neg = outputs["choice_neg"]

        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None

        # 1. Universal Complementarity Loss (Bivalence)
        # Prevents apathy (both 0) and contradiction (both 1).
        # -log( e+*(1 - e-) + e-*(1 - e+) )
        prob_comp = choice_pos * (1.0 - choice_neg) + choice_neg * (1.0 - choice_pos)
        log_comp = -torch.log(prob_comp + 1e-7)

        if mask_f is not None:
            nc_loss = (log_comp * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        else:
            nc_loss = log_comp.mean()

        me_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            truth = outputs["truth"]
            if active_mask is not None:
                # Mask out inactive truth values so they contribute 0 to XOR and 1 to (1 - t)
                masked_truth = torch.where(active_mask, truth, torch.zeros_like(truth))
            else:
                masked_truth = truth

            # 2. Exactly-One (XOR) Semantic Loss for Choice tasks
            # P(XOR) = sum_{i in Active} [ t_i * prod_{j in Active, j != i} (1 - t_j) ]
            N = masked_truth.size(1)
            p_not = 1.0 - masked_truth
            p_xor = torch.zeros_like(masked_truth[:, 0])

            for i in range(N):
                term = masked_truth[:, i].clone()
                for j in range(N):
                    if i != j:
                        term = term * p_not[:, j]
                p_xor = p_xor + term

            xor_loss_per_batch = -torch.log(p_xor + 1e-7)

            task_mask = (task_scalars > 0.5).float().squeeze(-1)
            me_loss = (xor_loss_per_batch * task_mask).mean()

        return nc_loss + me_loss
