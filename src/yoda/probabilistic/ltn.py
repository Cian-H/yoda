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

        # 1. Universal Complementarity Loss (Bivalence)
        # Prevents apathy (both 0) and contradiction (both 1).
        # -log( e+*(1 - e-) + e-*(1 - e+) )
        prob_comp = choice_pos * (1.0 - choice_neg) + choice_neg * (1.0 - choice_pos)
        nc_loss = -torch.log(prob_comp + 1e-7).mean()

        me_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            truth = outputs["truth"]
            # 2. Exactly-One (XOR) Semantic Loss for Choice tasks
            # P(XOR) = sum_i [ t_i * prod_{j != i} (1 - t_j) ]
            N = truth.size(1)
            p_not = 1.0 - truth
            p_xor = torch.zeros_like(truth[:, 0])
            
            for i in range(N):
                term = truth[:, i].clone()
                for j in range(N):
                    if i != j:
                        term = term * p_not[:, j]
                p_xor = p_xor + term

            xor_loss_per_batch = -torch.log(p_xor + 1e-7)

            task_mask = (task_scalars > 0.5).float().squeeze(-1)
            me_loss = (xor_loss_per_batch * task_mask).mean()

        return nc_loss + me_loss
