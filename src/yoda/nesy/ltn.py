import torch
from torch import nn


class LTNConstraintLoss(nn.Module):
    """Logic Tensor Network (LTN) constraints for Yoda Decision Engine."""

    def __init__(self):
        super().__init__()

    def _compute_grouped_xor(
        self,
        truth: torch.Tensor,
        group_ids: torch.Tensor,
        task_scalars: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Computes XOR semantic loss across candidate groups."""
        unique_groups = torch.unique(group_ids)
        group_xor_losses: list[torch.Tensor] = []
        for gid in unique_groups:
            mask = group_ids == gid
            if task_scalars.shape[0] == group_ids.shape[0]:
                g_task = task_scalars[mask][0].item()
            else:
                g_task = task_scalars[gid].item()

            if g_task > 0.5:
                g_truth = truth[mask]
                K = g_truth.numel()
                p_not = 1.0 - g_truth
                p_xor = torch.tensor(0.0, device=device, dtype=dtype)
                for i in range(K):
                    term = g_truth[i]
                    for j in range(K):
                        if i != j:
                            term = term * p_not[j]
                    p_xor = p_xor + term
                group_xor_losses.append(-torch.log(p_xor + 1e-7))

        if group_xor_losses:
            return torch.stack(group_xor_losses).mean()
        return torch.tensor(0.0, device=device, dtype=dtype)

    def _compute_legacy_xor(
        self,
        truth: torch.Tensor,
        task_scalars: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Computes legacy XOR semantic loss across fixed choice dimensions."""
        if active_mask is not None:
            masked_truth = torch.where(active_mask, truth, torch.zeros_like(truth))
        else:
            masked_truth = truth

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
        return (xor_loss_per_batch * task_mask).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        task_scalars: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the LTN constraint loss."""
        choice_pos = outputs["choice_pos"]
        choice_neg = outputs["choice_neg"]

        # Universal Complementarity Loss (Bivalence)
        prob_comp = choice_pos * (1.0 - choice_neg) + choice_neg * (1.0 - choice_pos)
        log_comp = -torch.log(prob_comp + 1e-7)

        if group_ids is not None:
            nc_loss = log_comp.mean()
            me_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
            if task_scalars is not None:
                me_loss = self._compute_grouped_xor(
                    outputs["truth"], group_ids, task_scalars, choice_pos.device, choice_pos.dtype
                )
            return nc_loss + me_loss

        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None
        if mask_f is not None:
            nc_loss = (log_comp * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        else:
            nc_loss = log_comp.mean()

        me_loss = torch.tensor(0.0, device=choice_pos.device, dtype=choice_pos.dtype)
        if task_scalars is not None:
            me_loss = self._compute_legacy_xor(outputs["truth"], task_scalars, active_mask)

        return nc_loss + me_loss


