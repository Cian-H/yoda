"""Categorical Focal Loss, Active-Mask Margin Loss, and Focal-Margin hybrid."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "FocalLoss",
    "FocalMarginLoss",
    "MarginLoss",
]


class FocalLoss(nn.Module):
    """Categorical Focal Loss for multi-class classification.

    FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the correct target class.
    Downweights easy examples to focus gradients on ambiguous candidates.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initializes FocalLoss.

        Args:
            gamma: Focusing exponent parameter. gamma=0 is equivalent to CrossEntropy.
            reduction: Reduction mode ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        logger.debug(
            "training.losses.focal_init",
            extra={"gamma": self.gamma, "reduction": self.reduction},
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes categorical focal loss.

        Args:
            logits: Class logits of shape (batch_size, num_classes) or flat (N,).
            target: Ground truth class indices or binary indicators.
            group_ids: Optional grouping tensor of shape (N,) for grouped candidate evaluation.

        Returns:
            Computed focal loss scalar or tensor depending on reduction mode.
        """
        if group_ids is not None:
            unique_groups = torch.unique(group_ids)
            group_losses: list[torch.Tensor] = []
            for gid in unique_groups:
                mask = group_ids == gid
                g_logits = logits[mask]
                g_target = target[mask]
                if g_target.dim() > 0 and g_target.shape == g_logits.shape:
                    target_idx = torch.argmax(g_target)
                else:
                    target_idx = g_target
                ce_loss = F.cross_entropy(
                    g_logits.unsqueeze(0), target_idx.unsqueeze(0), reduction="none"
                )
                pt = torch.exp(-ce_loss)
                fl = ((1.0 - pt) ** self.gamma) * ce_loss
                group_losses.append(fl.squeeze(0))
            if not group_losses:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            stacked = torch.stack(group_losses)
            if self.reduction == "mean":
                return stacked.mean()
            if self.reduction == "sum":
                return stacked.sum()
            return stacked

        ce_loss = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MarginLoss(nn.Module):
    """Multi-class margin ranking loss supporting active choice masks and grouped candidates.

    L_margin = (1 / (|A| - 1)) * sum_{j in A, j != y} relu(margin - (z_y - z_j))
    where A is the set of active choices, y is the target choice index,
    and z is the logit vector.
    """

    def __init__(self, margin: float = 0.2, reduction: str = "mean") -> None:
        """Initializes MarginLoss.

        Args:
            margin: Desired minimum score separation between target and competitors.
            reduction: Reduction mode ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

        logger.debug(
            "training.losses.margin_init",
            extra={"margin": self.margin, "reduction": self.reduction},
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes margin loss over active candidates or candidate groups.

        Args:
            logits: Class logits of shape (batch_size, num_classes) or flat (N,).
            target: Ground truth indices or binary indicators.
            active_mask: Optional boolean mask of shape (batch_size, num_classes).
            group_ids: Optional grouping tensor of shape (N,) for grouped candidate evaluation.

        Returns:
            Computed margin ranking loss scalar or tensor.
        """
        if group_ids is not None:
            unique_groups = torch.unique(group_ids)
            group_losses: list[torch.Tensor] = []
            for gid in unique_groups:
                mask = group_ids == gid
                g_logits = logits[mask]
                g_target = target[mask]
                if g_target.dim() > 0 and g_target.shape == g_logits.shape:
                    target_idx = torch.argmax(g_target)
                else:
                    target_idx = g_target
                target_val = g_logits[target_idx]
                competitors = torch.cat([g_logits[:target_idx], g_logits[target_idx + 1:]])
                if competitors.numel() == 0:
                    group_losses.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))
                else:
                    violations = F.relu(self.margin - (target_val - competitors))
                    group_losses.append(violations.mean())
            if not group_losses:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            stacked = torch.stack(group_losses)
            if self.reduction == "mean":
                return stacked.mean()
            if self.reduction == "sum":
                return stacked.sum()
            return stacked

        _, num_classes = logits.shape
        target_logits = logits.gather(1, target.unsqueeze(1))  # (batch_size, 1)

        # Margin violation: relu(margin - (z_target - z_comp))
        violations = F.relu(self.margin - (target_logits - logits))

        # Zero out the target index itself
        violations = violations.scatter(1, target.unsqueeze(1), 0.0)

        # Zero out inactive/padded candidates if mask provided
        if active_mask is not None:
            violations = violations.masked_fill(~active_mask, 0.0)
            active_count = active_mask.sum(dim=-1, keepdim=True).float()
            num_competitors = (active_count - 1.0).clamp(min=1.0)
        else:
            num_competitors = torch.tensor(
                max(1.0, float(num_classes - 1)), device=logits.device, dtype=logits.dtype
            )

        sample_loss = violations.sum(dim=-1) / num_competitors.squeeze(-1)

        if self.reduction == "mean":
            return sample_loss.mean()
        if self.reduction == "sum":
            return sample_loss.sum()
        return sample_loss


class FocalMarginLoss(nn.Module):
    """Hybrid loss combining Categorical Focal Loss with Active-Mask Margin Loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        margin: float = 0.2,
        margin_weight: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        """Initializes FocalMarginLoss.

        Args:
            gamma: Focusing exponent for FocalLoss.
            margin: Target separation margin for MarginLoss.
            margin_weight: Weight coefficient for MarginLoss term.
            reduction: Reduction mode ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, reduction=reduction)
        self.margin_loss = MarginLoss(margin=margin, reduction=reduction)
        self.margin_weight = float(margin_weight)

        logger.debug(
            "training.losses.focal_margin_init",
            extra={
                "gamma": gamma,
                "margin": margin,
                "margin_weight": self.margin_weight,
            },
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes joint Focal and Margin loss.

        Args:
            logits: Class logits of shape (batch_size, num_classes) or flat (N,).
            target: Ground truth indices or binary indicators.
            active_mask: Optional boolean mask of shape (batch_size, num_classes).
            group_ids: Optional grouping tensor of shape (N,) for grouped candidate evaluation.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        focal = self.focal(logits, target, group_ids=group_ids)
        margin = self.margin_loss(logits, target, active_mask=active_mask, group_ids=group_ids)
        total = focal + self.margin_weight * margin

        return total, {
            "focal_loss": focal,
            "margin_loss": margin,
        }

