"""Custom neural-symbolic activation functions."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SoftExp(nn.Module):
    """Numerically stable, vectorized Soft Exponential activation (Godfrey 2015).

    Smoothly interpolates:
    - Sub-linear / Logarithmic saturation (alpha < 0)
    - Linear identity (alpha -> 0)
    - Super-linear / Exponential amplification (alpha > 0)
    """

    def __init__(
        self,
        in_features: int | None = None,
        init_alpha: float = 0.0,
        eps: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes SoftExp.

        Args:
            in_features: Optional dimensionality for per-channel alpha. If None,
                shares a scalar alpha parameter.
            init_alpha: Initial alpha parameter value. Defaults to 0.0 (identity).
            eps: Epsilon threshold for smooth Taylor expansion near zero.
            device: Target execution device.
            dtype: Target execution data type.
        """
        super().__init__()
        self.eps = eps
        if in_features is not None:
            self.alpha = nn.Parameter(
                torch.full(
                    (in_features,),
                    init_alpha,
                    device=device,
                    dtype=dtype or torch.float32,
                )
            )
        else:
            self.alpha = nn.Parameter(
                torch.tensor(init_alpha, device=device, dtype=dtype or torch.float32)
            )

        logger.debug(
            "architecture.activations.softexp_init",
            extra={"in_features": in_features, "init_alpha": init_alpha},
        )

    def _linear_taylor(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Taylor expansion near zero (|alpha| < eps) guaranteeing non-zero gradients."""
        return x + 0.5 * a * (x**2)

    def _logarithmic_branch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Sub-linear logarithmic branch for alpha < -eps with NaN protection."""
        safe_a = torch.where(a < -self.eps, a, -torch.ones_like(a))
        arg = torch.clamp(1.0 - safe_a * (x + safe_a), min=1e-6)
        return -torch.log(arg) / safe_a

    def _exponential_branch(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Super-linear exponential branch for alpha > eps with overflow protection."""
        safe_a = torch.where(a > self.eps, a, torch.ones_like(a))
        scaled_x = torch.clamp(safe_a * x, max=20.0)
        return torch.expm1(scaled_x) / safe_a + safe_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes soft exponential activation elementwise across any input shape.

        Args:
            x: Input tensor of arbitrary shape `(*, D)` matching `in_features` or scalar.

        Returns:
            Transformed tensor of identical shape.
        """
        a = self.alpha
        near_zero = a.abs() < self.eps
        y_zero = self._linear_taylor(x, a)
        y_neg = self._logarithmic_branch(x, a)
        y_pos = self._exponential_branch(x, a)

        return torch.where(near_zero, y_zero, torch.where(a < 0.0, y_neg, y_pos))
