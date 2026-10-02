"""Unit tests for SoftExp and custom activation functions."""

import torch

from yoda.architecture.activations import SoftExp


class TestSoftExp:
    """Test suite for SoftExp activation function."""

    def test_initialization_scalar(self) -> None:
        """Verifies scalar alpha initialization."""
        act = SoftExp()
        assert act.alpha.shape == ()
        assert act.alpha.item() == 0.0
        assert act.alpha.requires_grad is True

    def test_initialization_vector(self) -> None:
        """Verifies per-channel vector alpha initialization."""
        dim = 16
        act = SoftExp(in_features=dim, init_alpha=0.1)
        assert act.alpha.shape == (dim,)
        assert torch.allclose(act.alpha, torch.full((dim,), 0.1))

    def test_gradient_alive_at_zero(self) -> None:
        """Verifies that alpha receives non-zero gradients at alpha=0.0."""
        act = SoftExp(init_alpha=0.0)
        x = torch.randn(10, requires_grad=True)
        out = act(x)
        loss = out.sum()
        loss.backward()

        assert act.alpha.grad is not None
        # d/d(alpha) of (x + 0.5 * alpha * x^2) is 0.5 * x^2 >= 0
        expected_grad = 0.5 * (x**2).sum()
        assert torch.allclose(act.alpha.grad, expected_grad, rtol=1e-4)

    def test_near_zero_smooth_continuity(self) -> None:
        """Verifies continuity across the eps boundary around alpha=0."""
        x = torch.tensor([0.5, -0.5, 1.0])
        act_zero = SoftExp(init_alpha=0.0)
        act_pos_eps = SoftExp(init_alpha=1.5e-4)
        act_neg_eps = SoftExp(init_alpha=-1.5e-4)

        y_zero = act_zero(x)
        y_pos = act_pos_eps(x)
        y_neg = act_neg_eps(x)

        assert torch.allclose(y_zero, x, atol=1e-5)
        assert torch.allclose(y_pos, y_zero, atol=1e-3)
        assert torch.allclose(y_neg, y_zero, atol=1e-3)

    def test_logarithmic_branch_nan_safe(self) -> None:
        """Verifies negative alpha does not produce NaNs even on challenging negative inputs."""
        act = SoftExp(init_alpha=-0.5)
        # Without clamp, x=-10 with alpha=-0.5 would cause 1 - a*(x+a) <= 0
        x_extreme = torch.tensor([-100.0, -10.0, -2.0, 0.0, 5.0, 50.0])
        out = act(x_extreme)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_exponential_branch_overflow_safe(self) -> None:
        """Verifies positive alpha does not overflow to inf on large positive inputs."""
        act = SoftExp(init_alpha=0.5)
        x_extreme = torch.tensor([10.0, 50.0, 100.0])
        out = act(x_extreme)

        assert not torch.isinf(out).any()
        assert not torch.isnan(out).any()

    def test_method_decomposition(self) -> None:
        """Verifies direct isolation of modular mathematical branch methods."""
        act = SoftExp(init_alpha=0.0)
        x = torch.tensor([1.0, 2.0])
        a = torch.tensor(0.0)

        # Taylor
        y_taylor = act._linear_taylor(x, a)
        assert torch.allclose(y_taylor, x)

        # Log branch
        a_neg = torch.tensor(-0.2)
        y_log = act._logarithmic_branch(x, a_neg)
        assert not torch.isnan(y_log).any()

        # Exp branch
        a_pos = torch.tensor(0.2)
        y_exp = act._exponential_branch(x, a_pos)
        assert not torch.isinf(y_exp).any()
