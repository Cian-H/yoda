"""Tests for Multihead Pooled Attention (MPA)."""

import pytest
import torch

from yoda.architecture.attention import MultiheadPooledAttention


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "embed_dim", "num_queries"),
    [
        (2, 16, 8, 4),
        (1, 32, 16, 2),
        (4, 10, 32, 1),
    ],
)
def test_multihead_pooled_attention_forward_shape(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_queries: int,
) -> None:
    """Verifies that MPA aggregates variable sequence lengths to fixed [B, embed_dim]."""
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim)
    model = MultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)

    out = model(x)

    assert out.shape == (batch_size, embed_dim)
    assert out.dtype.is_floating_point
    assert torch.isfinite(out).all()


def test_attention_hookpoint_recording() -> None:
    """Verifies that the attention matrix hookpoint captures attention maps."""
    torch.manual_seed(42)
    x = torch.randn(2, 12, 16)
    model = MultiheadPooledAttention(embed_dim=16, num_queries=3, num_heads=2)

    captured_weights: list[torch.Tensor] = []

    def hook_fn(
        _module: torch.nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured_weights.append(output)

    hook_handle = model.attn_matrix_hookpoint.register_forward_hook(hook_fn)
    try:
        _ = model(x)
        assert len(captured_weights) == 1
        # Shape should be [Batch, Heads, Num_queries, Seq_len]
        assert captured_weights[0].shape == (2, 2, 3, 12)
    finally:
        hook_handle.remove()


@pytest.mark.parametrize("activation", ["softmax", "linear", "relational", "time-series"])
def test_activation_registry_resolutions(activation: str) -> None:
    """Verifies that string-based activations in the registry execute cleanly."""
    x = torch.randn(2, 8, 16)
    model = MultiheadPooledAttention(embed_dim=16, num_queries=2, attn_activation=activation)
    out = model(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
