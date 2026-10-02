"""Tests for Phase 1 encoders (TextEncoder, SymbolicStateEncoder, ConstraintEncoder)."""

from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from yoda.architecture.encoders import (
    ConstraintEncoder,
    SymbolicStateEncoder,
    TextEncoder,
)


def test_text_encoder_dummy_default_shape() -> None:
    """Verifies dummy TextEncoder returns expected tensor shape with default dimension."""
    encoder = TextEncoder("dummy")
    texts = ["hello world", "test sample foo bar"]

    out = encoder(texts)

    assert isinstance(out, torch.Tensor)
    # 2 sentences; maximum word count is 4
    assert out.shape == (2, 4, 256)
    assert out.dtype.is_floating_point
    assert torch.isfinite(out).all()


def test_text_encoder_dummy_custom_target_dim() -> None:
    """Verifies dummy TextEncoder respects target_dim parameter."""
    target_dim = 64
    encoder = TextEncoder("dummy", target_dim=target_dim)
    texts = ["a b c"]

    out = encoder(texts)

    assert out.shape == (1, 3, target_dim)


def test_text_encoder_dummy_empty_and_whitespace_texts() -> None:
    """Verifies dummy TextEncoder handles empty strings, whitespace, and empty lists."""
    encoder = TextEncoder("dummy", target_dim=32)

    # Empty list of texts
    empty_batch_out = encoder([])
    assert empty_batch_out.shape == (0, 0, 32)

    # Empty string and whitespace
    out = encoder(["", "   "])
    assert out.shape == (2, 1, 32)


def test_text_encoder_hf_model_with_adapter() -> None:
    """Verifies HF integration and linear adapter when target_dim != hidden_size."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.config.hidden_size = 384

    # Setup tokenizer return dict
    mock_inputs = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    mock_tokenizer.return_value.to.return_value = mock_inputs

    # Setup model output
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 5, 384)
    mock_model.return_value = mock_output
    mock_model.to.return_value = mock_model

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        encoder = TextEncoder(model_name="mock-model", target_dim=128, device=torch.device("cpu"))
        assert isinstance(encoder.adapter, nn.Linear)
        assert encoder.adapter.in_features == 384
        assert encoder.adapter.out_features == 128

        out = encoder(["first test", "second test"])
        assert out.shape == (2, 5, 128)


def test_text_encoder_hf_model_identity_adapter() -> None:
    """Verifies Identity adapter when target_dim equals hidden_size or is None."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.config.hidden_size = 256
    mock_model.to.return_value = mock_model

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        encoder = TextEncoder(model_name="mock-model", target_dim=256)
        assert isinstance(encoder.adapter, nn.Identity)


def test_symbolic_state_encoder_formatting() -> None:
    """Verifies formatting of dictionary states into strings."""
    encoder = SymbolicStateEncoder(TextEncoder("dummy"))

    empty_formatted = encoder._format_dict({})
    assert empty_formatted == "empty"

    single_formatted = encoder._format_dict({"key1": "value1"})
    assert single_formatted == "key1: value1"

    multi_formatted = encoder._format_dict({"temp": 25.0, "status": "nominal"})
    assert multi_formatted == "temp: 25.0 | status: nominal"


def test_symbolic_state_encoder_forward() -> None:
    """Verifies SymbolicStateEncoder encodes a batch of dict states into a tensor."""
    encoder = SymbolicStateEncoder(TextEncoder("dummy", target_dim=128))
    states: list[dict[str, Any]] = [
        {"x": 1, "y": 2},
        {"status": "active"},
        {},
    ]

    out = encoder(states)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.shape[2] == 128
    assert torch.isfinite(out).all()


def test_constraint_encoder_formatting() -> None:
    """Verifies formatting of constraint lists into strings."""
    encoder = ConstraintEncoder(TextEncoder("dummy"))

    empty_formatted = encoder._format_constraints([])
    assert empty_formatted == "none"

    single_formatted = encoder._format_constraints(["x > 0"])
    assert single_formatted == "x > 0"

    multi_formatted = encoder._format_constraints(["x > 0", "y < 10", "z == 1"])
    assert multi_formatted == "x > 0 || y < 10 || z == 1"


def test_constraint_encoder_forward() -> None:
    """Verifies ConstraintEncoder encodes batches of constraints into a tensor."""
    encoder = ConstraintEncoder(TextEncoder("dummy", target_dim=64))
    constraints_batch: list[list[str]] = [
        ["temperature <= 100", "pressure >= 1.0"],
        ["speed < 50"],
        [],
    ]

    out = encoder(constraints_batch)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.shape[2] == 64
    assert torch.isfinite(out).all()


def test_architecture_package_exports() -> None:
    """Verifies that encoders are exported from the top-level yoda.architecture package."""
    import yoda.architecture as arch

    assert hasattr(arch, "TextEncoder")
    assert hasattr(arch, "SymbolicStateEncoder")
    assert hasattr(arch, "ConstraintEncoder")
