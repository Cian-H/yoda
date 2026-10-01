"""Tests for verifying package imports across all submodules in yoda."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "yoda",
        "yoda.architecture",
        "yoda.architecture.encoders",
        "yoda.architecture.schema",
        "yoda.nesy",
        "yoda.probabilistic",
        "yoda.training",
        "yoda.xai",
    ],
)
def test_submodule_imports(module_name: str) -> None:
    """Verify that every core submodule can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_public_api_reexport() -> None:
    """Verify that root yoda module re-exports primary entry points."""
    import yoda

    assert hasattr(yoda, "DecisionPayload")
    assert hasattr(yoda, "QueryContext")
