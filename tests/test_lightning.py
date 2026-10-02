"""Unit and integration tests for YodaLightningAdapter."""

from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from yoda.training.lightning import YodaLightningAdapter


class DummyDecisionEngine(nn.Module):
    """Minimal dummy decision engine for fast Lightning testing."""

    def __init__(self, embed_dim: int = 16, num_choices: int = 5) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_choices)

    def forward(
        self,
        queries: list[str],
        states: list[dict[str, Any]],
        constraints: list[list[str]],
        task_scalars: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        batch_size = len(queries)
        dev = self.linear.weight.device
        dummy_feat = torch.randn(batch_size, 16, device=dev)
        logits = self.linear(dummy_feat)
        truth = torch.sigmoid(logits)
        knowledge = torch.full_like(logits, 0.5)

        return {
            "logits": logits,
            "truth": truth,
            "knowledge": knowledge,
            "choice": logits.argmax(dim=-1),
            "choice_pos": truth,
            "choice_neg": 1.0 - truth,
        }


@pytest.fixture
def synthetic_batch() -> dict[str, Any]:
    return {
        "queries": ["Is this safe?", "Should we proceed?"],
        "states": [{"flag": True}, {"flag": False}],
        "constraints": [["opt_0", "none:opt_1"], ["opt_0", "opt_1"]],
        "target_indices": torch.tensor([0, 1]),
        "task_scalars": torch.tensor([[1.0], [1.0]]),
    }


def test_lightning_adapter_initialization_and_decoupling() -> None:
    engine: Any = DummyDecisionEngine()
    adapter = YodaLightningAdapter(model=engine, lr=2e-3, belnap_weight=0.2, ltn_weight=0.1)

    # Core decoupling invariant: the underlying model is a pure PyTorch nn.Module
    assert adapter.model is engine
    assert isinstance(adapter.model, nn.Module)
    assert not isinstance(engine, pl.LightningModule)


def test_lightning_adapter_steps_and_gradients(synthetic_batch: dict[str, Any]) -> None:
    engine: Any = DummyDecisionEngine()
    adapter = YodaLightningAdapter(model=engine, lr=1e-3)

    train_loss = adapter.training_step(synthetic_batch, batch_idx=0)
    assert isinstance(train_loss, torch.Tensor)
    assert train_loss.ndim == 0
    assert train_loss.item() > 0.0

    val_loss = adapter.validation_step(synthetic_batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)
    assert val_loss.ndim == 0
    assert val_loss.item() > 0.0

    optimizer = adapter.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_lightning_adapter_fast_dev_run(synthetic_batch: dict[str, Any]) -> None:
    engine: Any = DummyDecisionEngine()
    adapter = YodaLightningAdapter(model=engine, lr=1e-3)

    loader = DataLoader([synthetic_batch, synthetic_batch], batch_size=1, collate_fn=lambda x: x[0])
    trainer = pl.Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model=adapter, train_dataloaders=loader, val_dataloaders=loader)
