"""Tier 4 Acceptance Gate: Gradient Health Gate.

Asserts average gradient norms of reasoning_layers[0] > 1e-4 throughout
Epoch 1 backpropagation across 10 random initialization seeds.
"""

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data import collate_decision_batch
from yoda.training.trainer import YodaTrainer


def _create_synthetic_training_dataloader(
    seed: int, num_samples: int = 16, batch_size: int = 4
) -> DataLoader[dict[str, Any]]:
    """Creates a deterministic synthetic dataloader for multi-choice training."""
    torch.manual_seed(seed)
    samples = []
    for i in range(num_samples):
        target = i % 3
        samples.append(
            {
                "query": f"Evaluate environmental condition {i}",
                "state": {"temperature": 20 + i, "status": "active"},
                "constraints": [
                    f"Action Option 0: procedure {i}",
                    f"Action Option 1: procedure {i + 1}",
                    f"Action Option 2: procedure {i + 2}",
                    "none: Unused",
                    "none: Unused",
                ],
                "target_idx": target,
                "num_active": 3,
                "task_scalar": 1.0,
            }
        )
    return DataLoader(
        samples,  # type: ignore[arg-type]
        batch_size=batch_size,
        collate_fn=collate_decision_batch,
        shuffle=False,
    )


def compute_layer_grad_norm(module: torch.nn.Module) -> float:
    """Computes L2 norm of gradients across all parameters of a module."""
    total_norm_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm**2
    return total_norm_sq**0.5


@pytest.mark.parametrize("seed", list(range(10)))
def test_reasoning_layer_0_gradient_health_throughout_epoch_1(seed: int) -> None:
    """Acceptance Gate: Reasoning layer 0 gradient norms must stay > 1e-4 across 10 seeds."""
    torch.manual_seed(seed)

    engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=1,
        n_heads=2,
        num_reasoning_blocks=2,
    )

    trainer = YodaTrainer(
        model=engine,
        lr=1e-3,
        belnap_weight=0.1,
        ltn_weight=0.1,
        device="cpu",
    )

    dataloader = _create_synthetic_training_dataloader(seed=seed, num_samples=16, batch_size=4)

    engine.train()
    grad_norms: list[float] = []

    for batch in dataloader:
        trainer.optimizer.zero_grad()
        loss, _ = trainer._compute_loss_and_metrics(batch)
        loss.backward()

        # Measure gradient norm of reasoning_layers[0]
        layer0 = engine.reasoning_layers[0]
        norm = compute_layer_grad_norm(layer0)
        grad_norms.append(norm)

        trainer.optimizer.step()

    assert len(grad_norms) > 0, "Dataloader produced no batches"
    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    # Acceptance criterion: Average gradient norm strictly > 1e-4 throughout Epoch 1 backprop
    assert avg_grad_norm > 1e-4, (
        f"Seed {seed}: Epoch 1 reasoning_layers[0] gradient norm collapsed to {avg_grad_norm:.6e} "
        "(must remain > 1e-4 throughout Epoch 1 backprop)"
    )
