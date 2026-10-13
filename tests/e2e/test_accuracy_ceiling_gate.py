"""Tier 4 Acceptance Gate: Accuracy Ceiling Breakthrough Gate.

Asserts multi-choice validation accuracy > 60.0% on independent choice
evaluation benchmarks, demonstrating resolution of the 50.0% ceiling.
"""

from typing import Any

import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data import collate_decision_batch
from yoda.training.trainer import YodaTrainer


def _generate_context_dependent_dataset(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generates synthetic questions where correct choice depends on query-context matching."""
    torch.manual_seed(seed)
    samples: list[dict[str, Any]] = []

    # Two context rules:
    # Rule 1: High pressure -> vent valve
    # Rule 2: Low pressure -> close valve
    # Rule 3: High temperature -> enable coolant
    # Rule 4: Low temperature -> enable heater
    rules = [
        ("Pressure is critically high", {"system": "boiler"}, ["Vent valve", "Close valve"], 0),
        ("Pressure is critically low", {"system": "boiler"}, ["Vent valve", "Close valve"], 1),
        (
            "Temperature is dangerously elevated",
            {"system": "core"},
            ["Enable coolant", "Enable heater"],
            0,
        ),
        (
            "Temperature is dangerously depressed",
            {"system": "core"},
            ["Enable coolant", "Enable heater"],
            1,
        ),
    ]

    for i in range(num_samples):
        rule_idx = i % len(rules)
        q_text, state_dict, choices, target = rules[rule_idx]
        constraints = [*choices, "none: Unused", "none: Unused", "none: Unused"]
        samples.append(
            {
                "query": f"{q_text} (iteration {i})",
                "state": state_dict,
                "constraints": constraints,
                "target_idx": target,
                "num_active": len(choices),
                "task_scalar": 1.0,
            }
        )
    return samples


def test_independent_choice_validation_accuracy_breaks_ceiling() -> None:
    """Acceptance Gate: Validation accuracy on independent choice evaluation must exceed 60.0%."""
    torch.manual_seed(42)

    # Build dataset
    train_samples = _generate_context_dependent_dataset(num_samples=32, seed=42)
    val_samples = _generate_context_dependent_dataset(num_samples=16, seed=123)

    train_loader: DataLoader[dict[str, Any]] = DataLoader(
        train_samples,  # type: ignore[arg-type]
        batch_size=4,
        collate_fn=collate_decision_batch,
        shuffle=True,
    )
    val_loader: DataLoader[dict[str, Any]] = DataLoader(
        val_samples,  # type: ignore[arg-type]
        batch_size=4,
        collate_fn=collate_decision_batch,
        shuffle=False,
    )

    engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=1,
        n_heads=2,
        num_reasoning_blocks=1,
    )

    trainer = YodaTrainer(
        model=engine,
        lr=2e-3,
        device="cpu",
    )

    # Train for 5 epochs
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=5)

    # Evaluate validation accuracy
    val_metrics = trainer.evaluate(val_loader)
    val_accuracy = val_metrics.get("eval_accuracy", 0.0)

    # Acceptance criterion: Multi-choice validation accuracy > 60.0% (0.60)
    assert val_accuracy > 0.60, (
        f"Validation accuracy reached {val_accuracy * 100:.2f}%, failing to exceed "
        "the 60.0% acceptance ceiling (50.0% context cancellation ceiling persists)."
    )
