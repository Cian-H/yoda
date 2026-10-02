"""Tests for Yoda training pipeline, dataset loading, collation, and trainer."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.training import (
    YodaDecisionDataset,
    YodaTrainer,
    collate_decision_batch,
)


@pytest.fixture
def mock_jsonl_file(tmp_path: Path) -> Path:
    """Creates a temporary JSONL file with known items for testing dataset loading."""
    records = [
        # Valid item 1 (3 constraints, needs padding to 5)
        {
            "query": "Select the best option.",
            "context": {
                "symbolic_state": {"speed": 100, "status": "ok"},
            },
            "constraints": ["A1: First option", "A2: Second option", "A3: Third option"],
            "metadata": {
                "source": "n4ze3m_synth",
                "question_type": "choice",
                "target": "A2",
            },
        },
        # Valid item 2 (6 constraints, needs truncation to 5)
        {
            "query": "Select the fastest car.",
            "context": {
                "symbolic_state": {"weather": "clear"},
            },
            "constraints": [
                "C1: Option 1",
                "C2: Option 2",
                "C3: Option 3",
                "C4: Option 4",
                "C5: Option 5",
                "C6: Option 6",
            ],
            "metadata": {
                "source": "n4ze3m_synth",
                "question_type": "choice",
                "target": "C1",
            },
        },
        # Invalid item: source mismatch
        {
            "query": "Ignore this query.",
            "context": {"symbolic_state": {}},
            "constraints": ["X1: Opt 1", "X2: Opt 2"],
            "metadata": {
                "source": "nimble",
                "question_type": "choice",
                "target": "X1",
            },
        },
        # Invalid item: question_type mismatch
        {
            "query": "Is speed > 50?",
            "context": {"symbolic_state": {}},
            "constraints": ["true: Yes", "false: No"],
            "metadata": {
                "source": "n4ze3m_synth",
                "question_type": "noul",
                "target": True,
            },
        },
        # Invalid item: unmatchable target
        {
            "query": "Unmatchable target.",
            "context": {"symbolic_state": {}},
            "constraints": ["opt_a: A", "opt_b: B"],
            "metadata": {
                "source": "n4ze3m_synth",
                "question_type": "choice",
                "target": "opt_z",
            },
        },
        # Valid item 3 (exact string match on target)
        {
            "query": "Select exact match.",
            "context": {"symbolic_state": {"value": 42}},
            "constraints": ["Apple", "Banana", "Cherry"],
            "metadata": {
                "source": "n4ze3m_synth",
                "question_type": "choice",
                "target": "Banana",
            },
        },
    ]

    file_path = tmp_path / "test_data.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return file_path


@pytest.fixture
def dummy_engine() -> YodaDecisionEngine:
    """Creates a lightweight YodaDecisionEngine for fast testing."""
    torch.manual_seed(42)
    return YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=5,
        n_heads=2,
    )


class TestYodaDecisionDataset:
    """Verifies dataset loading, filtering, target matching, and constraint standardization."""

    def test_dataset_loading_and_filtering(self, mock_jsonl_file: Path) -> None:
        """Verifies filtering by source and question_type, plus matching and padding."""
        dataset = YodaDecisionDataset(
            file_path=mock_jsonl_file,
            source="n4ze3m_synth",
            question_type="choice",
            max_choices=5,
        )

        # Out of 6 records:
        # - Valid 1 (kept, target A2 -> idx 1)
        # - Valid 2 (kept, target C1 -> idx 0)
        # - Nimble source (filtered out)
        # - Noul question_type (filtered out)
        # - Unmatchable target (filtered out)
        # - Valid 3 (kept, target Banana -> idx 1)
        assert len(dataset) == 3

        # Check sample 0 (originally 3 constraints -> padded to 5)
        sample0 = dataset[0]
        assert sample0["query"] == "Select the best option."
        assert sample0["state"] == {"speed": 100, "status": "ok"}
        assert len(sample0["constraints"]) == 5
        assert sample0["constraints"][0] == "A1: First option"
        assert sample0["constraints"][1] == "A2: Second option"
        assert sample0["constraints"][2] == "A3: Third option"
        assert sample0["constraints"][3] == "none: Unused option"
        assert sample0["constraints"][4] == "none: Unused option"
        assert sample0["target_idx"] == 1

        # Check sample 1 (originally 6 constraints -> truncated to 5)
        sample1 = dataset[1]
        assert len(sample1["constraints"]) == 5
        assert sample1["target_idx"] == 0

        # Check sample 2 (exact string match)
        sample2 = dataset[2]
        assert sample2["target_idx"] == 1

    def test_dataset_max_samples(self, mock_jsonl_file: Path) -> None:
        """Verifies max_samples limits the loaded items."""
        dataset = YodaDecisionDataset(
            file_path=mock_jsonl_file,
            source="n4ze3m_synth",
            question_type="choice",
            max_samples=2,
        )
        assert len(dataset) == 2

    def test_dataset_choice_shuffling(self, mock_jsonl_file: Path) -> None:
        """Verifies shuffle_choices permutes active choices while preserving target mapping."""
        dataset_unshuffled = YodaDecisionDataset(
            file_path=mock_jsonl_file,
            source="n4ze3m_synth",
            question_type="choice",
            shuffle_choices=False,
        )
        dataset_shuffled = YodaDecisionDataset(
            file_path=mock_jsonl_file,
            source="n4ze3m_synth",
            question_type="choice",
            shuffle_choices=True,
        )

        # In unshuffled, sample 0 target is always at idx 1
        assert dataset_unshuffled[0]["target_idx"] == 1
        assert dataset_unshuffled[0]["constraints"][1] == "A2: Second option"

        # In shuffled, test multiple reads
        observed_indices = set()
        for _ in range(30):
            sample = dataset_shuffled[0]
            # Target choice string must always be the correct option!
            assert sample["constraints"][sample["target_idx"]] == "A2: Second option"
            # Padding must stay at the end
            assert sample["constraints"][3] == "none: Unused option"
            assert sample["constraints"][4] == "none: Unused option"
            observed_indices.add(sample["target_idx"])

        # Across 30 iterations, target should appear at multiple active positions (0, 1, 2)
        assert len(observed_indices) > 1
        assert observed_indices.issubset({0, 1, 2})

    def test_dataset_with_real_data(self) -> None:
        """Verifies dataset loading against data/processed/aggregated_eval.jsonl if present."""
        data_path = Path("data/processed/aggregated_eval.jsonl")
        if not data_path.exists():
            pytest.skip("aggregated_eval.jsonl not found")

        dataset = YodaDecisionDataset(
            file_path=data_path,
            source="n4ze3m_synth",
            question_type="choice",
            max_samples=20,
            max_choices=5,
        )
        assert len(dataset) == 20
        for i in range(len(dataset)):
            sample = dataset[i]
            assert isinstance(sample["query"], str)
            assert isinstance(sample["state"], dict)
            assert len(sample["constraints"]) == 5
            assert 0 <= sample["target_idx"] < 5


class TestCollateDecisionBatch:
    """Verifies collate_decision_batch aggregates list of samples into tensor batches."""

    def test_collate_shapes(self) -> None:
        batch_samples = [
            {
                "query": "Q1",
                "state": {"k1": 1},
                "constraints": ["c0", "c1", "c2", "c3", "c4"],
                "target_idx": 2,
            },
            {
                "query": "Q2",
                "state": {"k2": 2},
                "constraints": ["c0", "c1", "c2", "c3", "c4"],
                "target_idx": 0,
            },
        ]
        batch = collate_decision_batch(batch_samples)

        assert batch["queries"] == ["Q1", "Q2"]
        assert batch["states"] == [{"k1": 1}, {"k2": 2}]
        assert len(batch["constraints"]) == 2
        assert len(batch["constraints"][0]) == 5
        assert isinstance(batch["target_indices"], torch.Tensor)
        assert batch["target_indices"].shape == (2,)
        assert batch["target_indices"].dtype == torch.long
        assert batch["target_indices"].tolist() == [2, 0]
        assert "num_active" in batch
        assert "active_mask" in batch

    def test_collate_with_variable_criteria(self) -> None:
        """Verifies active_mask generation with variable criteria counts."""
        batch_samples = [
            {
                "query": "Q1",
                "state": {},
                "constraints": [
                    "c0",
                    "c1",
                    "none: Unused option",
                    "none: Unused option",
                    "none: Unused option",
                ],
                "target_idx": 1,
                "num_active": 2,
            },
            {
                "query": "Q2",
                "state": {},
                "constraints": ["c0", "c1", "c2", "c3", "none: Unused option"],
                "target_idx": 3,
                "num_active": 4,
            },
        ]
        batch = collate_decision_batch(batch_samples)
        assert batch["num_active"].tolist() == [2, 4]
        assert batch["active_mask"].shape == (2, 5)
        assert batch["active_mask"][0].tolist() == [True, True, False, False, False]
        assert batch["active_mask"][1].tolist() == [True, True, True, True, False]


class TestYodaTrainer:
    """Verifies YodaTrainer training loop, loss calculation, Belnap regularization, and eval."""

    @pytest.fixture
    def synthetic_loader(self) -> DataLoader[dict[str, Any]]:
        samples = [
            {
                "query": f"Query {i}",
                "state": {"sensor": i * 10},
                "constraints": [f"c{j}: Option {j}" for j in range(5)],
                "target_idx": i % 5,
            }
            for i in range(10)
        ]
        return DataLoader(samples, batch_size=4, shuffle=False, collate_fn=collate_decision_batch)

    def test_trainer_step_and_loss_decrease(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies that running training epochs decreases loss and computes gradients."""
        trainer = YodaTrainer(
            model=dummy_engine,
            lr=0.005,
            belnap_weight=0.2,
            ltn_weight=0.0,  # Disable LTN penalty to ensure simple CE convergence in 5 epochs
            device="cpu",
        )

        torch.manual_seed(42)
        initial_metrics = trainer.evaluate(synthetic_loader)
        initial_loss = initial_metrics["loss"]

        # Train for 15 epochs on small synthetic batch
        for _ in range(15):
            torch.manual_seed(42)
            trainer.train_epoch(synthetic_loader)

        torch.manual_seed(42)
        final_metrics = trainer.evaluate(synthetic_loader)
        final_loss = final_metrics["loss"]

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
        assert dummy_engine.decision_head.w_pos.weight.grad is not None
        assert dummy_engine.decision_head.w_neg.weight.grad is not None

    def test_trainer_evaluate_metrics(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies structure and value bounds of evaluate() metrics."""
        trainer = YodaTrainer(model=dummy_engine, device="cpu")
        metrics = trainer.evaluate(synthetic_loader)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mean_knowledge" in metrics

        assert metrics["loss"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["mean_knowledge"] <= 1.0

    def test_trainer_fit(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies fit() runs the expected number of epochs and returns history."""
        trainer = YodaTrainer(model=dummy_engine, lr=1e-3, device="cpu")
        history = trainer.fit(
            train_loader=synthetic_loader,
            eval_loader=synthetic_loader,
            epochs=2,
        )

        assert len(history) == 2
        for epoch_idx, record in enumerate(history, start=1):
            assert record["epoch"] == epoch_idx
            assert "train_loss" in record
            assert "train_accuracy" in record
            assert "eval_loss" in record
            assert "eval_accuracy" in record
            assert "eval_mean_knowledge" in record


def test_training_package_exports() -> None:
    """Verifies that YodaDecisionDataset, collate_decision_batch, and YodaTrainer are exported."""
    import yoda.training as training

    assert hasattr(training, "YodaDecisionDataset")
    assert hasattr(training, "collate_decision_batch")
    assert hasattr(training, "YodaTrainer")
    assert "YodaDecisionDataset" in training.__all__
    assert "collate_decision_batch" in training.__all__
    assert "YodaTrainer" in training.__all__
