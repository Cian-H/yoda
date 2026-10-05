"""Tests for Yoda training pipeline, dataset loading, collation, and trainer."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.training import (
    CyclicalConstraintScheduler,
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
            assert "lr" in record
            assert record["lr"] > 0.0

    def test_trainer_cosine_annealing_warm_restarts_scheduling_steps(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies CosineAnnealingWarmRestarts modifies learning rate per batch step."""
        trainer = YodaTrainer(
            model=dummy_engine,
            lr=1e-3,
            min_lr=1e-4,
            t0_epochs=2,
            t_mult=2,
            lr_decay=0.75,
            use_scheduler=True,
            device="cpu",
        )
        history = trainer.fit(train_loader=synthetic_loader, epochs=4)
        assert len(history) == 4
        # Scheduler should be active and stepped
        assert trainer.scheduler is not None
        assert isinstance(
            trainer.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        )
        assert trainer.scheduler.T_0 == 6
        assert trainer.scheduler.T_mult == 2
        assert trainer.scheduler.eta_min == 1e-4
        assert trainer.optimizer.param_groups[0]["lr"] != 1e-3

    def test_trainer_cosine_warm_restarts_decaying_amplitude(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies lengthening cycles and decaying peak learning rate upon restarts."""
        trainer = YodaTrainer(
            model=dummy_engine,
            lr=1e-3,
            min_lr=1e-4,
            t0_epochs=1,
            t_mult=2,
            lr_decay=0.75,
            use_scheduler=True,
            device="cpu",
        )
        history = trainer.fit(train_loader=synthetic_loader, epochs=3)
        assert len(history) == 3
        assert trainer.scheduler is not None
        assert pytest.approx(history[0]["lr"], rel=1e-4) == 7.5e-4
        assert pytest.approx(trainer.scheduler.base_lrs[0], rel=1e-4) == 5.625e-4
        assert pytest.approx(history[2]["lr"], rel=1e-4) == 5.625e-4

    def test_trainer_assertion_loss_telemetry(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies assertion loss is recorded in train_epoch, evaluate, and fit."""
        trainer_zero = YodaTrainer(model=dummy_engine, assertion_weight=0.0, device="cpu")
        trainer_weighted = YodaTrainer(model=dummy_engine, assertion_weight=0.5, device="cpu")

        assert trainer_weighted.assertion_weight == 0.5

        train_metrics = trainer_weighted.train_epoch(synthetic_loader)
        assert "assertion_loss" in train_metrics
        assert train_metrics["assertion_loss"] >= 0.0

        eval_metrics = trainer_weighted.evaluate(synthetic_loader)
        assert "assertion_loss" in eval_metrics
        assert eval_metrics["assertion_loss"] >= 0.0

        batch = next(iter(synthetic_loader))
        loss_zero, m_zero = trainer_zero._compute_loss_and_metrics(batch)
        loss_weighted, m_weighted = trainer_weighted._compute_loss_and_metrics(batch)

        assert "assertion_loss" in m_zero
        assert "assertion_loss" in m_weighted
        if m_weighted["assertion_loss"] > 0:
            assert loss_weighted.item() > loss_zero.item()

        history = trainer_weighted.fit(
            train_loader=synthetic_loader,
            eval_loader=synthetic_loader,
            epochs=1,
        )
        assert "train_assertion_loss" in history[0]
        assert "eval_assertion_loss" in history[0]

    def test_trainer_nan_loss_skip(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies train_epoch logs a warning and skips batch when loss is NaN."""
        from loguru import logger

        trainer = YodaTrainer(model=dummy_engine, device="cpu")
        original_compute = trainer._compute_loss_and_metrics

        def nan_compute(batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
            return (
                torch.tensor(float("nan"), requires_grad=True),
                {
                    "loss": float("nan"),
                    "ce_loss": 0.0,
                    "focal_loss": 0.0,
                    "margin_loss": 0.0,
                    "belnap_loss": 0.0,
                    "ltn_loss": 0.0,
                    "assertion_loss": 0.0,
                    "correct": 0.0,
                    "total": 1.0,
                    "knowledge_sum": 0.0,
                },
            )

        trainer._compute_loss_and_metrics = nan_compute  # type: ignore[method-assign]
        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            metrics = trainer.train_epoch(synthetic_loader)
            assert any("Loss is NaN" in w for w in warnings)
            assert metrics["loss"] == 0.0
        finally:
            logger.remove(handler_id)
            trainer._compute_loss_and_metrics = original_compute

    def test_trainer_loss_spike_warning(
        self,
        dummy_engine: YodaDecisionEngine,
    ) -> None:
        """Verifies train_epoch warns when batch loss spikes significantly above running_loss."""
        from loguru import logger

        trainer = YodaTrainer(model=dummy_engine, device="cpu")

        batch1 = {
            "queries": ["q1"],
            "states": [{"a": 1}],
            "constraints": [["c1", "c2"]],
            "target_indices": torch.tensor([0]),
        }
        batch2 = {
            "queries": ["q2"],
            "states": [{"a": 2}],
            "constraints": [["c1", "c2"]],
            "target_indices": torch.tensor([1]),
        }

        call_idx = 0

        def spike_compute(batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
            nonlocal call_idx
            val = 1.0 if call_idx == 0 else 10.0
            call_idx += 1
            loss_t = dummy_engine.decision_head.w_pos.weight.sum() * 0.0 + val
            return loss_t, {
                "loss": val,
                "ce_loss": val,
                "focal_loss": val,
                "margin_loss": 0.0,
                "belnap_loss": 0.0,
                "ltn_loss": 0.0,
                "assertion_loss": 0.0,
                "correct": 1.0,
                "total": 1.0,
                "knowledge_sum": 0.5,
            }

        trainer._compute_loss_and_metrics = spike_compute  # type: ignore[method-assign]
        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            trainer.train_epoch([batch1, batch2])  # type: ignore[arg-type]
            assert any("Loss spike detected" in w for w in warnings)
        finally:
            logger.remove(handler_id)

    def test_trainer_infinitesimal_gradient_warning(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies train_epoch warns when gradient norm is infinitesimal (< 1e-7)."""
        from loguru import logger

        trainer = YodaTrainer(model=dummy_engine, device="cpu")

        def zero_grad_compute(batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
            loss = dummy_engine.decision_head.w_pos.weight.sum() * 0.0
            return loss, {
                "loss": 0.0,
                "ce_loss": 0.0,
                "focal_loss": 0.0,
                "margin_loss": 0.0,
                "belnap_loss": 0.0,
                "ltn_loss": 0.0,
                "assertion_loss": 0.0,
                "correct": 1.0,
                "total": 1.0,
                "knowledge_sum": 0.0,
            }

        trainer._compute_loss_and_metrics = zero_grad_compute  # type: ignore[method-assign]
        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            trainer.train_epoch(synthetic_loader)
            assert any("Infinitesimal gradient detected" in w for w in warnings)
        finally:
            logger.remove(handler_id)

    def test_trainer_nan_gradient_skip(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies train_epoch skips step and logs warning when NaN gradient is detected."""
        from loguru import logger

        trainer = YodaTrainer(model=dummy_engine, device="cpu")

        param = next(dummy_engine.parameters())
        hook_handle = param.register_hook(lambda grad: torch.full_like(grad, float("nan")))

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            trainer.train_epoch(synthetic_loader)
            assert any("NaN gradient detected" in w for w in warnings)
        finally:
            hook_handle.remove()
            logger.remove(handler_id)
            trainer.optimizer.zero_grad()

    def test_trainer_cyclical_constraint_schedulers_fit(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies that YodaTrainer.fit initializes staggered constraint schedulers."""
        trainer = YodaTrainer(
            model=dummy_engine,
            ltn_weight=0.2,
            assertion_weight=0.1,
            t0_epochs=2,
            t_mult=2,
            use_scheduler=True,
            device="cpu",
        )
        # Check property aliases
        assert trainer.ltn_w == 0.2
        assert trainer.assertion_w == 0.1
        assert trainer.current_ltn_w == 0.2
        assert trainer.current_assertion_w == 0.1

        trainer.fit(train_loader=synthetic_loader, epochs=1)

        assert hasattr(trainer, "ltn_scheduler")
        assert hasattr(trainer, "assertion_scheduler")
        assert trainer.ltn_scheduler.max_weight == 0.2
        assert trainer.scheduler is not None
        assert trainer.ltn_scheduler.t0_steps == trainer.scheduler.T_0
        assert trainer.ltn_scheduler.start_step == 0

        assert trainer.assertion_scheduler.max_weight == 0.1
        assert trainer.assertion_scheduler.t0_steps == trainer.scheduler.T_0
        assert trainer.assertion_scheduler.start_step == trainer.scheduler.T_0

    def test_trainer_evaluate_strict_weights(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies evaluate sets current weights back to strict values."""
        trainer = YodaTrainer(
            model=dummy_engine,
            ltn_weight=0.3,
            assertion_weight=0.15,
            device="cpu",
        )
        # Simulate a dynamic weight state during training
        trainer.current_ltn_w = 0.01
        trainer.current_assertion_w = 0.02

        trainer.evaluate(synthetic_loader)

        assert trainer.current_ltn_w == 0.3
        assert trainer.current_assertion_w == 0.15

    def test_trainer_cyclical_weight_telemetry(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Verifies TensorBoard scalar logging of cyclical constraint weights."""
        trainer = YodaTrainer(
            model=dummy_engine,
            ltn_weight=0.2,
            assertion_weight=0.1,
            t0_epochs=1,
            t_mult=2,
            use_scheduler=True,
            tensorboard_dir=tmp_path / "tb",
            device="cpu",
        )
        trainer.fit(train_loader=synthetic_loader, epochs=2)
        trainer.close()
        assert (tmp_path / "tb").exists()

    def test_trainer_default_belnap_weight(
        self,
        dummy_engine: YodaDecisionEngine,
    ) -> None:
        """Verifies that YodaTrainer defaults belnap_weight to 1.0."""
        trainer = YodaTrainer(model=dummy_engine)
        assert trainer.belnap_weight == 1.0

    def test_trainer_total_loss_formula_legacy(
        self,
        dummy_engine: YodaDecisionEngine,
        synthetic_loader: DataLoader[dict[str, Any]],
    ) -> None:
        """Verifies legacy total_loss excludes focal_loss and matches the weighted sum formula."""
        dummy_engine.eval()
        trainer = YodaTrainer(
            model=dummy_engine,
            belnap_weight=1.5,
            margin_weight=0.25,
            ltn_weight=0.35,
            assertion_weight=0.1,
            device="cpu",
        )
        batch = next(iter(synthetic_loader))

        torch.manual_seed(123)
        loss, metrics = trainer._compute_loss_and_metrics(batch)

        assert "focal_loss" in metrics
        assert "ce_loss" in metrics
        assert "belnap_loss" in metrics
        assert "margin_loss" in metrics
        assert "ltn_loss" in metrics
        assert "assertion_loss" in metrics

        expected_total = (
            metrics["belnap_loss"] * 1.5
            + metrics["margin_loss"] * 0.25
            + metrics["ltn_loss"] * 0.35
            + metrics["assertion_loss"] * 0.1
        )
        assert pytest.approx(loss.item(), rel=1e-5) == expected_total

        # Verify focal_loss does not drive total_loss
        trainer_other_focal = YodaTrainer(
            model=dummy_engine,
            belnap_weight=1.5,
            margin_weight=0.25,
            ltn_weight=0.35,
            assertion_weight=0.1,
            focal_gamma=5.0,
            device="cpu",
        )
        torch.manual_seed(123)
        loss_other_focal, metrics_other_focal = trainer_other_focal._compute_loss_and_metrics(batch)
        assert pytest.approx(loss_other_focal.item(), rel=1e-5) == loss.item()
        assert metrics_other_focal["focal_loss"] != metrics["focal_loss"]

    def test_trainer_total_loss_formula_independent(
        self,
        dummy_engine: YodaDecisionEngine,
    ) -> None:
        """Verifies independent total_loss excludes focal_loss and matches the formula."""
        dummy_engine.eval()
        trainer = YodaTrainer(
            model=dummy_engine,
            independent_eval=True,
            belnap_weight=1.2,
            margin_weight=0.3,
            ltn_weight=0.4,
            assertion_weight=0.2,
            device="cpu",
        )
        batch = {
            "queries": ["Q1", "Q2"],
            "states": [{"k": 1}, {"k": 2}],
            "constraints": [["c1", "c2"], ["c3", "c4"]],
            "target_indices": torch.tensor([0, 1]),
            "task_scalars": torch.tensor([[1.0], [1.0]]),
            "candidates": ["c1", "c2", "c3", "c4"],
            "candidate_queries": ["Q1", "Q1", "Q2", "Q2"],
            "candidate_states": [{"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}],
            "candidate_labels": torch.tensor([1.0, 0.0, 0.0, 1.0]),
            "candidate_group_ids": torch.tensor([0, 0, 1, 1]),
            "candidate_task_scalars": torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        }
        torch.manual_seed(123)
        loss, metrics = trainer._compute_loss_and_metrics(batch)

        assert "focal_loss" in metrics
        assert "ce_loss" in metrics
        assert "belnap_loss" in metrics
        assert "margin_loss" in metrics
        assert "ltn_loss" in metrics
        assert "assertion_loss" in metrics

        expected_total = (
            metrics["belnap_loss"] * 1.2
            + metrics["margin_loss"] * 0.3
            + metrics["ltn_loss"] * 0.4
            + metrics["assertion_loss"] * 0.2
        )
        assert pytest.approx(loss.item(), rel=1e-5) == expected_total

        # Verify focal_loss does not drive total_loss
        trainer_other_focal = YodaTrainer(
            model=dummy_engine,
            independent_eval=True,
            belnap_weight=1.2,
            margin_weight=0.3,
            ltn_weight=0.4,
            assertion_weight=0.2,
            focal_gamma=5.0,
            device="cpu",
        )
        torch.manual_seed(123)
        loss_other, metrics_other = trainer_other_focal._compute_loss_and_metrics(batch)
        assert pytest.approx(loss_other.item(), rel=1e-5) == loss.item()
        assert metrics_other["focal_loss"] != metrics["focal_loss"]


class TestYodaLightningAdapterScheduler:
    """Verifies scheduler configuration in YodaLightningAdapter."""

    def test_configure_optimizers_with_total_steps(
        self,
        dummy_engine: YodaDecisionEngine,
    ) -> None:
        from yoda.training.lightning import YodaLightningAdapter

        adapter = YodaLightningAdapter(
            model=dummy_engine,
            lr=1e-3,
            use_scheduler=True,
            total_steps=100,
            pct_start=0.3,
        )
        opt_conf = adapter.configure_optimizers()
        assert isinstance(opt_conf, dict)
        assert "optimizer" in opt_conf
        assert "lr_scheduler" in opt_conf
        scheduler = opt_conf["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        assert opt_conf["lr_scheduler"]["interval"] == "step"

    def test_configure_optimizers_without_scheduler(
        self,
        dummy_engine: YodaDecisionEngine,
    ) -> None:
        from yoda.training.lightning import YodaLightningAdapter

        adapter = YodaLightningAdapter(
            model=dummy_engine,
            lr=1e-3,
            use_scheduler=False,
        )
        opt_conf = adapter.configure_optimizers()
        assert isinstance(opt_conf, torch.optim.Optimizer)


def test_training_package_exports() -> None:
    """Verifies that YodaDecisionDataset, collate_decision_batch, and YodaTrainer are exported."""
    import yoda.training as training

    assert hasattr(training, "CyclicalConstraintScheduler")
    assert hasattr(training, "YodaDecisionDataset")
    assert hasattr(training, "collate_decision_batch")
    assert hasattr(training, "YodaTrainer")
    assert hasattr(training, "FocalLoss")
    assert hasattr(training, "MarginLoss")
    assert hasattr(training, "FocalMarginLoss")
    assert "CyclicalConstraintScheduler" in training.__all__
    assert "YodaDecisionDataset" in training.__all__
    assert "collate_decision_batch" in training.__all__
    assert "YodaTrainer" in training.__all__
    assert "FocalLoss" in training.__all__
    assert "MarginLoss" in training.__all__
    assert "FocalMarginLoss" in training.__all__


def test_train_yoda_cli_assertion_weight() -> None:
    """Verifies --assertion-weight argument parsing and default."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.train_yoda import build_parser

    parser = build_parser()
    default_args = parser.parse_args([])
    assert default_args.assertion_weight == 0.1

    custom_args = parser.parse_args(["--assertion-weight", "0.35"])
    assert custom_args.assertion_weight == 0.35


def test_train_yoda_cli_dataset_splits() -> None:
    """Verifies dataset splitting CLI arguments and 80/10/10 defaults."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.train_yoda import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.eval_path is None
    assert args.test_path is None
    assert args.max_samples is None
    assert args.train_ratio == 0.8
    assert args.val_ratio == 0.1
    assert args.test_ratio == 0.1

    custom = parser.parse_args([
        "--max-samples",
        "5000",
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.15",
        "--test-ratio",
        "0.15",
        "--max-train-samples",
        "3000",
    ])
    assert custom.max_samples == 5000
    assert custom.train_ratio == 0.7
    assert custom.val_ratio == 0.15
    assert custom.test_ratio == 0.15
    assert custom.max_train_samples == 3000


def test_train_yoda_cli_cosine_scheduler_args() -> None:
    """Verifies CLI parsing of --t0-epochs, --t-mult, and --lr-decay arguments."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.train_yoda import build_parser

    parser = build_parser()
    default_args = parser.parse_args([])
    assert default_args.t0_epochs == 2
    assert default_args.t_mult == 2
    assert default_args.lr_decay == 0.75

    custom_args = parser.parse_args([
        "--t0-epochs",
        "3",
        "--t-mult",
        "4",
        "--lr-decay",
        "0.5",
    ])
    assert custom_args.t0_epochs == 3
    assert custom_args.t_mult == 4
    assert custom_args.lr_decay == 0.5


def test_cyclical_constraint_scheduler_init() -> None:
    """Verifies default and custom attribute assignment in CyclicalConstraintScheduler."""
    sched = CyclicalConstraintScheduler(max_weight=0.5, t0_steps=100)
    assert sched.max_weight == 0.5
    assert sched.t0_steps == 100
    assert sched.t_mult == 2
    assert sched.start_step == 0

    custom = CyclicalConstraintScheduler(
        max_weight=0.25,
        t0_steps=50,
        t_mult=3,
        start_step=25,
    )
    assert custom.max_weight == 0.25
    assert custom.t0_steps == 50
    assert custom.t_mult == 3
    assert custom.start_step == 25


def test_cyclical_constraint_scheduler_staggered_start() -> None:
    """Verifies that weights remain 0.0 before start_step."""
    sched = CyclicalConstraintScheduler(max_weight=1.0, t0_steps=10, start_step=10)
    for step in range(10):
        assert sched.get_weight(step) == 0.0
    assert sched.get_weight(10) == 0.0
    # Halfway through cycle 1 (step 15)
    assert pytest.approx(sched.get_weight(15), rel=1e-5) == 0.5


def test_cyclical_constraint_scheduler_warm_restarts() -> None:
    """Verifies cosine oscillation and cycle lengthening across warm restarts."""
    sched = CyclicalConstraintScheduler(max_weight=1.0, t0_steps=10, t_mult=2, start_step=0)

    # Cycle 1: length 10 (steps 0..9)
    assert sched.get_weight(0) == 0.0
    assert pytest.approx(sched.get_weight(5), rel=1e-5) == 0.5
    assert pytest.approx(sched.get_weight(9), rel=1e-3) == 0.9755

    # Cycle 2 restart at step 10: length 20 (steps 10..29)
    assert sched.get_weight(10) == 0.0
    assert pytest.approx(sched.get_weight(20), rel=1e-5) == 0.5

    # Cycle 3 restart at step 30: length 40 (steps 30..69)
    assert sched.get_weight(30) == 0.0
    assert pytest.approx(sched.get_weight(50), rel=1e-5) == 0.5

    # Cycle 4 restart at step 70
    assert sched.get_weight(70) == 0.0


def test_cyclical_constraint_scheduler_constant_cycle_length() -> None:
    """Verifies behavior when t_mult=1 produces periodic cycles of identical length."""
    sched = CyclicalConstraintScheduler(max_weight=0.8, t0_steps=6, t_mult=1, start_step=0)
    assert sched.get_weight(0) == 0.0
    assert pytest.approx(sched.get_weight(3), rel=1e-5) == 0.4
    assert sched.get_weight(6) == 0.0
    assert pytest.approx(sched.get_weight(9), rel=1e-5) == 0.4
    assert sched.get_weight(12) == 0.0


def test_cyclical_constraint_scheduler_zero_max_weight() -> None:
    """Verifies that 0.0 max_weight always returns 0.0."""
    sched = CyclicalConstraintScheduler(max_weight=0.0, t0_steps=10)
    for step in range(50):
        assert sched.get_weight(step) == 0.0


def test_train_yoda_cli_belnap_weight_default() -> None:
    """Verifies CLI parsing of --belnap-weight defaults to 1.0."""
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.train_yoda import build_parser

    parser = build_parser()
    default_args = parser.parse_args([])
    assert default_args.belnap_weight == 1.0


