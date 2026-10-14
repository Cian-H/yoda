"""Unit tests for logit_scale plumbing (engine constructor + train_yoda CLI)."""

from yoda.architecture.engine import YodaDecisionEngine


def test_engine_forwards_logit_scale_to_heads() -> None:
    engine = YodaDecisionEngine(
        text_model_name="dummy", embed_dim=32, num_choices=5, logit_scale=3.0
    )
    assert engine.logit_scale == 3.0
    assert engine.decision_head.logit_scale == 3.0
    assert engine.scalar_head.logit_scale == 3.0


def test_engine_default_logit_scale_is_8() -> None:
    engine = YodaDecisionEngine(text_model_name="dummy", embed_dim=32, num_choices=5)
    assert engine.decision_head.logit_scale == 8.0
    assert engine.scalar_head.logit_scale == 8.0


def test_train_yoda_cli_logit_scale() -> None:
    from experiments.train_yoda import build_parser

    assert build_parser().parse_args([]).logit_scale == 8.0
    assert build_parser().parse_args(["--logit-scale", "4.5"]).logit_scale == 4.5
