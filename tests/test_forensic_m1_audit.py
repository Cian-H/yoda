"""Forensic Auditor M1_1 Independent Verification Suite.

Validates authentic mathematical and neural operations, real PyTorch autograd,
absence of facades, and full gradient flow through the hardened architecture.
"""

import pytest
import torch

from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapState,
    BelnapTransformerBlock,
)
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy.ltn import LTNConstraintLoss


def test_forensic_attention_masking_invariance() -> None:
    """Verifies that padded tokens have zero effect on attention representations."""
    torch.manual_seed(1337)
    attn = BelnapAttention(d_model=32, n_heads=4)

    # 1 valid token, 3 padded tokens
    q = BelnapState(e_pos=torch.rand(1, 2, 32), e_neg=torch.rand(1, 2, 32))

    kv_pos_base = torch.rand(1, 1, 32)
    kv_neg_base = torch.rand(1, 1, 32)

    # Padding variant A: zeros
    kv_pos_a = torch.cat([kv_pos_base, torch.zeros(1, 3, 32)], dim=1)
    kv_neg_a = torch.cat([kv_neg_base, torch.zeros(1, 3, 32)], dim=1)
    kv_a = BelnapState(e_pos=kv_pos_a, e_neg=kv_neg_a)

    # Padding variant B: arbitrary random numbers
    kv_pos_b = torch.cat([kv_pos_base, torch.rand(1, 3, 32)], dim=1)
    kv_neg_b = torch.cat([kv_neg_base, torch.rand(1, 3, 32)], dim=1)
    kv_b = BelnapState(e_pos=kv_pos_b, e_neg=kv_neg_b)

    mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)

    out_a = attn(q, kv_a, mask=mask)
    out_b = attn(q, kv_b, mask=mask)

    assert torch.allclose(out_a.e_pos, out_b.e_pos, atol=1e-6)
    assert torch.allclose(out_a.e_neg, out_b.e_neg, atol=1e-6)


def test_forensic_autograd_gradient_flow_through_all_layers() -> None:
    """Verifies genuine PyTorch autograd gradient propagation through all
    hardened modules (MPA, BelnapAttention, BelnapFFN, LayerNorms, DecisionHead).
    """
    torch.manual_seed(42)
    engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=4,
        num_c_probes=4,
        num_k_probes=4,
        num_choices=3,
        num_reasoning_blocks=2,
        residual_weight=0.15,
        n_heads=2,
    )

    queries = ["alpha beta", "gamma delta epsilon zeta"]
    states = [{"x": 1}, {"y": 2, "z": 3}]
    constraints = [["rule1"], ["rule2", "rule3"]]

    out = engine(queries=queries, states=states, constraints=constraints)
    logits = out["logits"]
    assert logits.shape == (2, 3)

    # Backward pass
    loss = logits.sum()
    loss.backward()

    # Check gradients on all critical parameters
    # 1. Probes
    assert engine.q_mpa.raw_q_pos.grad is not None
    assert torch.norm(engine.q_mpa.raw_q_pos.grad) > 1e-5
    assert engine.c_mpa.raw_q_pos.grad is not None
    assert torch.norm(engine.c_mpa.raw_q_pos.grad) > 1e-5
    assert engine.k_mpa.raw_q_pos.grad is not None
    assert torch.norm(engine.k_mpa.raw_q_pos.grad) > 1e-5

    # 2. Decision Head
    assert engine.decision_head.w_pos.weight.grad is not None
    assert torch.norm(engine.decision_head.w_pos.weight.grad) > 1e-5
    assert engine.decision_head.w_neg.weight.grad is not None
    assert torch.norm(engine.decision_head.w_neg.weight.grad) > 1e-5

    # 3. Reasoning Blocks (BelnapAttention & BelnapFFN & LayerNorms)
    for _block_idx, block in enumerate(engine.reasoning_layers):
        assert isinstance(block, torch.nn.ModuleDict)
        ctx = block["context"]
        assert isinstance(ctx, BelnapTransformerBlock)
        assert ctx.self_attn.w_q_pos.weight.grad is not None
        assert ctx.cross_attn.w_q_pos.weight.grad is not None
        assert ctx.self_attn.norm_pos.weight.grad is not None
        assert ctx.ffn.w_1.weight.grad is not None
        assert ctx.ffn.norm_pos.weight.grad is not None


def test_forensic_contradiction_loss_autograd_behavior() -> None:
    """Verifies that contradiction loss computes authentic e^+ * e^- and
    drives gradients accurately according to Belnap semantics.
    """
    criterion = LTNConstraintLoss(decisiveness_weight=0.2, warmup_epochs=4)

    # At epoch 0, weight is 0.0 -> loss is 0.0, grads are 0.0
    criterion.step_epoch(0)
    assert criterion.current_decisiveness_weight == 0.0
    pos0 = torch.tensor([0.8, 0.9], requires_grad=True)
    neg0 = torch.tensor([0.8, 0.9], requires_grad=True)
    l0 = criterion({"choice_pos": pos0, "choice_neg": neg0, "truth": torch.tensor([0.5, 0.5])})
    assert l0.item() == 0.0
    l0.backward()
    assert torch.all(pos0.grad == 0.0)

    # At epoch 4 (w = 0.2), loss and grads are active
    criterion.step_epoch(4)
    assert pytest.approx(criterion.current_decisiveness_weight) == 0.2
    pos = torch.tensor([0.4, 0.7], requires_grad=True)
    neg = torch.tensor([0.3, 0.6], requires_grad=True)
    l4 = criterion({"choice_pos": pos, "choice_neg": neg, "truth": torch.tensor([0.5, 0.5])})

    # Expected value: ((0.4*0.3 + 0.7*0.6)/2) * 0.2 = ((0.12 + 0.42)/2) * 0.2 = 0.27 * 0.2 = 0.054
    assert pytest.approx(l4.item()) == 0.054
    l4.backward()

    # Analytical grad: d(loss)/d(pos_0) = (neg_0 / 2) * 0.2 = 0.3/2 * 0.2 = 0.03
    # Analytical grad: d(loss)/d(pos_1) = (neg_1 / 2) * 0.2 = 0.6/2 * 0.2 = 0.06
    assert pytest.approx(pos.grad[0].item()) == 0.03
    assert pytest.approx(pos.grad[1].item()) == 0.06
    assert pytest.approx(neg.grad[0].item()) == 0.04
    assert pytest.approx(neg.grad[1].item()) == 0.07


def test_forensic_deep_residual_gradient_retention() -> None:
    """Verifies that 6 cascaded residual joins with alpha=0.15 retain (1-0.15)^6 = 37.7%
    gradient without decay below 35%.
    """
    base = BelnapState(
        e_pos=torch.tensor([0.5], requires_grad=True),
        e_neg=torch.tensor([0.5], requires_grad=True),
    )
    delta = BelnapState(e_pos=torch.tensor([0.2]), e_neg=torch.tensor([0.2]))

    state = base
    for _ in range(6):
        state = BelnapTransformerBlock._residual_join(state, delta, alpha=0.15)

    loss = state.e_pos.sum() + state.e_neg.sum()
    loss.backward()

    retention = (1.0 - 0.15) ** 6
    assert pytest.approx(base.e_pos.grad.item(), rel=1e-4) == retention
    assert base.e_pos.grad.item() > 0.35
