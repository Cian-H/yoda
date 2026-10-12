"""Unit and property tests for BelnapMultiheadPooledAttention."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from yoda.architecture.attention import BelnapMultiheadPooledAttention
from yoda.architecture.belnap_transformer import BelnapState


def test_belnap_mpa_shapes() -> None:
    """Verifies that BelnapMultiheadPooledAttention outputs expected latent and state shapes."""
    batch_size = 2
    seq_len = 6
    embed_dim = 16
    num_queries = 4
    num_heads = 4

    mpa = BelnapMultiheadPooledAttention(
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_heads=num_heads,
    )

    kv_state = BelnapState(
        e_pos=torch.rand(batch_size, seq_len, embed_dim),
        e_neg=torch.rand(batch_size, seq_len, embed_dim),
    )

    latent, pooled_state = mpa(kv_state)

    assert latent.shape == (batch_size, embed_dim)
    assert pooled_state.e_pos.shape == (batch_size, num_queries, embed_dim)
    assert pooled_state.e_neg.shape == (batch_size, num_queries, embed_dim)
    assert pooled_state.truth.shape == (batch_size, num_queries, embed_dim)
    assert pooled_state.knowledge.shape == (batch_size, num_queries, embed_dim)

    # Invariant: All evidence, truth, and knowledge must be bounded in [0, 1]
    assert (pooled_state.e_pos >= 0.0).all() and (pooled_state.e_pos <= 1.0).all()
    assert (pooled_state.e_neg >= 0.0).all() and (pooled_state.e_neg <= 1.0).all()
    assert (pooled_state.truth >= 0.0).all() and (pooled_state.truth <= 1.0).all()
    assert (pooled_state.knowledge >= 0.0).all() and (pooled_state.knowledge <= 1.0).all()


def test_belnap_mpa_zero_knowledge_suppression() -> None:
    """Verifies that vacuous/uninformative inputs suppress to Neither (t=0.5, k=0)."""
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_queries = 4

    mpa = BelnapMultiheadPooledAttention(
        embed_dim=embed_dim, num_queries=num_queries, init_bias=-3.0
    )

    # Input state with zero knowledge (e+ = 0, e- = 0)
    zero_state = BelnapState(
        e_pos=torch.zeros(batch_size, seq_len, embed_dim),
        e_neg=torch.zeros(batch_size, seq_len, embed_dim),
    )

    _, pooled_state = mpa(zero_state)

    # Knowledge must be strictly suppressed near 0 when negative bias initialization is used
    assert pooled_state.knowledge.max().item() < 0.1
    # Truth should be neutral (0.5 within tight tolerance)
    assert torch.allclose(pooled_state.truth, torch.tensor(0.5), atol=0.05)

    # Default init_bias=0.0 preserves neutral gradient-rich knowledge around 0.5
    default_mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)
    _, default_pooled = default_mpa(zero_state)
    assert default_pooled.knowledge.max().item() > 0.4


def test_belnap_mpa_contradiction_preservation() -> None:
    """Verifies that conflicting evidence (True + False) preserves contradiction (Both)."""
    batch_size = 1
    embed_dim = 16
    num_queries = 2

    mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)

    # Token 0 is True (e+=1, e-=0), Token 1 is False (e+=0, e-=1)
    e_pos = torch.zeros(batch_size, 2, embed_dim)
    e_neg = torch.zeros(batch_size, 2, embed_dim)
    e_pos[0, 0, :] = 1.0
    e_neg[0, 1, :] = 1.0

    kv_state = BelnapState(e_pos=e_pos, e_neg=e_neg)
    _, pooled_state = mpa(kv_state)

    # Both positive and negative evidence must be present in the pooled representation
    assert pooled_state.e_pos.mean().item() > 0.02
    assert pooled_state.e_neg.mean().item() > 0.02
    # Knowledge must be preserved rather than collapsing to zero
    assert pooled_state.knowledge.mean().item() > 0.04
    # Both polarities must be concurrently active (contradiction invariant)
    assert torch.minimum(pooled_state.e_pos, pooled_state.e_neg).mean().item() > 0.02


def test_belnap_mpa_raw_tensor_input() -> None:
    """Verifies that raw float tensors are automatically adapted to Belnap states."""
    batch_size = 2
    seq_len = 8
    embed_dim = 16
    num_queries = 4

    mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)
    raw_x = torch.randn(batch_size, seq_len, embed_dim)

    latent, pooled_state = mpa(raw_x)

    assert latent.shape == (batch_size, embed_dim)
    assert pooled_state.e_pos.shape == (batch_size, num_queries, embed_dim)
    assert (pooled_state.e_pos >= 0.0).all() and (pooled_state.e_pos <= 1.0).all()


def test_belnap_mpa_masking() -> None:
    """Verifies that attention masks suppress masked positions."""
    batch_size = 1
    seq_len = 4
    embed_dim = 16
    num_queries = 2

    mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)

    # First 2 tokens have knowledge, last 2 tokens have knowledge
    e_pos = torch.ones(batch_size, seq_len, embed_dim) * 0.8
    e_neg = torch.zeros(batch_size, seq_len, embed_dim)
    kv_state = BelnapState(e_pos=e_pos, e_neg=e_neg)

    # Unmasked forward
    latent_unmasked, _state_unmasked = mpa(kv_state)

    # Mask out the last 2 positions
    mask = torch.zeros(batch_size, 1, num_queries, seq_len)
    mask[:, :, :, 2:] = float("-inf")

    latent_masked, _state_masked = mpa(kv_state, mask=mask)

    # Masking should change the aggregated representation
    assert not torch.allclose(latent_unmasked, latent_masked, atol=1e-4)


def test_belnap_mpa_gradient_flow() -> None:
    """Verifies end-to-end differentiability and smooth gradient backprop."""
    batch_size = 2
    seq_len = 4
    embed_dim = 16
    num_queries = 2

    mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)
    raw_x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    latent, pooled_state = mpa(raw_x)
    loss = latent.sum() + pooled_state.e_pos.sum() + pooled_state.truth.sum()
    loss.backward()

    assert raw_x.grad is not None
    assert not torch.isnan(raw_x.grad).any()
    assert (raw_x.grad.abs().sum() > 0.0).item()

    assert mpa.raw_q_pos.grad is not None
    assert not torch.isnan(mpa.raw_q_pos.grad).any()
    assert mpa.proj.weight.grad is not None


@settings(max_examples=10, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_belnap_mpa_hypothesis_bounds(pos_val: float, neg_val: float) -> None:
    """Property test ensuring pooled output truth and knowledge stay within [0, 1]."""
    mpa = BelnapMultiheadPooledAttention(embed_dim=8, num_queries=2, num_heads=2)
    e_pos = torch.full((1, 3, 8), pos_val)
    e_neg = torch.full((1, 3, 8), neg_val)
    state = BelnapState(e_pos=e_pos, e_neg=e_neg)

    _, pooled_state = mpa(state)

    assert (pooled_state.truth >= 0.0).all() and (pooled_state.truth <= 1.0).all()
    assert (pooled_state.knowledge >= 0.0).all() and (pooled_state.knowledge <= 1.0).all()


def test_belnap_mpa_include_projection_false() -> None:
    """Verifies that include_projection=False instantiates no proj layer.

    Also asserts that calling the module returns BelnapState directly.
    """
    batch_size = 2
    seq_len = 4
    embed_dim = 16
    num_queries = 3

    mpa = BelnapMultiheadPooledAttention(
        embed_dim=embed_dim,
        num_queries=num_queries,
        include_projection=False,
    )

    assert mpa.proj is None
    # Verify no proj parameters in state_dict or parameters()
    param_names = [name for name, _ in mpa.named_parameters()]
    assert not any(name.startswith("proj.") for name in param_names)
    assert any(name.startswith("input_proj.") for name in param_names)

    raw_x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    out = mpa(raw_x)

    # Must return BelnapState directly, not a tuple
    assert isinstance(out, BelnapState)
    assert out.e_pos.shape == (batch_size, num_queries, embed_dim)
    assert out.e_neg.shape == (batch_size, num_queries, embed_dim)

    loss = out.e_pos.sum() + out.truth.sum()
    loss.backward()
    assert raw_x.grad is not None
    assert mpa.raw_q_pos.grad is not None


def test_belnap_mpa_include_projection_false_raises_on_return_latent() -> None:
    """Verifies that requesting return_latent=True when include_projection=False raises error."""
    mpa = BelnapMultiheadPooledAttention(embed_dim=16, num_queries=2, include_projection=False)
    raw_x = torch.randn(2, 4, 16)

    with pytest.raises(RuntimeError, match="include_projection"):
        mpa(raw_x, return_latent=True)


def test_belnap_mpa_convenience_methods() -> None:
    """Verifies forward_pooled and forward_latent convenience methods."""
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_queries = 3

    mpa = BelnapMultiheadPooledAttention(
        embed_dim=embed_dim,
        num_queries=num_queries,
        include_projection=True,
    )

    raw_x = torch.randn(batch_size, seq_len, embed_dim)

    # forward_pooled returns BelnapState
    pooled = mpa.forward_pooled(raw_x)
    assert isinstance(pooled, BelnapState)
    assert pooled.e_pos.shape == (batch_size, num_queries, embed_dim)

    # forward_latent returns Tensor
    latent = mpa.forward_latent(raw_x)
    assert isinstance(latent, torch.Tensor)
    assert latent.shape == (batch_size, embed_dim)


def test_yoda_decision_engine_disables_mpa_projection() -> None:
    """Verifies that YodaDecisionEngine initializes MPAs with zero projection parameters."""
    from yoda.architecture.engine import YodaDecisionEngine

    engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=64,
        num_q_probes=4,
        num_c_probes=8,
        num_k_probes=6,
    )

    assert engine.q_mpa.proj is None
    assert engine.c_mpa.proj is None
    assert engine.k_mpa.proj is None

    # Forward pass executes cleanly
    queries = ["test query"]
    states = [{"key": "val"}]
    constraints = [["opt0", "opt1"]]
    out = engine(queries=queries, states=states, constraints=constraints)
    assert "logits" in out
    assert "choice" in out
