import pytest
import torch

from yoda.nesy import LTNConstraintLoss


def test_ltn_constraint_loss_forward():
    # Mock outputs
    outputs = {
        "choice_pos": torch.tensor(
            [
                [0.9, 0.1, 0.2],
                [0.8, 0.8, 0.1],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            requires_grad=True,
        ),
        "choice_neg": torch.tensor(
            [
                [0.1, 0.9, 0.8],
                [0.2, 0.2, 0.9],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            requires_grad=True,
        ),
        "truth": torch.tensor(
            [
                [0.9, 0.1, 0.2],  # one dominant
                [0.5, 0.5, 0.5],  # overlapping
                [0.1, 0.2, 0.1],  # no dominant
                [0.9, 0.8, 0.7],  # overlapping highly
            ],
            requires_grad=True,
        ),
    }

    # 0, 1 = score/null -> no ME. 2, 3 = choice -> ME
    task_scalars = torch.tensor([[-1.0], [0.0], [1.0], [1.0]])

    criterion = LTNConstraintLoss()
    loss = criterion(outputs, task_scalars)

    # Verify backward pass works: positive support XOR and contradiction penalty
    # propagate directly into choice_pos and choice_neg
    loss.backward()
    assert outputs["choice_pos"].grad is not None
    assert outputs["choice_neg"].grad is not None
    # Truth coordinate is no longer involved in XOR/OR loss
    assert outputs["truth"].grad is None

    # Expected NC Loss: mean of choice_pos * choice_neg
    # = mean of:
    # 0.09, 0.09, 0.16
    # 0.16, 0.16, 0.09
    # 0.0,  0.0,  0.0
    # 1.0,  1.0,  1.0
    # Sum: 3.75, num_elements: 12 -> 3.75 / 12 = 0.3125

    assert loss > 0.0


def test_ltn_constraint_loss_without_task_scalars():
    criterion = LTNConstraintLoss()
    outputs = {
        "choice_pos": torch.tensor([[0.5]]),
        "choice_neg": torch.tensor([[0.5]]),
        "truth": torch.tensor([[0.5]]),
    }
    loss = criterion(outputs, None)
    assert loss.item() > 0.0


def test_ltn_constraint_loss_with_active_mask():
    criterion = LTNConstraintLoss()
    active_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    choice_pos = torch.tensor(
        [
            [0.9, 0.1, 0.5, 0.5],
            [0.1, 0.8, 0.1, 0.5],
        ],
        requires_grad=True,
    )
    choice_neg = torch.tensor(
        [
            [0.1, 0.9, 0.5, 0.5],
            [0.9, 0.2, 0.9, 0.5],
        ],
        requires_grad=True,
    )
    truth = torch.tensor(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
        ],
        requires_grad=True,
    )

    task_scalars = torch.tensor([[1.0], [1.0]])

    outputs = {
        "choice_pos": choice_pos,
        "choice_neg": choice_neg,
        "truth": truth,
        "active_mask": active_mask,
    }

    loss = criterion(outputs, task_scalars, active_mask=active_mask)
    assert loss > 0.0
    loss.backward()

    # Inactive positions should receive zero gradient
    assert choice_pos.grad is not None
    assert choice_pos.grad[0, 2] == 0.0
    assert choice_pos.grad[0, 3] == 0.0
    assert choice_pos.grad[1, 3] == 0.0


def test_grouped_multi_choice_or_constraint() -> None:
    """Verifies At-Least-One (OR) constraint triggers for multi-choice tasks.

    Triggered when 0.25 < scalar <= 0.75.
    """
    criterion = LTNConstraintLoss()

    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    task_scalars = torch.tensor([[0.5], [0.5], [0.5], [1.0], [1.0], [1.0]])

    # Group 0: all zero (violates OR)
    # Group 1: all zero (violates XOR)
    choice_pos = torch.full((6,), 0.05, requires_grad=True)
    choice_neg = torch.full((6,), 0.95, requires_grad=True)
    truth = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], requires_grad=True)

    outputs = {
        "choice_pos": choice_pos,
        "choice_neg": choice_neg,
        "truth": truth,
        "logits": torch.zeros(6),
    }

    loss = criterion(outputs, task_scalars=task_scalars, group_ids=group_ids)
    assert loss > 0.0
    loss.backward()
    assert choice_pos.grad is not None
    assert choice_pos.grad[:3].abs().sum() > 0.0
    assert choice_pos.grad[3:].abs().sum() > 0.0


def test_boundedness_constraint_for_scoring() -> None:
    """Verifies operational box boundedness triggers only for score/null tasks (scalar <= 0.25)."""
    criterion = LTNConstraintLoss(bound_a=-3.0, bound_b=3.0)

    # Candidate 0, 1: score task (scalar = 0.0) -> unbounded logit should be penalized
    # Candidate 2, 3: choice task (scalar = 1.0) -> logit should NOT be penalized by boundedness
    logits = torch.tensor([5.0, -4.0, 10.0, -10.0], requires_grad=True)
    choice_pos = torch.full((4,), 0.5, requires_grad=True)
    choice_neg = torch.full((4,), 0.5, requires_grad=True)
    truth = torch.full((4,), 0.5, requires_grad=True)
    group_ids = torch.tensor([0, 0, 1, 1])
    task_scalars = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

    outputs = {
        "logits": logits,
        "choice_pos": choice_pos,
        "choice_neg": choice_neg,
        "truth": truth,
    }

    loss = criterion(outputs, task_scalars=task_scalars, group_ids=group_ids)
    assert loss > 0.0
    loss.backward()

    # Logits 0 and 1 violated [-3, 3] and must receive gradients
    assert logits.grad is not None
    assert logits.grad[0] > 0.0  # s > 3 -> penalize downward
    assert logits.grad[1] < 0.0  # s < -3 -> penalize upward
    # Logits 2 and 3 belong to choice task (scalar = 1.0) and must receive zero boundedness gradient
    assert logits.grad[2] == 0.0
    assert logits.grad[3] == 0.0


def test_hierarchical_implication_constraint() -> None:
    """Verifies Łukasiewicz hierarchical implication: t_child <= t_parent."""
    criterion = LTNConstraintLoss()

    # (0, 1) violated (0.9 > 0.2), (2, 3) satisfied (0.1 <= 0.8)
    truth = torch.tensor([0.9, 0.2, 0.1, 0.8], requires_grad=True)
    choice_pos = torch.full((4,), 0.5, requires_grad=True)
    choice_neg = torch.full((4,), 0.5, requires_grad=True)
    logits = torch.zeros(4)

    outputs = {
        "logits": logits,
        "choice_pos": choice_pos,
        "choice_neg": choice_neg,
        "truth": truth,
    }

    edges = [(0, 1), (2, 3)]
    loss = criterion(outputs, hierarchy_edges=edges)
    assert loss > 0.0
    loss.backward()

    # Violation on (0, 1): t_0 > t_1 -> grad on 0 should push down, grad on 1 push up
    assert truth.grad is not None
    assert truth.grad[0] > 0.0
    assert truth.grad[1] < 0.0
    # No violation on (2, 3): t_2 <= t_3 -> zero implication gradient
    assert truth.grad[2] == 0.0
    assert truth.grad[3] == 0.0


def test_ltn_init_decisiveness_weight() -> None:
    """Verifies default and custom decisiveness regularization weight and warmup."""
    criterion = LTNConstraintLoss()
    assert hasattr(criterion, "decisiveness_weight")
    assert criterion.decisiveness_weight == 0.1
    assert hasattr(criterion, "current_decisiveness_weight")
    assert criterion.current_decisiveness_weight == 0.1
    assert criterion.warmup_epochs == 0

    custom_criterion = LTNConstraintLoss(decisiveness_weight=0.25, warmup_epochs=3)
    assert custom_criterion.decisiveness_weight == 0.25
    assert custom_criterion.warmup_epochs == 3
    assert custom_criterion.current_decisiveness_weight == 0.0


def test_no_universal_bivalence_penalty() -> None:
    """Verifies non-bivalent states incur zero penalty when decisiveness weight is 0."""
    criterion = LTNConstraintLoss(decisiveness_weight=0.0)

    # Pure Ignorance: pos=0, neg=0 (Under old complementarity, this was heavily penalized)
    outputs_ignorance = {
        "choice_pos": torch.zeros((2, 3)),
        "choice_neg": torch.zeros((2, 3)),
        "truth": torch.zeros((2, 3)),
    }
    loss_ignorance = criterion(outputs_ignorance)
    assert torch.isclose(loss_ignorance, torch.tensor(0.0))

    # Pure Knowledge / Contradiction: pos=1, neg=1 (previously penalized by complementarity)
    outputs_knowledge = {
        "choice_pos": torch.ones((2, 3)),
        "choice_neg": torch.ones((2, 3)),
        "truth": torch.ones((2, 3)),
    }
    loss_knowledge = criterion(outputs_knowledge)
    assert torch.isclose(loss_knowledge, torch.tensor(0.0))


def test_decisiveness_penalty() -> None:
    """Verifies paraconsistent contradiction penalty (Both) and Belnap properties."""
    criterion = LTNConstraintLoss(decisiveness_weight=0.2)

    # Fence-sitting state: pos=0.5, neg=0.5 -> pos*neg = 0.25
    # Total per element = 0.25, mean = 0.25, loss = 0.25 * 0.2 = 0.05
    outputs_fence = {
        "choice_pos": torch.full((1, 4), 0.5),
        "choice_neg": torch.full((1, 4), 0.5),
        "truth": torch.full((1, 4), 0.5),
    }
    loss_fence = criterion(outputs_fence)
    assert torch.isclose(loss_fence, torch.tensor(0.05))

    # Decisive state: pos=1.0, neg=0.0 -> pos*neg=0.0 -> penalty = 0.0
    outputs_decisive = {
        "choice_pos": torch.ones((1, 4)),
        "choice_neg": torch.zeros((1, 4)),
        "truth": torch.ones((1, 4)),
    }
    loss_decisive = criterion(outputs_decisive)
    assert torch.isclose(loss_decisive, torch.tensor(0.0))

    # Neither state (uninformative): pos=0.0, neg=0.0 -> pos*neg=0.0 -> penalty = 0.0
    outputs_neither = {
        "choice_pos": torch.zeros((1, 4)),
        "choice_neg": torch.zeros((1, 4)),
        "truth": torch.full((1, 4), 0.5),
    }
    loss_neither = criterion(outputs_neither)
    assert torch.isclose(loss_neither, torch.tensor(0.0))

    # Both state (contradiction): pos=1.0, neg=1.0 -> pos*neg=1.0 -> penalty = 0.2
    outputs_both = {
        "choice_pos": torch.ones((1, 4)),
        "choice_neg": torch.ones((1, 4)),
        "truth": torch.full((1, 4), 0.5),
    }
    loss_both = criterion(outputs_both)
    assert torch.isclose(loss_both, torch.tensor(0.2))

    # Gradient pushes down positive evidence proportionally to conflicting negative evidence:
    pos = torch.tensor([0.4, 0.6], requires_grad=True)
    neg = torch.tensor([0.4, 0.6], requires_grad=True)
    outputs_grad = {
        "choice_pos": pos,
        "choice_neg": neg,
        "truth": torch.tensor([0.5, 0.5]),
    }
    loss = criterion(outputs_grad)
    loss.backward()
    # dL/d(pos) = (neg / N) * decisiveness_weight = [0.4/2 * 0.2, 0.6/2 * 0.2] = [0.04, 0.06] > 0
    assert pos.grad is not None
    assert pos.grad[0] > 0.0
    assert pos.grad[1] > 0.0
    assert pos.grad[1] > pos.grad[0]


def test_positive_support_neither_zero_penalty():
    """R3: Verifies unchosen Neither options (e+=0, e-=0) incur zero penalty in XOR and OR."""
    criterion = LTNConstraintLoss()

    # XOR with 1 chosen candidate and 3 unmentioned Neither candidates
    xor_pos = torch.tensor([1.0, 0.0, 0.0, 0.0])
    xor_loss = criterion._compute_xor(xor_pos)
    assert pytest.approx(xor_loss.item(), abs=1e-5) == 0.0

    # OR with 1 chosen candidate and 3 unmentioned Neither candidates
    or_pos = torch.tensor([1.0, 0.0, 0.0, 0.0])
    or_loss = criterion._compute_or(or_pos)
    assert pytest.approx(or_loss.item(), abs=1e-5) == 0.0

    # Aliases
    assert hasattr(criterion, "_compute_xor_positive_support")
    assert hasattr(criterion, "_compute_or_positive_support")
    assert pytest.approx(criterion._compute_xor_positive_support(xor_pos).item(), abs=1e-5) == 0.0
    assert pytest.approx(criterion._compute_or_positive_support(or_pos).item(), abs=1e-5) == 0.0
