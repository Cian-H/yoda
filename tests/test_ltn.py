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

    # Verify backward pass works
    loss.backward()
    assert outputs["choice_pos"].grad is not None
    assert outputs["choice_neg"].grad is not None
    assert outputs["truth"].grad is not None

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
    assert truth.grad is not None


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
    assert truth.grad[0] > 0.0
    assert truth.grad[1] < 0.0
    # No violation on (2, 3): t_2 <= t_3 -> zero implication gradient
    assert truth.grad[2] == 0.0
    assert truth.grad[3] == 0.0


def test_ltn_init_epistemic_weights() -> None:
    """Verifies default and custom epistemic regularization weights."""
    criterion = LTNConstraintLoss()
    assert hasattr(criterion, "gullibility_weight")
    assert criterion.gullibility_weight == 0.1
    assert hasattr(criterion, "ignorance_weight")
    assert criterion.ignorance_weight == 0.1

    custom_criterion = LTNConstraintLoss(gullibility_weight=0.25, ignorance_weight=0.35)
    assert custom_criterion.gullibility_weight == 0.25
    assert custom_criterion.ignorance_weight == 0.35


def test_no_universal_bivalence_penalty() -> None:
    """Verifies non-bivalent states incur zero penalty when epistemic weights are 0."""
    criterion = LTNConstraintLoss(gullibility_weight=0.0, ignorance_weight=0.0)

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


def test_epistemic_gullibility_penalty() -> None:
    """Verifies gullibility penalty scales with knowledge."""
    criterion = LTNConstraintLoss(gullibility_weight=0.2, ignorance_weight=0.0)

    # pos=1.0, neg=1.0 -> knowledge = 1.0, ignorance = 0.0
    outputs = {
        "choice_pos": torch.ones((1, 4)),
        "choice_neg": torch.ones((1, 4)),
        "truth": torch.full((1, 4), 0.5),
    }
    loss = criterion(outputs)
    assert torch.isclose(loss, torch.tensor(0.2))

    # Explicit knowledge in outputs dictionary
    outputs_explicit = {
        "choice_pos": torch.zeros((1, 2)),
        "choice_neg": torch.zeros((1, 2)),
        "truth": torch.zeros((1, 2)),
        "knowledge": torch.tensor([[0.5, 0.5]]),
    }
    loss_explicit = criterion(outputs_explicit)
    assert torch.isclose(loss_explicit, torch.tensor(0.1))


def test_epistemic_ignorance_penalty() -> None:
    """Verifies ignorance penalty scales with ignorance."""
    criterion = LTNConstraintLoss(gullibility_weight=0.0, ignorance_weight=0.15)

    # pos=0.0, neg=0.0 -> knowledge = 0.0, ignorance = 1.0
    outputs = {
        "choice_pos": torch.zeros((1, 4)),
        "choice_neg": torch.zeros((1, 4)),
        "truth": torch.full((1, 4), 0.5),
    }
    loss = criterion(outputs)
    assert torch.isclose(loss, torch.tensor(0.15))
