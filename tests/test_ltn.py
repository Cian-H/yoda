import torch

from yoda.probabilistic.ltn import LTNConstraintLoss


def test_ltn_constraint_loss_forward():
    # Mock outputs
    outputs = {
        "choice_pos": torch.tensor([
            [0.9, 0.1, 0.2],
            [0.8, 0.8, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ], requires_grad=True),
        "choice_neg": torch.tensor([
            [0.1, 0.9, 0.8],
            [0.2, 0.2, 0.9],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], requires_grad=True),
        "truth": torch.tensor([
            [0.9, 0.1, 0.2],  # one dominant
            [0.5, 0.5, 0.5],  # overlapping
            [0.1, 0.2, 0.1],  # no dominant
            [0.9, 0.8, 0.7],  # overlapping highly
        ], requires_grad=True)
    }

    # 0, 1 = score/null -> no ME. 2, 3 = choice -> ME
    task_scalars = torch.tensor([
        [-1.0],
        [0.0],
        [1.0],
        [1.0]
    ])

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

    # Expected ME Loss:
    # Batch 0: no mask
    # Batch 1: no mask
    # Batch 2: masked -> overlap=0.10, existence=(0.4-1.0)^2=0.36 -> 0.46
    # Batch 3: masked -> overlap=3.82, existence=(2.4-1.0)^2=1.96 -> 5.78
    # Mean ME over batch (4) = (0.46 + 5.78) / 4 = 6.24 / 4 = 1.56
    # Total expected loss = 0.3125 (NC) + 1.56 (ME) = 1.8725

    assert torch.allclose(loss, torch.tensor(1.8725))

def test_ltn_constraint_loss_without_task_scalars():
    criterion = LTNConstraintLoss()
    outputs = {
        "choice_pos": torch.tensor([[0.5]]),
        "choice_neg": torch.tensor([[0.5]]),
        "truth": torch.tensor([[0.5]]),
    }
    loss = criterion(outputs, None)
    assert loss.item() == 0.25  # Only NC loss: 0.5 * 0.5 = 0.25
