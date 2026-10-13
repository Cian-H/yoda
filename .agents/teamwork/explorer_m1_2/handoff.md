# Handoff Report: R2 Paraconsistent Contradiction Loss & Linear Warmup Schedule

**Author**: Explorer M1_2  
**Date**: 2026-10-13T20:15:00Z  
**Working Directory**: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2`  
**Milestone**: M1 (Phase 1 Architectural Hardening)  
**Target Files**:
- `src/yoda/nesy/ltn.py`
- `src/yoda/training/trainer.py`
- `src/yoda/training/lightning.py`
- `tests/test_ltn.py`
- `tests/test_contradiction_loss.py`

---

## 1. Observation

### Observation 1.1: Current Quadratic Decisiveness Penalty in `src/yoda/nesy/ltn.py`
In `src/yoda/nesy/ltn.py` lines 4–5 and lines 249–280, the decisiveness penalty is computed as:
```python
# Lines 4-5
# 1. Decisiveness Penalty:
#    (choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)).mean() * decisiveness_weight

# Lines 250-253 (grouped candidate evaluation path):
if group_ids is not None:
    decisiveness_penalty = (
        choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
    ).mean() * self.decisiveness_weight

# Lines 272-279 (legacy fixed-dimension evaluation path):
if mask_f is not None:
    denom = mask_f.sum().clamp(min=1.0)
    decisiveness_terms = choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
    decisiveness_penalty = (
        (decisiveness_terms * mask_f).sum() / denom
    ) * self.decisiveness_weight
else:
    decisiveness_penalty = (
        choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
    ).mean() * self.decisiveness_weight
```

### Observation 1.2: Mathematical Diagnosis from `.docs/todos/ltn-contradiction-penalty-and-warmup.md`
Quoted verbatim from `.docs/todos/ltn-contradiction-penalty-and-warmup.md` lines 8:
> "In `src/yoda/nesy/ltn.py`, `LTNConstraintLoss.forward` evaluates a decisiveness penalty defined as `decisiveness_terms = choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)`. Because `choice_pos = torch.sigmoid(raw_pos)`, the mathematical derivative (Jacobian) of the sigmoid activation is identically $\sigma(z)(1 - \sigma(z)) = \text{choice\_pos} \cdot (1 - \text{choice\_pos})$. Minimizing this term is literally penalizing the Jacobian of the decision head's activation function. At initialization ($\text{raw\_pos} \approx 0 \implies \text{choice\_pos} \approx 0.5$), the loss resides at its global maximum ($0.25$), incentivizing the optimizer on Step 1 to rapidly blow linear biases to $\pm 5.0$ to force outputs to $0$ or $1$. The moment $p(1 - p) \to 0$, upstream gradients through `raw_pos` and `raw_neg` collapse to zero, permanently cutting off all NeSy constraints (XOR, OR, hierarchy) from early training and contributing to the 36.5% accuracy plateau. Replacing this penalty with a true paraconsistent contradiction penalty ($Both$: $\text{choice\_pos} \times \text{choice\_neg}$) and introducing a linear warmup schedule resolves this failure mode."

### Observation 1.3: Interface Contract in `PROJECT.md`
Quoted verbatim from `PROJECT.md` lines 52–55:
```markdown
### `LTNConstraintLoss` ↔ `YodaDecisionEngine` & `YodaTrainer`
- `LTNConstraintLoss.forward(choice_pos, choice_neg, ...)`: Contradiction penalty computed as `(choice_pos * choice_neg).mean() * self.current_decisiveness_weight`.
- `LTNConstraintLoss.step_epoch(epoch)`: Updates `current_decisiveness_weight` according to linear warmup schedule.
- `_compute_xor(pos_support)` & `_compute_or(pos_support)`: Evaluates positive support, guaranteeing that $Neither$ $(e^+=0, e^-=0)$ incurs 0 loss.
```

### Observation 1.4: Warmup Schedule Requirement in Dispatch and Plan
Quoted from Dispatch:
> "Design the linear warmup schedule $\lambda(e) = \lambda_{\text{target}} \cdot \min(1.0, e / W)$ and its integration in both `YodaTrainer` and `YodaLightningAdapter`."

### Observation 1.5: Current `LTNConstraintLoss` Instantiation in Training Modules
In `src/yoda/training/trainer.py` line 194:
```python
self.ltn_criterion = LTNConstraintLoss()
```
And in `src/yoda/training/trainer.py` lines 402–410 & 713–715:
```python
# During loss computation:
if self.current_ltn_w > 0.0:
    hierarchy_edges = batch.get("hierarchy_edges")
    ltn_loss = self.ltn_criterion(
        out,
        task_scalars=task_scalars,
        active_mask=active_mask,
        hierarchy_edges=hierarchy_edges,
    )
...
# In fit():
for epoch in range(1, epochs + 1):
    train_metrics = self.train_epoch(train_loader)
```
Currently, `trainer.py` does NOT call any epoch progression on `self.ltn_criterion`.

In `src/yoda/training/lightning.py` line 81:
```python
self.ltn_criterion = LTNConstraintLoss()
```
Currently, `lightning.py` has no `on_train_epoch_start()` hook calling `step_epoch()`.

### Observation 1.6: Existing Test Assertions in `tests/test_ltn.py`
In `tests/test_ltn.py` lines 248–286 (`test_decisiveness_penalty`):
```python
def test_decisiveness_penalty() -> None:
    """Verifies decisiveness penalty penalizes fence-sitting at 0.5 and rewards 0.0 / 1.0."""
    criterion = LTNConstraintLoss(decisiveness_weight=0.2)

    # Fence-sitting state: pos=0.5, neg=0.5 -> pos*(1-pos) = 0.25, neg*(1-neg) = 0.25
    # Total per element = 0.5, mean = 0.5, loss = 0.5 * 0.2 = 0.1
    outputs_fence = {
        "choice_pos": torch.full((1, 4), 0.5),
        "choice_neg": torch.full((1, 4), 0.5),
        "truth": torch.full((1, 4), 0.5),
    }
    loss_fence = criterion(outputs_fence)
    assert torch.isclose(loss_fence, torch.tensor(0.1))

    # Decisive state: pos=1.0, neg=0.0 -> pos*(1-pos)=0.0, neg*(1-neg)=0.0 -> penalty = 0.0
    outputs_decisive = {
        "choice_pos": torch.ones((1, 4)),
        "choice_neg": torch.zeros((1, 4)),
        "truth": torch.ones((1, 4)),
    }
    loss_decisive = criterion(outputs_decisive)
    assert torch.isclose(loss_decisive, torch.tensor(0.0))

    # Gradient pushes away from 0.5:
    pos = torch.tensor([0.4, 0.6], requires_grad=True)
    neg = torch.tensor([0.4, 0.6], requires_grad=True)
    outputs_grad = {
        "choice_pos": pos,
        "choice_neg": neg,
        "truth": torch.tensor([0.5, 0.5]),
    }
    loss = criterion(outputs_grad)
    loss.backward()
    # At 0.4 (< 0.5): gradient is positive (pushes down towards 0)
    assert pos.grad is not None
    assert pos.grad[0] > 0.0
    # At 0.6 (> 0.5): gradient is negative (pushes up towards 1)
    assert pos.grad[1] < 0.0
```
Notice that line 50–57 of `tests/test_ltn.py` in `test_ltn_constraint_loss_forward` already had a comment anticipating NC Loss:
```python
# Expected NC Loss: mean of choice_pos * choice_neg
# = mean of:
# 0.09, 0.09, 0.16
# 0.16, 0.16, 0.09
# 0.0,  0.0,  0.0
# 1.0,  1.0,  1.0
# Sum: 3.75, num_elements: 12 -> 3.75 / 12 = 0.3125
```

---

## 2. Logic Chain

### Step 2.1: Mathematical Formulation of Paraconsistent Contradiction Loss
1. Under Belnap 4-valued logic $\mathcal{B}_4 = \{\mathbf{T}, \mathbf{F}, \mathbf{B}, \mathbf{N}\}$, every proposition has decoupled positive evidence $e^+ = \text{choice\_pos} \in [0, 1]$ and negative evidence $e^- = \text{choice\_neg} \in [0, 1]$.
2. The state of Contradiction ($Both$) is defined as simultaneous assertion of positive and negative support:
   $$\text{Both}(x) = e^+(x) \wedge e^-(x)$$
   Under the differentiable product t-norm $T(a, b) = a \cdot b$, the degree of contradiction is:
   $$\mathcal{C}(e^+, e^-) = e^+ \cdot e^-$$
3. Replacing the bivalent decisiveness penalty $p(1-p) + q(1-q)$ with $\mathcal{C}(e^+, e^-)$ produces the paraconsistent contradiction penalty:
   $$\mathcal{L}_{\text{contra}} = \frac{1}{N} \sum_{i=1}^N (e_i^+ \cdot e_i^-) \cdot \lambda_{\text{decisiveness}}$$
4. **Behavior across Belnap corners**:
   - **True ($\mathbf{T}$)**: $e^+ = 1, e^- = 0 \implies \mathcal{C} = 1 \cdot 0 = 0$ (Zero penalty, completely consistent).
   - **False ($\mathbf{F}$)**: $e^+ = 0, e^- = 1 \implies \mathcal{C} = 0 \cdot 1 = 0$ (Zero penalty, completely consistent).
   - **Neither ($\mathbf{N}$)**: $e^+ = 0, e^- = 0 \implies \mathcal{C} = 0 \cdot 0 = 0$ (Zero penalty! Crucial for unmentioned candidate options, resolving false penalties).
   - **Both ($\mathbf{B}$)**: $e^+ = 1, e^- = 1 \implies \mathcal{C} = 1 \cdot 1 = 1$ (Maximum penalty, strictly penalizes conflicting claims).
   - **Intermediate / Initialization**: $e^+ = 0.5, e^- = 0.5 \implies \mathcal{C} = 0.25$.
5. **Gradient Dynamics**:
   $$\frac{\partial \mathcal{L}_{\text{contra}}}{\partial e_i^+} = \frac{\lambda}{N} e_i^-, \qquad \frac{\partial \mathcal{L}_{\text{contra}}}{\partial e_i^-} = \frac{\lambda}{N} e_i^+$$
   When $e_i^- = 0$, $\frac{\partial \mathcal{L}_{\text{contra}}}{\partial e_i^+} = 0$: there is zero downward pull on consistent positive evidence!
   When $e_i^- > 0$, the gradient on $e_i^+$ is positive ($\ge 0$), pushing positive evidence down only in proportion to the conflicting negative evidence.
   Because $\frac{\partial \mathcal{L}}{\partial z^+} = \frac{\lambda}{N} e_i^- \cdot e_i^+ (1 - e_i^+)$, the gradient vanishes when $e_i^- = 0$, avoiding the artificial saturation that previously zeroed the Jacobian across upstream layers.

### Step 2.2: Linear Warmup Schedule Design
1. To ensure reasoning layers stabilize before constraint penalties exert force, we introduce a linear warmup schedule:
   $$\lambda(e) = \lambda_{\text{target}} \cdot \min\left(1.0, \frac{e}{W}\right) \quad \text{for } W > 0$$
   $$\lambda(e) = \lambda_{\text{target}} \quad \text{for } W \le 0$$
   where $W$ is `warmup_epochs`, $\lambda_{\text{target}}$ is `decisiveness_weight`, and $e$ is the current epoch index (0-indexed).
2. Progression:
   - At epoch $0$: $\lambda(0) = 0.0$ (disables contradiction penalty during initial parameter stabilization).
   - For $0 < e < W$: linearly ramps with slope $\frac{\lambda_{\text{target}}}{W}$.
   - For $e \ge W$: clamps at $\lambda_{\text{target}}$.
3. Contract `LTNConstraintLoss.step_epoch(epoch: int | None = None) -> float`:
   - If `epoch` is passed explicitly (e.g. `step_epoch(epoch)`), `self.current_epoch = max(0, epoch)`.
   - If `epoch is None`, auto-increments: `self.current_epoch += 1`.
   - Updates and returns `self.current_decisiveness_weight`.
4. Backward Compatibility:
   - Default `warmup_epochs = 0` in `LTNConstraintLoss.__init__`.
   - When `warmup_epochs = 0`, `current_decisiveness_weight = decisiveness_weight` at initialization, guaranteeing that standalone test cases not testing warmup continue to function without requiring explicit calls to `step_epoch`.

### Step 2.3: Integration in `YodaTrainer` and `YodaLightningAdapter`
1. **`YodaTrainer` (`src/yoda/training/trainer.py`)**:
   - `__init__`: accepts `decisiveness_weight: float = 0.1` and `decisiveness_warmup_epochs: int = 0`.
   - Passes both to `self.ltn_criterion = LTNConstraintLoss(decisiveness_weight=decisiveness_weight, warmup_epochs=decisiveness_warmup_epochs)`.
   - In `fit()`:
     The training loop is `for epoch in range(1, epochs + 1):`.
     Epoch index $0$ is reached on the first epoch via:
     `self.ltn_criterion.step_epoch(epoch - 1)`.
     This evaluates at $e = 0$ on Epoch 1, $e = 1$ on Epoch 2, etc.
   - Logs `self.ltn_criterion.current_decisiveness_weight` to TensorBoard (`self.writer.add_scalar("train/ltn_decisiveness_weight", ...)`) and records it in `history` / `epoch_record`.
2. **`YodaLightningAdapter` (`src/yoda/training/lightning.py`)**:
   - `__init__`: accepts `decisiveness_weight: float = 0.1` and `decisiveness_warmup_epochs: int = 0`.
   - Passes both to `self.ltn_criterion = LTNConstraintLoss(...)`.
   - Hook: adds `on_train_epoch_start(self) -> None`:
     ```python
     def on_train_epoch_start(self) -> None:
         """Updates LTN constraint loss warmup schedule at the start of each epoch."""
         self.ltn_criterion.step_epoch(self.current_epoch)
     ```
     PyTorch Lightning provides `self.current_epoch` which starts at $0$ for the first epoch, perfectly aligning with $\lambda(0) = 0.0$.
   - Logs `train/decisiveness_weight` via `self.log(...)`.

### Step 2.4: Adaptation of Existing Tests in `tests/test_ltn.py`
1. `test_decisiveness_penalty`:
   - Under old $p(1-p)+q(1-q)$ with weight 0.2, fence-sitting at $(0.5, 0.5)$ produced loss $0.1$. Under paraconsistent contradiction loss, $(0.5 \times 0.5) \times 0.2 = 0.05$. Assertions must check `0.05`.
   - Decisive state $(1.0, 0.0)$ still produces `0.0`.
   - Uninformative state $(0.0, 0.0)$ ($Neither$) produces `0.0`.
   - Contradiction state $(1.0, 1.0)$ ($Both$) produces $1.0 \times 0.2 = 0.2$ (old penalty incorrectly produced 0.0).
   - In gradient check: previously, for $pos = 0.6$, gradient was negative (pushing towards 1). Under contradiction loss with $neg = 0.6$, gradient $\frac{\partial L}{\partial pos}$ is $0.2 \times 0.6 / 2 = 0.06 > 0$, pushing positive evidence down to resolve the contradiction. Both `pos.grad[0]` and `pos.grad[1]` are strictly positive.
2. `test_ltn_init_decisiveness_weight`:
   - Add verification for `current_decisiveness_weight` and `warmup_epochs`.

### Step 2.5: Design of New Unit Tests in `tests/test_contradiction_loss.py`
A dedicated test suite `tests/test_contradiction_loss.py` is needed to thoroughly validate:
1. Four Belnap corner valuations ($\mathbf{T}, \mathbf{F}, \mathbf{N}, \mathbf{B}$).
2. Algebraic properties: commutativity and monotonicity.
3. Gradient behavior: no gradient on positive evidence when negative evidence is zero, linear scaling with opposing evidence.
4. Active mask and unrolled group evaluation support.
5. Linear warmup schedule progression across epochs and steps.
6. Integration with `YodaTrainer` and `YodaLightningAdapter`.

---

## 3. Caveats

1. **Phase 1 Scope**: This report focuses strictly on R2 (Paraconsistent Contradiction Loss & Linear Warmup). Phase 2 (R3 positive support XOR/OR reformulation) also touches `_compute_xor` and `_compute_or` in `ltn.py`, but R2 only modifies the contradiction/decisiveness penalty and warmup schedule. The separation is clean and isolated.
2. **Default Warmup in Tests vs CLI**: In `LTNConstraintLoss.__init__`, `warmup_epochs` defaults to 0 to prevent breaking tests that instantiate `LTNConstraintLoss` directly without running a multi-epoch training loop. In `experiments/train_yoda.py`, `--decisiveness-warmup-epochs` should be exposed with default `1` or `2` to activate warmup for training runs.
3. **Step-Level vs Epoch-Level Warmup**: While `step_epoch(epoch)` fulfills the required milestone contract, we also support step progression by allowing `step_epoch()` without arguments to auto-increment, or by calling `step_epoch` per epoch.

---

## 4. Conclusion & Implementation Plan

### 4.1 Exact Code Changes for `src/yoda/nesy/ltn.py`

#### Docstring Update (Lines 4–5):
```python
# Before:
1. Decisiveness Penalty:
   (choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)).mean() * decisiveness_weight

# After:
1. Paraconsistent Contradiction Penalty (Both):
   (choice_pos * choice_neg).mean() * current_decisiveness_weight
```

#### `__init__` and `step_epoch` Addition (Lines 37–63):
```python
# Before:
    def __init__(
        self,
        bound_a: float = -3.0,
        bound_b: float = 3.0,
        bound_weight: float = 1.0,
        hierarchy_weight: float = 1.0,
        decisiveness_weight: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.bound_a = float(bound_a)
        self.bound_b = float(bound_b)
        self.bound_weight = float(bound_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.decisiveness_weight = float(decisiveness_weight)
        self.eps = float(eps)

# After:
    def __init__(
        self,
        bound_a: float = -3.0,
        bound_b: float = 3.0,
        bound_weight: float = 1.0,
        hierarchy_weight: float = 1.0,
        decisiveness_weight: float = 0.1,
        warmup_epochs: int = 0,
        eps: float = 1e-7,
    ) -> None:
        """Initializes LTNConstraintLoss.

        Args:
            bound_a: Lower operational bound for continuous scoring / null tasks.
            bound_b: Upper operational bound for continuous scoring / null tasks.
            bound_weight: Multiplier weight for operational box boundedness loss.
            hierarchy_weight: Multiplier weight for hierarchical implication loss.
            decisiveness_weight: Target multiplier weight for contradiction penalty.
            warmup_epochs: Number of epochs for linear warmup from 0.0 to decisiveness_weight.
            eps: Epsilon for numerical stability inside log evaluations.
        """
        super().__init__()
        self.bound_a = float(bound_a)
        self.bound_b = float(bound_b)
        self.bound_weight = float(bound_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.decisiveness_weight = float(decisiveness_weight)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.current_epoch: int = 0
        self.current_decisiveness_weight: float = (
            0.0 if self.warmup_epochs > 0 else self.decisiveness_weight
        )
        self.eps = float(eps)

    def step_epoch(self, epoch: int | None = None) -> float:
        """Updates current_decisiveness_weight according to linear warmup schedule.

        Schedule: lambda(e) = lambda_target * min(1.0, e / W) when W > 0.

        Args:
            epoch: Optional epoch index (0-indexed). If None, increments internal counter.

        Returns:
            Updated current_decisiveness_weight.
        """
        if epoch is not None:
            self.current_epoch = max(0, int(epoch))
        else:
            self.current_epoch += 1

        if self.warmup_epochs <= 0:
            self.current_decisiveness_weight = self.decisiveness_weight
        else:
            progress = min(1.0, float(self.current_epoch) / float(self.warmup_epochs))
            self.current_decisiveness_weight = self.decisiveness_weight * progress

        return self.current_decisiveness_weight
```

#### `forward` Method Update (Lines 249–280):
```python
# Before:
        # Grouped candidate evaluation path
        if group_ids is not None:
            decisiveness_penalty = (
                choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
            ).mean() * self.decisiveness_weight
            ...

        # Legacy fixed-dimension evaluation path
        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None
        if mask_f is not None:
            denom = mask_f.sum().clamp(min=1.0)
            decisiveness_terms = choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
            decisiveness_penalty = (
                (decisiveness_terms * mask_f).sum() / denom
            ) * self.decisiveness_weight
        else:
            decisiveness_penalty = (
                choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
            ).mean() * self.decisiveness_weight

# After:
        # Grouped candidate evaluation path
        if group_ids is not None:
            decisiveness_penalty = (
                choice_pos * choice_neg
            ).mean() * self.current_decisiveness_weight
            ...

        # Legacy fixed-dimension evaluation path
        if active_mask is None:
            active_mask = outputs.get("active_mask")

        mask_f = active_mask.float() if active_mask is not None else None
        if mask_f is not None:
            denom = mask_f.sum().clamp(min=1.0)
            decisiveness_terms = choice_pos * choice_neg
            decisiveness_penalty = (
                (decisiveness_terms * mask_f).sum() / denom
            ) * self.current_decisiveness_weight
        else:
            decisiveness_penalty = (
                choice_pos * choice_neg
            ).mean() * self.current_decisiveness_weight
```

---

### 4.2 Exact Code Changes for `src/yoda/training/trainer.py`

#### `YodaTrainer.__init__` (Lines 85–100, 115–125, 158–162, 194):
```python
# In __init__ arguments:
        ltn_weight: float = 0.1,
        decisiveness_weight: float = 0.1,
        decisiveness_warmup_epochs: int = 0,
        assertion_weight: float = 0.0,
...
# In attribute assignments:
        self.decisiveness_weight = float(decisiveness_weight)
        self.decisiveness_warmup_epochs = max(0, int(decisiveness_warmup_epochs))
...
# In criterion instantiation:
        self.ltn_criterion = LTNConstraintLoss(
            decisiveness_weight=self.decisiveness_weight,
            warmup_epochs=self.decisiveness_warmup_epochs,
        )
```

#### `YodaTrainer.fit` (Lines 713–718):
```python
# Inside fit():
        try:
            for epoch in range(1, epochs + 1):
                # Update LTN decisiveness warmup schedule (epoch - 1 is 0-indexed)
                self.ltn_criterion.step_epoch(epoch - 1)
                train_metrics = self.train_epoch(train_loader)
                ...
                epoch_record["decisiveness_weight"] = self.ltn_criterion.current_decisiveness_weight
                ...
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/decisiveness_weight",
                        self.ltn_criterion.current_decisiveness_weight,
                        epoch,
                    )
```

---

### 4.3 Exact Code Changes for `src/yoda/training/lightning.py`

#### `YodaLightningAdapter.__init__` (Lines 35, 51, 65, 81):
```python
# In __init__ arguments:
        ltn_weight: float = 0.1,
        decisiveness_weight: float = 0.1,
        decisiveness_warmup_epochs: int = 0,
...
# In attribute assignments:
        self.decisiveness_weight = float(decisiveness_weight)
        self.decisiveness_warmup_epochs = max(0, int(decisiveness_warmup_epochs))
        self.ltn_criterion = LTNConstraintLoss(
            decisiveness_weight=self.decisiveness_weight,
            warmup_epochs=self.decisiveness_warmup_epochs,
        )
```

#### `YodaLightningAdapter.on_train_epoch_start` (Add new hook):
```python
    def on_train_epoch_start(self) -> None:
        """Updates LTN constraint loss warmup schedule at the start of each training epoch."""
        self.ltn_criterion.step_epoch(self.current_epoch)
        if self._trainer is not None:
            self.log(
                "train/decisiveness_weight",
                self.ltn_criterion.current_decisiveness_weight,
                on_step=False,
                on_epoch=True,
            )
```

---

### 4.4 Exact Changes for `tests/test_ltn.py`

Update `test_decisiveness_penalty` and `test_ltn_init_decisiveness_weight`:
```python
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
```

---

### 4.5 Full Specification for New `tests/test_contradiction_loss.py`

Create `tests/test_contradiction_loss.py` with the following structure:
```python
"""Unit tests for Paraconsistent Contradiction Loss and Linear Warmup Schedule."""

import pytest
import torch

from yoda.nesy.ltn import LTNConstraintLoss
from yoda.training.lightning import YodaLightningAdapter
from yoda.training.trainer import YodaTrainer


class TestBelnapCorners:
    """Verifies contradiction loss behavior across all 4 Belnap-Dunn valuation corners."""

    def test_true_consistent_zero_loss(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.ones((2, 3)),
            "choice_neg": torch.zeros((2, 3)),
            "truth": torch.ones((2, 3)),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_false_consistent_zero_loss(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.zeros((2, 3)),
            "choice_neg": torch.ones((2, 3)),
            "truth": torch.zeros((2, 3)),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_neither_unmentioned_zero_loss(self) -> None:
        """Neither (pos=0, neg=0) must incur ZERO loss, preventing false penalties."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.zeros((2, 3)),
            "choice_neg": torch.zeros((2, 3)),
            "truth": torch.full((2, 3), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_both_contradiction_maximum_loss(self) -> None:
        """Both (pos=1, neg=1) incurs maximal penalty equal to current_decisiveness_weight."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.ones((2, 3)),
            "choice_neg": torch.ones((2, 3)),
            "truth": torch.full((2, 3), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.2))


class TestContradictionGradients:
    """Verifies gradient properties of paraconsistent Both penalty."""

    def test_zero_gradient_when_conflicting_evidence_absent(self) -> None:
        """When neg=0, d(loss)/d(pos) must be exactly zero."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        pos = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        neg = torch.zeros(3)
        outputs = {
            "choice_pos": pos,
            "choice_neg": neg,
            "truth": torch.full((3,), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))
        loss.backward()
        assert pos.grad is not None
        assert torch.all(pos.grad == 0.0)

    def test_gradient_proportional_to_opposite_polarity(self) -> None:
        """Gradient on pos is strictly proportional to neg."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.5)
        pos = torch.full((4,), 0.5, requires_grad=True)
        neg = torch.tensor([0.1, 0.3, 0.5, 0.9])
        outputs = {
            "choice_pos": pos,
            "choice_neg": neg,
            "truth": torch.full((4,), 0.5),
        }
        loss = criterion(outputs)
        loss.backward()
        assert pos.grad is not None
        # dL/d(pos_i) = neg_i / 4 * 0.5
        expected_grads = neg / 4.0 * 0.5
        assert torch.allclose(pos.grad, expected_grads, atol=1e-6)

    def test_symmetry_commutativity(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.1)
        a = torch.rand(5, 5)
        b = torch.rand(5, 5)
        loss1 = criterion({"choice_pos": a, "choice_neg": b, "truth": torch.zeros(5, 5)})
        loss2 = criterion({"choice_pos": b, "choice_neg": a, "truth": torch.zeros(5, 5)})
        assert torch.isclose(loss1, loss2)


class TestLinearWarmupSchedule:
    """Verifies warmup progression across epochs."""

    def test_warmup_progression(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2, warmup_epochs=4)
        assert criterion.current_decisiveness_weight == 0.0

        # Epoch 0
        w0 = criterion.step_epoch(0)
        assert w0 == 0.0

        # Epoch 1
        w1 = criterion.step_epoch(1)
        assert pytest.approx(w1) == 0.05

        # Epoch 2
        w2 = criterion.step_epoch(2)
        assert pytest.approx(w2) == 0.10

        # Epoch 3
        w3 = criterion.step_epoch(3)
        assert pytest.approx(w3) == 0.15

        # Epoch 4
        w4 = criterion.step_epoch(4)
        assert pytest.approx(w4) == 0.20

        # Epoch 5 (saturated)
        w5 = criterion.step_epoch(5)
        assert pytest.approx(w5) == 0.20

    def test_auto_increment_step_epoch(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.3, warmup_epochs=3)
        assert criterion.step_epoch() == 0.1  # epoch 1
        assert criterion.step_epoch() == 0.2  # epoch 2
        assert pytest.approx(criterion.step_epoch()) == 0.3  # epoch 3
        assert pytest.approx(criterion.step_epoch()) == 0.3  # epoch 4
```

---

## 5. Verification Method

To independently verify the implementation after code editing:

1. **Unit Tests Execution**:
   ```bash
   uv run pytest tests/test_ltn.py tests/test_contradiction_loss.py -v
   ```
   **Expected Result**: All tests pass with zero failures. Specifically, `test_decisiveness_penalty` passes under the new paraconsistent contract, and `test_contradiction_loss.py` validates all 4 Belnap corner cases, gradients, and warmup progressions.

2. **Full Regression Test Suite**:
   ```bash
   uv run pytest tests/ -v
   ```
   **Expected Result**: All 181+ tests in `tests/` pass, including `test_independent_eval.py`, `test_training.py`, and `test_lightning.py`.

3. **Lint & Style Compliance**:
   ```bash
   uv run ruff check .
   ```
   **Expected Result**: Zero errors reported.

4. **Invalidation Conditions**:
   - If `outputs_neither` produces `loss > 0.0`, the paraconsistent contract is violated (penalty on unmentioned options).
   - If `outputs_both` produces `loss == 0.0`, the contradiction loss has reverted to quadratic Jacobian zeroing.
   - If `current_decisiveness_weight` does not ramp linearly from $0.0$ to $\lambda_{\text{target}}$ over `warmup_epochs`, the warmup schedule is invalidated.
