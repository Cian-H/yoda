# Handoff Report: R3 Epistemic Logit Synchronization & Knowledge Gating

## 1. Observation

### 1.1 Existing Implementations & Divergence
1. In `src/yoda/architecture/engine.py:91-144`, `BelnapDecisionHead.forward` calculates decision logits via raw unweighted pre-sigmoid difference:
   ```python
   # Line 91-101 (num_choices == 1)
   if self.num_choices == 1:
       raw_pos = self.w_pos(pooled_pos).squeeze(-1)
       raw_neg = self.w_neg(pooled_neg).squeeze(-1)
       choice_pos = torch.sigmoid(raw_pos)
       choice_neg = torch.sigmoid(raw_neg)
       logits = raw_pos - raw_neg
       choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
       truth = choice_state.truth
       knowledge = choice_state.knowledge
       choice = (logits > 0).long()
   ```
   ```python
   # Line 123 & 129 (num_choices > 1)
   if candidate_states is not None:
       ...
       logits = pos_affinity - neg_affinity
   else:
       raw_pos = self.w_pos(pooled_pos)
       raw_neg = self.w_neg(pooled_neg)
       choice_pos = torch.sigmoid(raw_pos)
       choice_neg = torch.sigmoid(raw_neg)
       logits = raw_pos - raw_neg
   ```
   In contrast, `src/yoda/architecture/belnap_transformer.py:473` gates decision transformer logits by epistemic knowledge coordinates:
   ```python
   logits = (choice_pos - choice_neg) * knowledge
   ```

2. When `self.num_choices == 1`, `BelnapDecisionHead.forward` exits at line 91 prior to evaluating `if candidate_states is not None:` (lines 110-123). As a consequence, candidate probes supplied in scalar independent mode are ignored, and pairwise bilinear alignment is bypassed.

3. When an input state represents pure ignorance ($Neither$, $e_{\text{pos}} = 0, e_{\text{neg}} = 0$), linear projection through `w_pos`/`w_neg` yields bias values near 0 ($\text{bias} \approx 0$). Applying `torch.sigmoid(0.0)` produces choice evidence $e_{\text{pos}} \approx 0.5, e_{\text{neg}} \approx 0.5$, artificially assigning choice knowledge $k = \frac{0.5 + 0.5}{2} = 0.5$ rather than $k \to 0$.

### 1.2 Verbatim Test Failures
Running `uv run pytest` produced 24 failures across the suite. The failures directly related to R3 and epistemic logit gating include:

1. `tests/e2e/test_paraconsistent_gate.py:74`:
   ```
   E AssertionError: Knowledge must be near 0, got 0.534935
   E assert 0.5349346399307251 < 0.01
   ```
   Verbatim test code in `tests/e2e/test_paraconsistent_gate.py:58-79`:
   ```python
   def test_low_knowledge_states_suppress_logits() -> None:
       head = BelnapDecisionHead(d_model=32, num_choices=1)
       low_k_state = BelnapState(
           e_pos=torch.ones(1, 32) * 1e-4,
           e_neg=torch.ones(1, 32) * 1e-4,
       )
       out = head(low_k_state)
       logit = out["logits"].item()
       knowledge = out["knowledge"].item()
       assert knowledge < 0.01
       assert abs(logit) < 0.25
   ```

2. `tests/e2e/test_tier2_boundary_corner.py:230`:
   ```
   E AssertionError: Zero-knowledge logit must be suppressed to 0, got 0.020482145249843597
   E assert 0.020482145249843597 < 0.001
   ```
   Verbatim test code in `tests/e2e/test_tier2_boundary_corner.py:222-230`:
   ```python
   def test_r3_boundary_zero_knowledge_logit_suppression(self) -> None:
       head = BelnapDecisionHead(d_model=32, num_choices=1)
       state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
       out = head(state)
       logit = out["logits"].item()
       assert abs(logit) < 1e-3
   ```

3. `tests/e2e/test_tier3_cross_feature.py:175`:
   ```
   E AssertionError: Bilinear scoring on zero-knowledge candidate must be suppressed
   E assert 0.2948966324329376 < 0.2
   ```
   Verbatim test code in `tests/e2e/test_tier3_cross_feature.py:165-177`:
   ```python
   def test_t3_r3_epistemic_logits_x_r4_bilinear_scoring(self) -> None:
       head = BelnapDecisionHead(d_model=32, num_choices=1)
       final_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
       cand_state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
       out = head(final_state, candidate_states=cand_state)
       assert abs(out["logits"].item()) < 0.2
   ```

---

## 2. Logic Chain

1. **Premise 1 (Admissibility of Epistemic Coordinates in $\mathcal{B}_4$)**:
   In the Belnap-Dunn bilattice $\mathcal{B}_4$, the knowledge coordinate $k = \frac{e_{\text{pos}} + e_{\text{neg}}}{2} \in [0, 1]$ represents the total mass of grounded evidence. The state $Neither = (0, 0)$ corresponds strictly to $k = 0$, while $Both = (1, 1)$ corresponds to $k = 1.0$, and $True = (1, 0)$ corresponds to $k = 0.5$.
   By definition, a decision system with zero evidence ($k = 0$) cannot possess positive support or refutation for any option, nor can it assert high classification confidence.

2. **Premise 2 (Failure of Sigmoid Projection without Grounding Gate)**:
   Per Observation 1.1(3), projecting a zero-evidence representation $e_{\text{pos}} = 0$ through an affine layer $W e + b$ yields pre-activation $z \approx b \approx 0$. Because $\sigma(0) = 0.5$, the choice evidence coordinates default to $(0.5, 0.5)$, producing a synthetic choice knowledge $k_{\text{choice}} = 0.5$. This violates the conservation of epistemic ignorance, causing `test_low_knowledge_states_suppress_logits` to fail with $k = 0.5349 \not< 0.01$.

3. **Premise 3 (Evidence Conservation Gating)**:
   To prevent synthetic evidence generation, the output choice evidence must be modulated by the candidate's intrinsic knowledge mass $k_{\text{cand}} = \frac{e_{\text{cand}, \text{pos}} + e_{\text{cand}, \text{neg}}}{2}$.
   In Belnap logic, a fully decisive option ($True$) possesses $e^+ = 1, e^- = 0$, giving $k_{\text{cand}} = 0.5$.
   Therefore, the scaling factor that maps $k_{\text{cand}} \in [0, 1]$ to an evidence preservation gate in $[0, 1]$ without attenuating decisive states is:
   $$g = \min(1.0, 2 \times k_{\text{cand}})$$
   - When $k_{\text{cand}} = 0$ ($Neither$): $g = 0.0$, yielding $e_{\text{choice}}^+ = 0, e_{\text{choice}}^- = 0, k_{\text{choice}} = 0.0$.
   - When $k_{\text{cand}} \approx 10^{-4}$: $g \approx 2 \times 10^{-4}$, ensuring $k_{\text{choice}} \le 2 \times 10^{-4} < 0.01$.
   - When $k_{\text{cand}} \ge 0.5$ ($True$, $False$, or $Both$): $g = 1.0$, transmitting the full dynamic range of $\sigma(z)$ with zero attenuation.

4. **Premise 4 (Epistemic Logit Gating)**:
   Per Observation 1.1(1) and Observation 1.2(2, 3), classification logits in `BelnapDecisionHead` are currently evaluated as $\text{logits} = \text{raw\_pos} - \text{raw\_neg}$ or $\text{pos\_affinity} - \text{neg\_affinity}$.
   Because pre-sigmoid affine differences $\Delta z$ are unconstrained in $(-\infty, +\infty)$, candidates with zero evidence retain residual logit differences (e.g., $0.0205$ or $0.2949$), resulting in acceptance test failures ($> 10^{-3}$ and $> 0.2$).
   Multiplying pre-sigmoid differences by the choice knowledge coordinate:
   $$\text{logits} = (\text{raw\_pos} - \text{raw\_neg}) \times \text{knowledge}$$
   or
   $$\text{logits} = (\text{pos\_affinity} - \text{neg\_affinity}) \times \text{knowledge}$$
   strictly guarantees:
   $$\lim_{k \to 0} \text{logits} = 0.0$$
   When logits evaluate to $0.0$, the resulting Softmax distribution over $K$ candidates is uniform ($p_i = 1/K$), preventing ungrounded focal loss penalties and spurious confidence.

5. **Premise 5 (Bilinear Candidate State Interception for Scalar Heads)**:
   Per Observation 1.1(2), moving the `if candidate_states is not None:` branch ahead of the scalar linear projection allows `num_choices == 1` scalar heads to compute pairwise bilinear alignment against 2D (`[B, D]`) or 3D (`[B, 1, D]`) candidate states. This resolves the R3 x R4 cross-interaction tested in `test_t3_r3_epistemic_logits_x_r4_bilinear_scoring`.

---

## 3. Caveats

1. **Downstream Phase 3 (R4) Candidate Caching Scope**:
   This plan focuses on logit synchronization and knowledge gating in `BelnapDecisionHead.forward`. The deduplication of context encoders in `YodaDecisionEngine` across grouped candidates is deferred to Phase 4 (M4, R5).
2. **Backpropagation of Product Rule**:
   Gating logits by knowledge $\text{logits} = \Delta z \cdot k$ introduces a product rule gradient $\frac{\partial \text{logits}}{\partial z} = k + \Delta z \frac{\partial k}{\partial z}$. When $k > 0$, the $k$ term guarantees positive gradient highways even at zero logit margin ($\Delta z = 0$). When $k = 0$, gradients are naturally zeroed because ungrounded inputs provide zero supervision signal.

---

## 4. Conclusion & Proposed Implementation

### 4.1 Target File: `src/yoda/architecture/engine.py`

Modify `BelnapDecisionHead.forward` (lines 84-145) to:
1. Extract candidate intrinsic knowledge mass $k_{\text{cand}}$.
2. Modulate choice evidence via gate $g = \text{clamp}(2 k_{\text{cand}}, \max=1.0)$.
3. Harmonize logit computation to $\text{logits} = \Delta \text{affinity} \times \text{knowledge}$.
4. Support 2D and 3D `candidate_states` for both $K > 1$ and $K = 1$.

#### Code Replacement Proposal for `src/yoda/architecture/engine.py`:
```python
    def forward(
        self,
        x: BelnapState,
        candidate_states: BelnapState | None = None,
    ) -> dict[str, torch.Tensor]:
        """Projects dual evidence states into decision logits and coordinates.

        If candidate_states is provided (shape: [batch_size, num_choices, d_model]
        or [batch_size, d_model]), computes pairwise candidate-level alignment scores.
        Otherwise falls back to linear projection heads over pooled evidence.
        Both scoring mechanisms are gated by epistemic knowledge coordinates.

        Args:
            x: Input BelnapState of shape `(batch_size, num_queries, d_model)` or
                `(batch_size, d_model)`.
            candidate_states: Optional per-candidate BelnapState of shape
                `(batch_size, num_choices, d_model)` or `(batch_size, d_model)`.

        Returns:
            Dictionary containing:
            - `logits`: Knowledge-gated decision logits.
            - `truth`: Choice truth coordinates in [0, 1].
            - `knowledge`: Choice knowledge coordinates in [0, 1].
            - `choice`: Selected choice indices.
            - `choice_pos`: Choice positive evidence in [0, 1].
            - `choice_neg`: Choice negative evidence in [0, 1].
        """
        if x.e_pos.dim() == 3:
            pooled_pos = x.e_pos.mean(dim=1)
            pooled_neg = x.e_neg.mean(dim=1)
        else:
            pooled_pos = x.e_pos
            pooled_neg = x.e_neg

        scale = self.d_model**0.5

        if candidate_states is not None:
            # Pairwise candidate alignment scoring
            if candidate_states.e_pos.dim() == 2:
                # 2D candidate states: [B, D] (e.g. scalar head evaluation)
                q_p = self.cand_query_pos(pooled_pos)            # [B, D]
                q_n = self.cand_query_neg(pooled_neg)            # [B, D]
                c_p = self.cand_opt_pos(candidate_states.e_pos)  # [B, D]
                c_n = self.cand_opt_neg(candidate_states.e_neg)  # [B, D]

                pos_affinity = (q_p * c_p + q_n * c_n).sum(dim=-1) / scale  # [B]
                neg_affinity = (q_p * c_n + q_n * c_p).sum(dim=-1) / scale  # [B]

                cand_k = (candidate_states.e_pos + candidate_states.e_neg).mean(dim=-1) / 2.0  # [B]
                raw_diff = pos_affinity - neg_affinity
            else:
                # 3D candidate states: [B, num_choices, D]
                q_p = self.cand_query_pos(pooled_pos).unsqueeze(1)  # [B, 1, D]
                q_n = self.cand_query_neg(pooled_neg).unsqueeze(1)  # [B, 1, D]
                c_p = self.cand_opt_pos(candidate_states.e_pos)     # [B, num_choices, D]
                c_n = self.cand_opt_neg(candidate_states.e_neg)     # [B, num_choices, D]

                pos_affinity = (q_p * c_p + q_n * c_n).sum(dim=-1) / scale  # [B, num_choices]
                neg_affinity = (q_p * c_n + q_n * c_p).sum(dim=-1) / scale  # [B, num_choices]

                cand_k = (candidate_states.e_pos + candidate_states.e_neg).mean(dim=-1) / 2.0  # [B, num_choices]
                raw_diff = pos_affinity - neg_affinity
                if self.num_choices == 1 and raw_diff.size(-1) == 1:
                    pos_affinity = pos_affinity.squeeze(-1)
                    neg_affinity = neg_affinity.squeeze(-1)
                    raw_diff = raw_diff.squeeze(-1)
                    cand_k = cand_k.squeeze(-1)

            cand_gate = (2.0 * cand_k).clamp(max=1.0)
            choice_pos = torch.sigmoid(pos_affinity) * cand_gate
            choice_neg = torch.sigmoid(neg_affinity) * cand_gate

        elif self.num_choices == 1:
            raw_pos = self.w_pos(pooled_pos).squeeze(-1)  # [B]
            raw_neg = self.w_neg(pooled_neg).squeeze(-1)  # [B]
            raw_diff = raw_pos - raw_neg

            state_k = (pooled_pos + pooled_neg).mean(dim=-1) / 2.0  # [B]
            cand_gate = (2.0 * state_k).clamp(max=1.0)
            choice_pos = torch.sigmoid(raw_pos) * cand_gate
            choice_neg = torch.sigmoid(raw_neg) * cand_gate

        else:
            raw_pos = self.w_pos(pooled_pos)  # [B, num_choices]
            raw_neg = self.w_neg(pooled_neg)  # [B, num_choices]
            raw_diff = raw_pos - raw_neg

            state_k = (pooled_pos + pooled_neg).mean(dim=-1, keepdim=True) / 2.0  # [B, 1]
            cand_gate = (2.0 * state_k).clamp(max=1.0)
            choice_pos = torch.sigmoid(raw_pos) * cand_gate
            choice_neg = torch.sigmoid(raw_neg) * cand_gate

        choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
        truth = choice_state.truth
        knowledge = choice_state.knowledge

        # Epistemic synchronization: logits gated by candidate knowledge coordinate
        logits = raw_diff * knowledge

        if self.num_choices == 1 or logits.dim() == 1:
            choice = (logits > 0).long()
        else:
            choice = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "truth": truth,
            "knowledge": knowledge,
            "choice": choice,
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
        }
```

---

### 4.2 Target File: `tests/test_engine.py`

Add targeted unit tests in `TestBelnapDecisionHead`:
```python
    def test_epistemic_logit_gating_zero_knowledge_suppression(self) -> None:
        """Verifies that zero knowledge (k=0) suppresses logits to 0 regardless of pre-activations."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state_zero = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
        out = head(state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_multi_choice_zero_evidence(self) -> None:
        """Verifies multi-choice head suppresses all choice logits when evidence is zero."""
        head = BelnapDecisionHead(d_model=32, num_choices=5)
        state_zero = BelnapState(e_pos=torch.zeros(3, 4, 32), e_neg=torch.zeros(3, 4, 32))
        out = head(state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_with_bilinear_candidate_states(self) -> None:
        """Verifies bilinear candidate scoring suppresses logits when candidate has zero evidence."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        final_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        cand_state_zero = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
        out = head(final_state, candidate_states=cand_state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_unattenuated_decisive_state(self) -> None:
        """Verifies that a decisive state (k=0.5, truth=1.0) transmits non-zero positive logits."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state_decisive = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state_decisive)
        assert out["knowledge"].item() > 0.4
        assert out["logits"].item() > 0.0
```

---

### 4.3 New Test File: `tests/test_knowledge_gating.py`

Create a dedicated test module verifying epistemic logit synchronization, monotonicity, Softmax entropy calibration, and gradient health:

```python
"""Unit and property tests for Epistemic Logit Synchronization and Knowledge Gating."""

import pytest
import torch
import torch.nn.functional as F

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import BelnapDecisionHead


class TestEpistemicLogitGating:
    """Verifies that decision head logits are gated by epistemic knowledge coordinates."""

    def test_zero_knowledge_suppression_single_choice(self) -> None:
        """Zero evidence (e_pos=0, e_neg=0) must produce exactly 0 knowledge and zero logits."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        assert out["knowledge"].item() == 0.0
        assert abs(out["logits"].item()) < 1e-4
        assert pytest.approx(out["truth"].item(), abs=1e-4) == 0.5  # Neither truth value is 0.5

    def test_near_zero_knowledge_suppression_low_evidence(self) -> None:
        """Near-zero evidence (k ~ 1e-4) must keep knowledge < 0.01 and logit < 0.25."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.ones(1, 32) * 1e-4, e_neg=torch.ones(1, 32) * 1e-4)
        out = head(state)
        assert out["knowledge"].item() < 0.01
        assert abs(out["logits"].item()) < 0.25

    def test_maximal_knowledge_unsuppressed_dynamic_range(self) -> None:
        """Decisive state (k=0.5, truth=1.0) must transmit unattenuated dynamic range."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        assert out["knowledge"].item() > 0.4
        assert out["logits"].item() > 0.0

    def test_bilinear_candidate_knowledge_gating_scalar(self) -> None:
        """Bilinear candidate head must gate logits by candidate knowledge."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        # Candidate 0 has full evidence, Candidate 1 has zero evidence
        cand_state = BelnapState(
            e_pos=torch.tensor([[1.0] * 32, [0.0] * 32]),
            e_neg=torch.tensor([[0.0] * 32, [0.0] * 32]),
        )
        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"]
        assert abs(logits[1].item()) < 1e-4, "Zero knowledge candidate must be suppressed to 0"
        assert logits[0].item() > logits[1].item()

    def test_bilinear_candidate_knowledge_gating_multi_choice(self) -> None:
        """Multi-choice candidate alignment must suppress zero-knowledge candidate positions."""
        head = BelnapDecisionHead(d_model=32, num_choices=3)
        query_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        # Candidate 0 has evidence; Candidate 1 and 2 are pure Neither
        cand_pos = torch.zeros(1, 3, 32)
        cand_pos[0, 0] = 1.0
        cand_neg = torch.zeros(1, 3, 32)
        cand_state = BelnapState(e_pos=cand_pos, e_neg=cand_neg)

        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"][0]
        assert abs(logits[1].item()) < 1e-4
        assert abs(logits[2].item()) < 1e-4
        assert logits[0].item() > logits[1].item()

    def test_uniform_softmax_probabilities_under_zero_knowledge(self) -> None:
        """Zero-knowledge candidates must yield uniform Softmax distributions (1/K)."""
        head = BelnapDecisionHead(d_model=32, num_choices=4)
        zero_state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(zero_state)
        probs = F.softmax(out["logits"], dim=-1)[0]
        for p in probs:
            assert pytest.approx(p.item(), abs=1e-3) == 0.25

    def test_gradient_flow_through_knowledge_gated_head(self) -> None:
        """Backward propagation through knowledge-gated logits must produce valid gradients."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(
            e_pos=torch.rand(2, 32, requires_grad=True),
            e_neg=torch.rand(2, 32, requires_grad=True),
        )
        out = head(state)
        loss = out["logits"].sum()
        loss.backward()

        assert head.w_pos.weight.grad is not None
        assert not torch.isnan(head.w_pos.weight.grad).any()
        assert head.w_neg.weight.grad is not None
        assert not torch.isnan(head.w_neg.weight.grad).any()
```

---

## 5. Verification Method

1. **Verify Target Acceptance & Unit Tests**:
   Execute the specific suite of boundary, acceptance, and regression tests:
   ```bash
   uv run pytest tests/e2e/test_paraconsistent_gate.py
   uv run pytest tests/e2e/test_tier1_feature_coverage.py -k test_r3_epistemic_logit_synchronization
   uv run pytest tests/e2e/test_tier2_boundary_corner.py -k test_r3_boundary_zero_knowledge_logit_suppression
   uv run pytest tests/e2e/test_tier3_cross_feature.py -k test_t3_r3_epistemic_logits_x_r4_bilinear_scoring
   uv run pytest tests/test_engine.py
   uv run pytest tests/test_knowledge_gating.py
   ```
2. **Lint and Style Compliance**:
   ```bash
   uv run ruff check src/yoda/architecture/engine.py tests/test_engine.py tests/test_knowledge_gating.py
   ```
3. **Invalidation Conditions**:
   - If `out["knowledge"] >= 0.01` when evaluating `BelnapState(1e-4, 1e-4)`.
   - If `abs(out["logits"]) >= 1e-3` when evaluating `BelnapState(0, 0)`.
   - If `out["logits"] <= 0.0` when evaluating decisive evidence `BelnapState(1.0, 0.0)`.
