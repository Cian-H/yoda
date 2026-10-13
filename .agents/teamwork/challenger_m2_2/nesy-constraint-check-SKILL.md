---
name: nesy-constraint-check
description: Verify neuro-symbolic logic invariants, continuous differentiable t-norm relaxations (LTN), and Kleene/Belnap truth valuations for decision payloads. Use when adding or testing rule constraints, checking boundary conditions with Hypothesis, or evaluating constraint violations.
---

# Neuro-Symbolic Constraint Verification

Protocol for implementing and verifying symbolic invariants, logic constraints, and continuous relaxations in Yoda.

## Core Concepts

- **Hard Constraints**: Boolean invariants that must never be violated by a decision.
- **Continuous Relaxations (LTN)**: Differentiable fuzzy logic operators (Product, Łukasiewicz, Gödel t-norms) mapping satisfaction to the unit interval `[0, 1]`.
- **Many-Valued Semantics**: Handling unknown or conflicting information via Kleene 3-valued (True, False, Unknown) or Belnap 4-valued (None, Both) logic.

## Verification Steps

1. **Schema Check**:
   Validate that `DecisionPayload.constraints` contains valid constraint expressions or references.

2. **Boundary Testing with Hypothesis**:
   Property-based testing must verify that continuous t-norms satisfy mathematical invariants:
   - Commutativity: `T(x, y) == T(y, x)`
   - Monotonicity: `x <= y` implies `T(x, z) <= T(y, z)`
   - Associativity: `T(x, T(y, z)) == T(T(x, y), z)`
   - Boundary conditions: `T(x, 1) == x`, `T(x, 0) == 0`

   Run:
   ```bash
   uv run pytest tests/test_nesy.py
   ```

3. **Violation Handling**:
   - If a constraint is violated, verify that the engine generates an explicit explanation and falls back to a safe fallback state rather than crashing.
