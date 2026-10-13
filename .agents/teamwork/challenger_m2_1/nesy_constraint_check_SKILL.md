# Neuro-Symbolic Constraint Verification (Local Copy)
Source: /home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md

## Core Concepts
- Hard Constraints: Boolean invariants that must never be violated by a decision.
- Continuous Relaxations (LTN): Differentiable fuzzy logic operators (Product, Łukasiewicz, Gödel t-norms) mapping satisfaction to the unit interval `[0, 1]`.
- Many-Valued Semantics: Handling unknown or conflicting information via Kleene 3-valued (True, False, Unknown) or Belnap 4-valued (None, Both) logic.

## Verification Steps
1. Schema Check: Validate that `DecisionPayload.constraints` contains valid constraint expressions or references.
2. Boundary Testing with Hypothesis: Property-based testing verifying mathematical invariants:
   - Commutativity: `T(x, y) == T(y, x)`
   - Monotonicity: `x <= y` implies `T(x, z) <= T(y, z)`
   - Associativity: `T(x, T(y, z)) == T(T(x, y), z)`
   - Boundary conditions: `T(x, 1) == x`, `T(x, 0) == 0`
3. Violation Handling: Explicit explanation and safe fallback state.
