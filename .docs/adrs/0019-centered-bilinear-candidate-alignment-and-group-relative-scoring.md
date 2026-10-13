# ADR 0019: Centered Bilinear Candidate Alignment and Group-Relative Scoring

## Status

Accepted

## Date

2026-10-13

## Context

Independent choice evaluation (`num_choices == 1`, ADR-0013) scored candidates with a linear
head over the pooled final state. Because the final state is a convex residual join of the
context-enriched query and the candidate representation, the context term is identical across
a question's candidates and cancels in grouped focal/margin losses (`z_y - z_j`): the model
ranked candidates only on generic plausibility, producing a hard 50% accuracy ceiling on
context-dependent benchmarks. Wiring bilinear candidate affinity into the independent path
exposed three further problems:

1. **Magnitude inflation**: raw `[0, 1]` evidence channels have ~0.5 mass, so any two encoded
   representations produce a large positive inner product. Unmentioned candidates scored
   high positive support and refutation affinity, landing at the contradictory Both midpoint
   instead of Neither, and violating the epistemic calibration gate (k < 0.20).
2. **Common-direction collapse**: per-candidate evidence targets (Belnap loss) are satisfiable
   by growing a candidate-common plausibility direction shared by the whole group, which
   moves every candidate's logit in lockstep and leaves within-group ranking undetermined.
3. **Washed contrast**: the Belnap MPA's sigmoid-bounded evidence channels contract
   per-candidate contrast by over an order of magnitude, starving the bilinear alignment
   of signal at initialization.

## Decision

1. **Centered truth-order bilinear affinity** (`BelnapDecisionHead`): the candidate
   contributes centered coordinates `eps = 2e - 1` while the query contributes raw evidence
   mass (confidence weighting), scaled by `1 / sqrt(d)`:
   `pos_aff = (q_pos . eps_c_pos + q_neg . eps_c_neg) / sqrt(d)` and the cross-polarity
   analogue for `neg_aff`. Neutral (~0.5) candidate evidence maps to ~0, so unrelated
   pairs produce zero affinity and decisive candidates transmit full signal.
   `logits = (pos_aff - neg_aff) * cand_gate * logit_scale`.
2. **Neither-at-zero evidence**: choice evidence is `sigmoid(a) * |tanh(a)| * cand_gate`,
   which rests at (0, 0) for zero affinity, keeps gradients alive on both affinity signs,
   and saturates to the correct polarity for decisive alignment. The candidate knowledge
   gate uses the information mass `(|e_pos| + |e_neg|) / 2`, identical to the knowledge
   coordinate for raw evidence and equal to the contrast magnitude for centered features.
3. **Group-relative candidate scoring** (`YodaDecisionEngine._center_candidate_states`):
   candidates sharing an identical `(query, state, task)` context form one decision
   group; each candidate's features are replaced by their deviation from the group mean
   (restored to the evidence range by a +0.5 midpoint offset that the head's `2e - 1`
   transform cancels exactly). The common plausibility direction is subtracted out, so
   within-group ranking is learned on distinctive signal. Single-candidate groups keep
   raw features to avoid dimensional collapse.
4. **Raw candidate features for the head**: the independent path pools candidate token
   embeddings directly (masked mean, polarity split `sigmoid(z)` / `sigmoid(-z)`); the
   Belnap MPA evidence stays reserved for the reasoning core, whose bounded channels wash
   out the per-candidate contrast the alignment needs.
5. **`logit_scale` (default 8.0)**: amplifies the ranking logits so grouped focal/margin
   losses receive useful gradients from step one, while evidence magnitudes keep their
   paraconsistent calibration independently of the ranking temperature.

## Consequences

- **Positive**: The 50% context-cancellation ceiling is broken — independent-choice
  validation accuracy on context-dependent benchmarks exceeds the 60% acceptance gate
  across random initialization seeds.
- **Positive**: Unmentioned candidates retain Neither coordinates (k < 0.20) without
  incurring constraint loss penalties; decisive candidates retain unattenuated dynamic
  range; low-knowledge candidates suppress to zero logits.
- **Positive**: Negation equivariance of the head's polarity handling is preserved:
  swapping candidate evidence polarities swaps the affinity signs exactly.
- **Negative**: The head's bilinear branch behavior differs numerically from the linear
  branches; head-level tests pin the linear branches (`logits = raw_diff * knowledge`)
  while engine-level gates pin the bilinear branch.
- **Negative**: Group-relative scoring requires identical `(query, state, task)` grouping,
  which is guaranteed by the unrolled collate contract; callers bypassing the collate
  with interleaved duplicates would form incorrect groups.

## Related

- Supersedes the linear scoring portion of [ADR-0013](./0013-independent-choice-assessment-architecture.md)
  for the independent path; [ADR-0020](./0020-dual-channel-negation-equivariant-ffn.md)
  covers the companion FFN refactor.
