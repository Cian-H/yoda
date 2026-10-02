# ADR 0011: Active Criteria Masking and Dynamic Candidate Epistemic Affinity Scoring

## Status
Accepted

## Date
2026-10-02

## Context
In realistic decision tasks, queries involve variable numbers of candidate criteria (from binary decisions up to dozens of candidate options). While the Parquet ETL pipeline standardizes batched tensor dimensions to a fixed `max_choices` (e.g. 5) by padding with `"none: Unused option"`:
1. **Logit & Softmax Distortion**: Passing unmasked logits to standard `nn.CrossEntropyLoss` includes padded choices in the softmax denominator, distorting probabilities and gradients.
2. **Pathology in Neuro-Symbolic Logic Losses**: In `LTNConstraintLoss`, calculating universal bivalence ($-\log(e^+(1-e^-) + e^-(1-e^+))$) and Exactly-One semantic loss ($P(\text{XOR})$) over inactive slots penalizes the model for padded options, violating the intended epistemic ignorance semantics.
3. **Fixed Head vs. Dynamic Candidate Scoring**: A static linear layer `nn.Linear(d_model, num_choices)` constrains evaluation to a fixed number of slots rather than evaluating arbitrary candidate sets dynamically.

## Decision
1. **Active Criteria Masking Across Data & Losses**:
   - `YodaParquetDataset` and `YodaDecisionDataset` expose `num_active` per sample.
   - `collate_decision_batch` constructs a boolean `active_mask: torch.Tensor` of shape `(batch_size, max_choices)`.
   - `YodaDecisionEngine` auto-detects or receives `active_mask`, masking inactive logits with `-1e9` and clamping inactive truth/knowledge/evidence coordinates to 0.0 (Belnap ignorance $\bot$).
   - `LTNConstraintLoss` conditions universal bivalence and Exactly-One XOR calculations strictly across active candidate indices per sample.
2. **Dynamic Candidate Epistemic Affinity Scoring**:
   - `ConstraintEncoder` adds `encode_candidates` to encode each candidate criterion individually into tensor representations of shape `(batch_size, num_candidates, embed_dim)`.
   - `YodaDecisionEngine` supports dynamic candidate evaluation via `use_candidate_affinity: bool`, projecting candidate embeddings into `candidate_states: BelnapState`.
   - `BelnapDecisionHead` evaluates bilateral epistemic affinity:
     $$\text{Affinity}(q, c_i) = \frac{q^+ \cdot c_i^+ + q^- \cdot c_i^-}{\sqrt{d}} - \frac{q^+ \cdot c_i^- + q^- \cdot c_i^+}{\sqrt{d}}$$
     allowing evaluation of arbitrary candidate counts $N$ dynamically.

## Consequences
- **Positive**: Softmax cross-entropy and LTN semantic loss strictly evaluate across active criteria.
- **Positive**: Inactive options receive 0 probability mass and cannot be chosen during inference.
- **Positive**: The engine can evaluate variable numbers of candidate criteria on the fly via dynamic epistemic affinity scoring without being constrained to a fixed $K$-way head.
- **Positive**: Backward compatibility is fully preserved for existing models and configurations.
