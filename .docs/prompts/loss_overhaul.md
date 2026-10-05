Task: Overhaul training losses to use Belnap loss as primary and add Decisiveness Penalty.

Changes:
- In `LTNConstraintLoss`, replaced gullibility/ignorance penalties with a `decisiveness_penalty` to penalize epistemic fence-sitting at 0.5.
- Promoted `belnap_loss` to the primary driving loss in `YodaTrainer` (default weight = 1.0).
- Excluded Cross-Entropy (`focal_loss`) from gradient optimization.
- Reformulated `total_loss` as: `(belnap_loss * belnap_weight) + (margin_loss * margin_weight) + (ltn_loss * current_ltn_w) + (assertion_loss * current_assertion_w)`.
- Updated CLI arguments and tests to reflect the new loss formula and default weights.
