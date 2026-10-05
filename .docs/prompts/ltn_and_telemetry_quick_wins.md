Task: Implement Quick Wins (Telemetry, Assertion Loss) and Major Project (Belnap LTN Epistemic Regularization).

Changes:
- Added NaN monitoring, loss spike monitoring, and infinitesimal gradient tracking to YodaTrainer.
- Implemented Assertion Loss margin penalty for confident incorrect predictions.
- Refactored LTNConstraintLoss to remove universal bivalence penalty, unlocking proper Belnap Knowledge/Ignorance states.
- Implemented Epistemic Regularization (Gullibility and Ignorance penalties) in LTN constraints.
