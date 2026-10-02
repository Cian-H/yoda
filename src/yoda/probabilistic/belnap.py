"""Fuzzy Belnap Bilattice implementation for continuous differentiable logic and uncertainty.

Implements Belnap 4-valued bilattice semantics over continuous evidence spaces
([0, 1] x [0, 1]), enabling neural networks to track positive (Truth) and negative
(Falsity) evidence independently without forcing total probability constraints.

Canonical Truth States (t, f):
- True:               (1.0, 0.0)
- False:              (0.0, 1.0)
- Both (Conflict):    (1.0, 1.0) - Overdetermined / logical contradiction
- Neither (Unknown):  (0.0, 0.0) - Underdetermined / complete ignorance
"""

import logging
from typing import NamedTuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BelnapEvidence(NamedTuple):
    """Pair of truth and falsity continuous evidence tensors."""

    t: torch.Tensor
    f: torch.Tensor


class BelnapBilattice:
    """Core mathematical operations on the continuous Belnap bilattice (L, <=_t, <=_k)."""

    # Canonical states as reference tuples
    TRUE: tuple[float, float] = (1.0, 0.0)
    FALSE: tuple[float, float] = (0.0, 1.0)
    BOTH: tuple[float, float] = (1.0, 1.0)
    NEITHER: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def negate(evidence: BelnapEvidence) -> BelnapEvidence:
        """Truth negation (-_t): inverts truth and falsity evidence.

        -(t, f) = (f, t)
        """
        return BelnapEvidence(t=evidence.f, f=evidence.t)

    @staticmethod
    def meet_t(a: BelnapEvidence, b: BelnapEvidence) -> BelnapEvidence:
        """Truth meet (logical AND / conjunction under <=_t).

        (t1, f1) ^_t (t2, f2) = (min(t1, t2), max(f1, f2))
        """
        return BelnapEvidence(
            t=torch.minimum(a.t, b.t),
            f=torch.maximum(a.f, b.f),
        )

    @staticmethod
    def join_t(a: BelnapEvidence, b: BelnapEvidence) -> BelnapEvidence:
        """Truth join (logical OR / disjunction under <=_t).

        (t1, f1) v_t (t2, f2) = (max(t1, t2), min(f1, f2))
        """
        return BelnapEvidence(
            t=torch.maximum(a.t, b.t),
            f=torch.minimum(a.f, b.f),
        )

    @staticmethod
    def meet_k(a: BelnapEvidence, b: BelnapEvidence) -> BelnapEvidence:
        """Knowledge meet (consensus operator under <=_k).

        Retains only shared/agreed information:
        (t1, f1) ^_k (t2, f2) = (min(t1, t2), min(f1, f2))
        """
        return BelnapEvidence(
            t=torch.minimum(a.t, b.t),
            f=torch.minimum(a.f, b.f),
        )

    @staticmethod
    def join_k(a: BelnapEvidence, b: BelnapEvidence) -> BelnapEvidence:
        """Knowledge join (gullibility / information pooling operator under <=_k).

        Combines all evidence from both sources:
        (t1, f1) v_k (t2, f2) = (max(t1, t2), max(f1, f2))
        """
        return BelnapEvidence(
            t=torch.maximum(a.t, b.t),
            f=torch.maximum(a.f, b.f),
        )

    @staticmethod
    def lukasiewicz_equivalence(a: BelnapEvidence, b: BelnapEvidence) -> torch.Tensor:
        """Continuous Lukasiewicz bi-implication across both bilattice axes.

        Computes continuous satisfiability in [0, 1]:
            sat_t = 1 - |t_a - t_b|
            sat_f = 1 - |f_a - f_b|
            sat = 0.5 * (sat_t + sat_f)
        """
        sat_t = 1.0 - torch.abs(a.t - b.t)
        sat_f = 1.0 - torch.abs(a.f - b.f)
        return 0.5 * (sat_t + sat_f)

    @staticmethod
    def decompose_uncertainty(evidence: BelnapEvidence) -> dict[str, torch.Tensor]:
        """Decomposes evidence into orthogonal uncertainty components.

        Returns:
            Dictionary with:
            - `conflict`: Degree of logical contradiction (min(t, f))
            - `ignorance`: Degree of missing information (1 - max(t, f))
            - `information`: Total evidence mass (0.5 * (t + f))
            - `polarity`: Net truth bias (t - f in [-1.0, 1.0])
        """
        conflict = torch.minimum(evidence.t, evidence.f)
        ignorance = torch.clamp(1.0 - torch.maximum(evidence.t, evidence.f), min=0.0)
        information = 0.5 * (evidence.t + evidence.f)
        polarity = evidence.t - evidence.f
        return {
            "conflict": conflict,
            "ignorance": ignorance,
            "information": information,
            "polarity": polarity,
        }


class FuzzyBelnapLoss(nn.Module):
    """Differentiable semantic loss module based on Belnap bilattice continuous satisfiability."""

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes FuzzyBelnapLoss.

        Args:
            reduction: Specifies reduction across batch ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        logger.debug("probabilistic.belnap.loss_init", extra={"reduction": reduction})

    def forward(
        self,
        pred_evidence: BelnapEvidence,
        target_evidence: BelnapEvidence,
    ) -> torch.Tensor:
        """Computes continuous bilattice semantic violation loss.

        Loss is 1 - satisfiability under continuous Lukasiewicz equivalence.

        Args:
            pred_evidence: Predicted Belnap evidence (t, f).
            target_evidence: Ground-truth or constraint target Belnap evidence (t, f).

        Returns:
            Scalar loss or unreduced tensor depending on `reduction`.
        """
        satisfaction = BelnapBilattice.lukasiewicz_equivalence(pred_evidence, target_evidence)
        loss = 1.0 - satisfaction

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
