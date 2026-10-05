"""Tests that retain explicit person correspondence."""
from dataclasses import dataclass
import numpy as np
from ._validation import positive_integer, variable


@dataclass(frozen=True)
class PairedResult:
    correlation: float
    pvalue: float
    n_people: int
    n_permutations: int


def paired_correlation(predicted, observed, *, n_permutations=999, seed=None):
    """Pearson association with person-shuffle inference, matching mapping keys.

    Each mapping must contain one scalar per unique person ID. Dictionary order
    is irrelevant. Person labels must match exactly; no silent inner join occurs.
    The two-sided test assumes exchangeable, independent people. This is not a
    symptom-specificity test or validation against a clinical threshold.
    """
    if not hasattr(predicted, "keys") or not hasattr(observed, "keys"):
        raise TypeError("predicted and observed must map person IDs to scores")
    if set(predicted) != set(observed) or len(predicted) < 3:
        raise ValueError("provide the same IDs for at least three people")
    ids = list(predicted)
    x = np.asarray([predicted[i] for i in ids], dtype=float)
    y = np.asarray([observed[i] for i in ids], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("each person must have one finite scalar score")
    variable(x, "predicted scores")
    variable(y, "observed scores")
    n = positive_integer(n_permutations, "n_permutations")
    rng = np.random.default_rng(seed)
    r = float(np.corrcoef(x, y)[0, 1])
    null = np.array([np.corrcoef(x, y[rng.permutation(len(y))])[0, 1] for _ in range(n)])
    p = float((1 + np.sum(np.abs(null) >= abs(r))) / (n + 1))
    return PairedResult(r, p, len(ids), n)
