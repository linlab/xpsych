"""Item-space geometry with explicit references and item-label inference."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata
from . import _geometry as kernels
from ._validation import matrix, square, positive_integer, variable, labels


@dataclass(frozen=True)
class RSAResult:
    """Upper-triangle association and two-sided permutation P value."""
    correlation: float
    pvalue: float
    n_permutations: int
    method: str


def covariance(activities):
    """Sample covariance over rows; columns must represent the same ordered items."""
    x = matrix(activities, "activities", min_rows=2, min_cols=2)
    return np.cov(x, rowvar=False, ddof=1)


def rdm(activities):
    """Correlation distance (1 - Pearson r) among item columns across observations.

    Constant item columns raise ValueError rather than silently yielding NaNs.
    Row permutation preserves this population geometry, not person correspondence.
    """
    x = matrix(activities, "activities", min_rows=2, min_cols=2)
    for j in range(x.shape[1]):
        variable(x[:, j], f"item column {j}")
    out = 1 - np.corrcoef(x, rowvar=False)
    np.fill_diagonal(out, 0)
    return out


def wording_null(item_embeddings):
    """Item cosine matrix for the isotropic embedding-space wording reference.

    Items are rows. This is a reference covariance up to scale; it does not model
    all nonpsychological variation and is not an empirical noise estimate.
    """
    a = matrix(item_embeddings, "item_embeddings", min_rows=2)
    norms = np.linalg.norm(a, axis=1)
    if np.any(norms == 0):
        raise ValueError("item embeddings must be nonzero")
    a = a / norms[:, None]
    return a @ a.T


def whiten(observed, reference, *, floor=1e-8):
    """Return R^(-1/2) C R^(-1/2) - I using eigenvalue flooring.

    This diagnoses departure from a declared reference; it does not recover
    personal symptom structure. Zero residual under C=R requires full rank above
    the floor. Rank-deficient references retain -1 along null directions.
    """
    c = square(observed, "observed", psd=True)
    r = square(reference, "reference", psd=True)
    if c.shape != r.shape:
        raise ValueError("observed and reference must have the same shape")
    if not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("floor must be between zero and one")
    if np.linalg.eigvalsh(r).max() <= 0:
        raise ValueError("reference must have a positive eigenvalue")
    return kernels.whiten(c, r, floor=floor)


def center_by_group(activities, groups, persons):
    """Subtract group means estimated from other people (e.g. question centring).

    String labels identify rows. For a group observed in only one person, its
    own mean is removed, matching the paper; this fallback is not out-of-person.
    """
    x = matrix(activities, "activities")
    groups = labels(groups, len(x), "groups")
    persons = labels(persons, len(x), "persons")
    _, gi = np.unique(groups, return_inverse=True)
    _, pi = np.unique(persons, return_inverse=True)
    _, pair = np.unique(np.column_stack([gi, pi]), axis=0, return_inverse=True)
    group_sum = np.zeros((gi.max() + 1, x.shape[1]))
    pair_sum = np.zeros((pair.max() + 1, x.shape[1]))
    np.add.at(group_sum, gi, x)
    np.add.at(pair_sum, pair, x)
    group_count = np.bincount(gi)[gi]
    pair_count = np.bincount(pair)[pair]
    other_count = group_count - pair_count
    reference = pair_sum[pair] / pair_count[:, None]
    has_others = other_count > 0
    reference[has_others] = ((group_sum[gi[has_others]] - pair_sum[pair[has_others]])
                             / other_count[has_others, None])
    return x - reference


def compare(first, second, *, control=None, method="spearman", n_permutations=999, seed=None):
    """Compare aligned item matrices with a two-sided item-label permutation test.

    Pearson is the paper's registered statistic; Spearman is a descriptive RSA
    choice. With control, Freedman–Lane residual item permutation is used.
    Spearman first ranks each upper triangle, including the control. Inference
    is conditional on these items and its exchangeability/reference assumptions.
    This is not a test of whether the correct person's scores were predicted.
    """
    a = square(first, "first", min_size=3)
    b = square(second, "second", min_size=3)
    if a.shape != b.shape:
        raise ValueError("item matrices must have the same shape and item order")
    n = positive_integer(n_permutations, "n_permutations")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    c = None if control is None else square(control, "control", min_size=3)
    if c is not None and c.shape != a.shape:
        raise ValueError("control must have the same shape")
    iu = np.triu_indices(len(a), 1)

    def prepare(m):
        v = m[iu]
        variable(v, "upper triangle")
        if method == "spearman":
            v = rankdata(v)
        out = np.zeros_like(m)
        out[iu] = v
        return out + out.T

    a, b = prepare(a), prepare(b)
    y, x = a[iu], b[iu]
    if c is None:
        design = np.ones((len(y), 1))
        fitted = np.zeros_like(y)
        residual_matrix = a
    else:
        c = prepare(c)
        design = np.column_stack([np.ones(len(y)), c[iu]])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        residual_matrix = np.zeros_like(a)
        residual_matrix[iu] = y - fitted
        residual_matrix += residual_matrix.T
    inverse = np.linalg.pinv(design)

    def residualize(v):
        return v - (v @ inverse.T) @ design.T

    xr, yr = residualize(x), residualize(y)
    if (np.linalg.norm(xr) <= 1e-12 * max(1, np.linalg.norm(x))
            or np.linalg.norm(yr) <= 1e-12 * max(1, np.linalg.norm(y))):
        raise ValueError("association is undefined: a residual is constant or zero")
    xr /= np.linalg.norm(xr)
    r = float(np.clip(yr @ xr / np.linalg.norm(yr), -1, 1))
    rng = np.random.default_rng(seed)
    exceed = 0
    # Bounded batches avoid a permutations × items × items allocation. The same
    # item permutation is applied to rows and columns, preserving dependence.
    for start in range(0, n, 64):
        orders = np.stack([rng.permutation(len(a)) for _ in range(min(64, n - start))])
        permuted = residual_matrix[orders[:, iu[0]], orders[:, iu[1]]] + fitted
        residuals = residualize(permuted)
        norms = np.linalg.norm(residuals, axis=1)
        if np.any(norms <= 1e-12 * np.maximum(1, np.linalg.norm(permuted, axis=1))):
            raise ValueError("a permuted association is undefined for this control; cannot compute P value")
        null = residuals @ xr / norms
        exceed += np.count_nonzero(np.abs(null) >= abs(r) - 1e-12)
    return RSAResult(r, float((1 + exceed) / (1 + n)), n, method)
