"""Numerical kernels for covariance geometry, scoring and permutation inference.

Kept stable for reproducible estimators. Applications use the validated public
interfaces in geometry.py and scoring.py.
"""
import numpy as np

def upper(M, k=1):
    """Upper-triangle entries (above the diagonal by default) as a vector."""
    i, j = np.triu_indices(M.shape[0], k)
    return M[i, j]

def cov_to_corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / d[:, None] / d[None, :]
    np.fill_diagonal(R, 1.0)
    return R

def sym_power(C, power, floor=1e-8):
    """C^power for a symmetric positive semi-definite matrix; eigenvalues below floor * max are floored."""
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, floor * max(w.max(), 1e-300))
    return (V * w ** power) @ V.T

def whiten(C, C_ref, floor=1e-8):
    """C_ref^{-1/2} C C_ref^{-1/2} - I: zero when C = C_ref."""
    Wh = sym_power(C_ref, -0.5, floor)
    G = Wh @ C @ Wh
    return (G + G.T) / 2 - np.eye(C.shape[0])

def deconvolve(C, K, N=None, ridge=0.0):
    """K^{-T} (C - N) K^{-1} with optional Tikhonov ridge (as a fraction of the mean eigenvalue of K^T K)."""
    k = C.shape[0]
    D = C if N is None else C - N
    KtK = K.T @ K
    lam = ridge * np.trace(KtK) / k
    Kinv = np.linalg.solve(KtK + lam * np.eye(k), K.T)
    S = Kinv.T @ D @ Kinv
    return (S + S.T) / 2

def deconvolve_wording(C, M, ridge=0.0):
    """Recover corr(z) when C = g^2 M R M + c M (K = M symmetric, isotropic unrelated variation).

    M^{-1} C M^{-1} = g^2 R + c M^{-1}. The diagonal (R_jj = 1) gives g^2 and c by least squares; returns
    (R_hat, g2, c). R_hat is the estimate of R scaled to a unit diagonal.
    """
    k = C.shape[0]
    lam = ridge * np.trace(M) / k
    Mi = np.linalg.inv(M + lam * np.eye(k))
    S = Mi @ C @ Mi
    S = (S + S.T) / 2
    X = np.column_stack([np.ones(k), np.diag(Mi)])
    (g2, c), *_ = np.linalg.lstsq(X, np.diag(S), rcond=None)
    R = (S - c * Mi) / g2 if g2 > 0 else S - c * Mi
    return cov_to_corr(R) if g2 > 0 else R, float(g2), float(c)

def center_by_group(P, groups, persons):
    """Subtract each window's group mean (e.g. its eliciting question), estimated without the window's own person.

    P: windows x items; groups, persons: length-n labels. Groups seen in only one person keep their own mean
    removed (that person's windows in the group are centred on themselves).
    """
    P = np.asarray(P, float)
    groups = np.asarray(groups); persons = np.asarray(persons)
    out = np.empty_like(P)
    for g in np.unique(groups):
        ig = groups == g
        tot = P[ig].sum(0); cnt = ig.sum()
        for p in np.unique(persons[ig]):
            ip = ig & (persons == p)
            n_other = cnt - ip.sum()
            mean = (tot - P[ip].sum(0)) / n_other if n_other > 0 else P[ip].mean(0)
            out[ip] = P[ip] - mean
    return out

def split_half_reliability(P, geometry, n_splits=50, rng=None):
    """Reliability of a geometry estimated from windows (rows of P): random half splits, correlation of the two
    halves' upper triangles, Spearman-Brown corrected. geometry: function(windows x items) -> k x k matrix.
    Returns (mean split-half r, Spearman-Brown full-length reliability)."""
    rng = np.random.default_rng(rng)
    n = P.shape[0]
    rs = []
    for _ in range(n_splits):
        idx = rng.permutation(n); a, b = idx[: n // 2], idx[n // 2:]
        rs.append(np.corrcoef(upper(geometry(P[a])), upper(geometry(P[b])))[0, 1])
    r = float(np.nanmean(rs))
    return r, float(2 * r / (1 + r)) if r > -1 else float("nan")

def partial_corr(y, x, z):
    """Correlation of y and x after regressing both on z (vectors; z may be 2-D with columns)."""
    Z = np.column_stack([np.ones(len(y)), z])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])

def mantel(M1, M2, control=None, n_perm=2000, rng=None):
    """RSA correlation of two k x k matrices (upper triangles), optionally partial on a third, with an item-
    permutation null (rows and columns of M1 permuted together). Returns (r, two-sided p)."""
    rng = np.random.default_rng(rng)
    k = M1.shape[0]
    y, x = upper(M1), upper(M2)
    z = None if control is None else upper(control)
    stat = (lambda a: np.corrcoef(a, x)[0, 1]) if z is None else (lambda a: partial_corr(a, x, z))
    r = stat(y)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(k)
        null[b] = stat(upper(M1[np.ix_(p, p)]))
    pval = (1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm)
    return float(r), float(pval)

def keyed_scores(P, key):
    """Scale scores from item activities with a signed key: dict scale -> list of (item index, polarity)."""
    return {s: P[:, [i for i, _ in v]] @ np.array([pol for _, pol in v], float) for s, v in key.items()}

def mantel_partial_fl(M1, M2, control, n_perm=2000, rng=None):
    """Partial association of M1 with M2 given control (k x k matrices; upper triangles), with a Freedman-Lane item
    permutation: M1 is split into its fit on control plus residual, the residual matrix is permuted (rows and columns
    together) and added back. Better calibrated than permuting M1 itself when control is structured.
    Returns (partial r, two-sided p)."""
    rng = np.random.default_rng(rng)
    k = M1.shape[0]
    iu = np.triu_indices(k, 1)
    y, x, z = M1[iu], M2[iu], control[iu]
    Z = np.column_stack([np.ones_like(z), z])
    fit = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    R = np.zeros((k, k)); R[iu] = y - fit; R = R + R.T
    Fm = np.zeros((k, k)); Fm[iu] = fit; Fm = Fm + Fm.T
    r = partial_corr(y, x, z)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(k)
        null[b] = partial_corr((Fm + R[np.ix_(p, p)])[iu], x, z)
    return float(r), float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm))
