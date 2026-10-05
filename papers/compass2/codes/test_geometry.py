"""Test geometry."""
import numpy as np

import geometry as G

def rand_corr(k, rng, rank=3):
    F = rng.standard_normal((k, rank))
    S = F @ F.T + np.diag(rng.uniform(0.3, 1.0, k))
    return G.cov_to_corr(S)

def unit_items(D, k, rng, clusters=3, spread=0.6):
    cent = rng.standard_normal((clusters, D))
    A = np.repeat(cent, int(np.ceil(k / clusters)), axis=0)[:k] + spread * rng.standard_normal((k, D))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A.T

def test_upper_and_corr():
    M = np.arange(16.0).reshape(4, 4)
    assert np.allclose(G.upper(M), [1, 2, 3, 6, 7, 11])
    R = G.cov_to_corr(np.array([[4.0, 2.0], [2.0, 9.0]]))
    assert np.allclose(R, [[1, 1 / 3], [1 / 3, 1]])

def test_whiten_zero_under_null_and_scale():
    rng = np.random.default_rng(0)
    C = np.cov(rng.standard_normal((500, 6)).T) + np.eye(6)
    assert np.allclose(G.whiten(C, C), 0, atol=1e-10)
    assert np.allclose(G.whiten(2.5 * C, C), 1.5 * np.eye(6), atol=1e-10)

def test_deconvolve_exact():
    rng = np.random.default_rng(1)
    k = 7
    K = rng.standard_normal((k, k)) + 3 * np.eye(k)
    S = rand_corr(k, rng); N = np.diag(rng.uniform(0.1, 0.5, k))
    C = K.T @ S @ K + N
    assert np.allclose(G.deconvolve(C, K, N), S, atol=1e-10)

def test_deconvolve_wording_exact():
    rng = np.random.default_rng(2)
    A = unit_items(32, 9, rng)
    M = A.T @ A
    R = rand_corr(9, rng)
    for g2, c in [(1.0, 0.0), (0.3, 2.0), (2.0, 0.5)]:
        Rh, g2h, ch = G.deconvolve_wording(g2 * M @ R @ M + c * M, M)
        assert np.allclose(Rh, R, atol=1e-8) and abs(g2h - g2) < 1e-8 and abs(ch - c) < 1e-8

def test_linear_scorer_identity():
    """Sample covariance of p = A^T e converges to A^T Sigma A; isotropic windows give corr = A^T A (the null)."""
    rng = np.random.default_rng(3)
    D, k, n = 24, 6, 200_000
    A = unit_items(D, k, rng)
    L = rng.standard_normal((D, D)) / np.sqrt(D)
    Sigma = L @ L.T + 0.2 * np.eye(D)
    E = rng.standard_normal((n, D)) @ np.linalg.cholesky(Sigma).T
    assert np.allclose(np.cov((E @ A).T), A.T @ Sigma @ A, atol=0.02)
    E0 = rng.standard_normal((n, D))
    assert np.allclose(np.corrcoef((E0 @ A).T), A.T @ A, atol=0.01)

def test_center_by_group_leave_person_out():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(5), 40); persons = np.tile(np.arange(20), 10)
    effect = rng.standard_normal((5, 3)) * 5
    P = effect[groups] + rng.standard_normal((200, 3))
    Pc = G.center_by_group(P, groups, persons)
    gm = np.array([Pc[groups == g].mean(0) for g in range(5)])
    assert np.abs(gm).max() < 0.5

    P2 = P.copy(); P2[persons == 0] += 100.0
    Pc2 = G.center_by_group(P2, groups, persons)
    assert np.allclose(Pc2[persons == 0] - Pc[persons == 0], 100.0)

def test_partial_corr_and_mantel():
    rng = np.random.default_rng(5)
    z = rng.standard_normal(300); x = rng.standard_normal(300)
    assert abs(G.partial_corr(x + 2 * z, x, z) - 1.0) < 1e-10
    k = 12
    M = rand_corr(k, rng)
    r, p = G.mantel(M, M, n_perm=500, rng=0)
    assert r > 0.999 and p < 0.01
    N1, N2 = rand_corr(k, rng), rand_corr(k, rng)
    r0, p0 = G.mantel(N1, N2, n_perm=500, rng=0)
    assert 0.0 <= p0 <= 1.0

def test_keyed_scores():
    P = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    s = G.keyed_scores(P, {"Task": [(0, 1), (2, -1)], "Bond": [(1, 1)]})
    assert np.allclose(s["Task"], [-2.0, 0.0]) and np.allclose(s["Bond"], [2.0, 0.5])

def test_split_half_reliability():
    rng = np.random.default_rng(6)
    S = rand_corr(8, rng, rank=2)
    P = rng.standard_normal((4000, 8)) @ np.linalg.cholesky(S).T
    r, sb = G.split_half_reliability(P, lambda X: np.corrcoef(X.T), n_splits=20, rng=0)
    assert r > 0.9 and sb > r
    r0, _ = G.split_half_reliability(rng.standard_normal((4000, 8)), lambda X: np.corrcoef(X.T), n_splits=20, rng=0)
    assert abs(r0) < 0.3

def test_mantel_partial_fl_detects_and_ignores():
    rng = np.random.default_rng(7)
    k = 14
    Z = rand_corr(k, rng); X = rand_corr(k, rng)
    Y = 0.8 * Z + 0.6 * X + 0.05 * rand_corr(k, rng)
    r, p = G.mantel_partial_fl(Y, X, Z, n_perm=500, rng=0)
    assert r > 0.5 and p < 0.01
    Y0 = 0.8 * Z + 0.3 * rand_corr(k, rng)
    r0, p0 = G.mantel_partial_fl(Y0, X, Z, n_perm=500, rng=0)
    assert 0.0 <= p0 <= 1.0

def test_geometry_agreement_ignores_who_said_what():

    rng = np.random.default_rng(5)
    L = rng.standard_normal((120, 6)); Y = L @ rng.standard_normal((6, 6)) + rng.standard_normal((120, 6))
    geo = lambda y: np.corrcoef(G.upper(np.corrcoef(L.T)), G.upper(np.corrcoef(y.T)))[0, 1]
    assert abs(geo(Y) - geo(Y[rng.permutation(120)])) < 1e-12

def test_deconvolution_exact_at_zero_ridge_only():

    rng = np.random.default_rng(6)
    A = rng.standard_normal((40, 8)); A /= np.linalg.norm(A, axis=0); M = A.T @ A
    Rt = G.cov_to_corr(np.cov(rng.standard_normal((50, 8)).T))
    C = 1.0 * M @ Rt @ M + 2.0 * M
    R0, g2, c = G.deconvolve_wording(C, M, ridge=0.0)
    assert np.allclose(R0, Rt, atol=1e-8) and abs(g2 - 1) < 1e-8 and abs(c - 2) < 1e-8
    R1, _, _ = G.deconvolve_wording(C, M, ridge=0.1)
    assert np.abs(R1 - Rt).max() > 0.01

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
