"""Compare estimators and reference assumptions using synthetic data."""
import paths as P
import json

import numpy as np

import geometry as G
import sim_wording_null as S

OUT = S.OUT.parent / "sim_estimators.json"
D, K = S.D, S.K_ITEMS

def inverse_fit(C, M):
    """Exact deconvolution with diagnostics: fit g2, c on the diagonal of M^-1 C M^-1 (valid at ridge 0), subtract
    c M^-1. Returns (R_hat or None if inadmissible, diagnostics, R_psd): admissible means g2 > 0 and a positive diagonal
    after noise subtraction; R_psd projects the noise-subtracted matrix onto the positive semidefinite cone (eigenvalues
    clipped at a small positive floor) before normalizing, a constrained alternative that is always defined."""
    k = len(M); Mi = np.linalg.inv(M); S_ = Mi @ C @ Mi; S_ = (S_ + S_.T) / 2
    X = np.column_stack([np.ones(k), np.diag(Mi)]); (g2, c), *_ = np.linalg.lstsq(X, np.diag(S_), rcond=None)
    D_ = S_ - c * Mi
    diag = {"g2_nonpositive": bool(g2 <= 0), "diag_nonpositive": bool((np.diag(D_) <= 0).any()),
            "indefinite": bool(np.linalg.eigvalsh(D_).min() < 0)}
    w, V = np.linalg.eigh(D_); floor = 1e-6 * max(w.max(), 1e-12)
    Dp = (V * np.clip(w, floor, None)) @ V.T; R_psd = G.cov_to_corr(Dp)
    ok = not diag["g2_nonpositive"] and not diag["diag_nonpositive"]
    return (G.cov_to_corr(D_ / g2) if ok else None), diag, R_psd

def part_a(rng, reps=200):
    A, lab = S.make_items(rng); M = A.T @ A; R = S.make_truth(A, lab, rng)
    out = {"cond_M": float(np.linalg.cond(M))}
    for n in [189, 86]:
        for g in [0.0, 0.1, 0.2, 0.4]:
            acc = {k: {"r_truth": [], "beyond_wording": []} for k in ["naive", "whitened", "inverse_admissible", "inverse_psd"]}
            fails = {"g2_nonpositive": 0, "diag_nonpositive": 0, "indefinite": 0, "inadmissible": 0}
            for _ in range(reps):
                m = S.person_means(A, R, n, 155, g, rng); C = np.cov(m.T)
                Rinv, dg, Rpsd = inverse_fit(C, M)
                for k_ in ["g2_nonpositive", "diag_nonpositive", "indefinite"]:
                    fails[k_] += dg[k_]
                fails["inadmissible"] += Rinv is None
                est = {"naive": G.cov_to_corr(C), "whitened": G.whiten(C, M), "inverse_psd": Rpsd}
                if Rinv is not None:
                    est["inverse_admissible"] = Rinv
                for k_, E in est.items():
                    acc[k_]["r_truth"].append(np.corrcoef(G.upper(E), G.upper(R))[0, 1])
                    acc[k_]["beyond_wording"].append(G.partial_corr(G.upper(E), G.upper(R), G.upper(M)))
            out[f"n={n},g={g}"] = {k_: {s_: ([float(np.nanmean(v)), float(np.nanstd(v))] if len(v) else [None, None]) for s_, v in d_.items()}
                                   for k_, d_ in acc.items()}
            out[f"n={n},g={g}"]["inverse_failures"] = {k_: int(v) for k_, v in fails.items()}
            out[f"n={n},g={g}"]["inverse_admissible_n"] = len(acc["inverse_admissible"]["r_truth"])
    return out

def simulate_windows(A, R, Ut, n, wins, g, tau_b, tau_w, rng, g_within=0.0, Uw=None):
    """Windows e = g A z_i + tau_b * topic_i + u_i + tau_w * topic_it + g_within A z_it + noise. Topics are drawn along the
    columns of Ut; z_it (window-level item states with correlation R) adds genuine within-person symptom dynamics."""
    Lr = np.linalg.cholesky(R)
    z = rng.standard_normal((n, K)) @ Lr.T
    person = g * z @ A.T + tau_b * rng.standard_normal((n, Ut.shape[1])) @ Ut.T + rng.standard_normal((n, D))
    P, who = [], []
    for i in range(n):
        Uwi = Ut if Uw is None else Uw
        e = person[i] + tau_w * rng.standard_normal((wins, Uwi.shape[1])) @ Uwi.T + rng.standard_normal((wins, D))
        if g_within:
            e = e + g_within * (rng.standard_normal((wins, K)) @ Lr.T) @ A.T
        P.append(e @ A); who.extend([i] * wins)
    return np.vstack(P), np.array(who)

SCENARIOS = {
    "matched nuisance": (1.5, 1.5, 0.0, False),
    "between-person nuisance stronger": (3.0, 1.5, 0.0, False),
    "nuisance in different directions": (1.5, 1.5, 0.0, True),
    "genuine within-person dynamics": (0.0, 0.0, 0.6, False),
}

def part_b(rng, reps_null=200, reps_power=100, n=189, wins=55, n_perm=499):
    A, lab = S.make_items(rng); M = A.T @ A; R = S.make_truth(A, lab, rng)
    cent = np.array([A[:, lab == c].mean(1) for c in range(lab.max() + 1)]).T
    Ut = cent[:, :3] / np.linalg.norm(cent[:, :3], axis=0); Uo = cent[:, 3:6] / np.linalg.norm(cent[:, 3:6], axis=0)
    res = {}
    for sc_name, (tb, tw, gw, other) in SCENARIOS.items():
        res[sc_name] = {}
        for g, reps in [(0.0, reps_null), (0.4, reps_power)]:
            rej = {"wording_reference": 0, "within_person_reference": 0}; rs = {"wording_reference": [], "within_person_reference": []}
            for _ in range(reps):
                Pw, who = simulate_windows(A, R, Ut, n, wins, g, tb, tw, rng, g_within=gw, Uw=Uo if other else None)
                prof = np.array([Pw[who == i].mean(0) for i in range(n)]); C = np.cov(prof.T)
                Sw = sum(np.cov(Pw[who == i].T) * (wins - 1) for i in range(n)) / (n * (wins - 1))
                for name, ref in [("wording_reference", M), ("within_person_reference", Sw)]:
                    W = G.whiten(C, ref); ctrl = G.cov_to_corr(ref)
                    r, pv = G.mantel_partial_fl(W, R, ctrl, n_perm=n_perm, rng=rng)
                    rej[name] += pv < 0.05; rs[name].append(r)
            res[sc_name][f"g={g}"] = {k: dict(rejection_rate=rej[k] / reps, reps=reps, mean_partial_r=float(np.mean(rs[k]))) for k in rej}
    return res

def part_c(rng, n=189):
    """Person shuffle in simulation: speech and self-reports share the same latent item levels z; the geometry-agreement
    statistic is unchanged by shuffling who provided which self-report, whereas a person-paired correlation is not."""
    A, lab = S.make_items(rng); M = A.T @ A; R = S.make_truth(A, lab, rng)
    z = rng.standard_normal((n, K)) @ np.linalg.cholesky(R).T
    m = (0.4 * z @ A.T + rng.standard_normal((n, D)) + rng.standard_normal((n, D)) / np.sqrt(155)) @ A
    Y = z + 0.5 * rng.standard_normal(z.shape)
    geo = lambda Yx: float(np.corrcoef(G.upper(G.cov_to_corr(np.cov(m.T))), G.upper(np.corrcoef(Yx.T)))[0, 1])
    paired = lambda Yx: float(np.corrcoef(m.mean(1), Yx.mean(1))[0, 1])
    sh = [rng.permutation(n) for _ in range(200)]
    return dict(geometry_true=geo(Y), geometry_shuffled_max_abs_change=float(max(abs(geo(Y[q]) - geo(Y)) for q in sh)),
                paired_true=paired(Y), paired_shuffled_95=np.quantile([paired(Y[q]) for q in sh], [.025, .975]).tolist())

def main():
    rng = np.random.default_rng(20261004)
    out = {"A_estimators": part_a(rng), "B_anisotropic_nuisance": part_b(rng), "C_person_permutation": part_c(rng)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["B_anisotropic_nuisance"], indent=1)); print(out["C_person_permutation"])
    for k in ["n=189,g=0.0", "n=189,g=0.4"]:
        a = out["A_estimators"][k]
        print(k, {e: a[e]["r_truth"] for e in ["naive", "whitened", "inverse_admissible", "inverse_psd"]}, a["inverse_failures"], a["inverse_admissible_n"])

if __name__ == "__main__":
    main()
