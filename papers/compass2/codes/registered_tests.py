"""Registered symptom tests. Run with discovery, confirmation, or --selftest."""
import json
import sys

import numpy as np
import pandas as pd

import geometry as G
from xpsych import scoring as XS

import paths as P

SCORERS = ["minilm", "bge", "nli"]
SYMPTOMS = ["PHQ-8", "PCL-C"]
N_BOOT, N_PERM, SEED = 2000, 10000, 20261003

def profiles(Pw, groups, persons, centre=True):
    X = G.center_by_group(Pw, groups, persons) if centre else np.asarray(Pw, float)
    ids = np.unique(persons)
    return ids, np.array([X[persons == i].mean(0) for i in ids])

def corr(X):
    return np.corrcoef(X.T)

def h1(naive, Y, M, rng, alpha=0.05):
    stat = lambda a, b: np.corrcoef(G.upper(corr(a)), G.upper(M))[0, 1] - np.corrcoef(G.upper(corr(a)), G.upper(corr(b)))[0, 1]
    obs = stat(naive, Y)
    n = len(Y); boot = [stat(naive[i], Y[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(r_naive_wording=float(np.corrcoef(G.upper(corr(naive)), G.upper(M))[0, 1]),
                r_naive_selfreport=float(np.corrcoef(G.upper(corr(naive)), G.upper(corr(Y)))[0, 1]),
                difference=float(obs), interval=[float(lo), float(hi)], interval_level=1 - alpha, supported=bool(lo > 0))

def h2(prof, Y, M, K=None, rng=None):
    W = G.whiten(np.cov(prof.T), M)
    control = M if K is None else np.stack([M, (K + K.T) / 2])
    if K is None:
        r, p = G.mantel_partial_fl(W, corr(Y), M, n_perm=N_PERM, rng=rng)
    else:
        iu = np.triu_indices(len(M), 1)
        Z = np.column_stack([M[iu], ((K + K.T) / 2)[iu]])
        r = G.partial_corr(W[iu], corr(Y)[iu], Z)
        Ks = (K + K.T) / 2

        Zf = np.column_stack([np.ones(len(iu[0])), Z])
        fit = Zf @ np.linalg.lstsq(Zf, W[iu], rcond=None)[0]
        R = np.zeros_like(W); R[iu] = W[iu] - fit; R = R + R.T
        F = np.zeros_like(W); F[iu] = fit; F = F + F.T
        k = len(M); null = np.empty(N_PERM)
        for b in range(N_PERM):
            q = rng.permutation(k); null[b] = G.partial_corr((F + R[np.ix_(q, q)])[iu], corr(Y)[iu], Z)
        p = float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + N_PERM))
    return dict(partial_r=float(r), p_freedman_lane=float(p), supported_uncorrected=bool(r > 0 and p < 0.05))

def h3(prof, Y, rng, alpha=0.05):
    def stat(a, b):
        za = (a - a.mean(0)) / a.std(0); zb = (b - b.mean(0)) / b.std(0)
        C = za.T @ zb / len(a)
        same = np.diag(C).mean(); diff = C[~np.eye(len(C), dtype=bool)].mean()
        return same - diff, same, diff
    d, same, diff = stat(prof, Y)
    n = len(Y); boot = [stat(prof[i], Y[i])[0] for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(mean_same_item_r=float(same), mean_different_item_r=float(diff), difference=float(d),
                interval=[float(lo), float(hi)], interval_level=1 - alpha, supported=bool(lo > 0))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def load_corpus(split):
    name = "daicwoz" if split == "discovery" else "edaic_new"
    W = pd.read_parquet(P.DERIVED / f"windows_{name}.parquet")
    lab = pd.read_csv(P.EDAIC_LABELS)
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    s = items[items["instrument"].isin(SYMPTOMS)]
    lab = lab.set_index("Participant")[s["key"].tolist()]
    return name, W, lab, items

def run(split):
    rng = np.random.default_rng(SEED)
    name, W, lab, items = load_corpus(split)
    s = items["instrument"].isin(SYMPTOMS).to_numpy()
    groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
    Ms = {m: (lambda E: XS.dot_activities(E, E).astype(float))(np.load(P.DERIVED / f"emb_{m}_items.npy").astype(np.float32)[s]) for m in ["minilm", "bge"]}
    K = np.array(json.loads((P.RESULTS / "nli_calibration.json").read_text())["K"])
    out = {}
    for sc in SCORERS:
        if sc == "nli":
            Pw = np.load(P.DERIVED / f"nli_{name}.npy").astype(float); M = (Ms["minilm"] + Ms["bge"]) / 2
        else:
            E = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32)
            A = np.load(P.DERIVED / f"emb_{sc}_items.npy").astype(np.float32)[s]
            Pw = XS.dot_activities(E, A).astype(float); M = Ms[sc]
        ids, prof = profiles(Pw, groups, persons, centre=True)
        _, naive = profiles(Pw, groups, persons, centre=False)
        keep = np.isin(ids, lab.index)
        Y = lab.loc[ids[keep]].to_numpy(float); prof, naive = prof[keep], naive[keep]
        res = dict(persons=int(keep.sum()))
        conf = split == "confirmation"
        if sc != "nli":
            res["H1"] = h1(naive, Y, M, rng, alpha=0.05 / 2 if conf else 0.05)
        res["H2"] = h2(prof, Y, M, K if sc == "nli" else None, rng)
        res["H3"] = h3(prof, Y, rng, alpha=0.05 / 3 if conf else 0.05)
        out[sc] = res
    if split == "confirmation":
        for h in ["H2"]:
            adj = holm([out[sc][h]["p_freedman_lane"] for sc in SCORERS])
            for sc, a in zip(SCORERS, adj):
                out[sc][h]["p_holm"] = a; out[sc][h]["supported"] = bool(out[sc][h]["partial_r"] > 0 and a < 0.05)
    (P.RESULTS / f"registered_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

def selftest():
    """Synthetic check that the pipeline runs and behaves: signal present -> H2/H3 supported; absent -> not."""
    rng = np.random.default_rng(1)
    import sim_wording_null as S
    A, lab = S.make_items(rng); M = A.T @ A; R = S.make_truth(A, lab, rng)
    for g in [0.0, 0.6]:
        z = rng.standard_normal((189, 25)) @ np.linalg.cholesky(R).T
        mu = g * z @ A.T + rng.standard_normal((189, S.D))
        prof = mu @ A
        Y = z + 0.5 * rng.standard_normal(z.shape)
        global N_PERM, N_BOOT
        N_PERM, N_BOOT = 499, 300
        print(f"g={g}:", "H1", h1(prof, Y, M, rng)["difference"].__round__(2), "| H2", h2(prof, Y, M, None, rng), "| H3",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in h3(prof, Y, rng).items() if k in ("difference", "supported")})

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "discovery")
