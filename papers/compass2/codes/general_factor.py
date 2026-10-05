"""General factor."""
from xpsych import scoring as XS

import paths as P
import json

import numpy as np

import registered_tests as R

RNG = np.random.default_rng(20261003)

def z(X):
    return (X - X.mean(0)) / X.std(0)

def pc1(X):
    Xz = z(X); u, s, vt = np.linalg.svd(Xz, full_matrices=False)
    g = Xz @ vt[0]
    return g if np.corrcoef(g, Xz.mean(1))[0, 1] >= 0 else -g

def stats(prof, Y):
    gs, gl = pc1(Y), pc1(prof)
    resid = lambda X, g: z(X) - np.outer(g / np.linalg.norm(g), (g / np.linalg.norm(g)) @ z(X))
    Lr, Yr = resid(prof, gs), resid(Y, gs)
    C = z(Lr).T @ z(Yr) / len(Y)
    same, diff = np.diag(C).mean(), C[~np.eye(len(C), dtype=bool)].mean()
    return dict(r_glang_gself=float(np.corrcoef(gl, gs)[0, 1]),
                r_langmean_phq=float(np.corrcoef(z(prof).mean(1), Y[:, :8].sum(1))[0, 1]),
                r_langmean_pcl=float(np.corrcoef(z(prof).mean(1), Y[:, 8:].sum(1))[0, 1]),
                specific_same=float(same), specific_diff=float(diff), specific_delta=float(same - diff))

def boot(prof, Y, key, n=2000):
    v = [stats(prof[i], Y[i])[key] for i in (RNG.integers(0, len(Y), len(Y)) for _ in range(n))]
    return [float(x) for x in np.quantile(v, [0.025, 0.975])]

out = {}
for split in ["discovery", "confirmation"]:
    name, W, lab, items = R.load_corpus(split)
    import pandas as pd, paths as P
    s = items["instrument"].isin(R.SYMPTOMS).to_numpy()
    groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
    out[split] = {}
    for sc in R.SCORERS:
        if sc == "nli":
            Pw = np.load(P.DERIVED / f"nli_{name}.npy").astype(float)
        else:
            E = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32)
            A = np.load(P.DERIVED / f"emb_{sc}_items.npy").astype(np.float32)[s]; Pw = (XS.dot_activities(E, A)).astype(float)
        ids, prof = R.profiles(Pw, groups, persons, centre=True)
        keep = np.isin(ids, lab.index); Y = lab.loc[ids[keep]].to_numpy(float); prof = prof[keep]
        st = stats(prof, Y)
        st["ci_r_glang_gself"] = boot(prof, Y, "r_glang_gself")
        st["ci_specific_delta"] = boot(prof, Y, "specific_delta")
        out[split][sc] = st
(P.RESULTS / "general_factor.json").write_text(json.dumps(out, indent=1))
for split, d in out.items():
    for sc, v in d.items():
        print(f"{split:12s} {sc:6s} r(g_lang,g_self) {v['r_glang_gself']:.2f} {[round(x,2) for x in v['ci_r_glang_gself']]} | r(lang mean, PHQ) {v['r_langmean_phq']:.2f} PCL {v['r_langmean_pcl']:.2f} | specific Δ {v['specific_delta']:.3f} {[round(x,3) for x in v['ci_specific_delta']]}")
