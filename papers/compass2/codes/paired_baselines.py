"""Paired baselines."""
import json

import numpy as np
from scipy.stats import rankdata

from xpsych import scoring as XS

import paths as P
import registered_tests as R

RNG = np.random.default_rng(20261004)
N_BOOT = 2000
OUT = P.RESULTS / "paired_baselines.json"

def z(X):
    return (X - X.mean(0)) / X.std(0)

def pc1(X):
    Z = z(X); u, s, vt = np.linalg.svd(Z, full_matrices=False); g = Z @ vt[0]
    return g if np.corrcoef(g, Z.mean(1))[0, 1] >= 0 else -g

def partial(x, y, Z):
    X = np.column_stack([np.ones(len(x)), Z])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])

def stats(prof, Y, B):
    score, phq = z(prof)[:, :8].mean(1), Y[:, :8].sum(1)
    gl, gs = pc1(prof), pc1(Y)
    return {"phq8_score_r": float(np.corrcoef(score, phq)[0, 1]), "phq8_score_partial_r": partial(score, phq, B),
            "general_factor_r": float(np.corrcoef(gl, gs)[0, 1]), "general_factor_partial_r": partial(gl, gs, B)}

def main():
    out = {}
    for split in ["discovery", "confirmation"]:
        name, W, lab, items = R.load_corpus(split)
        s = items["instrument"].isin(R.SYMPTOMS).to_numpy()
        groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
        words = W["text"].astype(str).str.split().str.len().to_numpy()
        E = np.load(P.DERIVED / f"emb_minilm_{name}.npy").astype(np.float32)
        A = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float32)[s]
        topic_w = (XS.dot_activities(E, A)).astype(float).mean(1)
        ids = np.unique(persons); keep = np.isin(ids, lab.index); ids_k = ids[keep]
        raw = np.column_stack([[np.sum(persons == i) for i in ids_k], [words[persons == i].sum() for i in ids_k],
                               [topic_w[persons == i].mean() for i in ids_k]]).astype(float)
        B = np.column_stack([rankdata(c) for c in raw.T]); B_log = np.column_stack([np.log(raw[:, 0]), np.log(raw[:, 1]), raw[:, 2]])
        Y = lab.loc[ids_k].to_numpy(float); phq = Y[:, :8].sum(1)
        rk = lambda a, b: float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
        names = ["windows", "words", "topic_activation"]
        base = {k: rk(raw[:, j], phq) for j, k in enumerate(names)}
        bci = {}
        for j, k in enumerate(names):
            v = [rk(raw[i, j], phq[i]) for i in (RNG.integers(0, len(phq), len(phq)) for _ in range(N_BOOT))]
            bci[k] = [float(x) for x in np.quantile(v, [.025, .975])]
        out[split] = {"n": int(len(ids_k)), "baselines_r_phq8": base, "baselines_r_phq8_ci": bci}
        for sc in R.SCORERS:
            if sc == "nli":
                Pw = np.load(P.DERIVED / f"nli_{name}.npy").astype(float)
            else:
                Es = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32)
                As = np.load(P.DERIVED / f"emb_{sc}_items.npy").astype(np.float32)[s]; Pw = (XS.dot_activities(Es, As)).astype(float)
            ids2, prof = R.profiles(Pw, groups, persons, centre=True)
            assert np.array_equal(ids2, ids)
            prof = prof[keep]
            st = stats(prof, Y, B)
            log_check = stats(prof, Y, B_log)
            boots = [stats(prof[i], Y[i], B[i]) for i in (RNG.integers(0, len(Y), len(Y)) for _ in range(N_BOOT))]
            st["ci"] = {k: [float(x) for x in np.quantile([b[k] for b in boots], [.025, .975])] for k in st}
            st["log_count_check"] = {k: log_check[k] for k in ["phq8_score_partial_r", "general_factor_partial_r"]}
            out[split][sc] = st
    OUT.write_text(json.dumps(out, indent=1))
    for split, d in out.items():
        print(split, d["n"], {k: round(v, 3) for k, v in d["baselines_r_phq8"].items()})
        for sc in R.SCORERS:
            v = d[sc]; print(f"  {sc:6s}", {k: (round(v[k], 3), [round(x, 3) for x in v["ci"][k]]) for k in v["ci"]}, v["log_count_check"])

if __name__ == "__main__":
    main()
