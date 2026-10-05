"""Imposed structure."""
import json
import sys

import numpy as np
import pandas as pd

import geometry as G
from xpsych import scoring as XS

import paths as P

SYMPTOMS = ["PHQ-8", "PCL-C"]
RNG = np.random.default_rng(20261003)

def eta2(P_, labels):
    """Share of variance of each column explained by group means."""
    df = pd.DataFrame(P_); g = df.groupby(np.asarray(labels)).transform("mean")
    return ((g - df.mean()) ** 2).sum().to_numpy() / ((df - df.mean()) ** 2).sum().to_numpy()

def person_geometry(P_, persons, idx=None):
    persons = np.asarray(persons)
    if idx is not None:
        P_, persons = P_[idx], persons[idx]
    ids = np.unique(persons)
    m = np.array([P_[persons == i].mean(0) for i in ids])
    return np.cov(m.T)

def split_half(P_, persons, M, reps=20):
    """Population geometry from two disjoint halves of every person's windows."""
    persons = np.asarray(persons); out = {"raw": [], "whitened": []}
    for _ in range(reps):
        a = np.zeros(len(persons), bool)
        for i in np.unique(persons):
            w = np.flatnonzero(persons == i); a[RNG.choice(w, len(w) // 2, replace=False)] = True
        Ca, Cb = person_geometry(P_, persons, a), person_geometry(P_, persons, ~a)
        out["raw"].append(np.corrcoef(G.upper(G.cov_to_corr(Ca)), G.upper(G.cov_to_corr(Cb)))[0, 1])
        out["whitened"].append(np.corrcoef(G.upper(G.whiten(Ca, M)), G.upper(G.whiten(Cb, M)))[0, 1])
    sb = lambda r: 2 * r / (1 + r)
    return {k: dict(split_half=float(np.mean(v)), spearman_brown=float(sb(np.mean(v)))) for k, v in out.items()}

def within_person(P_, persons, min_windows=40):
    """Split-half reliability of person-level geometry: raw (shares wording, so inflated) and of the person's
    deviation from everyone else's window covariance (whitened against the leave-one-person-out reference)."""
    persons = np.asarray(persons); raw, dev = [], []
    for i in np.unique(persons):
        X = P_[persons == i]
        if len(X) < min_windows:
            continue
        C_ref = np.cov(P_[persons != i].T)
        idx = RNG.permutation(len(X)); a, b = X[idx[: len(X) // 2]], X[idx[len(X) // 2:]]
        raw.append(np.corrcoef(G.upper(np.corrcoef(a.T)), G.upper(np.corrcoef(b.T)))[0, 1])
        dev.append(np.corrcoef(G.upper(G.whiten(np.cov(a.T), C_ref)), G.upper(G.whiten(np.cov(b.T), C_ref)))[0, 1])
    med = lambda v: float(np.median(v)) if v else None
    return dict(persons=len(raw), median_split_half_raw=med(raw), median_split_half_deviation=med(dev),
                share_deviation_reliability_above_0=float(np.mean(np.array(dev) > 0)) if dev else None)

def corpus(name, model, items, Ei):
    W = pd.read_parquet(P.DERIVED / f"windows_{name}.parquet")
    E = np.load(P.DERIVED / f"emb_{model}_{name}.npy").astype(np.float32)
    s = items["instrument"].isin(SYMPTOMS).to_numpy()
    A = Ei[s].astype(np.float32); M = (XS.dot_activities(A, A)).astype(float)
    Pw = (XS.dot_activities(E, A)).astype(float)
    grp = W["group"].fillna("none").to_numpy(); per = W["pid"].to_numpy()
    Pc = G.center_by_group(Pw, grp, per)
    rM = lambda X: float(np.corrcoef(G.upper(X), G.upper(M))[0, 1])
    C_raw, C_c = person_geometry(Pw, per), person_geometry(Pc, per)
    return dict(
        windows=int(len(W)), persons=int(len(np.unique(per))),
        eta2_group_median=float(np.median(eta2(Pw, grp))), eta2_group_range=[float(x) for x in np.quantile(eta2(Pw, grp), [0, 1])],
        eta2_person_median=float(np.median(eta2(Pw, per))),
        r_wording={"window_raw": rM(np.corrcoef(Pw.T)), "window_centred": rM(np.corrcoef(Pc.T)),
                   "person_raw": rM(G.cov_to_corr(C_raw)), "person_centred": rM(G.cov_to_corr(C_c)),
                   "person_centred_whitened": rM(G.whiten(C_c, M))},
        effective_rank={"wording": G_eff(M), "person_centred": G_eff(G.cov_to_corr(C_c))},
        reliability_population={"raw": split_half(Pw, per, M), "centred": split_half(Pc, per, M)},
        reliability_within_person=within_person(Pc, per))

def G_eff(C):
    w = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())

def main(model):
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    Ei = np.load(P.DERIVED / f"emb_{model}_items.npy")
    res = {c: corpus(c, model, items, Ei) for c in ["daicwoz", "edaic_new"]}
    (P.RESULTS / f"imposed_structure_{model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "minilm")
