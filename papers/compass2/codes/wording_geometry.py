"""Wording geometry."""
import json
import sys

import numpy as np
import pandas as pd

from xpsych import scoring as XS

import paths as P

DUPES = [("PHQ8_3_Sleep", "PCL-C_13_Sleep"), ("PHQ8_7_Concentration", "PCL-C_15_Concentration"),
         ("PHQ8_1_NoInterest", "PCL-C_9_NoInterest")]

def participation_ratio(M):
    w = np.clip(np.linalg.eigvalsh(M), 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())

def symptoms(items, E):
    s = items["instrument"].isin(["PHQ-8", "PCL-C"]).to_numpy()
    A = E[s].astype(float); M = XS.dot_activities(A, A)
    inst = items.loc[s, "instrument"].to_numpy(); keys = items.loc[s, "key"].tolist()
    iu = np.triu_indices(len(A), 1)
    within = M[iu][inst[iu[0]] == inst[iu[1]]]; between = M[iu][inst[iu[0]] != inst[iu[1]]]
    dup = {f"{a}~{b}": float(M[keys.index(a), keys.index(b)]) for a, b in DUPES}
    others = between[~np.isin(np.arange(len(between)), [])]
    rank_dup = {k: float((between >= v).mean()) for k, v in dup.items()}
    return dict(items=int(s.sum()), effective_rank=participation_ratio(M), mean_cos_within=float(within.mean()),
                mean_cos_between=float(between.mean()), near_duplicates=dup, near_duplicate_top_share=rank_dup)

def wai(items, E, form):
    s = (items["instrument"] == form).to_numpy()
    A = E[s].astype(float); sc = items.loc[s, "scale"].to_numpy(); pol = items.loc[s, "polarity"].to_numpy()
    M = XS.dot_activities(A, A); out = {}
    for scale in ["Task", "Bond", "Goal"]:
        m = sc == scale
        idx = np.flatnonzero(m); f = idx[pol[idx] > 0]; r = idx[pol[idx] < 0]
        ff = M[np.ix_(f, f)][np.triu_indices(len(f), 1)].mean()
        fr = M[np.ix_(f, r)].mean()
        cen = A[m].mean(0); cen /= np.linalg.norm(cen)
        out[scale] = dict(forward=int(len(f)), reverse=int(len(r)), cos_forward_forward=float(ff),
                          cos_forward_reverse=float(fr), centroid_load_forward=float((A[f] @ cen).mean()),
                          centroid_load_reverse=float((A[r] @ cen).mean()))
    return out

def main(models):
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    res = {}
    for m in models:
        f = P.DERIVED / f"emb_{m}_items.npy"
        if not f.exists():
            print("missing", f); continue
        E = np.load(f)
        res[m] = dict(symptoms=symptoms(items, E), **{form: wai(items, E, form) for form in ["WAI-C", "WAI-T", "WAI-O"]})
    old = json.loads((P.RESULTS / "wording_geometry.json").read_text()) if (P.RESULTS / "wording_geometry.json").exists() else {}
    old.update(res)
    (P.RESULTS / "wording_geometry.json").write_text(json.dumps(old, indent=1))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1:] or ["minilm", "bge"])
