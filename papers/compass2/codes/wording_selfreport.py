"""Wording selfreport."""
import json

import numpy as np

import geometry as G
from xpsych import scoring as XS

import paths as P
import registered_tests as R

def main():
    out = {}
    for split in ["discovery", "confirmation"]:
        name, W, lab, items = R.load_corpus(split)
        s = items["instrument"].isin(R.SYMPTOMS).to_numpy()
        groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
        out[split] = {}
        for sc in ["minilm", "bge"]:
            E = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32)
            A = np.load(P.DERIVED / f"emb_{sc}_items.npy").astype(np.float32)[s]
            M = (XS.dot_activities(A, A)).astype(float); Pw = (XS.dot_activities(E, A)).astype(float)
            ids, naive = R.profiles(Pw, groups, persons, centre=False)
            keep = np.isin(ids, lab.index); Y = lab.loc[ids[keep]].to_numpy(float); naive = naive[keep]
            iu = np.triu_indices(len(M), 1)
            Ln, Ys = R.corr(naive)[iu], R.corr(Y)[iu]
            out[split][sc] = dict(r_selfreport_wording=float(np.corrcoef(Ys, M[iu])[0, 1]),
                                  r_naive_selfreport=float(np.corrcoef(Ln, Ys)[0, 1]),
                                  partial_naive_selfreport_given_wording=G.partial_corr(Ln, Ys, M[iu]))
    (P.RESULTS / "wording_selfreport.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
