"""Hub sources."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xpsych import scoring as XS

import paths as P
import registered_tests as RT

def kendall_tau_a(x, y):
    """Kendall's tau-a (ties count as neither concordant nor discordant), recommended when an RDM has many ties."""
    x, y = np.asarray(x), np.asarray(y); n = len(x)
    dx = np.sign(x[:, None] - x[None, :]); dy = np.sign(y[:, None] - y[None, :])
    return float((dx * dy).sum() / (n * (n - 1)))

def main():
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    s = items["instrument"].isin(RT.SYMPTOMS).to_numpy()
    A = {m: np.load(P.DERIVED / f"emb_{m}_items.npy").astype(np.float32)[s] for m in ["minilm", "bge"]}
    rdms, rel = {}, {}
    rng = np.random.default_rng(20261003)
    iu0 = np.triu_indices(25, 1)

    def split_half(X, n_split=200):
        """Spearman-Brown split-half reliability of an RDM over random halves of participants (its noise ceiling)."""
        rs = []
        for _ in range(n_split):
            idx = rng.permutation(len(X)); h = len(X) // 2
            a, b = (1 - RT.corr(X[idx[:h]]))[iu0], (1 - RT.corr(X[idx[h:2 * h]]))[iu0]
            rs.append(spearmanr(a, b)[0])
        r = float(np.mean(rs)); return 2 * r / (1 + r)
    for split, tag in [("discovery", "DAIC-WOZ"), ("confirmation", "E-DAIC")]:
        name, W, lab, _ = RT.load_corpus(split)
        groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
        for sc, sname in [("minilm", "embedding"), ("nli", "entailment")]:
            if sc == "nli":
                Pw = np.load(P.DERIVED / f"nli_{name}.npy").astype(float)
            else:
                E = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32); Pw = (XS.dot_activities(E, A[sc])).astype(float)
            ids, naive = RT.profiles(Pw, groups, persons, centre=False)
            keep = np.isin(ids, lab.index)
            rdms[f"language, {sname} ({tag})"] = 1 - RT.corr(naive[keep]); rel[f"language, {sname} ({tag})"] = split_half(naive[keep])
            if sc == "minilm":
                Yk = lab.loc[ids[keep]].to_numpy(float)
                rdms[f"self-report ({tag})"] = 1 - RT.corr(Yk); rel[f"self-report ({tag})"] = split_half(Yk)
    rdms["item wording (MiniLM)"] = 1 - (XS.dot_activities(A["minilm"], A["minilm"]))
    rdms["item wording (bge)"] = 1 - (XS.dot_activities(A["bge"], A["bge"]))
    K = np.array(json.loads((P.RESULTS / "nli_calibration.json").read_text())["K"]); K = (K + K.T) / 2
    rdms["entailment probe geometry"] = 1 - RT.corr(K)
    clusters = np.array([0] * 8 + [1] * 5 + [2] * 7 + [3] * 5)
    rdms["theory: scale clusters"] = (clusters[:, None] != clusters[None, :]).astype(float)
    names = list(rdms)
    iu = np.triu_indices(25, 1)
    S = np.array([[spearmanr(rdms[a][iu], rdms[b][iu])[0] for b in names] for a in names])
    Tau = np.array([[kendall_tau_a(rdms[a][iu], rdms[b][iu]) for b in names] for a in names])
    np.savez(P.DERIVED / "hub_rdms.npz", names=np.array(names), **{f"r{k}": rdms[n] for k, n in enumerate(names)})
    (P.RESULTS / "hub_sources.json").write_text(json.dumps({"names": names, "spearman": S.tolist(), "kendall_tau_a": Tau.tolist(),
                                                           "split_half_reliability": rel}, indent=1))
    print("reliability:", {k: round(v, 2) for k, v in rel.items()})
    print(pd.DataFrame(S, index=names, columns=[n[:14] for n in names]).round(2).to_string())

if __name__ == "__main__":
    main()
