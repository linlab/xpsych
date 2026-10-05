"""Symptom sensitivity."""
import json

import numpy as np
import pandas as pd

import geometry as G
from xpsych import scoring as XS

import paths as P
import registered_tests as RT

SEED = 20261004
N_SHUF, N_BOOT, N_PERM = 1000, 2000, 5000

def data(split, sc):
    name, W, lab, items = RT.load_corpus(split)
    s = items["instrument"].isin(RT.SYMPTOMS).to_numpy()
    groups = W["group"].fillna("none").to_numpy(); persons = W["pid"].to_numpy()
    A = {m: np.load(P.DERIVED / f"emb_{m}_items.npy").astype(np.float32)[s] for m in ["minilm", "bge"]}
    Ms = {m: (XS.dot_activities(A[m], A[m])).astype(float) for m in A}
    if sc == "nli":
        Pw = np.load(P.DERIVED / f"nli_{name}.npy").astype(float); M = (Ms["minilm"] + Ms["bge"]) / 2
    else:
        E = np.load(P.DERIVED / f"emb_{sc}_{name}.npy").astype(np.float32); Pw = (XS.dot_activities(E, A[sc])).astype(float); M = Ms[sc]
    X = G.center_by_group(Pw, groups, persons)
    ids = np.unique(persons)
    prof = np.array([X[persons == i].mean(0) for i in ids]); naive = np.array([Pw[persons == i].mean(0) for i in ids])
    keep = np.isin(ids, lab.index); ids_k = ids[keep]
    Y = lab.loc[ids_k].to_numpy(float)

    scat, dof = [], []
    for i in ids_k:
        xi = X[persons == i]; xc = xi - xi.mean(0)
        scat.append(xc.T @ xc); dof.append(max(len(xi) - 1, 0))
    scat, dof = np.array(scat), np.array(dof, float)
    Sw = scat.sum(0) / dof.sum()
    return dict(prof=prof[keep], naive=naive[keep], Y=Y, M=M, Sw=Sw, ids=ids_k, scat=scat, dof=dof)

def pc1(X):
    Z = (X - X.mean(0)) / X.std(0); u, s, vt = np.linalg.svd(Z, full_matrices=False); g = Z @ vt[0]
    return g if np.corrcoef(g, Z.mean(1))[0, 1] >= 0 else -g

def paired_stats(prof, Y):
    za = (prof - prof.mean(0)) / prof.std(0); zb = (Y - Y.mean(0)) / Y.std(0)
    Cx = za.T @ zb / len(Y)
    return {"general_factor_r": float(np.corrcoef(pc1(prof), pc1(Y))[0, 1]),
            "phq8_score_r": float(np.corrcoef(za[:, :8].mean(1), Y[:, :8].sum(1))[0, 1]),
            "item_same_minus_diff": float(np.diag(Cx).mean() - Cx[~np.eye(len(Cx), dtype=bool)].mean())}

def shuffle_demo(d, rng):
    iu = np.triu_indices(25, 1)
    geo = lambda Y: float(np.corrcoef(G.upper(RT.corr(d["naive"])), G.upper(RT.corr(Y)))[0, 1])
    true = {"geometry_agreement": geo(d["Y"]), **paired_stats(d["prof"], d["Y"])}
    null = {k: [] for k in true}
    for _ in range(N_SHUF):
        Ys = d["Y"][rng.permutation(len(d["Y"]))]
        null["geometry_agreement"].append(geo(Ys))
        for k, v in paired_stats(d["prof"], Ys).items():
            null[k].append(v)
    return {k: dict(true=true[k], null_mean=float(np.mean(v)), null_95=np.quantile(v, [.025, .975]).tolist(),
                    p=float((1 + np.sum(np.abs(np.array(v)) >= abs(true[k]))) / (1 + N_SHUF))) for k, v in null.items()}

def fl_multi(W, Rs, controls, rng, blocks=None, n_perm=N_PERM):
    """Freedman-Lane partial association of W with Rs given several control matrices (as in the registered H2 with two
    controls); item permutations of the residual, optionally restricted within blocks. Returns (partial r, p)."""
    k = len(W); iu = np.triu_indices(k, 1)
    y, x = W[iu], Rs[iu]; Zc = np.column_stack([c[iu] for c in controls])
    Z = np.column_stack([np.ones(len(y)), Zc])
    fit = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    Rm = np.zeros((k, k)); Rm[iu] = y - fit; Rm = Rm + Rm.T; Fm = np.zeros((k, k)); Fm[iu] = fit; Fm = Fm + Fm.T
    r = G.partial_corr(y, x, Zc); null = np.empty(n_perm)
    for b in range(n_perm):
        if blocks is None:
            q = rng.permutation(k)
        else:
            q = np.arange(k)
            for blk in blocks:
                q[blk] = rng.permutation(blk)
        null[b] = G.partial_corr((Fm + Rm[np.ix_(q, q)])[iu], x, Zc)
    return float(r), float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm))

def h2_boot_ci(d, rng, K=None):
    iu = np.triu_indices(25, 1); n = len(d["Y"]); vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        W = G.whiten(np.cov(d["prof"][i].T), d["M"]); Rs = RT.corr(d["Y"][i])
        Zc = d["M"][iu] if K is None else np.column_stack([d["M"][iu], K[iu]])
        vals.append(G.partial_corr(W[iu], Rs[iu], Zc))
    return np.nanquantile(vals, [.025, .975]).tolist()

def noise_normalized(d, rng, K=None):
    """Whitening by the within-speaker reference S_w. Three control sets, so the change of reference can be separated
    from the change of controls: (i) Corr(S_w) only; (ii) the registered controls (M, and K for the entailment scorer);
    (iii) registered controls plus Corr(S_w). Bootstrap intervals re-estimate S_w from the resampled people (they remain
    conditional on the estimated question means)."""
    iu = np.triu_indices(25, 1)
    C = np.cov(d["prof"].T); Wn = G.whiten(C, d["Sw"]); Rs = RT.corr(d["Y"]); Swc = G.cov_to_corr(d["Sw"])
    reg = [d["M"]] if K is None else [d["M"], K]
    sets = {"reference_only": [Swc], "registered_controls": reg, "registered_plus_reference": reg + [Swc]}
    out = dict(r_reference_vs_wording=float(np.corrcoef(G.upper(Swc), G.upper(d["M"]))[0, 1]))
    for name, ctr in sets.items():
        r, pv = fl_multi(Wn, Rs, ctr, rng)
        out[name] = dict(partial_r=r, p_freedman_lane=pv)
    n = len(d["Y"]); vals = {k: [] for k in sets}
    for _ in range(N_BOOT // 2):
        i = rng.integers(0, n, n)
        Swb = d["scat"][i].sum(0) / d["dof"][i].sum(); Swbc = G.cov_to_corr(Swb)
        Wb = G.whiten(np.cov(d["prof"][i].T), Swb); Rb = RT.corr(d["Y"][i])
        bsets = {"reference_only": [Swbc], "registered_controls": reg, "registered_plus_reference": reg + [Swbc]}
        for k_, ctr in bsets.items():
            vals[k_].append(G.partial_corr(Wb[iu], Rb[iu], np.column_stack([c[iu] for c in ctr])))
    for k_ in sets:
        out[k_]["interval"] = np.nanquantile(vals[k_], [.025, .975]).tolist()
    return out

def main():
    rng = np.random.default_rng(SEED)
    K = np.array(json.loads((P.RESULTS / "nli_calibration.json").read_text())["K"]); K = (K + K.T) / 2
    out = {"shuffle": {}, "noise_normalized": {}, "h2_interval": {}, "h2_block_permutation": {}}
    blocks = [np.arange(8), np.arange(8, 25)]
    for split in ["discovery", "confirmation"]:
        for sc in ["minilm", "bge", "nli"]:
            d = data(split, sc); key = f"{split}:{sc}"
            out["shuffle"][key] = shuffle_demo(d, rng)
            Kc = K if sc == "nli" else None
            out["noise_normalized"][key] = noise_normalized(d, rng, Kc)
            out["h2_interval"][key] = h2_boot_ci(d, rng, Kc)
            W = G.whiten(np.cov(d["prof"].T), d["M"]); ctr = [d["M"]] if Kc is None else [d["M"], Kc]
            r_b, p_b = fl_multi(W, RT.corr(d["Y"]), ctr, rng, blocks=blocks, n_perm=10000)
            reg_r = json.loads((P.RESULTS / f"registered_{split}.json").read_text())[sc]["H2"]["partial_r"]
            assert abs(r_b - reg_r) < 1e-9, f"block test statistic {r_b} differs from registered H2 {reg_r}"
            r_m, p_m = fl_multi(W, RT.corr(d["Y"]), [d["M"]], rng, blocks=blocks)
            out["h2_block_permutation"][key] = dict(partial_r=r_b, p=p_b, controls="registered (M" + (", K)" if Kc is not None else ")"),
                                                    m_only_partial_r=r_m, m_only_p=p_m)
            print(key, "done", flush=True)

    name, W_, lab, items = RT.load_corpus("confirmation")
    res = {}
    s = items["instrument"].isin(RT.SYMPTOMS).to_numpy()
    for sc in ["minilm", "bge", "nli"]:
        d = data("confirmation", sc); m = d["ids"] != 683
        prof, naive, Y = d["prof"][m], d["naive"][m], d["Y"][m]
        r = {"persons": int(m.sum())}
        if sc != "nli":
            r["H1"] = RT.h1(naive, Y, d["M"], rng, alpha=0.05 / 2)
        r["H2"] = RT.h2(prof, Y, d["M"], K if sc == "nli" else None, rng)
        r["H3"] = RT.h3(prof, Y, rng, alpha=0.05 / 3)
        res[sc] = r
    out["confirmation_without_683"] = res
    (P.RESULTS / "symptom_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["h2_interval", "h2_block_permutation"]}, indent=1))

if __name__ == "__main__":
    main()
