"""Compass1 reexamined."""
import json
import sys

import numpy as np
import pandas as pd

from xpsych import scoring as XS

import paths as P

SCALES = ["Task", "Bond", "Goal"]

def eta2(y, g):
    df = pd.DataFrame(dict(y=y, g=g)); m = df.groupby("g")["y"].transform("mean")
    return float(((m - df["y"].mean()) ** 2).sum() / ((df["y"] - df["y"].mean()) ** 2).sum())

def main(model):
    W = pd.read_parquet(P.DERIVED / "windows_alexst.parquet")
    E = np.load(P.DERIVED / f"emb_{model}_alexst.npy").astype(np.float32)
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    Ei = np.load(P.DERIVED / f"emb_{model}_items.npy").astype(np.float32)
    forms = {"client": "WAI-C-COMPASS2025", "therapist": "WAI-T"}
    rows = []
    for role, form in forms.items():
        s = (items["instrument"] == form).to_numpy()
        A = Ei[s]; sc = items.loc[s, "scale"].to_numpy(); pol = items.loc[s, "polarity"].to_numpy()
        m = (W["role"] == role).to_numpy()
        Pw = XS.dot_activities(E[m], A)
        df = pd.DataFrame({"pid": W.loc[m, "pid"].to_numpy()})
        for scale in SCALES:
            j = sc == scale
            df[f"keyed_{scale}"] = XS.apply_key(Pw[:, j], pol[j])
            df[f"topic_{scale}"] = Pw[:, j].mean(1)
        g = df.groupby("pid").mean(); g["role"] = role; rows.append(g)
    S = pd.concat(rows).reset_index()
    tags = W.groupby("pid")["tags"].first()
    S["tags"] = S["pid"].map(tags)
    out = dict(sessions=int(S["pid"].nunique()), sessions_with_several_tags=int((tags.str.contains(";")).sum()),
               tag_counts={k: int(v) for k, v in tags.str.split(";").explode().value_counts().items()})
    for role in forms:
        R = S[S["role"] == role]
        single = R[~R["tags"].str.contains(";")]
        res = {}
        for scale in SCALES:
            k, t = R[f"keyed_{scale}"].to_numpy(), R[f"topic_{scale}"].to_numpy()
            r2 = float(np.corrcoef(k, t)[0, 1] ** 2)
            ks, ts = single[f"keyed_{scale}"].to_numpy(), single[f"topic_{scale}"].to_numpy()
            b = np.polyfit(ts, ks, 1); resid = ks - np.polyval(b, ts)
            res[scale] = dict(r2_keyed_by_topic=r2, slope=float(b[0]),
                              eta2_collection_keyed=eta2(ks, single["tags"]), eta2_collection_topic=eta2(ts, single["tags"]),
                              eta2_collection_keyed_beyond_topic=eta2(resid, single["tags"]))
        out[role] = res
    piv = S.pivot_table(index="pid", columns="role", values=[f"keyed_{s}" for s in SCALES] + [f"topic_{s}" for s in SCALES]).dropna()
    out["sessions_with_both_roles"] = int(len(piv))
    out["role_offset_therapist_minus_client"] = {
        s: dict(keyed=float((piv[(f"keyed_{s}", "therapist")] - piv[(f"keyed_{s}", "client")]).mean()),
                topic=float((piv[(f"topic_{s}", "therapist")] - piv[(f"topic_{s}", "client")]).mean()),
                r_keyed_gap_with_topic_gap=float(np.corrcoef(piv[(f"keyed_{s}", "therapist")] - piv[(f"keyed_{s}", "client")],
                                                             piv[(f"topic_{s}", "therapist")] - piv[(f"topic_{s}", "client")])[0, 1]))
        for s in SCALES}
    (P.RESULTS / f"compass1_reexamined_{model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "minilm")
