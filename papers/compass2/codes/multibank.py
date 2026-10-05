"""Multibank."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from xpsych import scoring as XS

import paths as P

torch.set_num_threads(4)
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-large-en-v1.5"}
ITEMS = P.ITEMS
DOMAIN = {"PHQ-8": "symptoms", "PCL-C": "symptoms", "WAI-C": "alliance", "WAI-T": "alliance", "WAI-O": "alliance",
          "MITI": "interaction", "VR-CoDES": "interaction", "IPIP50": "personality", "TIPI": "personality",
          "UCLA3": "process", "UCLA-LS3": "process", "DJG6": "process", "EQ-D": "process", "RRS10": "process",
          "SCS-SF": "process", "ERQ": "process", "RDoC": "framework", "WHO-5": "process", "RSES": "process", "PSS-10": "process"}
RENAME = {"C-SSRS Full (Since Last Visit)": "C-SSRS", "C-SSRS Screener (Recent)": "C-SSRS", "C-SSRS Screener (Since Last Contact)": "C-SSRS",
          "Rosenberg Self-Esteem Scale": "RSES"}

def build():
    rows = []
    ib = pd.read_csv(ITEMS / "item_bank.csv")
    for _, r in ib[ib["instrument"].isin(["PHQ-8", "PCL-C", "WAI-C", "WAI-T", "WAI-O"])].iterrows():
        rows.append(dict(bank=r["instrument"], key=r["key"], text=r["text"]))
    cb = pd.read_csv(ITEMS / "banks" / "coding_personality_banks.csv")
    cb = cb[~cb["category"].fillna("").str.contains("Zimmermann et al. 2011|Del Piccolo et al. 2011")]
    for i, r in cb.iterrows():
        rows.append(dict(bank=r["bank"], key=f"{r['bank']}_{r['code_or_item']}", text=r["text"]))
    rd = pd.read_csv(ITEMS / "rdoc" / "rdoc_constructs.csv")
    rd = rd[(rd["level"] != "domain") & (rd["definition"].fillna("").str.len() > 0)]
    for _, r in rd.iterrows():
        rows.append(dict(bank="RDoC", key=f"RDoC::{r['construct']}::{r['subconstruct'] if isinstance(r['subconstruct'], str) else ''}",
                         text=r["definition"], rdoc_domain=r["domain"]))
    extra = ITEMS / "banks" / "verified_instruments.csv"
    if extra.exists():
        vi = pd.read_csv(extra)
        vi = vi[vi["verified"].astype(str).str.lower().isin(["yes", "partial"]) & vi["text"].notna()]
        it_ = vi["item"].astype(str).str.lower()
        vi = vi[~it_.str.startswith("stem") & ~it_.str.startswith("instruction") & ~vi["key"].astype(str).str.endswith("_DIFF")]
        for _, r in vi.iterrows():
            bank = RENAME.get(r["instrument"], r["instrument"])
            rows.append(dict(bank=bank, key=r["key"], text=r["text"]))
            DOMAIN.setdefault(bank, "symptoms")
    df = pd.DataFrame(rows)
    df = df[df["text"].fillna("").str.strip().str.len() > 0].drop_duplicates(["bank", "text"]).reset_index(drop=True)
    df["domain"] = df["bank"].map(DOMAIN).fillna("other")
    counts = df["bank"].value_counts()
    for bank, n_expected in {"AUDIT": 10, "PHQ-9": 9, "GAD-7": 7, "PHQ-8": 8, "PCL-C": 17, "WAI-O": 36, "VR-CoDES": 34}.items():
        assert counts.get(bank, 0) == n_expected, f"{bank}: {counts.get(bank, 0)} rows, expected {n_expected}"
    df.to_csv(ITEMS / "multibank.csv", index=False)
    return df

def pr(M):
    w = np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())

def cmds(D, k=2):
    n = len(D); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:k]
    return V[:, i] * np.sqrt(np.clip(w[i], 0, None))

def main():
    df = build(); out = dict(items=int(len(df)), banks={k: int(v) for k, v in df["bank"].value_counts().items()})
    for short, rid in MODELS.items():
        E = SentenceTransformer(P.model_path(rid), revision=None, device="cpu").encode(df["text"].tolist(), normalize_embeddings=True, batch_size=32)
        np.save(P.DERIVED / f"emb_{short}_multibank.npy", E.astype(np.float16))
        Ec = E - E.mean(0)
        res = {}
        for b, g in df.groupby("bank"):
            ix = g.index.to_numpy()
            Eb, Ebc = E[ix], Ec[ix]
            res[b] = dict(n=int(len(ix)), effective_rank=pr(XS.dot_activities(Eb, Eb)), effective_rank_mean_removed=pr(XS.dot_activities(Ebc, Ebc)),
                          mean_cos=float((XS.dot_activities(Eb, Eb))[np.triu_indices(len(ix), 1)].mean()) if len(ix) > 1 else None)
        M = XS.dot_activities(E, E); np.fill_diagonal(M, -np.inf); nn = M.argmax(1)
        banks = df["bank"].to_numpy(); doms = df["domain"].to_numpy()
        np.fill_diagonal(M, 1.0)
        bl = sorted(df["bank"].unique())
        between = {a: {b: float(M[np.ix_(banks == a, banks == b)].mean()) for b in bl} for a in bl}
        nn_other_bank = {b: float((banks[nn][banks == b] != b).mean()) for b in bl}
        nn_other_domain = {b: float((doms[nn][banks == b] != doms[banks == b][0]).mean()) for b in bl}
        X = cmds(np.sqrt(np.clip(2 - 2 * (XS.dot_activities(E, E)), 0, None)))
        np.save(P.DERIVED / f"mds_{short}_multibank.npy", X)

        first = ~df.duplicated("text", keep="first").to_numpy()
        Md = (XS.dot_activities(E[first], E[first])); np.fill_diagonal(Md, -np.inf); nnd = Md.argmax(1)
        bd, dd = banks[first], doms[first]
        nn_dedup = {b: float((bd[nnd][bd == b] != b).mean()) for b in sorted(set(bd))}
        nn_dom_dedup = {b: float((dd[nnd][bd == b] != dd[bd == b][0]).mean()) for b in sorted(set(bd))}
        out[short] = dict(per_bank=res, between_bank_mean_cos=between, nn_in_other_bank=nn_other_bank, nn_in_other_domain=nn_other_domain,
                          nn_in_other_bank_dedup=nn_dedup, nn_in_other_domain_dedup=nn_dom_dedup, duplicate_texts=int((~first).sum()),
                          rank_vs_n_corr=float(np.corrcoef([v["n"] for v in res.values()], [v["effective_rank"] for v in res.values()])[0, 1]),
                          all_items_effective_rank=pr(XS.dot_activities(E, E)), all_items_effective_rank_mean_removed=pr(XS.dot_activities(Ec, Ec)))
    (P.RESULTS / "multibank.json").write_text(json.dumps(out, indent=1))
    for short in MODELS:
        print(short, "all items rank", round(out[short]["all_items_effective_rank"], 1), "| mean-removed", round(out[short]["all_items_effective_rank_mean_removed"], 1))
        for b, v in sorted(out[short]["per_bank"].items(), key=lambda kv: kv[1]["n"]):
            print(f"   {b:9s} n={v['n']:3d} rank {v['effective_rank']:.1f} | mean-removed {v['effective_rank_mean_removed']:.1f} | nn other bank {out[short]['nn_in_other_bank'][b]:.2f}")

if __name__ == "__main__":
    main()
