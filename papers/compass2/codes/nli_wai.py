"""Nli wai."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xpsych.text import score_nli_pairs

import paths as P

torch.set_num_threads(int(os.environ.get("NLI_THREADS", "4")))
RID, REV = P.NLI
SEG, BATCH, CHUNK = 4, 32, 500

def segments(W):
    rows = []
    for pid, g in W.sort_values(["pid", "widx"]).groupby("pid", sort=True):
        g = g[g["role"].isin(["therapist", "client"])]
        for s in range(0, len(g), SEG):
            b = g.iloc[s:s + SEG]
            text = "\n".join(f"{r.capitalize()}: {t}" for r, t in zip(b["role"], b["text"]))
            rows.append(dict(pid=pid, seg=s // SEG, first_widx=int(b["widx"].iloc[0]), n_turns=len(b), text=text))
    return pd.DataFrame(rows)

def score(tok, model, ie, ic, premises, hyps):
    return score_nli_pairs(tok, model, premises, hyps, entailment_index=ie,
                           contradiction_index=ic, batch_size=BATCH, max_length=256,
                           truncation="only_first", sort_by_length=True)


def main(corpora):
    tok = AutoTokenizer.from_pretrained(RID, revision=REV)
    model = AutoModelForSequenceClassification.from_pretrained(RID, revision=REV).eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    ie, ic = lab["entailment"], lab["contradiction"]
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    hyps = items[items["instrument"] == "WAI-O"]["hypothesis"].tolist()
    segs = {c: segments(pd.read_parquet(P.DERIVED / f"windows_{c}.parquet")) for c in corpora}
    total = sum(len(s) for s in segs.values()); done = 0
    for c in corpora:
        S = segs[c]
        S.drop(columns=["text"]).to_parquet(P.DERIVED / f"nliwai_{c}_segments.parquet", index=False)
        out = P.DERIVED / f"nliwai_{c}.npy"
        if out.exists():
            done += len(S); continue
        parts = []
        for st in range(0, len(S), CHUNK):
            ck = P.DERIVED / f"nliwai_{c}.part{st // CHUNK:04d}.npy"
            if ck.exists():
                parts.append(np.load(ck))
            else:
                parts.append(score(tok, model, ie, ic, S["text"].iloc[st:st + CHUNK].tolist(), hyps).astype(np.float16))
                np.save(ck, parts[-1])
            done += len(parts[-1])
            print("Scored", done, "of", total, flush=True)
        np.save(out, np.concatenate(parts))
        for ck in P.DERIVED.glob(f"nliwai_{c}.part*.npy"):
            ck.unlink()
        print(time.strftime("%H:%M:%S"), c, len(S), flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or ["annomi", "hlqc"])
