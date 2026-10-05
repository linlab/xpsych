"""Nli score."""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xpsych.text import score_nli_pairs
from xpsych.scoring import pool_item_activities

import paths as P

torch.set_num_threads(4)
RID, REV = P.NLI
CHUNK, BATCH = 2000, 64

def load():
    tok = AutoTokenizer.from_pretrained(RID, revision=REV)
    model = AutoModelForSequenceClassification.from_pretrained(RID, revision=REV).eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return tok, model, lab["entailment"], lab["contradiction"]

NEUTRAL = ["I went to the grocery store yesterday.", "The weather was nice this morning.", "I work as an accountant.",
           "My brother lives in Ohio.", "We had pasta for dinner.", "I take the bus to work.", "The meeting was moved to Tuesday.",
           "I grew up in a small town.", "My car needs an oil change.", "I like watching basketball.", "We painted the kitchen last year.",
           "I usually read the news online.", "My daughter started school in September.", "The library closes at six.",
           "I bought a new phone.", "We visited my parents for the holidays.", "I have two cats.", "The traffic was heavy today.",
           "I studied engineering in college.", "My friend is getting married next month.", "I cook most of my meals at home.",
           "The park near my house has a lake.", "I listen to podcasts in the car.", "We play cards on Fridays.",
           "I moved to this city five years ago.", "My job involves a lot of email.", "I was born in March.",
           "The store was out of milk.", "I am learning to play the guitar.", "We are planning a trip to the coast.",
           "My neighbor has a big dog.", "I watched a movie last night.", "The bank is next to the post office.",
           "I drink coffee in the morning.", "My sister is a nurse.", "We replaced the old fence.", "I walk to the train station.",
           "The restaurant was busy.", "I fixed the bike over the weekend.", "My office is on the third floor."]

def atoms_of(items):
    s = items[items["instrument"].isin(["PHQ-8", "PCL-C"])]
    atoms, owner = [], []
    for j, a in enumerate(s["atoms"]):
        for x in a.split(" || "):
            atoms.append(x); owner.append(j)
    return s, atoms, np.array(owner)

def to_items(A, owner, k):
    return pool_item_activities(A, owner, k)

def score(tok, model, ie, ic, premises, hypotheses):
    return score_nli_pairs(tok, model, premises, hypotheses, entailment_index=ie,
                           contradiction_index=ic, batch_size=BATCH, max_length=256,
                           truncation=True, sort_by_length=True)


def progress(done, total):
    print("Scored", done, "of", total, flush=True)

def bench():
    tok, model, ie, ic = load()
    prem = ["I have not been sleeping well and I feel tired all the time."] * 8 + ["Work has been fine, nothing special."] * 8
    hyp = pd.read_csv(P.ITEMS / "item_bank.csv").query("instrument in ['PHQ-8','PCL-C']")["hypothesis"].tolist()[:16]
    t = time.time(); s = score(tok, model, ie, ic, prem, hyp); dt = time.time() - t
    print(f"{len(prem) * len(hyp)} pairs in {dt:.1f} s = {len(prem) * len(hyp) / dt:.0f} pairs/s")
    print("sleep-item score, sleep text vs neutral text:", round(float(s[0, 2]), 2), round(float(s[8, 2]), 2))

def calibrate():
    tok, model, ie, ic = load()
    s, atoms, owner = atoms_of(pd.read_csv(P.ITEMS / "item_bank.csv"))
    halo = to_items(score(tok, model, ie, ic, NEUTRAL, atoms), owner, len(s))
    K = to_items(score(tok, model, ie, ic, s["text"].tolist(), atoms), owner, len(s))
    res = dict(neutral_mean_per_item=dict(zip(s["key"], np.round(halo.mean(0), 3).tolist())),
               neutral_mean=float(halo.mean()), neutral_share_below_minus_half=float((halo < -0.5).mean()),
               K=np.round(K, 3).tolist(), K_diag_mean=float(np.diag(K).mean()),
               K_offdiag_mean=float(K[~np.eye(len(s), dtype=bool)].mean()), keys=s["key"].tolist())
    (P.RESULTS / "nli_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("K", "keys", "neutral_mean_per_item")}, indent=1))

def main(corpora):
    tok, model, ie, ic = load()
    s, atoms, owner = atoms_of(pd.read_csv(P.ITEMS / "item_bank.csv"))
    texts = {c: pd.read_parquet(P.DERIVED / f"windows_{c}.parquet")["text"].tolist() for c in corpora}
    total = sum(map(len, texts.values())); done = 0
    for c in corpora:
        out = P.DERIVED / f"nli_{c}.npy"
        if out.exists():
            done += len(texts[c]); continue
        parts = []
        for st in range(0, len(texts[c]), CHUNK):
            ck = P.DERIVED / f"nli_atoms_{c}.part{st // CHUNK:04d}.npy"
            if ck.exists():
                parts.append(np.load(ck))
            else:
                parts.append(score(tok, model, ie, ic, texts[c][st:st + CHUNK], atoms).astype(np.float16)); np.save(ck, parts[-1])
            done += len(parts[-1]); progress(done, total)
        A = np.concatenate(parts)
        np.save(P.DERIVED / f"nli_atoms_{c}.npy", A); np.save(out, to_items(A.astype(np.float32), owner, len(s)).astype(np.float16))
        for ck in P.DERIVED.glob(f"nli_atoms_{c}.part*.npy"):
            ck.unlink()
        print(time.strftime("%H:%M:%S"), c, len(texts[c]), flush=True)
    progress(total, total)

if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    elif "--calibrate" in sys.argv:
        calibrate()
    else:
        main([a for a in sys.argv[1:]] or ["daicwoz", "edaic_new"])
