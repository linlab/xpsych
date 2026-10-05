"""Embed windows."""
import json
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import paths as P

torch.set_num_threads(4)
CORPORA = ["daicwoz", "edaic_new", "annomi", "alexst"]
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-large-en-v1.5"}
BATCH = 64

def progress(done, total, label):
    print("Scored", done, "of", total, flush=True)

def main():
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    texts = {c: pd.read_parquet(P.DERIVED / f"windows_{c}.parquet")["text"].tolist() for c in CORPORA}
    total = len(MODELS) * (sum(map(len, texts.values())) + len(items))
    done = 0
    for short, rid in MODELS.items():
        model = SentenceTransformer(P.model_path(rid), revision=None, device="cpu")
        jobs = [("items", items["text"].tolist())] + [(c, texts[c]) for c in CORPORA]
        for name, tx in jobs:
            out = P.DERIVED / f"emb_{short}_{name}.npy"
            if out.exists():
                done += len(tx); continue
            parts = []
            for i in range(0, len(tx), 4096):
                parts.append(model.encode(tx[i:i + 4096], batch_size=BATCH, normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False).astype(np.float16))
                done += len(tx[i:i + 4096]); progress(done, total, "COMPASS 2.0: sentence embeddings (MiniLM, bge-large)")
            np.save(out, np.concatenate(parts))
            print(time.strftime("%H:%M:%S"), short, name, len(tx), flush=True)
    progress(total, total, "COMPASS 2.0: sentence embeddings (MiniLM, bge-large)")

if __name__ == "__main__":
    main()
