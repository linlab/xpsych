"""Embed extra."""
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import paths as P

torch.set_num_threads(4)
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-large-en-v1.5"}

for corpus in sys.argv[1:] or ["hlqc"]:
    texts = pd.read_parquet(P.DERIVED / f"windows_{corpus}.parquet")["text"].tolist()
    for short, rid in MODELS.items():
        out = P.DERIVED / f"emb_{short}_{corpus}.npy"
        if out.exists():
            continue
        model = SentenceTransformer(P.model_path(rid), revision=None, device="cpu")
        np.save(out, model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float16))
        print(short, corpus, len(texts), flush=True)
