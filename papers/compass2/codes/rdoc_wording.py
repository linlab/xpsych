"""Rdoc wording."""
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from xpsych import scoring as XS

import paths as P

torch.set_num_threads(2)
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-large-en-v1.5"}
R = pd.read_csv(P.ITEMS / "rdoc" / "rdoc_constructs.csv")
R = R[(R["level"] != "domain") & (R["definition"].fillna("").str.len() > 0)].reset_index(drop=True)
dom = R["domain"].to_numpy()
out = dict(n_definitions=int(len(R)), per_domain={k: int(v) for k, v in R["domain"].value_counts().items()})
for short, rid in MODELS.items():
    E = SentenceTransformer(P.model_path(rid), revision=None, device="cpu").encode(R["definition"].tolist(), normalize_embeddings=True)
    np.save(P.DERIVED / f"emb_{short}_rdoc.npy", E.astype(np.float16))
    M = XS.dot_activities(E, E); iu = np.triu_indices(len(M), 1)
    same = dom[iu[0]] == dom[iu[1]]
    w = np.clip(np.linalg.eigvalsh(M), 0, None)
    Mn = M.copy(); np.fill_diagonal(Mn, -np.inf); nn = Mn.argmax(1)
    chance = float(sum((dom == d).sum() * ((dom == d).sum() - 1) for d in set(dom)) / (len(dom) * (len(dom) - 1)))
    out[short] = dict(effective_rank=float(w.sum() ** 2 / (w ** 2).sum()), mean_cos_within_domain=float(M[iu][same].mean()),
                      mean_cos_between_domain=float(M[iu][~same].mean()), nearest_neighbour_same_domain=float((dom[nn] == dom).mean()),
                      nearest_neighbour_chance=chance)
(P.RESULTS / "rdoc_wording.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
