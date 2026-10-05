"""Frozen pre-refactor NLI loops for numerical and batch-order regression checks."""

import numpy as np
import torch

def score_symptoms(tok, model, ie, ic, premises, hypotheses):
    out = np.empty((len(premises), len(hypotheses)), np.float32)
    order = np.argsort([len(x) for x in premises])
    pairs = [(i, j) for i in order for j in range(len(hypotheses))]
    for s in range(0, len(pairs), 64):
        b = pairs[s:s + 64]
        enc = tok([premises[i] for i, _ in b], [hypotheses[j] for _, j in b], truncation=True, max_length=256, padding=True, return_tensors='pt')
        pr = torch.softmax(model(**enc).logits, -1).numpy()
        for (i, j), p in zip(b, pr):
            out[i, j] = p[ie] - p[ic]
    return out

def score_alliance(tok, model, ie, ic, premises, hyps):
    out = np.empty((len(premises), len(hyps)), np.float32)
    order = np.argsort([len(x) for x in premises])
    pairs = [(i, j) for i in order for j in range(len(hyps))]
    for s in range(0, len(pairs), 32):
        b = pairs[s:s + 32]
        enc = tok([premises[i] for i, _ in b], [hyps[j] for _, j in b], truncation='only_first', max_length=256, padding=True, return_tensors='pt')
        pr = torch.softmax(model(**enc).logits, -1).numpy()
        for (i, j), p in zip(b, pr):
            out[i, j] = p[ie] - p[ic]
    return out
