"""COMPASS item activity profiles and signed construct read-outs."""
import numpy as np
from ._validation import matrix, labels
from .instruments import ItemBank



def cosine_activities(window_embeddings, item_embeddings):
    """Cosine activity, windows × items; both embedding arrays use rows as observations."""
    e = matrix(window_embeddings, "window_embeddings")
    a = matrix(item_embeddings, "item_embeddings")
    if e.shape[1] != a.shape[1]:
        raise ValueError("window and item embedding dimensions must match")
    en = np.linalg.norm(e, axis=1)
    an = np.linalg.norm(a, axis=1)
    if np.any(en == 0) or np.any(an == 0):
        raise ValueError("embeddings must be nonzero")
    return (e / en[:, None]) @ (a / an[:, None]).T


def stance_activities(probabilities, *, entailment_index, contradiction_index):
    """P(entailment) - P(contradiction) from windows × items × classes probabilities."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 3 or any(s == 0 for s in p.shape) or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite windows × items × classes array")
    for i in (entailment_index, contradiction_index):
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < p.shape[2]:
            raise ValueError("class indices must identify existing columns")
    if entailment_index == contradiction_index:
        raise ValueError("entailment and contradiction indices must differ")
    if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError("class probabilities must be in [0, 1] and sum to one")
    return p[..., entailment_index] - p[..., contradiction_index]


def aggregate_by_person(activities, person_ids):
    """Return (IDs, mean profiles), in first-observed person order."""
    x = matrix(activities, "activities")
    ids = labels(person_ids, len(x), "person_ids")
    order = tuple(dict.fromkeys(ids.tolist()))
    return order, np.stack([x[ids == person].mean(0) for person in order])


class Compass:
    """Apply an item bank to profiles or to text through an explicitly supplied scorer.

    A scorer implements score(texts, bank) -> windows × items activities. Models
    are never loaded implicitly. Custom local scorers can implement this method.
    """
    def __init__(self, bank, scorer=None):
        if not isinstance(bank, ItemBank):
            raise TypeError("bank must be an ItemBank")
        self.bank = bank
        self.scorer = scorer

    def score(self, activities):
        return self.bank.score(activities)

    def activities(self, texts):
        if self.scorer is None:
            raise ValueError("provide a scorer to score text, or pass activities to score()")
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings, not a single string")
        texts = tuple(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("texts must be nonempty strings")
        x = matrix(self.scorer.score(texts, self.bank), "scorer output")
        if x.shape != (len(texts), len(self.bank.items)):
            raise ValueError("scorer output must have one row per text and one column per item")
        return x

    def score_texts(self, texts):
        return self.score(self.activities(texts))


def dot_activities(observation_embeddings, item_embeddings):
    """Dot products in shared embedding coordinates, preserving input precision.

    Both inputs have embeddings as rows. No normalization or dtype conversion
    is performed. This is useful for already normalized cached embeddings, where
    normalizing again after storage rounding would change the recorded results.
    Use cosine_activities for arbitrary, unnormalized embeddings.
    """
    e, a = np.asarray(observation_embeddings), np.asarray(item_embeddings)
    for value in (e, a):
        if value.ndim != 2 or value.shape[1] == 0 or value.dtype.kind not in 'fiu':
            raise ValueError("embeddings must be numeric 2-D arrays")
        if not np.isfinite(value).all():
            raise ValueError("embeddings must be finite")
    if e.shape[1] != a.shape[1]:
        raise ValueError("observation and item embedding dimensions must match")
    return e @ a.T


def apply_key(activities, weights, *, reduction="sum"):
    """Apply one ordered construct key as an elementwise weighted sum or mean.

    Preserves NumPy input precision and reduction order. With reduction='mean',
    divide by the number of supplied columns, not the sum of signed weights.
    Pass only the construct's items when computing a subscale mean.
    """
    x, w = np.asarray(activities), np.asarray(weights)
    if x.ndim != 2 or w.ndim != 1 or x.shape[1] != len(w) or not len(w):
        raise ValueError("provide observations × items and one weight per item")
    if x.dtype.kind not in 'fiu' or w.dtype.kind not in 'fiu':
        raise ValueError("activities and weights must be numeric")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("activities and weights must be finite")
    weighted = x * w
    if reduction == "sum":
        return weighted.sum(1)
    if reduction == "mean":
        return weighted.mean(1)
    raise ValueError("reduction must be sum or mean")


def pool_item_activities(activities, owners, n_items):
    """Max-pool atomic hypothesis activities into their owning item columns.

    owners[j] is the zero-based item index of hypothesis column j. Every item
    must have at least one hypothesis. Dtype and within-item max are preserved.
    """
    from ._validation import positive_integer
    k = positive_integer(n_items, "n_items")
    x, owner = np.asarray(activities), np.asarray(owners)
    if x.ndim != 2 or owner.ndim != 1 or x.shape[1] != len(owner):
        raise ValueError("owners must identify each activity column")
    if owner.dtype.kind not in 'iu' or set(owner.tolist()) != set(range(k)):
        raise ValueError("owners must cover every item index, with no out-of-range values")
    if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError("activities must be finite numeric values")
    return np.stack([x[:, owner == j].max(1) for j in range(k)], 1)
