"""Optional local neural scorers. Importing xpsych does not import these backends.

Loading defaults to local files only. Set local_files_only=False explicitly to
allow a model download; input text is still scored in the local process.
"""
import numpy as np
from .scoring import cosine_activities
from ._validation import positive_integer


class EmbeddingScorer:
    """Cosine scoring with any encoder exposing encode(list[str]) -> 2-D array."""
    def __init__(self, encoder):
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model, *, revision=None, device="cpu", local_files_only=True):
        """Load a SentenceTransformer; install xpsych[text] first."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install xpsych[text] to load an embedding model") from exc
        return cls(SentenceTransformer(model, revision=revision, device=device,
                                       local_files_only=local_files_only))

    def score(self, texts, bank):
        e = self.encoder.encode(list(texts))
        a = self.encoder.encode([i.text for i in bank.items])
        return cosine_activities(e, a)


def nli_indices(id2label):
    """Require meaningful NLI labels instead of guessing LABEL_0/1/2 order."""
    names = {str(v).lower(): int(k) for k, v in id2label.items()}
    if "entailment" not in names or "contradiction" not in names:
        raise ValueError("model config must identify entailment and contradiction labels")
    ie, ic = names["entailment"], names["contradiction"]
    if ie == ic or ie < 0 or ic < 0:
        raise ValueError("invalid NLI label indices")
    return ie, ic


class NLIScorer:
    """Stance scoring with a local Hugging Face sequence-classification model.

    Every Item must supply a hypothesis. This general adapter scores one
    hypothesis per item. Paper-specific atomic-hypothesis maxima and four-turn
    segmentation remain explicit in papers/compass2 rather than hidden defaults.
    """
    def __init__(self, tokenizer, model, *, batch_size=32, max_length=256):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.batch_size = positive_integer(batch_size, "batch_size")
        self.max_length = positive_integer(max_length, "max_length")
        self.ie, self.ic = nli_indices(model.config.id2label)

    @classmethod
    def from_pretrained(cls, model, *, revision=None, device="cpu", local_files_only=True,
                        batch_size=32, max_length=256):
        """Load local tokenizer and safetensors weights; install xpsych[text] first."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise ImportError("Install xpsych[text] to load an NLI model") from exc
        kwargs = dict(revision=revision, local_files_only=local_files_only)
        tok = AutoTokenizer.from_pretrained(model, **kwargs)
        net = AutoModelForSequenceClassification.from_pretrained(model, use_safetensors=True, **kwargs)
        return cls(tok, net.to(device), batch_size=batch_size, max_length=max_length)

    def score(self, texts, bank):
        if any(i.hypothesis is None for i in bank.items):
            raise ValueError("NLI scoring requires an explicit hypothesis for every item")
        hypotheses = [i.hypothesis for i in bank.items]
        return score_nli_pairs(
            self.tokenizer, self.model, texts, hypotheses, entailment_index=self.ie,
            contradiction_index=self.ic, batch_size=self.batch_size, max_length=self.max_length,
        )


def score_nli_pairs(tokenizer, model, premises, hypotheses, *, entailment_index,
                    contradiction_index, batch_size=32, max_length=256,
                    truncation=True, sort_by_length=False):
    """Score all premise/hypothesis pairs with explicit inference settings.

    Returns float32 P(entailment)-P(contradiction) activities, in input order.
    Length sorting changes batches but restores output order. Preserve sorting,
    batch size and truncation when reproducing a recorded model run.
    """
    import torch
    batch_size = positive_integer(batch_size, "batch_size")
    max_length = positive_integer(max_length, "max_length")
    for name, texts in [('premises', premises), ('hypotheses', hypotheses)]:
        if isinstance(texts, str) or any(not isinstance(t, str) for t in texts):
            raise ValueError(f"{name} must be a sequence of strings")
    if truncation not in (True, 'only_first'):
        raise ValueError("truncation must be True or only_first")
    ie, ic = entailment_index, contradiction_index
    for index in (ie, ic):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("class indices must be distinct nonnegative integers")
    if ie == ic:
        raise ValueError("class indices must be distinct nonnegative integers")
    n_items = len(hypotheses)
    out = np.empty((len(premises), n_items), dtype=np.float32)
    order = np.argsort([len(t) for t in premises]) if sort_by_length else np.arange(len(premises))
    total = len(premises) * n_items
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            pairs = [(int(order[i // n_items]), i % n_items)
                     for i in range(start, min(total, start + batch_size))]
            enc = tokenizer([premises[i] for i, _ in pairs], [hypotheses[j] for _, j in pairs],
                            padding=True, truncation=truncation, max_length=max_length,
                            return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            probs = model(**enc).logits.softmax(-1)
            if max(ie, ic) >= probs.shape[-1]:
                raise ValueError("NLI labels do not match model output columns")
            values = (probs[:, ie] - probs[:, ic]).cpu().numpy()
            for (i, j), value in zip(pairs, values):
                out[i, j] = value
    return out
