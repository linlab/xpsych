# Local text scoring

Install the optional backend with `python -m pip install '.[text]'` from the checkout.
The core package remains usable without these dependencies.

```python
import xpsych as xp
from xpsych.text import NLIScorer

bank = xp.instruments.ipip50()
scorer = NLIScorer.from_pretrained("/path/to/local/nli-model", device="cpu")
scores = xp.compass(
    ["I enjoy meeting people and often start conversations."], bank, scorer=scorer,
)
```

`EmbeddingScorer.from_pretrained(path_or_id, revision=None, device="cpu",
local_files_only=True)` loads a SentenceTransformer. It computes cosine similarity
between text embeddings and item-text embeddings. A custom encoder can instead be
passed directly to `EmbeddingScorer(encoder)` if it implements `encode(texts)`.

`NLIScorer.from_pretrained` has the same model-loading options, plus `batch_size=32`
and `max_length=256`. It requires meaningful entailment and contradiction labels in
the model configuration. Generic LABEL_0 labels are rejected rather than guessed.
It scores pairs in bounded batches with inference mode, retaining input order.
Long inputs are truncated; choose and document segmentation before scoring.

Downloads require explicitly setting `local_files_only=False`. Inference is local;
there is no hosted scoring API. Models and their licences are external dependencies.
Pin their revision and record preprocessing when comparing results across runs.

The general NLI adapter uses one hypothesis per item. COMPASS 2.0's symptom scorer
uses maxima over atomic hypotheses for some items, and its alliance scorer uses
four-turn exchanges. These are paper-specific procedures in `papers/compass2`.
The simple example above is therefore not a substitute for exact paper reproduction.

`score_nli_pairs(tokenizer, model, premises, hypotheses, entailment_index=...,
contradiction_index=..., batch_size=32, max_length=256, truncation=True,
sort_by_length=False)` exposes the shared pair-scoring operation. It returns a
float32 premises × hypotheses array. For reproduction, retain the original batch
size, length sorting and truncation: symptoms use 64 pairs and truncation=True;
alliance uses 32 pairs and truncation="only_first". Output rows always retain the
input order. Pass a model in evaluation mode, as the loaders and NLIScorer do.
