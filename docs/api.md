# API reference

## Two main entry points

```python
import xpsych as xp

bank = xp.instruments.ipip50()
scores = xp.compass(activities, bank)
rdm = xp.rsa(activities)
comparison = xp.rsa(activities, other_profiles, seed=7)
comparison = xp.rsa(rdm, other_rdm, input_type="rdm", seed=7)
```

`xp.compass(data, bank, scorer=None)` returns `{construct: score_array}`. Without a
scorer, data is a numeric observations × items matrix. With a scorer it is a
sequence of texts: `xp.compass(texts, bank, scorer=scorer)`.

`xp.rsa(first, second=None, input_type="profiles", control=None,
method="spearman", n_permutations=999, seed=None)` returns an item RDM with one
profile matrix, or an `RSAResult` with two sources. With `input_type="rdm"`, both
inputs must be precomputed item matrices. A control is always an aligned item
matrix in the same distance convention. Sources must share item-column order;
row counts may differ. These comparisons do not match people.

## Instruments (`xpsych.instruments`)

`ipip50()` returns the real 50-item IPIP Big-Five Factor Markers and their five
10-item scoring keys. Items are grouped by the official key, with IDs such as
`extraversion_1`; these are not interleaved questionnaire item numbers. Match data
to `bank.ids`. Wording and signs follow the
[official source](https://ipip.ori.org/newBigFive5broadKey.htm).
For NLI, hypotheses prepend "I " to the item text with its initial word lowercased.
IPIP items and scales are public domain. COMPASS signed sums are not standard
1–5 questionnaire totals; negative questionnaire responses instead require `6 - x`.

## Items and banks (`xpsych.instruments`)

`Item(id, text, hypothesis=None)` stores user-supplied wording and optional NLI
hypothesis. IDs must be unique within a bank.

`ItemBank(items, keys)` defines ordered items and a mapping of construct names to
`{item_id: signed_weight}` mappings. Unknown IDs, nonfinite weights and zero-only
keys raise errors. `bank.score(activities)` returns `{construct: array}`. Rows are
preserved. Signed sums do not implement Likert reverse-scoring or normalization.

## COMPASS (`xpsych.scoring`)

`Compass(bank, scorer=None)` exposes `score(activities)`, `activities(texts)` and
`score_texts(texts)`. A custom scorer needs `score(texts, bank)`, returning a finite
array with one row per text and one column per bank item. No model loads by default.

`cosine_activities(window_embeddings, item_embeddings)` normalizes rows and returns
windows × items cosine scores. Zero vectors and mismatched dimensions raise errors.

`stance_activities(probabilities, entailment_index=..., contradiction_index=...)`
converts windows × items × classes probabilities to signed stance. Class indices
are explicit; invalid probabilities raise errors.

`aggregate_by_person(activities, person_ids)` returns `(ids, mean_profiles)`, keeping
first-seen ID order. String IDs are required; each row receives equal weight.

`dot_activities(observation_embeddings, item_embeddings)` computes row-wise dot
products without renormalizing or changing NumPy dtypes. Use it for reproducing
already-normalized, precision-rounded caches; use `cosine_activities` for general
embeddings. Integer inputs retain NumPy's integer arithmetic.

`apply_key(activities, weights, reduction="sum")` applies one ordered key using
an elementwise product followed by sum or mean. It preserves input precision and
reduction order. A mean divides by the number of supplied item columns, not the
sum of signed weights. Pass only the subscale's columns when taking its mean.

`pool_item_activities(activities, owners, n_items)` takes a maximum over each item's
atomic-hypothesis columns. `owners` must cover all zero-based item indices. This
matches the symptom scorer's explicit aggregation rule.

## Psychometric RSA (`xpsych.geometry`)

`covariance(activities)` returns sample covariance over rows, with ddof=1.

`rdm(activities)` returns 1 minus item-column Pearson correlation. It rejects
constant item columns. At least two observations and two items are required.

`wording_null(item_embeddings)` returns the cosine matrix of item rows.

`whiten(observed, reference, floor=1e-8)` returns the symmetric reference residual.
Both matrices must be positive semidefinite and have the same shape. See the
[reference assumptions](concepts.md); this does not recover latent symptoms.

`center_by_group(activities, groups, persons)` centres each row on the group's mean
from other people. String row labels are required. For singleton-person groups,
it removes that person's group mean, matching the paper's explicit fallback.

`compare(first, second, control=None, method="spearman", n_permutations=999, seed=None)`
returns `RSAResult(correlation, pvalue, n_permutations, method)`. Matrices must have
identical item order and at least three items. Diagonals are ignored. P values
are two-sided with a plus-one correction. The public implementation treats
correlation differences within 1e-12 as numerical ties. Select `method="pearson"` to match the
registered COMPASS 2.0 statistic. A fixed seed reproduces the permutation stream.

## Person-level validation (`xpsych.validation`)

`paired_correlation(predicted, observed, n_permutations=999, seed=None)` accepts two
mappings from person ID to scalar score, and returns
`PairedResult(correlation, pvalue, n_people, n_permutations)`. IDs must match exactly
and there must be at least three people with variable finite scores. This avoids
silently pairing arrays with different row orders.

## Numerical kernels

`_geometry` is an internal numerical kernel module, not the supported
application API. It retains the original paper's edge-case behavior and includes
experimental deconvolution routines. The paper adapter imports it so reproductions
preserve the audited estimator and permutation streams.
