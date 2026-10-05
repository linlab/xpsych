# What is being measured?

An activity matrix has observations in rows and instrument items in columns.
Observations may be answer windows, speaker turns or person-mean profiles. Its
meaning depends on how those rows were produced. Always preserve item order.

COMPASS applies signed scoring keys to these profiles. Cosine similarity measures
related content; it need not distinguish endorsement from denial. NLI stance uses
P(entailment) minus P(contradiction), but remains dependent on hypotheses, context,
model and domain. Neither read-out is automatically calibrated to a questionnaire's
response scale or clinical thresholds.

Psychometric RSA compares relationships among the same items across sources. A
correlation RDM is one minus the correlation between item columns. Comparing upper
triangles can describe agreement between speech, self-report, wording and theory.
Permuting person rows does not change the item correlations. It is therefore
necessary to test person correspondence separately when making person-level claims.

For unit item embeddings A (columns), linear embedding activity has covariance
Aᵀ Σ A. Under isotropic embedding variation this is proportional to Aᵀ A. This
wording null predicts structure without personal symptom information. It is a
reference model, not a complete account of every nuisance. `wording_null` accepts
item embeddings as rows and normalizes them before constructing their cosine matrix.

Whitening computes R⁻¹ᐟ² C R⁻¹ᐟ² − I. Scale and reference choice matter. It transforms
signal as well as nuisance and is not an identified estimator of personal symptom
structure. Eigenvalues are floored for numerical stability. Equality to zero under
C=R requires a full-rank reference above that floor. A within-speaker reference is
only a nuisance model under additional assumptions about within/between variation.

`geometry.compare` uses item-label permutation, jointly permuting rows and columns.
With a control matrix it uses Freedman–Lane residual permutation. Pearson matches
the registered paper test; Spearman ranks the upper triangles first. Inference is
conditional on the fixed instrument and its exchangeability assumptions. A constant
triangle or residual is undefined, and the public API raises an error.

`paired_correlation` matches scalar scores by person ID and shuffles those pairings.
It assumes exchangeable independent people, and does not account for sessions
nested within people or source videos. The paper's specialised resampling and
sensitivity analyses remain in its reproduction folder. No single high correlation
establishes clinical validity, specificity, transportability or measurement invariance.
