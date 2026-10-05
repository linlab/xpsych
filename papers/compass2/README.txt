COMPASS 2.0: psychometric representational similarity analysis

This package contains the analysis and plotting code, reference aggregate results,
public symptom items and hypotheses, RDoC definitions, and input schemas. It does
not contain participant transcripts, questionnaire responses, licensed instrument
texts or model weights.

Install and check

Use Python 3.11. From this directory, run:

    python -m pip install -r requirements.txt
    python run.py test

The checks use synthetic data and need no transcript access. The reference JSON
files in expected/ contain aggregate numerical results for comparison. They are
not a substitute for independently rerunning the analyses. The values stage also
requires prepared row-level files and instrument inputs, so run it only after the
full pipeline described below; it does not operate on reference aggregates alone.

Citation: Baihan Lin (2026), COMPASS 2.0: psychometric representational similarity
analysis distinguishes symptom structure from personal signal. arXiv: TBA.

Inputs for a full reproduction

Obtain the original corpora under their applicable access conditions. DAIC-WOZ and
E-DAIC require access through USC ICT (https://dcapswoz.ict.usc.edu/). Psychotherapy
transcripts require an Alexander Street subscription. Obtain AnnoMI and the high-
and low-quality counselling corpus from the sources cited in the manuscript.
Keep these inputs local. The following layout is expected under input/:

    daicwoz/data/<participant>_P/<participant>_TRANSCRIPT.csv
    edaic/data/<participant>_P/<participant>_Transcript.csv
    edaic/labels/detailed_lables.csv
    alexst/<collection>-transcripts/*.txt
    annomi/AnnoMI-full.csv
    highlowquality_counseling/HighLowQualityCounseling/labels.csv
    highlowquality_counseling/HighLowQualityCounseling/urls.csv
    highlowquality_counseling/HighLowQualityCounseling/transcripts/<id>
    items/item_bank.csv
    items/banks/coding_personality_banks.csv
    items/banks/verified_instruments.csv
    items/rdoc/rdoc_constructs.csv

The spelling detailed_lables.csv follows the source data. Input schemas are in
inputs/. The ordered item_index.csv preserves the item identifiers and scoring
keys. symptom_items.csv supplies the PHQ-8 and PCL-C texts and atomic hypotheses.
RDoC definitions are supplied separately. The complete instrument-bank CSVs,
including the exact WAI forms and wording used in analysis, must be obtained from
the author subject to the instrument permissions. Filling the schemas with other
wording or reordering rows will not reproduce the reported results. Use a writable
copy of the item inputs: multibank.py creates multibank.csv there.

Place the model files specified in manuscript Methods in models/minilm,
models/bge and models/nli. These correspond to sentence-transformers/all-MiniLM-L6-v2,
BAAI/bge-large-en-v1.5 and cross-encoder/nli-deberta-v3-base. Use the exact model
revisions identified in Methods; substituting current weights can change results.
Models run locally on CPU. Neural scoring is substantially slower than the checks.

Existing locations can be supplied through PSYCHOMETRIC_DATA, PSYCHOMETRIC_DAICWOZ,
PSYCHOMETRIC_EDAIC, PSYCHOMETRIC_ALEXST, PSYCHOMETRIC_ITEMS and PSYCHOMETRIC_MODELS.
Set PSYCHOMETRIC_DERIVED, PSYCHOMETRIC_RESULTS and PSYCHOMETRIC_FIGURES to choose
output folders. Defaults are derived/, results/ and figures/ in this package.
Use an empty derived/ directory for a fresh run, since scoring reuses cached arrays.

Run the analyses

    python run.py all

The stages can also be run separately, in this order:

    python run.py test
    python run.py simulate
    python run.py prepare
    python run.py score
    python run.py analyse
    python run.py plot
    python run.py values

The plot stage also computes the illustrative method-figure statistics needed by
the values stage. Results are written as JSON, intermediate arrays and locally
processed transcripts are stored in derived/, and plotting generates PDF and PNG
figures. Keep derived/ private under the source-data access conditions.

registered_tests.py implements H1-H3; alliance_tests.py implements the pre-specified
alliance comparisons. The general-factor, paired-baseline and sensitivity scripts
implement the exploratory analyses. The original identifier-based counselling
exclusion is deliberately retained for reproducing the registered sample. The
transcript-screen alternatives are implemented in alliance_sensitivity.py, as
described in Supplementary Note 2. Geometric comparison and whitening functions
are in geometry.py; simulation scripts evaluate their assumptions.

Reference comparisons allow small floating-point differences across systems.
Use the sample sizes, effect estimates, intervals and adjusted P values together;
do not judge reproducibility solely by whether a significance threshold is crossed.
