# Reproduce COMPASS 2.0

Citation: Baihan Lin (2026), *COMPASS 2.0: psychometric representational similarity
analysis distinguishes symptom structure from personal signal*. **arXiv: TBA**.

From the repository root, install xpsych and the pinned paper dependencies in a
Python 3.11 environment:

```bash
python -m pip install .
python -m pip install -r papers/compass2/requirements.txt
python papers/compass2/run.py test
```

Synthetic checks require no participant data or models. Full reproduction requires
local authorized inputs, the exact model revisions and original instrument wording.
Read `papers/compass2/README.txt` before running other stages.

```bash
python papers/compass2/run.py simulate
python papers/compass2/run.py prepare
python papers/compass2/run.py score
python papers/compass2/run.py analyse
python papers/compass2/run.py plot
python papers/compass2/run.py values
```

`all` runs these stages in order, starting with tests. Paths can be configured with
the PSYCHOMETRIC_* environment variables described in that README. Scoring uses
local model files and may take substantial time. Use a new derived directory to
avoid reusing cached arrays. Reference outputs in `expected/` are comparison targets.

The scripts call xpsych for embedding projections, signed key reductions,
atomic-hypothesis pooling, NLI pair scoring and numerical geometry. They retain
the original cached precision, item order, model batching, length sorting and
truncation choices. Dataset preparation, hypotheses, registration-specific choices,
source-overlap sensitivity analyses and plotting remain paper-specific.

Installing the wheel alone does not install this directory; use the repository or
source distribution for paper reproduction. The `values` stage requires prepared
row-level and instrument files as well as aggregate results. Reference JSONs can
be inspected directly without running that stage.
