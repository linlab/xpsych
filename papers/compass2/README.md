# COMPASS 2.0 reproduction

This folder accompanies **COMPASS 2.0: psychometric representational similarity
analysis distinguishes symptom structure from personal signal**, by Baihan Lin.
Citation: Lin, Baihan (2026), [arXiv:2610.06615](https://arxiv.org/abs/2610.06615).

Install xpsych from the repository root first:

```bash
python -m pip install .
python -m pip install -r papers/compass2/requirements.txt
python papers/compass2/run.py test
```

See [README.txt](README.txt) for source-data access, model setup and the full
reproduction workflow. The scripts cover corpus preparation, scoring, statistical
analyses and figure generation, using xpsych for shared numerical operations.

## Output reference files

The JSON files in `expected/` are aggregate comparison targets. The `values` stage
requires prepared row-level files and instrument inputs. Run it after the full
pipeline. The reference JSONs can also be inspected directly.
