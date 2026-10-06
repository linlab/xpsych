# xpsych: A Python toolbox for explainable psychiatry

**Author: Baihan Lin** — [doerlbh@gmail.com](mailto:doerlbh@gmail.com)

Use `import xpsych as xp`, then `xp.compass(...)` for construct scores and
`xp.rsa(...)` for psychometric representational similarity analysis. Both use
ordered item profiles. Scores test a different question from population geometry.

Install from PyPI with `python -m pip install --pre xpsych`; use
`python -m pip install --pre 'xpsych[text]'` for the optional local embedding and
entailment adapters. For a development checkout, use `python -m pip install .`. NumPy and SciPy are the core dependencies.
Python 3.10+ is supported; exact paper reproduction uses Python 3.11.

Start with the real public-domain IPIP-50 inventory using `xp.instruments.ipip50()`.
See [API](api.md), [architecture and UML](architecture.md), [text scoring](text-scoring.md),
and [paper reproduction](reproduction.md).
