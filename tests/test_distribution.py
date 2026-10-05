"""Integration checks for examples, offline imports and the paper adapter."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_core_import_does_not_load_neural_libraries():
    subprocess.run([sys.executable, "-c", "import xpsych, sys; "
                    "assert not {'torch', 'transformers', 'sentence_transformers'} & sys.modules.keys()"], check=True)


def test_synthetic_example_runs():
    subprocess.run([sys.executable, str(ROOT / 'examples/quickstart.py')], check=True)


def test_paper_adapter_is_the_package_core():
    subprocess.run([sys.executable, "-c", "import geometry; from xpsych import _geometry; "
                    "assert geometry.whiten is _geometry.whiten"],
                   cwd=ROOT / 'papers/compass2/codes', check=True)
