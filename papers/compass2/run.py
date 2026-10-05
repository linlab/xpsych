"""Reproduce analyses from local, authorized inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "test": [["test_geometry.py"], ["registered_tests.py", "--selftest"]],
    "simulate": [["sim_wording_null.py"], ["calibrate_fl.py"], ["sim_estimators.py"]],
    "prepare": [["corpus.py"], ["corpus_alliance.py"]],
    "score": [["embed_windows.py"], ["embed_extra.py", "hlqc"], ["nli_score.py", "--calibrate"],
              ["nli_score.py", "daicwoz", "edaic_new"], ["nli_wai.py", "annomi", "hlqc"],
              ["rdoc_wording.py"], ["multibank.py"], ["method_examples.py"]],
    "analyse": [["wording_geometry.py"], ["imposed_structure.py", "minilm"],
                ["registered_tests.py", "discovery"], ["registered_tests.py", "confirmation"],
                ["alliance_tests.py", "discovery"], ["alliance_tests.py", "confirmation"],
                ["general_factor.py"], ["wording_selfreport.py"], ["compass1_reexamined.py", "minilm"],
                ["symptom_sensitivity.py"], ["alliance_sensitivity.py"], ["paired_baselines.py"], ["hub_sources.py"]],
    "values": [["paper_numbers.py"]],
    "plot": [["figures.py"]],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    for key, value in list(os.environ.items()):
        if key.startswith("PSYCHOMETRIC_"):
            os.environ[key] = str(Path(value).expanduser().resolve())
    stages = ["test", "simulate", "prepare", "score", "analyse", "plot", "values"] if args.stage == "all" else [args.stage]
    for stage in stages:
        for command in STAGES[stage]:
            print("Running", " ".join(command), flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT / "codes", check=True)

if __name__ == "__main__":
    main()
