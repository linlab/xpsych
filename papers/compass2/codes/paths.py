"""Local input and output locations. Override them with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PSYCHOMETRIC_DATA", ROOT / "input"))
ITEMS = Path(os.environ.get("PSYCHOMETRIC_ITEMS", DATA / "items"))
DAICWOZ = Path(os.environ.get("PSYCHOMETRIC_DAICWOZ", DATA / "daicwoz"))
EDAIC = Path(os.environ.get("PSYCHOMETRIC_EDAIC", DATA / "edaic"))
EDAIC_LABELS = EDAIC / "labels" / "detailed_lables.csv"
ALEXST = Path(os.environ.get("PSYCHOMETRIC_ALEXST", DATA / "alexst"))
DERIVED = Path(os.environ.get("PSYCHOMETRIC_DERIVED", ROOT / "derived"))
RESULTS = Path(os.environ.get("PSYCHOMETRIC_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("PSYCHOMETRIC_FIGURES", ROOT / "figures"))
MODELS = Path(os.environ.get("PSYCHOMETRIC_MODELS", ROOT / "models"))
MODEL_FOLDERS = {"sentence-transformers/all-MiniLM-L6-v2": "minilm", "BAAI/bge-large-en-v1.5": "bge"}
EMBEDDERS = {k: None for k in MODEL_FOLDERS}
NLI = (str(MODELS / "nli"), None)

def model_path(identifier):
    folder = MODELS / MODEL_FOLDERS[identifier]
    if not folder.is_dir():
        raise FileNotFoundError(f"Place the model weights specified in Methods at {folder}")
    return str(folder)

for directory in [DERIVED, RESULTS]:
    directory.mkdir(parents=True, exist_ok=True)
