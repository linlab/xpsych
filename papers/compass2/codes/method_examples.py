"""Method examples."""
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xpsych import scoring as XS

import paths as P

torch.set_num_threads(2)
UTTER = ["I lie awake most nights and can't get back to sleep.",
         "Nothing I used to enjoy feels worth doing now.",
         "I'm exhausted all the time, even after resting.",
         "Mostly I just feel low and hopeless."]
ITEMS = ["Little interest or pleasure in doing things",
         "Feeling down, depressed, or hopeless",
         "Trouble falling or staying asleep, or sleeping too much",
         "Feeling tired or having little energy"]
STANCE_UTT = "I feel I can tell my therapist anything."
STANCE_ITEMS = {"forward": "My therapist and I trust each other.", "reverse": "I do not trust my therapist."}

def cmds(D, k=2):
    n = len(D); J = np.eye(n) - 1 / n; B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:k]
    return V[:, i] * np.sqrt(np.clip(w[i], 0, None))

@torch.inference_mode()
def main():
    rid = "sentence-transformers/all-MiniLM-L6-v2"
    st = SentenceTransformer(P.model_path(rid), revision=None, device="cpu")
    E = st.encode(UTTER + ITEMS + [STANCE_UTT] + list(STANCE_ITEMS.values()), normalize_embeddings=True)
    U, I = E[:4], E[4:8]
    X = cmds(np.sqrt(np.clip(2 - 2 * (XS.dot_activities(E[:8], E[:8])), 0, None)))
    out = dict(utterances=UTTER, items=ITEMS, cos_utt_item=(XS.dot_activities(U, I)).tolist(), cos_item_item=(XS.dot_activities(I, I)).tolist(), mds=X.tolist())
    s, f, r = E[8], E[9], E[10]
    nid, rev = P.NLI
    tok = AutoTokenizer.from_pretrained(nid, revision=rev)
    mod = AutoModelForSequenceClassification.from_pretrained(nid, revision=rev).eval()
    lab = {v.lower(): int(k) for k, v in mod.config.id2label.items()}
    enc = tok([STANCE_UTT] * 2, list(STANCE_ITEMS.values()), return_tensors="pt", padding=True)
    pr = torch.softmax(mod(**enc).logits, -1).numpy()
    nli = (pr[:, lab["entailment"]] - pr[:, lab["contradiction"]]).tolist()
    out["stance"] = dict(utterance=STANCE_UTT, items=STANCE_ITEMS, cosine=[float(s @ f), float(s @ r)], entailment=nli)
    (P.RESULTS / "method_examples.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["stance"], indent=1)); print(np.round(XS.dot_activities(U, I), 2))

if __name__ == "__main__":
    main()
