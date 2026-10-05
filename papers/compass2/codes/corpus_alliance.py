"""Load counselling sessions, retaining the original identifier-based exclusion for reproduction; transcript-based sensitivity checks are in alliance_sensitivity.py."""
import json
import re

import pandas as pd

import paths as P
from corpus import clean

ROOT = P.DATA / "highlowquality_counseling" / "HighLowQualityCounseling"

def youtube_id(url):
    m = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", str(url))
    return m.group(1) if m else None

def hlqc():
    lab = pd.read_csv(ROOT / "labels.csv").set_index("id")["label"]
    url = pd.read_csv(ROOT / "urls.csv").set_index("id")["url"]
    ann = pd.read_csv(P.DATA / "annomi" / "AnnoMI-full.csv", usecols=["video_url"])
    ann_ids = {youtube_id(u) for u in ann["video_url"].unique()}
    rows = []
    for sid in sorted(lab.index):
        f = ROOT / "transcripts" / sid
        if not f.exists():
            continue
        turns = []
        for ln in f.read_text(errors="ignore").splitlines():
            m = re.match(r"^\s*([TC])\s*:\s*(.*)$", ln)
            if m:
                turns.append(("therapist" if m.group(1) == "T" else "client", clean(m.group(2))))
            elif turns and ln.strip():
                turns[-1] = (turns[-1][0], clean(turns[-1][1] + " " + ln))
        n = len(turns)
        yid = youtube_id(url.get(sid))
        for w, (role, text) in enumerate(turns):
            if text:
                rows.append(dict(corpus="hlqc", pid=sid, widx=w, role=role, group=None, text=text,
                                 phase=min(9, int(10 * w / max(n, 1))), quality=lab[sid], url=url.get(sid),
                                 youtube=yid, in_annomi=yid in ann_ids))
    return pd.DataFrame(rows)

def main():
    df = hlqc()
    df.to_parquet(P.DERIVED / "windows_hlqc.parquet", index=False)
    s = df.groupby("pid").agg(in_annomi=("in_annomi", "first"), turns=("widx", "size"))
    out = dict(sessions=int(len(s)), turns=int(len(df)), turns_per_session_median=float(s["turns"].median()),
               sessions_also_in_annomi=int(s["in_annomi"].sum()), sessions_independent_of_annomi=int((~s["in_annomi"]).sum()),
               roles={k: int(v) for k, v in df["role"].value_counts().items()})
    (P.RESULTS / "corpus_structure_hlqc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
