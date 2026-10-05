"""Corpus."""
import csv
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

import paths as P

GAP = 3.0
INTERROG = {"how", "what", "why", "when", "where", "who", "whom", "which", "do", "does", "did", "are", "is", "was",
            "were", "have", "has", "had", "can", "could", "would", "will", "should", "tell", "describe", "explain"}
CLIENT = {"PATIENT", "CLIENT", "PT", "MALE CLIENT", "FEMALE CLIENT", "PARTICIPANT", "RESPONDENT"}
THERAPIST = re.compile(r"THERAPIST|COUNSELOR|COUNSELLOR|ANALYST|INTERVIEWER|DOCTOR|^DR\b|PSYCHIATRIST|PSYCHOLOGIST|^TH$|^T$")
SPEAKER = re.compile(r"^\s*([A-Z][A-Z .'\-]{0,24}?)\s*:\s*(.*)$")
NOISE = re.compile(r"<[^>]*>|\[[^\]]*\]")

def clean(s):
    return re.sub(r"\s+", " ", NOISE.sub(" ", s or "")).strip()

def is_prompt(line):
    w = line.lower().split()
    return bool(w) and (w[0] in INTERROG or "tell me" in line.lower())

def qkey(line):
    return re.sub(r"[^a-z ]", "", line.lower()).strip()

def daicwoz():
    rows = []
    for f in sorted(glob.glob(str(P.DAICWOZ / "data" / "*_P" / "*_TRANSCRIPT.csv"))):
        pid = int(Path(f).name.split("_")[0])
        lines = list(csv.DictReader(open(f, newline="", errors="replace"), delimiter="\t"))
        t_end = max((float(r["stop_time"]) for r in lines if r.get("stop_time")), default=1.0)
        q, buf, w = None, [], 0

        def flush():
            nonlocal buf, w
            text = clean(" ".join(r["value"] for r in buf))
            if text:
                rows.append(dict(corpus="daicwoz", pid=pid, widx=w, role="participant", group=q, text=text,
                                 start=float(buf[0]["start_time"]), stop=float(buf[-1]["stop_time"]),
                                 phase=min(9, int(10 * float(buf[0]["start_time"]) / t_end))))
                w += 1
            buf = []

        for r in lines:
            who = (r.get("speaker") or "").strip().lower()
            if who == "ellie":
                flush()
                v = clean(r.get("value"))
                if is_prompt(v):
                    q = qkey(v)
            elif who == "participant":
                buf.append(r)
        flush()
    return pd.DataFrame(rows)

def edaic_new():
    rows = []
    for pid in range(600, 719):
        fs = glob.glob(str(P.EDAIC / "data" / f"{pid}_P" / f"{pid}_Transcript.csv"))
        if not fs:
            continue
        seg = list(csv.DictReader(open(fs[0], newline="", errors="replace")))
        if not seg:
            continue
        t_end = max(float(s["End_Time"]) for s in seg)
        groups, cur = [], [seg[0]]
        for a, b in zip(seg, seg[1:]):
            if float(b["Start_Time"]) - float(a["End_Time"]) >= GAP:
                groups.append(cur); cur = []
            cur.append(b)
        groups.append(cur)
        for w, g in enumerate(groups):
            text = clean(" ".join(s["Text"] for s in g))
            if text:
                ph = min(9, int(10 * float(g[0]["Start_Time"]) / t_end))
                rows.append(dict(corpus="edaic_new", pid=pid, widx=w, role="participant", group=f"phase{ph}", text=text,
                                 start=float(g[0]["Start_Time"]), stop=float(g[-1]["End_Time"]), phase=ph))
    return pd.DataFrame(rows)

def alexst():
    files = {}
    for d in sorted(glob.glob(str(P.ALEXST / "*-transcripts"))):
        tag = Path(d).name.replace("-transcripts", "")
        for f in sorted(os.listdir(d)):
            s = open(os.path.join(d, f), errors="ignore").read()
            k = hashlib.sha1(re.sub(r"\s+", " ", s).encode()).hexdigest()[:12]
            files.setdefault(k, dict(text=s, tags=set(), name=f"{tag}/{f}"))["tags"].add(tag)
    rows = []
    for k, r in sorted(files.items()):
        turns, who, buf = [], None, []
        for ln in r["text"].splitlines():
            m = SPEAKER.match(ln)
            if m and m.group(1).strip() != "BEGIN TRANSCRIPT":
                if who is not None and buf:
                    turns.append((who, " ".join(buf)))
                who, buf = m.group(1).strip(), [m.group(2)]
            elif who is not None and ln.strip():
                buf.append(ln.strip())
        if who is not None and buf:
            turns.append((who, " ".join(buf)))
        n = len(turns)
        for w, (tag, text) in enumerate(turns):
            role = "client" if tag in CLIENT else ("therapist" if THERAPIST.search(tag) else "other")
            text = clean(text)
            if text and role != "other":
                rows.append(dict(corpus="alexst", pid=k, widx=w, role=role, group=None, text=text,
                                 phase=min(9, int(10 * w / max(n, 1))), tags=";".join(sorted(r["tags"]))))
    return pd.DataFrame(rows)

def annomi():
    a = pd.read_csv(P.DATA / "annomi" / "AnnoMI-full.csv")
    a = a.drop_duplicates(["transcript_id", "utterance_id"]).sort_values(["transcript_id", "utterance_id"])
    n = a.groupby("transcript_id")["utterance_id"].transform("count")
    out = pd.DataFrame(dict(corpus="annomi", pid=a["transcript_id"].astype(str), widx=a["utterance_id"],
                            role=a["interlocutor"].map({"client": "client", "therapist": "therapist"}),
                            group=None, text=a["utterance_text"].map(clean),
                            phase=(10 * a["utterance_id"] / n).clip(upper=9).astype(int),
                            quality=a["mi_quality"], url=a["video_url"]))
    return out[out["text"].str.len() > 0]

def write(df, name):
    P.DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(P.DERIVED / f"windows_{name}.parquet", index=False)

def describe(df, name):
    """Structure only: counts and lengths, never text."""
    words = df["text"].str.split().str.len()
    by = df.groupby("pid").size()
    out = dict(corpus=name, windows=int(len(df)), units=int(df["pid"].nunique()),
               windows_per_unit_median=float(by.median()), words_per_window_median=float(words.median()),
               roles={k: int(v) for k, v in df["role"].value_counts().items()})
    if df["group"].notna().any():
        g = df["group"].value_counts()
        out.update(groups=int(g.size), windows_with_group=float(df["group"].notna().mean()),
                   share_in_groups_seen_in_20plus_units=float(df[df["group"].isin(
                       df.groupby("group")["pid"].nunique().loc[lambda s: s >= 20].index)].shape[0] / len(df)))
    return out

def main():
    stats = []
    for name, fn in [("daicwoz", daicwoz), ("edaic_new", edaic_new), ("alexst", alexst), ("annomi", annomi)]:
        df = fn(); write(df, name); stats.append(describe(df, name))
        print(json.dumps(stats[-1]))
    (P.RESULTS / "corpus_structure.json").write_text(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
