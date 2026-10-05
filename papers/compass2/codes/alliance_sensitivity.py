"""Alliance sensitivity."""
import json
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import alliance_tests as AT
import nli_wai as NW
import paths as P
from corpus_alliance import ROOT

RNG = np.random.default_rng(20261004)
N_BOOT = 5000

def source_id(url):
    u = str(url) if isinstance(url, str) else ""
    m = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", u)
    if m:
        return "yt:" + m.group(1)
    m = re.search(r"vimeo\.com/(?:video/)?(\d+)", u)
    if m:
        return "vimeo:" + m.group(1)
    return None

def grams(text, n=5):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

TEXT_THRESHOLD = 0.3

def provenance():
    """Per HLQC session: source-ID status (match, distinct, unresolved), transcript screen (text match at containment >=
    TEXT_THRESHOLD, otherwise screen-negative), and the combined decision. A screen-negative result is not proof of
    independence: it means no AnnoMI session contains much of the same word sequence."""
    url = pd.read_csv(ROOT / "urls.csv").set_index("id")["url"]
    ann = pd.read_csv(P.DATA / "annomi" / "AnnoMI-full.csv", usecols=["video_url"])
    ann_src = {s_ for s_ in (source_id(u) for u in ann["video_url"].unique()) if s_}
    H = pd.read_parquet(P.DERIVED / "windows_hlqc.parquet"); A = pd.read_parquet(P.DERIVED / "windows_annomi.parquet")
    ag = {pid: grams(" ".join(g["text"])) for pid, g in A.groupby("pid")}
    rows = []
    for pid, g in H.groupby("pid"):
        src = source_id(url.get(pid)); hg = grams(" ".join(g["text"]))
        cont = max((len(hg & a) / max(len(hg), 1) for a in ag.values()), default=0.0)
        id_status = "ID unresolved" if src is None else ("ID match" if src in ann_src else "ID distinct")
        text_status = "text match" if cont >= TEXT_THRESHOLD else "screen-negative"
        rows.append(dict(pid=pid, raw_url=str(url.get(pid)) if isinstance(url.get(pid), str) else "", source=src, quality=g["quality"].iloc[0],
                         registered_excluded=bool(g["in_annomi"].iloc[0]), id_status=id_status, containment=round(cont, 4), text_status=text_status,
                         eligible_screen_negative=bool(id_status != "ID match" and text_status == "screen-negative")))
    D = pd.DataFrame(rows)
    reg = D[~D["registered_excluded"]]
    summary = dict(threshold=TEXT_THRESHOLD,
                   counts=D.groupby(["registered_excluded", "id_status", "text_status"]).size().rename("n").reset_index().to_dict("records"),
                   registered_sample_containment_max_below=float(reg[reg["containment"] < TEXT_THRESHOLD]["containment"].max()),
                   registered_sample_containment_min_above=float(reg[reg["containment"] >= TEXT_THRESHOLD]["containment"].min()))
    return D, summary

def auc_ci_cluster(sess, clusters, n=N_BOOT):
    y = (sess["quality"] == "high").to_numpy(); st = sess["stance_total"].to_numpy(); co = sess["cos_total"].to_numpy()
    cl = np.asarray(clusters); uc = np.unique(cl); idx = {c: np.flatnonzero(cl == c) for c in uc}
    a1, d = [], []
    for _ in range(n):
        pick = np.concatenate([idx[c] for c in RNG.choice(uc, len(uc))])
        yy = y[pick]
        if yy.all() or (~yy).all():
            continue
        s1 = AT.auc(st[pick], yy); a1.append(s1); d.append(s1 - AT.auc(co[pick], yy))
    return dict(auc_stance=AT.auc(st, y), auc_cos=AT.auc(co, y), interval_stance=np.quantile(a1, [.025, .975]).tolist(),
                interval_difference=np.quantile(d, [.025, .975]).tolist(), sessions=int(len(y)), clusters=int(len(uc)))

def truncation_mask(corpus, tok):
    W = pd.read_parquet(P.DERIVED / f"windows_{corpus}.parquet"); S = NW.segments(W)
    items = pd.read_csv(P.ITEMS / "item_bank.csv"); hyps = items[items["instrument"] == "WAI-O"]["hypothesis"].tolist()
    hl = [len(tok(h)["input_ids"]) for h in hyps]
    pl = [len(tok(t)["input_ids"]) for t in S["text"]]
    return np.array([p + max(hl) - 1 > 256 for p in pl]), S

def stance_without(corpus, mask):
    items = pd.read_csv(P.ITEMS / "item_bank.csv"); o = items[items["instrument"] == "WAI-O"]
    pol = o["polarity"].to_numpy(float)
    S = pd.read_parquet(P.DERIVED / f"nliwai_{corpus}_segments.parquet"); X = np.load(P.DERIVED / f"nliwai_{corpus}.npy").astype(float)
    seg = pd.DataFrame({"pid": S["pid"], "st": (X * pol).mean(1)})[~mask]
    return seg.groupby("pid")["st"].mean()

def main():
    D, summary = provenance()
    out = {"provenance": summary}
    url = pd.read_csv(ROOT / "urls.csv").set_index("id")["url"]
    ann_url = pd.read_parquet(P.DERIVED / "windows_annomi.parquet").groupby("pid")["url"].first()
    cl = lambda pids: [source_id(url.get(p)) or f"session:{p}" for p in pids]
    hl = AT.session_scores("hlqc"); conf = hl[~hl["in_annomi"].astype(bool)].copy()
    disc = AT.session_scores("annomi"); disc_cl = [source_id(ann_url.get(p)) or f"session:{p}" for p in disc.index]
    Dp = D.set_index("pid")
    sets = {
        "registered_184": conf,
        "registered_screen_negative": conf.loc[[p for p in conf.index if Dp.loc[p, "text_status"] == "screen-negative"]],
        "all_screen_negative": hl.loc[[p for p in hl.index if Dp.loc[p, "eligible_screen_negative"]]],
    }
    out["discovery_cluster"] = auc_ci_cluster(disc, disc_cl)
    for name, ss in sets.items():
        out[name] = auc_ci_cluster(ss, cl(ss.index))
    tok = AutoTokenizer.from_pretrained(P.NLI[0], revision=P.NLI[1])
    mask, Sg = truncation_mask("hlqc", tok)
    in_conf = Sg["pid"].isin(conf.index).to_numpy()
    st = stance_without("hlqc", mask)
    s2 = conf.copy(); s2["stance_total"] = st.reindex(s2.index); s2 = s2.dropna(subset=["stance_total"])
    out["registered_184_untruncated"] = dict(segments=int(in_conf.sum()), segments_over_limit_share=float(mask[in_conf].mean()),
                                             **auc_ci_cluster(s2, cl(s2.index)))
    dmask, _ = truncation_mask("annomi", tok)
    out["discovery_segments_over_limit_share"] = float(dmask.mean())
    keys = ["registered_184", "registered_screen_negative", "all_screen_negative", "registered_184_untruncated"]
    out["range_stance_auc"] = [min(out[k]["auc_stance"] for k in keys), max(out[k]["auc_stance"] for k in keys)]
    out["range_difference_lower_bound"] = [min(out[k]["interval_difference"][0] for k in keys), max(out[k]["interval_difference"][0] for k in keys)]
    (P.RESULTS / "alliance_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["range_stance_auc", "range_difference_lower_bound"]}, indent=1))
    for k in keys:
        v = out[k]; print(k, v["sessions"], v["clusters"], round(v["auc_stance"], 3), [round(x, 3) for x in v["interval_stance"]], round(v["auc_cos"], 3), [round(x, 3) for x in v["interval_difference"]])

if __name__ == "__main__":
    main()
