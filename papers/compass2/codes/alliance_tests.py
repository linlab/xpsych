"""Pre-specified alliance tests. Run with discovery or confirmation."""
import json
import sys

import numpy as np
import pandas as pd

from xpsych import scoring as XS

import paths as P

N_BOOT, N_PERM, SEED = 5000, 10000, 20261003
SCALES = ["Task", "Bond", "Goal"]

def auc(s, y):
    """Probability that a random high-quality session outscores a random low-quality one (ties count 1/2)."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def strat_boot(y, rng):
    hi, lo = np.flatnonzero(y), np.flatnonzero(~y)
    return np.concatenate([rng.choice(hi, len(hi)), rng.choice(lo, len(lo))])

def perm_p(s, y, rng):
    obs = auc(s, y) - 0.5
    null = np.array([auc(s, rng.permutation(y)) - 0.5 for _ in range(N_PERM)])
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + N_PERM))

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def keyed(X, pol, scale, cols=None):
    """Session-free helper: rows x items scores -> dict of keyed means (total and per subscale)."""
    out = {"total": XS.apply_key(X, pol, reduction="mean")}
    for sc in SCALES:
        j = scale == sc
        out[sc] = XS.apply_key(X[:, j], pol[j], reduction="mean")
    return out

def session_scores(corpus):
    items = pd.read_csv(P.ITEMS / "item_bank.csv")
    o = items[items["instrument"] == "WAI-O"]
    pol, scale = o["polarity"].to_numpy(float), o["scale"].to_numpy()
    W = pd.read_parquet(P.DERIVED / f"windows_{corpus}.parquet")
    E = np.load(P.DERIVED / f"emb_minilm_{corpus}.npy").astype(np.float32)
    Ei = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float32)
    m = W["role"].isin(["therapist", "client"]).to_numpy()
    W, E = W[m].reset_index(drop=True), E[m]
    C = XS.dot_activities(E, Ei[o.index.to_numpy()]).astype(float)
    cos = keyed(C, pol, scale)
    turn = pd.DataFrame({"pid": W["pid"], **{f"cos_{k}": v for k, v in cos.items()}, "topic": C.mean(1)})

    rep = np.full(len(W), np.nan)
    for role, form in {"client": "WAI-C-COMPASS2025", "therapist": "WAI-T"}.items():
        f = items[items["instrument"] == form]; r = (W["role"] == role).to_numpy()
        rep[r] = XS.apply_key(XS.dot_activities(E[r], Ei[f.index.to_numpy()]), f["polarity"].to_numpy(float), reduction="mean")
    turn["compass_role"] = W["role"].to_numpy(); turn["compass"] = rep
    sess = turn.groupby("pid").agg(**{c: (c, "mean") for c in turn.columns if c.startswith("cos_") or c == "topic"},
                                   n_turns=("topic", "size"))
    sess["compass2025"] = turn.groupby(["pid", "compass_role"])["compass"].mean().unstack().mean(1)
    S = pd.read_parquet(P.DERIVED / f"nliwai_{corpus}_segments.parquet")
    X = np.load(P.DERIVED / f"nliwai_{corpus}.npy").astype(float)
    assert len(S) == len(X), "segments and scores differ in length"
    st = keyed(X, pol, scale)
    seg = pd.DataFrame({"pid": S["pid"], **{f"stance_{k}": v for k, v in st.items()}})
    sess = sess.join(seg.groupby("pid").mean(), how="inner")
    q = W.groupby("pid").first()
    sess["quality"] = q.loc[sess.index, "quality"]
    if "in_annomi" in q:
        sess["in_annomi"] = q.loc[sess.index, "in_annomi"]
    return sess

def analyse(sess, rng):
    y = (sess["quality"] == "high").to_numpy()
    st, co = sess["stance_total"].to_numpy(), sess["cos_total"].to_numpy()
    res = dict(sessions=int(len(y)), high=int(y.sum()), low=int((~y).sum()))
    a_st, a_co = auc(st, y), auc(co, y)
    boots = [strat_boot(y, rng) for _ in range(N_BOOT)]
    b_st = np.array([auc(st[i], y[i]) for i in boots]); b_co = np.array([auc(co[i], y[i]) for i in boots])
    lo, hi = np.quantile(b_st, [0.025, 0.975])
    res["A1"] = dict(auc_stance=a_st, interval=[float(lo), float(hi)], supported=bool(lo > 0.5))
    d = b_st - b_co; lo2, hi2 = np.quantile(d, [0.025, 0.975])
    res["A2"] = dict(auc_stance=a_st, auc_keyed_cosine=a_co, difference=a_st - a_co, interval=[float(lo2), float(hi2)],
                     tested=res["A1"]["supported"], supported=bool(res["A1"]["supported"] and lo2 > 0))
    sec = {}
    for score in ["stance", "cos"]:
        ps = []
        for sc in SCALES:
            s = sess[f"{score}_{sc}"].to_numpy(); b = np.array([auc(s[i], y[i]) for i in boots[:2000]])
            sec[f"{score}_{sc}"] = dict(auc=auc(s, y), interval=np.quantile(b, [0.025, 0.975]).tolist(), p=perm_p(s, y, rng))
            ps.append(sec[f"{score}_{sc}"]["p"])
        for sc, a in zip(SCALES, holm(ps)):
            sec[f"{score}_{sc}"]["p_holm"] = a
    c = sess["compass2025"].to_numpy(); b = np.array([auc(c[i], y[i]) for i in boots[:2000]])
    sec["compass2025_total"] = dict(auc=auc(c, y), interval=np.quantile(b, [0.025, 0.975]).tolist())
    ln = np.log(sess["n_turns"].to_numpy(float))
    sec["correlations"] = {f"{a}~{b}": float(np.corrcoef(sess[a], sess[b] if b != "log_turns" else ln)[0, 1])
                           for a in ["stance_total", "cos_total", "compass2025"] for b in ["topic", "log_turns"]}
    sec["correlations"]["stance_total~cos_total"] = float(np.corrcoef(st, co)[0, 1])
    res["secondary"] = sec
    return res

def run(split):
    rng = np.random.default_rng(SEED)
    if split == "discovery":
        sess = session_scores("annomi")
    else:
        sess = session_scores("hlqc")
        sess = sess[~sess["in_annomi"].astype(bool)]
    out = analyse(sess, rng)
    (P.RESULTS / f"alliance_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

def selftest():
    """Synthetic sessions: stance tracks quality, keyed cosine tracks talk volume only. No real label is read."""
    rng = np.random.default_rng(1)
    n = 180; y = rng.random(n) < 0.6
    sess = pd.DataFrame({"quality": np.where(y, "high", "low"), "n_turns": rng.integers(20, 200, n)})
    vol = rng.normal(size=n); sess["topic"] = vol
    for sc in ["total"] + SCALES:
        sess[f"stance_{sc}"] = 0.8 * y + rng.normal(size=n)
        sess[f"cos_{sc}"] = vol + 0.1 * rng.normal(size=n)
    sess["compass2025"] = vol
    r = analyse(sess, np.random.default_rng(2))
    assert 0.65 < r["A1"]["auc_stance"] < 0.85 and r["A1"]["supported"], r["A1"]
    assert abs(r["A2"]["auc_keyed_cosine"] - 0.5) < 0.12 and r["A2"]["supported"], r["A2"]
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1) < 1e-12 and abs(auc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-12

    fails = 0
    for k in range(40):
        g = np.random.default_rng(100 + k); yy = g.random(n) < 0.6
        s0 = pd.DataFrame({"quality": np.where(yy, "high", "low"), "n_turns": g.integers(20, 200, n), "topic": g.normal(size=n),
                           "compass2025": g.normal(size=n), **{f"{a}_{b}": g.normal(size=n) for a in ["stance", "cos"] for b in ["total"] + SCALES}})
        global N_BOOT, N_PERM
        nb, npm = N_BOOT, N_PERM; N_BOOT, N_PERM = 400, 200
        fails += analyse(s0, g)["A1"]["supported"]
        N_BOOT, N_PERM = nb, npm
    print("selftest passed; null A1 false-positive count", fails, "of 40 (expected ~1)")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    selftest() if arg == "--selftest" else run(arg)
