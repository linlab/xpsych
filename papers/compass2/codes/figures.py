"""Figures."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import geometry as G
import house_style as H
import paths as P
import registered_tests as RT
import figcheck
import overlapcheck

PROBLEMS = {}
ROW_SPAN_OK = {"fig1_hub", "fig2_method", "fig4_interviews_registered", "fig5_alliance", "fig6_probe_banks"}
_save = H.save

def _checked_save(fig, name, **kw):
    basic = [m for m in figcheck.check(fig) if not m.startswith("text-data")]
    if name in ROW_SPAN_OK:
        basic = [m for m in basic if not m.startswith("row-align")]
    PROBLEMS[name] = sorted(set(basic) | set(overlapcheck.strict(fig)))
    _save(fig, name, **kw)

H.save = _checked_save

H.setup()
RES = P.RESULTS
SYM = ["PHQ-8", "PCL-C"]
DOMCOL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#333333"]

def load(name):
    return json.loads((RES / name).read_text())

def items():
    return pd.read_csv(P.ITEMS / "item_bank.csv")

def short_labels(keys):
    return [("PHQ " if k.startswith("PHQ") else "PCL ") + k.split("_", 2)[-1].replace("Future_Cut_Short", "Future").replace("TroubleRemembering", "Remembering")
            .replace("ThoughtAvoidance", "AvoidThought").replace("ActivityAvoidance", "AvoidActivity") for k in keys]

def fig1():
    it = items(); s = it["instrument"].isin(SYM).to_numpy()
    E = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float32)[s]; M = E @ E.T
    sim = load("sim_wording_null.json")["part1_population_geometry"]["grid_mean_sd"]
    gs = [0.0, 0.1, 0.2, 0.4]
    fig = plt.figure(figsize=(H.W2, 58 * H.MM), layout="constrained")
    gsp = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    a = fig.add_subplot(gsp[0]); b = fig.add_subplot(gsp[1]); c = fig.add_subplot(gsp[2])
    im = a.imshow(M, cmap="Greys", vmin=0, vmax=1)
    a.axhline(7.5, color=H.ACCENT, lw=0.6); a.axvline(7.5, color=H.ACCENT, lw=0.6)
    a.set_xticks([3.5, 16]); a.set_xticklabels(["PHQ-8", "PCL-C"]); a.set_yticks([3.5, 16]); a.set_yticklabels(["PHQ-8", "PCL-C"], rotation=90, va="center")
    a.tick_params(length=0)
    for (i, j) in [(2, 20), (6, 22), (0, 16)]:
        a.plot(j, i, "o", ms=3.2, mfc="none", mec=H.ACCENT, mew=0.8)
    a.set_title("Item-text similarity (wording)")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, ticks=[0, 0.5, 1]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("cosine", fontsize=6)
    for name, col, mk in [("naive", H.MUTED, "o"), ("whitened_N0", H.ACCENT, "s")]:
        m = [sim[f"g={g},n=189,w=155"][name]["r_truth"][0] for g in gs]; e = [sim[f"g={g},n=189,w=155"][name]["r_truth"][1] for g in gs]
        b.errorbar(gs, m, yerr=e, color=col, marker=mk, ms=3.2, mfc="white", lw=1, capsize=1.5, label="naive" if name == "naive" else "whitened")
        m2 = [sim[f"g={g},n=189,w=155"][name]["r_truth_beyond_wording"][0] for g in gs]
        c.plot(gs, m2, color=col, marker=mk, ms=3.2, mfc="white", lw=1)
    rtw = load("sim_wording_null.json")["part1_population_geometry"]["r_truth_vs_wording"]
    b.axhline(rtw, color=H.FAINT, lw=0.7, ls="--"); b.text(0.41, rtw + 0.02, "wording ~ truth", ha="right", va="bottom", fontsize=5.5, color=H.MUTED)
    b.set_xlabel("symptom signal in speech, g"); b.set_ylabel("r with true symptom structure"); b.set_ylim(-0.1, 0.62)
    b.set_title("Apparent validity without signal"); b.legend(loc="lower right", fontsize=5.5)
    c.axhline(0, color=H.FAINT, lw=0.6)
    c.set_xlabel("symptom signal in speech, g"); c.set_ylabel("partial r beyond wording"); c.set_ylim(-0.05, 0.3)
    c.set_title("Beyond wording (simulation)")
    c.legend(handles=[Line2D([], [], color=H.MUTED, marker="o", mfc="white", ms=3.2, lw=1, label="naive"),
                      Line2D([], [], color=H.ACCENT, marker="s", mfc="white", ms=3.2, lw=1, label="whitened")], loc="upper left", fontsize=5.5)
    for ax in (b, c):
        ax.set_xticks(gs)
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c")])
    H.save(fig, "fig3_wording_null")

def eta2(P_, labels):
    df = pd.DataFrame(P_); g = df.groupby(np.asarray(labels)).transform("mean")
    return ((g - df.mean()) ** 2).sum().to_numpy() / ((df - df.mean()) ** 2).sum().to_numpy()

def fig2():
    it = items(); s = it["instrument"].isin(SYM).to_numpy()
    W = pd.read_parquet(P.DERIVED / "windows_daicwoz.parquet")
    E = np.load(P.DERIVED / "emb_minilm_daicwoz.npy").astype(np.float32)
    A = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float32)[s]; M = (A @ A.T).astype(float)
    Pw = (E @ A.T).astype(float); grp = W["group"].fillna("none").to_numpy(); per = W["pid"].to_numpy()
    eq, ep = eta2(Pw, grp), eta2(Pw, per)
    ids, naive = RT.profiles(Pw, grp, per, centre=False)
    lab = pd.read_csv(P.EDAIC_LABELS).set_index("Participant")[it.loc[s, "key"].tolist()]
    keep = np.isin(ids, lab.index); naive = naive[keep]
    Y = lab.loc[ids[keep]].to_numpy(float); Remp = RT.corr(Y); Gn = RT.corr(naive)
    disc, conf, gf = load("registered_discovery.json"), load("registered_confirmation.json"), load("general_factor.json")
    fig = plt.figure(figsize=(H.W2, 128 * H.MM), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.1, 1])
    a, b, c = sf[0].subplots(1, 3)
    d, e, f6 = sf[1].subplots(1, 3, width_ratios=[1.75, 1, 1])
    order = np.argsort(eq)
    a.scatter(eq[order], np.arange(25), s=7, color=H.INK, label="eliciting question", zorder=3)
    a.scatter(ep[order], np.arange(25), s=7, facecolor="white", edgecolor=H.MUTED, label="person", zorder=3)
    a.set_yticks(np.arange(25)); a.set_yticklabels(np.array(short_labels(it.loc[s, "key"].tolist()))[order], fontsize=5)
    a.set_xlabel("share of item-activity variance"); a.set_title("The script outweighs the person"); a.legend(loc="lower right", fontsize=5.5)
    iu = np.triu_indices(25, 1)
    b.scatter(M[iu], Gn[iu], s=3, color=H.INK, alpha=0.55, lw=0)
    b.set_xlabel("item-text similarity"); b.set_ylabel("language geometry (naive)")
    b.set_title(f"Language follows wording (r = {np.corrcoef(M[iu], Gn[iu])[0, 1]:.2f})")
    c.scatter(Remp[iu], Gn[iu], s=3, color=H.MUTED, alpha=0.55, lw=0)
    c.set_xlabel("self-reported item correlation"); c.set_ylabel("language geometry (naive)")
    c.set_title(f"...more than self-report (r = {np.corrcoef(Remp[iu], Gn[iu])[0, 1]:.2f})")

    rows = []
    for split, D in [("discovery", disc), ("confirmation", conf)]:
        for sc, nm in [("minilm", "MiniLM"), ("bge", "bge-large"), ("nli", "NLI")]:
            v = D[sc]
            if "H1" in v:
                rows.append(("H1  wording − self-report", nm, split, v["H1"]["difference"], v["H1"]["interval"], v["H1"]["supported"]))
            rows.append(("H2  beyond wording (partial r)", nm, split, v["H2"]["partial_r"], None, v["H2"].get("supported", v["H2"]["supported_uncorrected"])))
            rows.append(("H3  same − different item", nm, split, v["H3"]["difference"], v["H3"]["interval"], v["H3"]["supported"]))
    hyps = ["H1  wording − self-report", "H2  beyond wording (partial r)", "H3  same − different item"]
    y = 0; yt, yl = [], []
    for h in hyps:
        for nm in ["MiniLM", "bge-large", "NLI"]:
            drawn = False
            for split, off, mk in [("discovery", 0.18, "o"), ("confirmation", -0.18, "s")]:
                r = [q for q in rows if q[0] == h and q[1] == nm and q[2] == split]
                if not r:
                    continue
                _, _, _, est, ci, sup = r[0]; drawn = True
                col = H.ACCENT if sup else H.MUTED
                if ci:
                    d.plot(ci, [y + off] * 2, color=col, lw=0.9)
                d.plot(est, y + off, marker=mk, ms=3.4, color=col, mfc=col if sup else "white", mew=0.8)
            if drawn:
                yt.append(y); yl.append(f"{h.split()[0]}  {nm}"); y -= 1
        y -= 0.6
    d.axvline(0, color=H.FAINT, lw=0.7)
    d.set_yticks(yt); d.set_yticklabels(yl, fontsize=5.5); d.set_xlabel("estimate (95% interval; Bonferroni in confirmation)")
    d.set_title("Registered tests")
    d.legend(handles=[Line2D([], [], color=H.INK, marker="o", ls="", ms=3.4, label="discovery (DAIC-WOZ, 189)"),
                      Line2D([], [], color=H.INK, marker="s", ls="", ms=3.4, mfc="white", label="confirmation (E-DAIC, 86)"),
                      Line2D([], [], color=H.ACCENT, lw=1, label="supported"), Line2D([], [], color=H.MUTED, lw=1, label="not supported")],
             loc="center right", fontsize=5.0, ncol=1)

    xs = np.arange(3)
    for k, (split, mk, off) in enumerate([("discovery", "o", -0.27), ("confirmation", "s", 0.09)]):
        g = [gf[split][sc]["r_glang_gself"] for sc in ["minilm", "bge", "nli"]]
        gi = [gf[split][sc]["ci_r_glang_gself"] for sc in ["minilm", "bge", "nli"]]
        sp = [gf[split][sc]["specific_delta"] for sc in ["minilm", "bge", "nli"]]
        spi = [gf[split][sc]["ci_specific_delta"] for sc in ["minilm", "bge", "nli"]]
        for x, v, ci in zip(xs + off, g, gi):
            e.plot([x, x], ci, color=H.ACCENT, lw=0.9); e.plot(x, v, marker=mk, color=H.ACCENT, ms=3.4, mfc=H.ACCENT if k == 0 else "white")
        for x, v, ci in zip(xs + off + 0.16, sp, spi):
            e.plot([x, x], ci, color=H.MUTED, lw=0.9); e.plot(x, v, marker=mk, color=H.MUTED, ms=3.2, mfc=H.MUTED if k == 0 else "white")
    e.axhline(0, color=H.FAINT, lw=0.6)
    e.set_xticks(xs); e.set_xticklabels(["MiniLM", "bge", "NLI"]); e.set_ylabel("r, or same − different contrast")
    e.set_title("Distress, not symptoms")
    e.legend(handles=[Line2D([], [], color=H.ACCENT, marker="o", lw=0.9, ms=3.2, label="general factor: language vs self-report"),
                      Line2D([], [], color=H.MUTED, marker="o", lw=0.9, ms=3.2, label="item-specific (same − different)")], loc="upper left", fontsize=5.2)
    e.set_ylim(-0.1, 1.0); e.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

    ss = load("symptom_sensitivity.json")["shuffle"]
    stats = [("geometry_agreement", "geometry\nagreement"), ("general_factor_r", "general\nfactors"), ("phq8_score_r", "PHQ-8\nscore")]
    for k, (split, mk, off) in enumerate([("discovery", "o", -0.14), ("confirmation", "s", 0.14)]):
        v = ss[f"{split}:nli"]
        for j, (st, _) in enumerate(stats):
            lo, hi = v[st]["null_95"]
            f6.plot([j + off] * 2, [lo, hi], color=H.FAINT, lw=3.2, solid_capstyle="butt", zorder=1)
            f6.plot(j + off, v[st]["true"], marker=mk, ms=3.6, color=H.ACCENT, mfc=H.ACCENT if k == 0 else "white", zorder=3)
    f6.axhline(0, color=H.FAINT, lw=0.6)
    f6.text(0, 0.18, "identical after\nshuffling", ha="center", va="top", fontsize=5.0, color=H.MUTED)
    f6.set_xticks(range(3)); f6.set_xticklabels([l for _, l in stats], fontsize=5.5); f6.set_ylim(-0.3, 0.8)
    f6.set_ylabel("r with self-report"); f6.set_title("Shuffling who said what")
    f6.legend(handles=[Line2D([], [], color=H.ACCENT, marker="o", ls="", ms=3.4, label="true pairing"),
                       Line2D([], [], color=H.FAINT, lw=3.2, label="shuffled people (95%)")], loc="upper right", fontsize=5.0)
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e"), (f6, "f")], tol_col=0.01)
    H.save(fig, "fig4_interviews_registered")

def compass_sessions():
    W = pd.read_parquet(P.DERIVED / "windows_alexst.parquet")
    E = np.load(P.DERIVED / "emb_minilm_alexst.npy").astype(np.float32)
    it = items(); Ei = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float32)
    out = {}
    for role, form in {"client": "WAI-C-COMPASS2025", "therapist": "WAI-T"}.items():
        s = (it["instrument"] == form).to_numpy(); A = Ei[s]; sc = it.loc[s, "scale"].to_numpy(); pol = it.loc[s, "polarity"].to_numpy()
        m = (W["role"] == role).to_numpy(); Pw = E[m] @ A.T
        df = pd.DataFrame({"pid": W.loc[m, "pid"].to_numpy()})
        for scale in ["Task", "Bond", "Goal"]:
            j = sc == scale; df[f"keyed_{scale}"] = (Pw[:, j] * pol[j]).sum(1); df[f"topic_{scale}"] = Pw[:, j].mean(1)
        out[role] = df.groupby("pid").mean()
    return out

def alliance_aucs():
    """AUCs (high vs low quality) for stance, keyed cosine and the replica of our earlier scoring, per available split.
    Intervals: the registered ones where they exist (stance, replica); keyed cosine gets a descriptive stratified-bootstrap
    interval computed here with the registered functions and seed."""
    import alliance_tests as AT
    out = {}
    for split, corpus in [("discovery", "annomi"), ("confirmation", "hlqc")]:
        f = P.RESULTS / f"alliance_{split}.json"
        if not f.exists():
            continue
        a = json.loads(f.read_text())
        sess = AT.session_scores(corpus)
        if split == "confirmation":
            sess = sess[~sess["in_annomi"].astype(bool)]
        y = (sess["quality"] == "high").to_numpy(); co = sess["cos_total"].to_numpy()
        rng = np.random.default_rng(AT.SEED)
        bs = [AT.auc(co[i], y[i]) for i in (AT.strat_boot(y, rng) for _ in range(2000))]
        out[split] = {"stance": (a["A1"]["auc_stance"], a["A1"]["interval"]),
                      "keyed cosine": (a["A2"]["auc_keyed_cosine"], list(np.quantile(bs, [0.025, 0.975]))),
                      "our earlier method": (a["secondary"]["compass2025_total"]["auc"], a["secondary"]["compass2025_total"]["interval"])}
    return out

def fig3():
    wg = load("wording_geometry.json")["minilm"]; sim = load("sim_wording_null.json")["part3_reverse_keying"]
    S = compass_sessions(); cl, th = S["client"], S["therapist"]; both = cl.join(th, lsuffix="_c", rsuffix="_t", how="inner")
    aucs = alliance_aucs()
    fig = plt.figure(figsize=(H.W2, 104 * H.MM), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1, 1])
    a, b, c = sf[0].subplots(1, 3)
    d, e = sf[1].subplots(1, 2, width_ratios=[1, 1.6])
    xs = np.arange(3)
    ff = [wg["WAI-C"][s]["cos_forward_forward"] for s in ["Task", "Bond", "Goal"]]
    fr = [wg["WAI-C"][s]["cos_forward_reverse"] for s in ["Task", "Bond", "Goal"]]
    a.bar(xs - 0.18, ff, 0.34, color=H.MUTED, label="forward–forward"); a.bar(xs + 0.18, fr, 0.34, color=H.ACCENT, label="forward–reverse")
    a.set_xticks(xs); a.set_xticklabels(["Task", "Bond", "Goal"]); a.set_ylabel("mean item-text cosine (WAI-C)"); a.set_ylim(0, 0.95)
    a.set_title("Reverse ≈ forward items"); a.legend(loc="upper left", fontsize=5.3)
    b.scatter(cl["topic_Bond"], cl["keyed_Bond"], s=2.5, color=H.INK, alpha=0.5, lw=0)
    r2 = np.corrcoef(cl["topic_Bond"], cl["keyed_Bond"])[0, 1] ** 2
    b.text(0.04, 0.96, f"R² = {r2:.2f}\n{len(cl)} sessions", transform=b.transAxes, va="top", fontsize=5.5)
    b.set_xlabel("relationship-topic activation"); b.set_ylabel("keyed Bond score (client)")
    b.set_title("Bond ≈ relationship talk")
    gk = both["keyed_Bond_t"] - both["keyed_Bond_c"]; gt = both["topic_Bond_t"] - both["topic_Bond_c"]
    c.scatter(gt, gk, s=2.5, color=H.INK, alpha=0.5, lw=0)
    c.text(0.04, 0.96, f"r = {np.corrcoef(gt, gk)[0, 1]:.2f}\n{len(both)} sessions", transform=c.transAxes, va="top", fontsize=5.5)
    c.set_xlabel("therapist − client topic activation"); c.set_ylabel("therapist − client keyed Bond")
    c.set_title("Misalignment ≈ talk gap")
    kc = [sim[s]["r_keyed_cosine_vs_alliance"] for s in ["Task", "Bond", "Goal"]]; ks = [sim[s]["r_keyed_stance_vs_alliance"] for s in ["Task", "Bond", "Goal"]]
    d.bar(xs - 0.18, kc, 0.34, color=H.MUTED, label="keyed cosine"); d.bar(xs + 0.18, ks, 0.34, color=H.ACCENT, label="stance-aware")
    d.set_xticks(xs); d.set_xticklabels(["Task", "Bond", "Goal"]); d.set_ylabel("r with true alliance"); d.set_ylim(0, 1.18)
    d.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); d.set_title("Simulation: stance recovers alliance")
    d.legend(loc="upper center", ncol=2, fontsize=5, columnspacing=0.8, handlelength=1.2)
    cats = ["stance", "keyed cosine", "our earlier method"]
    e.axhline(0.5, color=H.FAINT, lw=0.7, ls="--"); e.text(2.45, 0.505, "chance", fontsize=5.2, color=H.MUTED, ha="right", va="bottom")
    for k, (split, mk, off, lab) in enumerate([("discovery", "o", -0.12, "discovery (AnnoMI)"), ("confirmation", "s", 0.12, "confirmation (second corpus)")]):
        if split not in aucs:
            continue
        for j, cname in enumerate(cats):
            v, ci = aucs[split][cname]
            col = H.ACCENT if cname == "stance" else H.MUTED
            e.plot([j + off] * 2, ci, color=col, lw=0.9)
            e.plot(j + off, v, marker=mk, ms=3.6, color=col, mfc=col if k == 0 else "white", label=lab if j == 0 else None)
    e.set_xticks(range(3)); e.set_xticklabels(cats); e.set_xlim(-0.5, 2.5); e.set_ylim(0.3, 1.0)
    e.set_ylabel("AUC, high- vs low-quality sessions"); e.set_title("Known groups: high- vs low-quality counselling")
    hs = [Line2D([], [], color=H.INK, marker="o", ls="", ms=3.4, label="discovery (AnnoMI, 133)")]
    if "confirmation" in aucs:
        hs.append(Line2D([], [], color=H.INK, marker="s", ls="", ms=3.4, mfc="white", label="confirmation (second corpus, 184)"))
    e.legend(handles=hs, loc="upper left", fontsize=5.2)
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e")])
    H.save(fig, "fig5_alliance")

def fig4():
    mb = load("multibank.json"); df = pd.read_csv(P.ITEMS / "multibank.csv")
    X = np.load(P.DERIVED / "mds_minilm_multibank.npy")
    doms = ["symptoms", "alliance", "interaction", "personality", "process", "framework"]
    fig = plt.figure(figsize=(H.W2, 92 * H.MM), layout="constrained")
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.25, 1])
    a = fig.add_subplot(gsp[0, 0]); b = fig.add_subplot(gsp[1, 0]); c = fig.add_subplot(gsp[:, 1])
    for k, dm in enumerate(doms):
        m = (df["domain"] == dm).to_numpy()
        a.scatter(X[m, 0], X[m, 1], s=5, color=DOMCOL[k], edgecolor="white", lw=0.25, alpha=0.9, label=f"{dm} ({m.sum()})")
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("MDS 1 (item-text distance)"); a.set_ylabel("MDS 2")
    a.set_title(f"{len(df)} items from {df['bank'].nunique()} probe banks")
    a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5, markerscale=1.6, handletextpad=0.2, frameon=False)
    pb = mb["bge"]["per_bank"]
    order = sorted(pb, key=lambda q: (pb[q]["n"], q))
    n = np.array([pb[q]["n"] for q in order]); raw = np.array([pb[q]["effective_rank"] for q in order])
    mr = np.array([pb[q]["effective_rank_mean_removed"] for q in order]); xs_b = np.arange(len(order))
    b.vlines(xs_b, raw, mr, color=H.FAINT, lw=0.6, zorder=1)
    b.scatter(xs_b, raw, s=7, color=H.ACCENT, label="raw (bge)", zorder=3)
    b.scatter(xs_b, mr, s=7, facecolor="white", edgecolor=H.MUTED, lw=0.6, label="shared direction removed", zorder=3)
    b.set_yscale("log"); b.set_yticks([1, 2, 5, 10, 20]); b.set_yticklabels(["1", "2", "5", "10", "20"]); b.minorticks_off(); b.set_ylim(1, 26)
    marks = [int(np.argmax(n >= v)) for v in (3, 10, 20, 36, 50)]
    b.set_xticks(marks); b.set_xticklabels([str(n[i]) for i in marks]); b.set_xlim(-1, len(order))
    b.set_xlabel("probe banks ordered by number of items"); b.set_ylabel("effective dimensions")
    b.set_title("Embeddings compress every bank"); b.legend(loc="upper left", fontsize=5.2)
    dom_of = df.groupby("bank")["domain"].first(); size = df["bank"].value_counts()
    ord_ = sorted(pb, key=lambda q: (doms.index(dom_of[q]) if dom_of[q] in doms else 9, -size[q]))
    v = [mb["minilm"]["nn_in_other_bank"][o] for o in ord_]; w = [mb["minilm"]["nn_in_other_domain"][o] for o in ord_]
    yy = np.arange(len(ord_))
    c.barh(yy - 0.2, v, color=H.MUTED, height=0.38, label="another instrument")
    c.barh(yy + 0.2, w, color=H.ACCENT, height=0.38, label="another domain")
    c.set_yticks(yy); c.set_yticklabels(ord_, fontsize=5.0); c.invert_yaxis()
    for t, o in zip(c.get_yticklabels(), ord_):
        t.set_color(DOMCOL[doms.index(dom_of[o])] if dom_of[o] in doms else H.INK)
    c.set_xlabel("share of items whose nearest\nneighbour lies in ..."); c.set_xlim(0, 1.75); c.set_xticks([0, 0.5, 1]); c.set_title("Overlap stays within domains")
    c.legend(loc="lower right", fontsize=5.2)
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c")])
    H.save(fig, "fig6_probe_banks")

def ed1():
    im = load("imposed_structure_minilm.json")
    fig = plt.figure(figsize=(H.W2, 50 * H.MM), layout="constrained")
    gsp = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1]); a, b, c = [fig.add_subplot(gsp[i]) for i in range(3)]
    cats = [("window_raw", "windows"), ("person_raw", "people,\nnaive"), ("person_centred", "people,\ncentred"), ("person_centred_whitened", "people,\nwhitened")]
    xs = np.arange(len(cats))
    for k, (corp, lab, col) in enumerate([("daicwoz", "DAIC-WOZ", H.INK), ("edaic_new", "E-DAIC", H.ACCENT)]):
        a.bar(xs + (k - 0.5) * 0.36, [im[corp]["r_wording"][q] for q, _ in cats], 0.34, color=col, label=lab)
        rel = im[corp]["reliability_population"]["centred"]
        b.bar(np.arange(2) + (k - 0.5) * 0.36, [rel["raw"]["spearman_brown"], rel["whitened"]["spearman_brown"]], 0.34, color=col)
        wp = im[corp]["reliability_within_person"]
        c.bar(np.arange(2) + (k - 0.5) * 0.36, [wp["median_split_half_raw"], wp["median_split_half_deviation"]], 0.34, color=col)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in cats], fontsize=5.5); a.set_ylabel("r with item-text similarity"); a.set_ylim(0, 1)
    a.set_title("Wording in the geometry"); a.legend(loc="upper right", fontsize=5.3)
    b.set_xticks([0, 1]); b.set_xticklabels(["raw", "whitened"]); b.set_ylabel("split-half reliability"); b.set_ylim(0, 1); b.set_title("Population geometry")
    c.set_xticks([0, 1]); c.set_xticklabels(["own geometry\n(raw)", "deviation\n(whitened)"]); c.set_ylabel("median split-half reliability"); c.set_ylim(0, 1); c.set_title("Within a person")
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c")])
    H.save(fig, "ed1_script_reliability")

def ed2():
    rd = load("rdoc_wording.json")
    df = pd.read_csv(P.ITEMS / "multibank.csv"); r = df[df["bank"] == "RDoC"]
    E = np.load(P.DERIVED / "emb_minilm_multibank.npy").astype(np.float32)[r.index.to_numpy()]
    D = np.sqrt(np.clip(2 - 2 * (E @ E.T), 0, None)); n = len(D); Jm = np.eye(n) - 1 / n; B = -0.5 * Jm @ (D ** 2) @ Jm
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:2]; X = V[:, i] * np.sqrt(np.clip(w[i], 0, None))
    dom = r["key"].str.split("::").str[1].map(lambda c: None)
    src = pd.read_csv(P.ITEMS / "rdoc" / "rdoc_constructs.csv")
    src = src[(src["level"] != "domain") & (src["definition"].fillna("").str.len() > 0)].drop_duplicates("definition")
    dmap = dict(zip(src["definition"], src["domain"])); dom = r["text"].map(dmap).fillna("other").to_numpy()
    fig = plt.figure(figsize=(H.W2, 62 * H.MM), layout="constrained")
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.6, 1]); a, b = fig.add_subplot(gsp[0]), fig.add_subplot(gsp[1])
    for k, dm in enumerate(sorted(set(dom))):
        m = dom == dm; a.scatter(X[m, 0], X[m, 1], s=11, color=H.CAT[k % len(H.CAT)], edgecolor="white", lw=0.4, label=dm)
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("MDS 1"); a.set_ylabel("MDS 2"); a.set_title(f"{n} RDoC construct definitions")
    a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.2, frameon=False)
    vals = [rd["minilm"]["nearest_neighbour_same_domain"], rd["bge"]["nearest_neighbour_same_domain"]]
    b.bar([0, 1], vals, 0.5, color=H.ACCENT); b.axhline(rd["minilm"]["nearest_neighbour_chance"], color=H.MUTED, lw=0.8, ls="--")
    b.text(0.5, rd["minilm"]["nearest_neighbour_chance"] + 0.02, "chance", fontsize=5.5, color=H.MUTED, ha="center")
    b.set_xticks([0, 1]); b.set_xticklabels(["MiniLM", "bge"]); b.set_ylim(0, 1); b.set_ylabel("nearest neighbour in the same domain")
    b.set_title("Definitions cluster by domain")
    H.label_panels(fig, [(a, "a"), (b, "b")])
    H.save(fig, "ed2_rdoc")

def ed3():
    se = load("sim_estimators.json"); sim = load("sim_wording_null.json")
    fig = plt.figure(figsize=(H.W2, 58 * H.MM), layout="constrained")
    gsp = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.35]); a, b, c = [fig.add_subplot(gsp[i]) for i in range(3)]
    ests = [("naive", "naive"), ("whitened", "whitened"), ("inverse_admissible", "inverse,\nadmissible fits"), ("inverse_psd", "inverse,\nPSD-projected")]
    for k, (gk, mk) in enumerate([("n=189,g=0.2", "o"), ("n=189,g=0.4", "s")]):
        A_ = se["A_estimators"][gk]
        for j, (e_, lab) in enumerate(ests):
            m, sd = A_[e_]["r_truth"]
            if m is None:
                continue
            a.errorbar(j + (k - 0.5) * 0.28, m, yerr=sd, fmt=mk, color=H.ACCENT if e_ == "whitened" else H.MUTED, ms=3.5,
                       mfc="white" if k else None, capsize=2, lw=1, label=("g = 0.2" if j == 0 and k == 0 else ("g = 0.4" if j == 0 else None)))
    na = [se["A_estimators"][g_]["inverse_admissible_n"] for g_ in ["n=189,g=0.2", "n=189,g=0.4"]]
    labs = ["naive", "whitened", f"inverse,\nadmissible\n({na[0]}, {na[1]}/200)", "inverse,\nPSD"]
    a.axhline(0, color=H.FAINT, lw=0.6); a.set_xticks(range(4)); a.set_xticklabels(labs, fontsize=5.0)
    a.set_ylim(-0.25, 0.62); a.set_ylabel("r with true structure"); a.set_title("Exact inversion often fails"); a.legend(loc="upper right", fontsize=5.2)
    ps = sim["part2_interview_script"]
    for k, (gg, lab) in enumerate([("g=0.0", "g = 0"), ("g=0.3", "g = 0.3")]):
        for j, (mode, col) in enumerate([("raw", H.MUTED), ("centred", H.ACCENT)]):
            m, sd = ps[gg][mode]
            b.errorbar(k + (j - 0.5) * 0.3, m, yerr=sd, fmt="o", color=col, ms=3.5, capsize=2, lw=1, label=mode if k == 0 else None)
    b.axhline(0, color=H.FAINT, lw=0.6); b.set_xticks([0, 1]); b.set_xticklabels(["g = 0", "g = 0.3"]); b.set_xlim(-0.6, 1.6)
    b.set_ylabel("partial r with truth beyond wording"); b.set_title("Centring costs little"); b.legend(loc="upper left", fontsize=5.3)
    B_ = se["B_anisotropic_nuisance"]
    names = [("matched nuisance", "matched\nnuisance"), ("between-person nuisance stronger", "stronger\nbetween"),
             ("nuisance in different directions", "different\ndirections"), ("genuine within-person dynamics", "within-person\ndynamics")]
    xs = np.arange(len(names))
    for j, (ref, lab, col) in enumerate([("wording_reference", "wording reference", H.MUTED), ("within_person_reference", "within-speaker reference", H.ACCENT)]):
        c.bar(xs + (j - 0.5) * 0.36, [B_[nm]["g=0.0"][ref]["rejection_rate"] for nm, _ in names], 0.34, color=col, label=lab)
    c.axhline(0.05, color=H.INK, lw=0.6, ls="--"); c.set_xticks(xs); c.set_xticklabels([l for _, l in names], fontsize=5.0)
    c.set_ylabel("false positives (P < 0.05, no person signal)"); c.set_ylim(0, 1.18); c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.set_title("Calibration depends on the speech"); c.legend(loc="upper left", fontsize=5.0)
    H.label_panels(fig, [(a, "a"), (b, "b"), (c, "c")])
    H.save(fig, "ed3_simulation")

if __name__ == "__main__":
    import fig_hub, fig_method
    fig_hub.main(); fig_method.main(); fig1(); fig2(); fig3(); fig4(); ed1(); ed2(); ed3()
    print("figures written; letter problems:", H.LETTER_PROBLEMS)
    for k, v in PROBLEMS.items():
        print(k, len(v), "layout problems", *v[:12], sep="\n  ")
