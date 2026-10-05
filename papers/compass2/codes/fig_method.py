"""Fig method."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

import geometry as G
import house_style as H
import paths as P
import registered_tests as RT

DIV = LinearSegmentedColormap.from_list("div", list(H.DIVERGING))
MONO = "Courier New"
ITEMCOL = [H.CAT[0], H.CAT[1], H.CAT[2], H.CAT[3]]

def card(ax, x0, y0, w, h, fc="#f6f6f6"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.008,rounding_size=0.025", fc=fc, ec=H.FAINT, lw=0.5,
                                transform=ax.transAxes, clip_on=False, gid="card", zorder=0))

def arrow(fig, xy0, xy1, color=H.MUTED):
    fig.add_artist(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color))

def mat(ax, M, cmap="Greys", vmin=0, vmax=1, ticks=None, tickcols=None):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color(H.FAINT)
    if ticks is not None:
        ax.set_yticks(range(len(ticks))); ax.set_yticklabels(ticks, fontsize=5.0)
        ax.tick_params(length=0, pad=1.5)
        for t, c in zip(ax.get_yticklabels(), tickcols):
            t.set_color(c)

def main():
    ex = json.loads((P.RESULTS / "method_examples.json").read_text())
    it = pd.read_csv(P.ITEMS / "item_bank.csv"); s = it["instrument"].isin(["PHQ-8", "PCL-C"]).to_numpy()
    A = np.load(P.DERIVED / "emb_minilm_items.npy").astype(np.float64)[s]
    A8 = A[:8]; M8 = A8 @ A8.T
    short8 = ["interest", "mood", "sleep", "energy", "appetite", "failure", "concentration", "psychomotor"]
    col8 = [ITEMCOL[0], ITEMCOL[1], ITEMCOL[2], ITEMCOL[3]] + [H.MUTED] * 4

    rng = np.random.default_rng(3)
    T, q_of = 36, np.repeat([2, 1, 3], 12)
    Z = rng.standard_normal((T, A.shape[1])) / np.sqrt(A.shape[1])
    E = Z + 0.30 * A8[q_of].reshape(T, -1)
    Pw = E @ A8.T
    Pz = (Pw - Pw.mean(0)) / Pw.std(0)
    Pc = Pw - np.array([Pw[q_of == q].mean(0) for q in q_of])
    Gl = np.corrcoef(Pc.T)
    r_ge = np.corrcoef(G.upper(Gl), G.upper(M8))[0, 1]

    name, W, lab, items = RT.load_corpus("discovery")
    Ew = np.load(P.DERIVED / "emb_minilm_daicwoz.npy").astype(np.float32)
    Pd = (Ew @ A.T.astype(np.float32)).astype(float)
    ids, prof = RT.profiles(Pd, W["group"].fillna("none").to_numpy(), W["pid"].to_numpy(), centre=True)
    keep = np.isin(ids, lab.index); Y = lab.loc[ids[keep]].to_numpy(float); prof = prof[keep]
    M = A @ A.T
    Wh = G.whiten(np.cov(prof.T), M); np.fill_diagonal(Wh, np.nan)
    Rs = RT.corr(Y); np.fill_diagonal(Rs, np.nan)
    Mo = M.copy(); np.fill_diagonal(Mo, np.nan)
    h2 = json.loads((P.RESULTS / "registered_discovery.json").read_text())["minilm"]["H2"]
    Pn = np.load(P.DERIVED / "nli_daicwoz.npy").astype(float)
    idn, profn = RT.profiles(Pn, W["group"].fillna("none").to_numpy(), W["pid"].to_numpy(), centre=True)
    kn = np.isin(idn, lab.index); profn = profn[kn]; Yn = lab.loc[idn[kn]].to_numpy(float)
    zs = (profn - profn.mean(0)) / profn.std(0)
    score_lang = zs[:, :8].mean(1); score_self = Yn[:, :8].sum(1)
    r_score = float(np.corrcoef(score_lang, score_self)[0, 1])

    fig = plt.figure(figsize=(H.W2, 104 * H.MM))

    TOP, BOT = 0.895, 0.545
    a = fig.add_axes([0.0, BOT, 0.25, TOP - BOT]); a.axis("off"); a.set_xticks([]); a.set_yticks([])
    card(a, 0.0, 0.56, 0.98, 0.44)
    lines = [("T", "How have you been sleeping?", None), ("P", "I lie awake most nights and", 0), ("", "can't get back to sleep.", None),
             ("T", "And during the day?", None), ("P", "I'm exhausted all the time,", 1), ("", "even after resting.", None)]
    for k, (who, txt, dot) in enumerate(lines):
        y = 0.948 - k * 0.067
        a.text(0.05, y, (who + ":" if who else "") , fontsize=5.0, family=MONO, color=H.MUTED, va="center", transform=a.transAxes)
        a.text(0.13, y, txt, fontsize=5.0, family=MONO, va="center", transform=a.transAxes)
        if dot is not None:
            a.plot(0.025, y, "o", ms=2.6, color=H.INK, transform=a.transAxes, clip_on=False)
    a.text(0.0, 1.015, "patient narrative (transcript)", fontsize=5.6, color=H.MUTED, transform=a.transAxes, va="bottom")
    card(a, 0.0, 0.0, 0.98, 0.44)
    a.text(0.0, 0.455, "construct probes (questionnaire items)", fontsize=5.6, color=H.MUTED, transform=a.transAxes, va="bottom")
    for k, txt in enumerate(ex["items"]):
        y = 0.365 - k * 0.092
        a.plot(0.04, y, "o", ms=3.0, color=ITEMCOL[k], transform=a.transAxes, clip_on=False)
        a.text(0.08, y, txt if len(txt) < 44 else txt[:41] + "…", fontsize=5.0, va="center", transform=a.transAxes)

    b = fig.add_axes([0.27, BOT, 0.165, TOP - BOT]); b.axis("off"); b.set_xticks([]); b.set_yticks([])
    X = np.array(ex["mds"]); X = (X - X.min(0)) / (X.max(0) - X.min(0)); X = 0.12 + 0.76 * X
    card(b, 0.0, 0.0, 1.0, 1.0, fc="#efefef")
    C = np.array(ex["cos_utt_item"])
    for i in range(4):
        for j in range(4):
            if C[i, j] > 0.3:
                b.plot([X[i, 0], X[4 + j, 0]], [X[i, 1], X[4 + j, 1]], ls=(0, (2, 1.5)), lw=0.4 + 1.6 * C[i, j], color=ITEMCOL[j],
                       alpha=0.85, transform=b.transAxes)
    for i in range(4):
        b.plot(*X[i], "o", ms=4, color=H.INK, transform=b.transAxes, zorder=3)
    for j in range(4):
        b.plot(*X[4 + j], "o", ms=5.5, color=ITEMCOL[j], mec="white", mew=0.6, transform=b.transAxes, zorder=4)
    b.text(0.5, -0.06, "utterances (black), items (colour);\nline width = similarity", fontsize=5.0, ha="center", va="top",
           color=H.MUTED, transform=b.transAxes)

    c = fig.add_axes([0.51, 0.565, 0.16, 0.33])
    c.imshow(Pz.T, cmap=DIV, vmin=-2.5, vmax=2.5, aspect="auto")
    c.set_yticks(range(8)); c.set_yticklabels(short8, fontsize=5.0); c.tick_params(length=0, pad=1.5)
    for t, cc in zip(c.get_yticklabels(), col8):
        t.set_color(cc)
    c.set_xticks([5.5, 17.5, 29.5]); c.set_xticklabels(["Q sleep", "Q mood", "Q energy"], fontsize=5.0)
    for xb in [11.5, 23.5]:
        c.axvline(xb, color="white", lw=1.2)
    c.set_xlabel("speech windows by eliciting question\n(before centring)", fontsize=5.0, labelpad=2)
    for sp in c.spines.values():
        sp.set_visible(False)

    d = fig.add_axes([0.71, 0.60, 0.12, 0.27])
    mat(d, Gl, vmin=-0.2, vmax=1)
    d.set_xlabel("item geometry\n(items × items)", fontsize=5.0, labelpad=2)
    ee = fig.add_axes([0.895, 0.615, 0.1, 0.255])
    ee.scatter(score_self, score_lang, s=2.2, color=H.INK, alpha=0.6, lw=0)
    ee.set_xlabel("PHQ-8, self-report", fontsize=5.0, labelpad=1.5); ee.set_ylabel("PHQ-8, language", fontsize=5.0, labelpad=1.5)
    ee.tick_params(labelsize=5.0, length=1.5, pad=1); ee.set_yticks([])
    ee.text(0.04, 0.97, f"r = {r_score:.2f}", transform=ee.transAxes, fontsize=5.0, va="top", color=H.ACCENT)

    e1 = fig.add_axes([0.015, 0.08, 0.12, 0.28]); e2 = fig.add_axes([0.195, 0.08, 0.12, 0.28])
    mat(e1, Gl, vmin=-0.2, vmax=1); mat(e2, M8, vmin=-0.2, vmax=1)
    e1.set_xlabel("speech with no\nsymptom signal", fontsize=5.0, labelpad=2); e2.set_xlabel("item-text\nsimilarity", fontsize=5.0, labelpad=2)
    fig.text(0.165, 0.22, "≈", fontsize=10, ha="center", va="center")
    fig.text(0.165, 0.285, f"r = {r_ge:.2f}", fontsize=5.4, ha="center", color=H.ACCENT)

    f = fig.add_axes([0.325, 0.03, 0.215, 0.34]); f.set_xlim(-3.6, 7.6); f.set_ylim(-3.6, 2.4); f.axis("off"); f.set_xticks([]); f.set_yticks([]); f.set_aspect("equal")
    rr = np.random.default_rng(5); Zc = rr.standard_normal((140, 2)); Zc = Zc[np.hypot(*Zc.T) < 1.9][:90]
    Lc = np.linalg.cholesky(np.array([[1, 0.85], [0.85, 1]])); Pc = Zc @ Lc.T
    f.scatter(Pc[:, 0] - 1.0, Pc[:, 1], s=2, color=H.MUTED, lw=0)
    f.add_patch(Ellipse((-1.0, 0), 2 * np.sqrt(1.85) * 1.6, 2 * np.sqrt(0.15) * 1.6, angle=45, fill=False, ec=H.INK, lw=0.6))
    f.scatter(Zc[:, 0] * 0.75 + 4.6, Zc[:, 1] * 0.75, s=2, color=H.ACCENT, lw=0)
    f.add_patch(Ellipse((4.6, 0), 2.4, 2.4, fill=False, ec=H.ACCENT, lw=0.6))
    f.annotate("", xy=(3.0, 0), xytext=(1.4, 0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=H.INK, mutation_scale=6))
    f.text(2.2, 0.35, r"$M^{-1/2}$", fontsize=5.6, ha="center", va="bottom")
    f.text(-1.0, -3.5, "items that read alike\nco-vary", fontsize=5.0, ha="center", va="bottom", color=H.MUTED)
    f.text(4.6, -3.5, "what departs\nfrom the wording", fontsize=5.0, ha="center", va="bottom", color=H.ACCENT)
    fig.text(0.335, 0.395, r"$W = M^{-1/2}\,C\,M^{-1/2} - I$", fontsize=5.6, va="center", color=H.MUTED)

    g1 = fig.add_axes([0.575, 0.15, 0.095, 0.21]); g2 = fig.add_axes([0.695, 0.15, 0.095, 0.21])
    lim = np.nanpercentile(np.abs(Wh), 98)
    mat(g1, Wh, cmap=DIV, vmin=-lim, vmax=lim); mat(g2, Rs, vmin=-0.2, vmax=1)
    g1.set_xlabel("whitened\nlanguage", fontsize=5.0, labelpad=2); g2.set_xlabel("self-report", fontsize=5.0, labelpad=2)
    fig.text(0.6825, 0.255, "~", fontsize=9, ha="center", va="center")
    fig.text(0.6825, 0.055, f"partial r given wording = {h2['partial_r']:.2f}\nitem-permutation test (DAIC-WOZ)", fontsize=5.0,
             ha="center", va="center", color=H.MUTED)

    hx = fig.add_axes([0.855, 0.10, 0.13, 0.25])
    st = ex["stance"]; xs = np.array([0, 1])
    hx.bar(xs - 0.19, [st["cosine"][0], st["entailment"][0]], 0.36, color=H.MUTED, label="forward item")
    hx.bar(xs + 0.19, [st["cosine"][1], st["entailment"][1]], 0.36, color=H.ACCENT, label="reversed item")
    hx.axhline(0, color=H.INK, lw=0.5); hx.set_ylim(-1.05, 1.05); hx.set_yticks([-1, 0, 1])
    hx.set_xticks(xs); hx.set_xticklabels(["cosine", "entailment"], fontsize=5.2)
    hx.legend(loc="lower left", fontsize=5.0, handlelength=0.9, borderaxespad=0.1)
    fig.text(0.835, 0.392, "“I feel I can tell my\ntherapist anything.”", fontsize=5.0, va="center", color=H.MUTED, style="italic", linespacing=1.0)

    arrow(fig, (0.252, 0.73), (0.266, 0.73)); arrow(fig, (0.437, 0.73), (0.452, 0.73)); arrow(fig, (0.676, 0.73), (0.703, 0.73))
    fig.add_artist(FancyArrowPatch((0.672, 0.855), (0.888, 0.855), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                   color=H.MUTED, connectionstyle="arc3,rad=-0.22"))
    fig.add_artist(plt.Line2D([0.0, 1.0], [0.478, 0.478], transform=fig.transFigure, color=H.GRID, lw=0.8))
    fig.text(0.0, 0.997, "Psychometric representational similarity analysis: language read through the geometry of an instrument",
             fontsize=5.8, color=H.MUTED, va="top")
    fig.text(0.0, 0.468, "What COMPASS 2.0 adds: separating the questionnaire from the person", fontsize=5.8, color=H.MUTED, va="top")
    row1 = [(0.0, "a", "Language and probes"), (0.27, "b", "Embedding space"), (0.475, "c", "Item activity profiles"),
            (0.695, "d", "Psychometric RSA"), (0.855, "e", "COMPASS")]
    row2 = [(0.0, "f", "The wording null"), (0.325, "g", "Whitening by the wording"), (0.555, "h", "Test against the person"),
            (0.83, "i", "Stance, not topic")]
    for x, y, eq in [(0.288, 0.922, r"$s(x,a)$: cosine or entailment"), (0.493, 0.922, r"$P_{tj}=s(x_t,a_j)$"),
                     (0.713, 0.922, r"$C=\mathrm{cov}(P)$"), (0.873, 0.922, r"$\hat{s}_t=\Sigma_j\,k_j P_{tj}$"),
                     (0.018, 0.395, r"null: $C\propto A^{\top}A=M$"), (0.573, 0.395, r"$\rho=r(W,R\mid M)$")]:
        fig.text(x, y, eq, fontsize=5.6, va="center", color=H.MUTED)
    for y, row in [(0.955, row1), (0.428, row2)]:
        for x, l, t in row:
            fig.text(x, y, l, fontsize=8, fontweight="bold", va="center")
            fig.text(x + 0.018, y, t, fontsize=7, va="center")
    H.save(fig, "fig2_method")
    (P.RESULTS / "method_figure.json").write_text(json.dumps({"r_signal_free_geometry_vs_wording": float(r_ge),
                                                               "windows": int(T), "questions": 3, "seed": 3,
                                                               "r_language_phq8_score_vs_selfreport_daic": r_score}, indent=1))
    return r_ge

if __name__ == "__main__":
    H.setup(); print("r(geometry of signal-free speech, wording) =", round(main(), 2))
