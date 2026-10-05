"""Fig hub."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

import house_style as H
import paths as P

SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + H.BLUES[1:])
SHORT = {"language, embedding (DAIC-WOZ)": "speech · embedding (D)", "language, embedding (E-DAIC)": "speech · embedding (E)",
         "item wording (MiniLM)": "item wording · MiniLM", "item wording (bge)": "item wording · bge",
         "self-report (DAIC-WOZ)": "self-report (D)", "self-report (E-DAIC)": "self-report (E)",
         "language, entailment (DAIC-WOZ)": "speech · entailment (D)", "language, entailment (E-DAIC)": "speech · entailment (E)",
         "entailment probe geometry": "entailment probes", "theory: scale clusters": "theory · scale clusters"}
SPOKES = [("language, embedding (DAIC-WOZ)", "speech\n(embedding scorer)"), ("self-report (DAIC-WOZ)", "self-report\n(questionnaire answers)"),
          ("theory: scale clusters", "theory\n(scale clusters)"), ("entailment probe geometry", "scorer's probes\n(entailment model)"),
          ("language, entailment (DAIC-WOZ)", "speech\n(entailment scorer)"), ("item wording (MiniLM)", "item wording\n(text similarity)")]

def main():
    hub = json.loads((P.RESULTS / "hub_sources.json").read_text())
    z = np.load(P.DERIVED / "hub_rdms.npz", allow_pickle=True)
    names = list(hub["names"]); R = {n: z[f"r{k}"] for k, n in enumerate(names)}
    S = np.array(hub["spearman"])
    fig = plt.figure(figsize=(H.W2, 92 * H.MM))

    cx, cy, rx, ry = 0.245, 0.455, 0.185, 0.335
    box = FancyBboxPatch((cx - 0.09, cy - 0.065), 0.18, 0.13, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#f2f2f2",
                         ec=H.FAINT, lw=0.6, transform=fig.transFigure, gid="card")
    fig.add_artist(box)
    fig.text(cx, cy + 0.018, "psychometric RDM", fontsize=6.6, ha="center", va="center", fontweight="bold")
    fig.text(cx, cy - 0.025, "dissimilarity between items\n(items × items) from any source", fontsize=5.2, ha="center", va="center",
             color=H.MUTED)
    sz = 0.105
    for k, (key, lab) in enumerate(SPOKES):
        ang = np.pi / 2 - k * np.pi / 3
        x, y = cx + rx * np.cos(ang), cy + ry * np.sin(ang)
        fig.add_artist(plt.Line2D([cx + 0.098 * np.cos(ang), x - 0.058 * np.cos(ang)], [cy + 0.072 * np.sin(ang), y - 0.075 * np.sin(ang)],
                                  transform=fig.transFigure, color=H.FAINT, lw=0.8, zorder=0))
        ax = fig.add_axes([x - sz / 2, y - sz * 0.62, sz, sz * 1.24 * 0.92])
        M = R[key].copy(); np.fill_diagonal(M, 0)
        ax.imshow(M, cmap="Greys_r", vmin=0, vmax=max(1.0, np.nanpercentile(M, 99)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color(H.FAINT)
        above = np.sin(ang) > 0.1
        ax.text(0.5, 1.06 if above else -0.06, lab, transform=ax.transAxes, fontsize=5.2, ha="center",
                va="bottom" if above else "top", linespacing=1.0)
    fig.text(0.0, 0.975, "a", fontsize=8, fontweight="bold", va="center")
    fig.text(0.018, 0.975, "One currency for psychometric representations", fontsize=7, va="center")
    fig.text(0.018, 0.935, "25 PHQ-8 and PCL-C items, DAIC-WOZ (189 participants); darker = more similar", fontsize=5.2, va="center",
             color=H.MUTED)

    order = list(SHORT)
    ix = [names.index(n) for n in order]
    Sb = S[np.ix_(ix, ix)]
    side = 0.36
    b = fig.add_axes([0.625, 0.235, side, side * H.W2 / (92 * H.MM)])
    rel = hub.get("split_half_reliability", {})
    Sd = Sb.copy(); np.fill_diagonal(Sd, np.nan)
    b.imshow(Sd, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(ix)):
        for j in range(len(ix)):
            if i != j:
                b.text(j, i, f"{Sb[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=5.0,
                       color="white" if Sb[i, j] > 0.6 else H.INK, gid="cell")
            else:
                rv = rel.get(order[i])
                b.text(j, i, f"{rv:.2f}".replace("0.", ".") if rv is not None else "–", ha="center", va="center", fontsize=5.0,
                       color=H.ACCENT, fontstyle="italic", gid="cell")
    b.set_xticks(range(len(ix))); b.set_xticklabels([SHORT[n] for n in order], rotation=90, fontsize=5.0)
    b.set_yticks(range(len(ix))); b.set_yticklabels([SHORT[n] for n in order], fontsize=5.0)
    b.tick_params(length=0, pad=1.5)
    for sp in b.spines.values():
        sp.set_visible(False)
    fig.text(0.47, 0.975, "b", fontsize=8, fontweight="bold", va="center")
    fig.text(0.488, 0.975, "Comparing representations", fontsize=7, va="center")
    fig.text(0.488, 0.935, "Spearman correlation between RDMs; diagonal: split-half reliability across participants;", fontsize=5.2, va="center", color=H.MUTED)
    fig.text(0.488, 0.902, "D, DAIC-WOZ; E, E-DAIC", fontsize=5.2, va="center", color=H.MUTED)
    H.save(fig, "fig1_hub")

if __name__ == "__main__":
    H.setup(); main()
