"""House style."""
from pathlib import Path
import paths as P

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
W1, W2 = 89 * MM, 165 * MM

INK, MUTED, FAINT, GRID = "#1b1b1b", "#6b6b6b", "#bdbdbd", "#ececec"

CAT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
BLUES = ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
DIVERGING = ("#2166ac", "#f0f0f0", "#b2182b")
ACCENT, BAND = "#D55E00", "#fbe3d3"

def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "lines.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
        "axes.titlepad": 3, "axes.titlelocation": "left"})

def label(ax, letter, title="", x=-0.02, y=1.02):
    """Bold panel letter with an optional plain title, left-aligned above the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
    if title:
        ax.text(x + 0.02, y, title, transform=ax.transAxes, fontsize=7, va="bottom", ha="left")

def save(fig, name, folder=None):
    """Write PDF and PNG copies of a figure."""
    out = P.FIGURES if folder is None else Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

LETTER_PROBLEMS = []

def label_panels(fig, items, size=8, tol_row=0.06, tol_col=0.06):
    """Bold lowercase panel labels at the top-left of each panel's full extent (axes plus tick labels, axis labels,
    titles and legends). Panels whose axes frames share a top edge form a row and get one common label height;
    panels whose frames share a left edge form a column and get one common label x. Use with
    layout="constrained"; call after all drawing, before saving. items: list of (axes, "a") pairs."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rec = [(lab, ax.get_tightbbox(r).transformed(inv), ax.get_window_extent(r).transformed(inv)) for ax, lab in items]

    def groups(key, tol):
        out = []
        for item in sorted(rec, key=key):
            for g in out:
                if abs(key(g[0]) - key(item)) < tol:
                    g.append(item)
                    break
            else:
                out.append([item])
        return out
    ys = {lab: max(f.y1 for _, f, _ in g) for g in groups(lambda t: -t[2].y1, tol_row) for lab, _, _ in g}
    xs = {lab: min(f.x0 for _, f, _ in g) for g in groups(lambda t: t[2].x0, tol_col) for lab, _, _ in g}
    W = fig.get_size_inches()[0] * 25.4
    for lab, full, _ in rec:
        fig.text(xs[lab], ys[lab], lab, fontsize=size, fontweight="bold", ha="left", va="top")
        gap = (full.x0 - xs[lab]) * W
        if gap > 3:
            LETTER_PROBLEMS.append(f"letter-far: '{lab}' sits {gap:.0f} mm left of its panel (column shared with a "
                                   "wider panel; pass a smaller tol_col)")
