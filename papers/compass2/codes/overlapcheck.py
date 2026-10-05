"""Overlapcheck."""
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.text import Text

import figcheck as F

def _data_pts(ax, r):
    saved = []
    for p in ax.patches:
        if p.get_gid() == "card" and p.get_visible():
            p.set_visible(False); saved.append(p)
    pts = F._data_points(ax, r)
    for p in saved:
        p.set_visible(True)
    ims = []
    for im in ax.get_images():
        if isinstance(im, AxesImage) and im.get_visible():
            bb = im.get_window_extent(r)
            gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 12), np.linspace(bb.y0, bb.y1, 12))
            ims.append(np.column_stack([gx.ravel(), gy.ravel()]))
    return np.vstack([pts] + ims) if ims else pts

def _on_card(t, r, cards):
    bb = t.get_window_extent(r)
    return any(c.x0 <= bb.x0 and c.x1 >= bb.x1 and c.y0 <= bb.y0 and c.y1 >= bb.y1 for c in cards)

def strict(fig, name=""):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    axes = [a for a in fig.axes if a.get_visible()]
    frames = {a: a.get_window_extent(r) for a in axes}
    tight = {a: a.get_tightbbox(r) for a in axes}
    shown = {a: a.axison for a in axes}
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            if shown[a] and shown[b] and F._shrink(frames[a], 1).overlaps(F._shrink(frames[b], 1)):
                out.append(f"panel-panel: {name} {a.get_label() or i} / {b.get_label()}")
    for a in axes:
        if not shown[a]:
            continue
        for b in axes:
            if b is a or not shown[b]:
                continue
            if F._shrink(tight[a], 1.5).overlaps(F._shrink(frames[b], 1.5)) and not F._shrink(frames[a], 1).overlaps(frames[b]):
                out.append(f"label-panel: {name} labels of panel at x={frames[a].x0:.0f} run into panel at x={frames[b].x0:.0f}")
    cards = [p.get_window_extent(r) for a in axes for p in a.patches if p.get_gid() == "card"]
    texts = [(t, bb) for t, bb in F._texts(fig, r)]
    lt = F._legend_texts(fig)
    free = [(t, bb) for t, bb in texts if id(t) not in lt]
    W, Hh = fig.bbox.width, fig.bbox.height
    for t, bb in free:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > Hh + 1:
            out.append(f"off-canvas: {name} {t.get_text()[:30]!r}")
    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            (t1, b1), (t2, b2) = free[i], free[j]
            if F._rotated_tick(t1) and F._rotated_tick(t2):
                continue
            if F._shrink(b1).overlaps(F._shrink(b2)):
                out.append(f"text-text: {name} {t1.get_text()[:28]!r} / {t2.get_text()[:28]!r}")
    pts = {a: _data_pts(a, r) for a in axes}
    ticklabels = {id(t) for a in axes for t in a.get_xticklabels() + a.get_yticklabels()}
    axlabels = {id(a.xaxis.label) for a in axes} | {id(a.yaxis.label) for a in axes} | {id(a.title) for a in axes}\
        | {id(a._left_title) for a in axes} | {id(a._right_title) for a in axes}
    for t, bb in free:
        if id(t) in ticklabels or id(t) in axlabels or _on_card(t, r, cards) or t.get_gid() == "cell":
            continue
        sb = F._shrink(bb, 1.0)
        for a in axes:
            if F._inside(sb, pts[a]):
                out.append(f"text-data: {name} {t.get_text()[:30]!r}")
                break
    for a in axes:
        lg = a.get_legend()
        if lg is None or not lg.get_visible():
            continue
        lb = F._shrink(lg.get_window_extent(r), 1.0)
        if F._inside(lb, pts[a]):
            out.append(f"legend-data: {name} legend of panel at x={frames[a].x0:.0f}")
        for t, bb in free:
            if lb.overlaps(F._shrink(bb)):
                out.append(f"legend-text: {name} legend over {t.get_text()[:30]!r}")
    return out
