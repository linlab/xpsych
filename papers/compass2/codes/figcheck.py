"""Figcheck."""
import numpy as np
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox

TOL = 0.5

def _shrink(bb, d=TOL):
    return Bbox.from_extents(bb.x0 + d, bb.y0 + d, bb.x1 - d, bb.y1 - d)

def _inside(bb, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    return np.any((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))

def _densify(xy, step=2.0):
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 2:
        return xy
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.vstack(out)

def _data_points(ax, renderer):
    """Display-coordinate samples of everything plotted in an axes (excluding the legend's own artists)."""
    pts = []
    for ln in ax.get_lines():
        if not ln.get_visible() or len(ln.get_xydata()) == 0:
            continue
        xy = ln.get_transform().transform(ln.get_xydata())
        pts.append(_densify(xy) if ln.get_linestyle() not in ("None", "", " ") else xy)
    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, LineCollection):
            for seg in c.get_segments():
                if len(seg):
                    pts.append(_densify(c.get_transform().transform(seg)))
        elif isinstance(c, PathCollection):
            off = c.get_offsets()
            if len(off):
                pts.append(c.get_offset_transform().transform(off))
        elif isinstance(c, PolyCollection):
            for p in c.get_paths():
                tp = p.transformed(c.get_transform())
                v = tp.vertices
                if not len(v):
                    continue
                gx, gy = np.meshgrid(np.linspace(v[:, 0].min(), v[:, 0].max(), 16), np.linspace(v[:, 1].min(), v[:, 1].max(), 16))
                g = np.column_stack([gx.ravel(), gy.ravel()])
                pts.append(g[tp.contains_points(g)])
        else:
            try:
                bb = c.get_window_extent(renderer)
                if np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all() and bb.width > 0:
                    pts.append([[bb.x0, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1], [bb.x1, bb.y0]])
            except Exception:
                pass
    for p in ax.patches:
        if not p.get_visible() or p is ax.patch:
            continue
        bb = p.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0:
            gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 8), np.linspace(bb.y0, bb.y1, 8))
            pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    return np.vstack([np.asarray(p, float).reshape(-1, 2) for p in pts]) if pts else np.empty((0, 2))

def _undrawn_ticklabels(fig):
    """Tick labels that exist but are not drawn (outside the view limits)."""
    skip = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            drawn = set()
            for tk in axis._update_ticks():
                drawn |= {id(tk.label1), id(tk.label2)}
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tk.label1, tk.label2):
                    if id(lab) not in drawn:
                        skip.add(id(lab))
    return skip

def _rotated_tick(t):
    return t.get_rotation() % 90 != 0

def _texts(fig, renderer):
    out = []
    skip = _undrawn_ticklabels(fig)
    for t in fig.findobj(Text):
        if id(t) in skip:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb))
    return out

def _legend_texts(fig):
    ids = set()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            ids |= {id(t) for t in lg.get_texts()}
            ids.add(id(lg.get_title()))
    for lg in fig.legends:
        ids |= {id(t) for t in lg.get_texts()}
    return ids

def check(fig, name=""):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    issues = []
    lt = _legend_texts(fig)
    texts = [(t, bb) for t, bb in _texts(fig, r) if id(t) not in lt]
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None or not lg.get_visible():
            continue
        lb = _shrink(lg.get_window_extent(r), 1.0)
        if _inside(lb, _data_points(ax, r)):
            issues.append(f"legend-data: {name} {ax.get_title(loc='left')[:40]!r}")
        for t, bb in texts:
            if lb.overlaps(_shrink(bb)):
                issues.append(f"legend-text: {name} legend in {ax.get_title(loc='left')[:30]!r} over {t.get_text()[:30]!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (t1, b1), (t2, b2) = texts[i], texts[j]
            if _rotated_tick(t1) and _rotated_tick(t2):
                continue
            if _shrink(b1).overlaps(_shrink(b2)):
                issues.append(f"text-text: {name} {t1.get_text()[:28]!r} / {t2.get_text()[:28]!r}")
    for ax in fig.axes:
        pts = _data_points(ax, r)
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip() and getattr(t, "_allow_data", False) is False:
                if _inside(_shrink(t.get_window_extent(r), 1.0), pts):
                    issues.append(f"text-data: {name} {t.get_text()[:30]!r}")
    main = [a for a in fig.axes if not getattr(a, "_colorbar", False) and a.get_label() != "<colorbar>"]
    rows = []
    for a in main:
        p = a.get_position()
        for row in rows:
            q = row[0].get_position()
            if min(p.y1, q.y1) - max(p.y0, q.y0) > 0.5 * min(p.height, q.height):
                row.append(a); break
        else:
            rows.append([a])
    for row in rows:
        if len(row) > 1:
            tops = [a.get_window_extent(r).y1 for a in row]; bots = [a.get_window_extent(r).y0 for a in row]
            if max(tops) - min(tops) > 1.5 * fig.dpi / 100 or max(bots) - min(bots) > 1.5 * fig.dpi / 100:
                issues.append(f"row-align: {name} tops {np.round(np.ptp(tops), 1)} px, bottoms {np.round(np.ptp(bots), 1)} px")
    return issues

def place_letters(fig, items, size=8, pad_pt=2):
    """items: list of (ax, letter). Letter at the panel's tight-box left edge; one baseline per row."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rows = []
    for ax, s in items:
        p = ax.get_position()
        for row in rows:
            q = row[0][0].get_position()
            if min(p.y1, q.y1) - max(p.y0, q.y0) > 0.5 * min(p.height, q.height):
                row.append((ax, s)); break
        else:
            rows.append([(ax, s)])
    pos = {}
    for row in rows:
        top = max(ax.get_tightbbox(r).y1 for ax, _ in row)
        for ax, s in row:
            x = ax.get_tightbbox(r).x0
            tt = [t for t in (ax.title, ax._left_title) if t.get_text()]
            w = 9.0 * fig.dpi / 72
            if tt:
                x = min(x, min(t.get_window_extent(r).x0 for t in tt) - w)
            pos[(id(ax), s)] = [x, top + pad_pt * fig.dpi / 72]
    xs = sorted(pos.items(), key=lambda kv: kv[1][0])
    col = []
    for k, v in xs:
        if col and v[0] - col[0][1][0] < 0.06 * fig.bbox.width:
            col.append((k, v))
        else:
            if col:
                m = min(c[1][0] for c in col)
                for c in col:
                    pos[c[0]][0] = m
            col = [(k, v)]
    if col:
        m = min(c[1][0] for c in col)
        for c in col:
            pos[c[0]][0] = m
    for (axid, s), (x, y) in pos.items():
        fx, fy = inv.transform((x, y))
        fig.text(fx, fy, s, fontsize=size, fontweight="bold", ha="left", va="top")

def _legend_ok(ax, lg, r):
    lb = _shrink(lg.get_window_extent(r), 1.0)
    if _inside(lb, _data_points(ax, r)):
        return False
    lt = {id(t) for t in lg.get_texts()}
    for t, bb in _texts(ax.figure, r):
        if id(t) not in lt and lb.overlaps(_shrink(bb)):
            return False
    for other in ax.figure.axes:
        if other is not ax and lb.overlaps(_shrink(other.get_window_extent(r))):
            return False
    return True

LOCS = ["upper left", "upper right", "lower right", "lower left", "center right", "center left", "upper center", "lower center"]

def fix_legends(fig, headroom=(0.0, 0.15, 0.3, 0.45)):
    """Move any legend that overlaps data, text or another panel to the first clear position: inside the axes,
    then with added headroom above the data, then below the axes. Keeps handles, labels, font size and columns."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None or _legend_ok(ax, lg, r):
            continue
        labels = [t.get_text() for t in lg.get_texts()]
        orig = dict(zip(*ax.get_legend_handles_labels()[::-1]))
        handles = [orig.get(l, h) for l, h in zip(labels, lg.legend_handles)]
        kw = dict(frameon=lg.get_frame_on(), fontsize=lg._fontsize, ncol=lg._ncols, handlelength=lg.handlelength,
                  borderaxespad=0.3)
        y0, y1 = ax.get_ylim()
        log = ax.get_yscale() == "log"
        placed = False
        for h in headroom:
            if h:
                if log:
                    ax.set_ylim(y0, y1 * (y1 / y0) ** h) if y1 > y0 else None
                else:
                    ax.set_ylim(y0, y1 + h * (y1 - y0)) if y1 > y0 else ax.set_ylim(y0 + h * (y0 - y1), y1)
            for loc in (LOCS if not h else ["upper left", "upper right", "upper center"]):
                new = ax.legend(handles, labels, loc=loc, **kw)
                fig.canvas.draw()
                if _legend_ok(ax, new, r):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            ax.set_ylim(y0, y1)
            new = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.28), **{**kw, "ncol": len(labels)})
            fig.canvas.draw()
