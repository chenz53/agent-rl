"""A small, dependency-free SVG plotting and diagram toolkit for the book's figures.

The remote host has PyTorch but no matplotlib, and the book must not install
packages, so figures are written as hand-built SVG. Every figure:
  * uses one validated categorical palette in fixed slot order (never cycled),
  * carries its own light and dark tokens (prefers-color-scheme inside the SVG),
  * uses thin marks (2px lines, >=8px markers with a surface ring, <=24px bars
    with 4px rounded data ends), hairline solid gridlines, one y-axis per panel,
  * has a legend for >=2 series plus selective direct labels, and native hover
    tooltips (<title>) on data marks.

Typical use (inside scripts/figures/chNN.py):

    from svgplot import Figure
    fig = Figure(720, 360, title='Clipped surrogate', subtitle='A > 0 and A < 0')
    ax = fig.axes(0, 0, 1, 1, xlabel='ratio r', ylabel='objective')
    ax.line([...], [...], label='A > 0', series=1)
    ax.legend()
    fig.save('ppo-clip')            # -> tutorial/figures/ppo-clip.svg
"""
from __future__ import annotations

import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tutorial' / 'figures'

# Categorical slots (light, dark). Order is the CVD-safety mechanism: assign in order.
SERIES = [
    ('#2a78d6', '#3987e5'),  # 1 blue
    ('#eb6834', '#d95926'),  # 2 orange
    ('#1baf7a', '#199e70'),  # 3 aqua
    ('#eda100', '#c98500'),  # 4 yellow
    ('#e87ba4', '#d55181'),  # 5 magenta
    ('#008300', '#008300'),  # 6 green
    ('#6250d6', '#9085e9'),  # 7 violet
    ('#e34948', '#e66767'),  # 8 red
]
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
       '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
DIV_NEG = '#e34948'  # red pole
DIV_POS = '#2a78d6'  # blue pole
INK = {
    'surface': ('#fcfcfb', '#1a1a19'),
    'ink': ('#0b0b0b', '#f0efec'),
    'ink2': ('#52514e', '#c3c2b7'),
    'muted': ('#898781', '#898781'),
    'grid': ('#e1e0d9', '#2c2c2a'),
    'axis': ('#c3c2b7', '#383835'),
    'neutral': ('#f0efec', '#383835'),
    'panel': ('#f4f3ef', '#232322'),
}


def _css():
    light = ';'.join(f'--{k}:{v[0]}' for k, v in INK.items())
    dark = ';'.join(f'--{k}:{v[1]}' for k, v in INK.items())
    light += ';' + ';'.join(f'--s{i + 1}:{c[0]}' for i, c in enumerate(SERIES))
    dark += ';' + ';'.join(f'--s{i + 1}:{c[1]}' for i, c in enumerate(SERIES))
    return f"""
svg{{{light};font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
@media (prefers-color-scheme: dark){{svg{{{dark}}}}}
.bg{{fill:var(--surface)}} .t{{fill:var(--ink)}} .t2{{fill:var(--ink2)}} .m{{fill:var(--muted)}}
.title{{font-size:15px;font-weight:600;fill:var(--ink)}} .sub{{font-size:12px;fill:var(--ink2)}}
.lab{{font-size:12px;fill:var(--ink2)}} .tick{{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}}
.grid{{stroke:var(--grid);stroke-width:1}} .axis{{stroke:var(--axis);stroke-width:1}}
.box{{fill:var(--panel);stroke:var(--axis);stroke-width:1}} .edge{{stroke:var(--ink2);stroke-width:1.5;fill:none}}
.ring{{stroke:var(--surface);stroke-width:2}} .note{{font-size:11px;fill:var(--ink2)}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}}
"""


def esc(text):
    return html.escape(str(text), quote=True)


def color(series):
    """Accept a slot number (1..8) or any CSS colour / var()."""
    if isinstance(series, int):
        if not 1 <= series <= len(SERIES):
            raise ValueError('only 8 categorical slots; fold extra series into "Other" or facet')
        return f'var(--s{series})'
    return series


def nice_ticks(lo, hi, n=5):
    if hi == lo:
        hi = lo + 1
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v):
    if v == 0:
        return '0'
    if abs(v) >= 1000 or abs(v) < 0.01:
        return f'{v:.0e}' if abs(v) < 0.01 else f'{v:,.0f}'
    return f'{v:g}'


class Axes:
    def __init__(self, fig, x, y, w, h, xlabel='', ylabel='', title='', xlim=None, ylim=None,
                 xlog=False, ylog=False, xticks=None, yticks=None, xticklabels=None):
        self.fig, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog
        self.xticks, self.yticks, self.xticklabels = xticks, yticks, xticklabels
        self.items = []      # deferred draw calls (need limits first)
        self.legend_items = []
        self.xs, self.ys = [], []
        self.categorical = None

    # ---- scales -------------------------------------------------------
    def _limits(self):
        xlo, xhi = self.xlim or (min(self.xs), max(self.xs))
        if self.ylim:
            ylo, yhi = self.ylim
        else:
            ylo, yhi = min(self.ys), max(self.ys)
            pad = (yhi - ylo) * 0.06 or 1
            ylo, yhi = (ylo - pad if ylo < 0 or ylo - pad > 0 else 0) if not self.ylog else ylo, yhi + pad
            if min(self.ys) >= 0 and not self.ylog:
                ylo = 0
        return xlo, xhi, ylo, yhi

    def sx(self, v):
        xlo, xhi = self._xl
        if self.xlog:
            v, xlo, xhi = math.log10(v), math.log10(xlo), math.log10(xhi)
        return self.x + (v - xlo) / ((xhi - xlo) or 1) * self.w

    def sy(self, v):
        ylo, yhi = self._yl
        if self.ylog:
            v, ylo, yhi = math.log10(v), math.log10(ylo), math.log10(yhi)
        return self.y + self.h - (v - ylo) / ((yhi - ylo) or 1) * self.h

    # ---- marks --------------------------------------------------------
    def line(self, xs, ys, label=None, series=1, dashed=False, markers=False, end_label=True, width=2,
             tip_fmt='{label}: x={x:g}, y={y:.3g}'):
        xs, ys = list(map(float, xs)), list(map(float, ys))
        self.xs += xs
        self.ys += ys
        self.items.append(('line', dict(xs=xs, ys=ys, label=label, c=color(series), dashed=dashed,
                                        markers=markers, end_label=end_label, width=width, tip=tip_fmt)))
        if label:
            self.legend_items.append((label, color(series), 'line'))

    def band(self, xs, lo, hi, series=1, opacity=0.12):
        xs, lo, hi = list(map(float, xs)), list(map(float, lo)), list(map(float, hi))
        self.xs += xs
        self.ys += lo + hi
        self.items.append(('band', dict(xs=xs, lo=lo, hi=hi, c=color(series), op=opacity)))

    def scatter(self, xs, ys, label=None, series=1, r=4, tips=None):
        xs, ys = list(map(float, xs)), list(map(float, ys))
        self.xs += xs
        self.ys += ys
        self.items.append(('scatter', dict(xs=xs, ys=ys, c=color(series), r=r, label=label, tips=tips)))
        if label:
            self.legend_items.append((label, color(series), 'dot'))

    def bars(self, categories, groups, value_labels=False, fmt_value='{:.2g}'):
        """groups: list of (label, values, series). Grouped vertical bars."""
        self.categorical = list(categories)
        self.xs += [0, len(categories)]
        for _, values, _ in groups:
            self.ys += list(map(float, values))
        self.ys.append(0.0)
        self.items.append(('bars', dict(groups=groups, value_labels=value_labels, fmt=fmt_value)))
        if len(groups) > 1:
            for label, _, s in groups:
                self.legend_items.append((label, color(s), 'box'))

    def hline(self, y, label=None, style='axis'):
        self.ys.append(float(y))
        self.items.append(('hline', dict(y=float(y), label=label, style=style)))

    def vline(self, x, label=None):
        self.xs.append(float(x))
        self.items.append(('vline', dict(x=float(x), label=label)))

    def text(self, x, y, s, anchor='start', cls='note', dx=0, dy=0):
        self.items.append(('text', dict(x=x, y=y, s=s, anchor=anchor, cls=cls, dx=dx, dy=dy)))

    def legend(self, where='top'):
        self.fig._legend_axes.append((self, where))

    # ---- rendering ----------------------------------------------------
    def render(self):
        out = []
        if self.categorical is not None:
            self._xl = (0, len(self.categorical))
            ylo = min(0.0, min(self.ys))
            yhi = self.ylim[1] if self.ylim else max(self.ys) * 1.1
            self._yl = (self.ylim[0] if self.ylim else ylo, yhi)
        else:
            xlo, xhi, ylo, yhi = self._limits()
            self._xl, self._yl = (xlo, xhi), (ylo, yhi)
        ylo, yhi = self._yl
        if self.title:
            out.append(f'<text class="lab" x="{self.x}" y="{self.y - 10}" font-weight="600">{esc(self.title)}</text>')
        yt = self.yticks or (nice_ticks(ylo, yhi) if not self.ylog else
                             [10 ** k for k in range(math.ceil(math.log10(ylo)), math.floor(math.log10(yhi)) + 1)])
        for t in yt:
            py = self.sy(t)
            out.append(f'<line class="grid" x1="{self.x}" x2="{self.x + self.w}" y1="{py:.1f}" y2="{py:.1f}"/>')
            out.append(f'<text class="tick" x="{self.x - 6}" y="{py + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
        base = self.sy(0) if ylo <= 0 <= yhi and not self.ylog else self.y + self.h
        out.append(f'<line class="axis" x1="{self.x}" x2="{self.x + self.w}" y1="{base:.1f}" y2="{base:.1f}"/>')
        if self.categorical is not None:
            for i, c in enumerate(self.categorical):
                out.append(f'<text class="tick" x="{self.sx(i + 0.5):.1f}" y="{self.y + self.h + 16}" text-anchor="middle">{esc(c)}</text>')
        else:
            xlo, xhi = self._xl
            xt = self.xticks or (nice_ticks(xlo, xhi) if not self.xlog else
                                 [10 ** k for k in range(math.ceil(math.log10(xlo)), math.floor(math.log10(xhi)) + 1)])
            for i, t in enumerate(xt):
                px = self.sx(t)
                lab = self.xticklabels[i] if self.xticklabels else fmt(t)
                out.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{self.y + self.h}" y2="{self.y + self.h + 4}"/>')
                out.append(f'<text class="tick" x="{px:.1f}" y="{self.y + self.h + 16}" text-anchor="middle">{esc(lab)}</text>')
        if self.xlabel:
            out.append(f'<text class="lab" x="{self.x + self.w / 2}" y="{self.y + self.h + 34}" text-anchor="middle">{esc(self.xlabel)}</text>')
        if self.ylabel:
            cx, cy = self.x - 44, self.y + self.h / 2
            out.append(f'<text class="lab" x="{cx}" y="{cy}" text-anchor="middle" transform="rotate(-90 {cx} {cy})">{esc(self.ylabel)}</text>')
        out.append(f'<clipPath id="clip{id(self)}"><rect x="{self.x}" y="{self.y - 2}" width="{self.w}" height="{self.h + 4}"/></clipPath>')
        for kind, d in self.items:
            out += getattr(self, '_draw_' + kind)(d)
        return out

    def _draw_band(self, d):
        pts = [(self.sx(x), self.sy(v)) for x, v in zip(d['xs'], d['hi'])]
        pts += [(self.sx(x), self.sy(v)) for x, v in reversed(list(zip(d['xs'], d['lo'])))]
        p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
        return [f'<polygon points="{p}" fill="{d["c"]}" fill-opacity="{d["op"]}" stroke="none" clip-path="url(#clip{id(self)})"/>']

    def _draw_line(self, d):
        pts = ' '.join(f'{self.sx(x):.1f},{self.sy(y):.1f}' for x, y in zip(d['xs'], d['ys']))
        dash = ' stroke-dasharray="6 4"' if d['dashed'] else ''
        out = [f'<polyline points="{pts}" fill="none" stroke="{d["c"]}" stroke-width="{d["width"]}" '
               f'stroke-linejoin="round" stroke-linecap="round"{dash} clip-path="url(#clip{id(self)})"/>']
        n = len(d['xs'])
        idx = range(n) if d['markers'] else ([n - 1] if d['end_label'] and d['label'] else [])
        for i in idx:
            x, y = d['xs'][i], d['ys'][i]
            tip = esc(d['tip'].format(label=d['label'] or '', x=x, y=y))
            out.append(f'<circle class="ring" cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="4" fill="{d["c"]}"><title>{tip}</title></circle>')
        if d['end_label'] and d['label']:
            out.append(f'<text class="lab" x="{self.sx(d["xs"][-1]) + 8:.1f}" y="{self.sy(d["ys"][-1]) + 4:.1f}">{esc(d["label"])}</text>')
        return out

    def _draw_scatter(self, d):
        out = []
        for i, (x, y) in enumerate(zip(d['xs'], d['ys'])):
            tip = esc(d['tips'][i]) if d['tips'] else f'{esc(d["label"] or "")} ({x:g}, {y:.3g})'
            out.append(f'<circle class="ring" cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{d["r"]}" fill="{d["c"]}"><title>{tip}</title></circle>')
        return out

    def _draw_bars(self, d):
        out, groups = [], d['groups']
        slot = self.w / len(self.categorical)
        bw = min(24, (slot * 0.7 - 2 * (len(groups) - 1)) / len(groups))
        total = bw * len(groups) + 2 * (len(groups) - 1)
        base = self.sy(0)
        for gi, (label, values, s) in enumerate(groups):
            for i, v in enumerate(values):
                x0 = self.x + slot * i + (slot - total) / 2 + gi * (bw + 2)
                top = self.sy(float(v))
                y0, hgt = min(top, base), abs(base - top)
                r = min(4, hgt / 2, bw / 2)
                # Rounded data end, square at the baseline.
                if v >= 0:
                    path = (f'M{x0:.1f},{base:.1f} V{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} '
                            f'H{x0 + bw - r:.1f} Q{x0 + bw:.1f},{y0:.1f} {x0 + bw:.1f},{y0 + r:.1f} V{base:.1f} Z')
                else:
                    yb = y0 + hgt
                    path = (f'M{x0:.1f},{base:.1f} V{yb - r:.1f} Q{x0:.1f},{yb:.1f} {x0 + r:.1f},{yb:.1f} '
                            f'H{x0 + bw - r:.1f} Q{x0 + bw:.1f},{yb:.1f} {x0 + bw:.1f},{yb - r:.1f} V{base:.1f} Z')
                tip = esc(f'{label}, {self.categorical[i]}: {float(v):.4g}')
                out.append(f'<path d="{path}" fill="{color(s)}"><title>{tip}</title></path>')
                if d['value_labels']:
                    ty = (top - 5) if v >= 0 else (top + 13)
                    out.append(f'<text class="tick" x="{x0 + bw / 2:.1f}" y="{ty:.1f}" text-anchor="middle">{esc(d["fmt"].format(float(v)))}</text>')
        return out

    def _draw_hline(self, d):
        py = self.sy(d['y'])
        out = [f'<line class="{d["style"]}" x1="{self.x}" x2="{self.x + self.w}" y1="{py:.1f}" y2="{py:.1f}" stroke-dasharray="4 3"/>']
        if d['label']:
            out.append(f'<text class="note" x="{self.x + self.w - 4}" y="{py - 5:.1f}" text-anchor="end">{esc(d["label"])}</text>')
        return out

    def _draw_vline(self, d):
        px = self.sx(d['x'])
        out = [f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{self.y}" y2="{self.y + self.h}" stroke-dasharray="4 3"/>']
        if d['label']:
            out.append(f'<text class="note" x="{px + 5:.1f}" y="{self.y + 12}">{esc(d["label"])}</text>')
        return out

    def _draw_text(self, d):
        return [f'<text class="{d["cls"]}" x="{self.sx(d["x"]) + d["dx"]:.1f}" y="{self.sy(d["y"]) + d["dy"]:.1f}" '
                f'text-anchor="{d["anchor"]}">{esc(d["s"])}</text>']


class Figure:
    """A titled SVG canvas holding plot axes and/or diagram primitives."""

    def __init__(self, width, height, title='', subtitle=''):
        self.W, self.H, self.title, self.subtitle = width, height, title, subtitle
        self.top = 20 + (22 if title else 0) + (18 if subtitle else 0)
        self.body = []
        self._axes = []
        self._legend_axes = []

    # Plot area in fractions of the area below the title.
    def axes(self, fx, fy, fw, fh, margin=(64, 28, 60, 52), **kw):
        left, top, right, bottom = margin
        if kw.get('title'):
            top += 18  # room for the panel title above the legend row
        ah = self.H - self.top
        x = fx * self.W + left
        y = self.top + fy * ah + top
        ax = Axes(self, x, y, fw * self.W - left - right, fh * ah - top - bottom, **kw)
        self._axes.append(ax)
        return ax

    # ---- diagram primitives -------------------------------------------
    def rect(self, x, y, w, h, fill=None, cls='box', rx=6, tip=None, stroke=None):
        f = f' fill="{color(fill)}"' if fill else ''
        s = f' stroke="{color(stroke)}" stroke-width="1.5"' if stroke else ''
        t = f'<title>{esc(tip)}</title>' if tip else ''
        self.body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"{f}{s}>{t}</rect>')

    def text(self, x, y, s, cls='t', size=12, anchor='start', weight=None, fill=None, mono=False):
        wt = f' font-weight="{weight}"' if weight else ''
        fl = f' style="fill:{color(fill)}"' if fill else ''
        c = cls + (' mono' if mono else '')
        lines = str(s).split('\n')
        for i, line in enumerate(lines):
            self.body.append(f'<text class="{c}" x="{x}" y="{y + i * (size + 3)}" font-size="{size}" '
                             f'text-anchor="{anchor}"{wt}{fl}>{esc(line)}</text>')

    def node(self, x, y, w, h, label, sub=None, fill=None, stroke=None, tip=None, size=12):
        """A rounded box with a centred label (and optional second line)."""
        self.rect(x, y, w, h, fill=fill, stroke=stroke, tip=tip)
        cy = y + h / 2 + (0 if sub else 4)
        self.text(x + w / 2, cy - (4 if sub else 0), label, anchor='middle', weight=600, size=size)
        if sub:
            self.text(x + w / 2, cy + 12, sub, cls='t2', anchor='middle', size=11)

    def arrow(self, x1, y1, x2, y2, label=None, curve=0, cls='edge', stroke=None, dashed=False):
        mid = ((x1 + x2) / 2 - curve * (y2 - y1) / 100, (y1 + y2) / 2 + curve * (x2 - x1) / 100)
        s = f' style="stroke:{color(stroke)}"' if stroke else ''
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        self.body.append(f'<path class="{cls}" d="M{x1},{y1} Q{mid[0]:.1f},{mid[1]:.1f} {x2},{y2}"{s}{dash} marker-end="url(#arrow)"/>')
        if label:
            self.text(mid[0], mid[1] - 6, label, cls='t2', size=11, anchor='middle')

    def tokens(self, x, y, items, cell=None, h=26, gap=2, size=11):
        """Draw a token strip. items: list of (text, fill_or_None, tip_or_None). Returns end x."""
        cx = x
        for text, fill, tip in items:
            w = cell or max(26, 7.2 * len(text) + 12)
            if fill is None:
                self.rect(cx, y, w, h, rx=4, tip=tip)
                self.text(cx + w / 2, y + h / 2 + 4, text, anchor='middle', size=size, mono=True)
            else:
                self.body.append(f'<rect x="{cx}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color(fill)}" '
                                 f'fill-opacity="0.18" stroke="{color(fill)}" stroke-width="1.5">'
                                 + (f'<title>{esc(tip)}</title>' if tip else '') + '</rect>')
                self.text(cx + w / 2, y + h / 2 + 4, text, anchor='middle', size=size, mono=True)
            cx += w + gap
        return cx

    def heatmap(self, x, y, w, h, matrix, row_labels=(), col_labels=(), diverging=False, vmin=None, vmax=None,
                fmt_cell=None, tip_fmt='{r}, {c}: {v:.3g}'):
        rows, cols = len(matrix), len(matrix[0])
        flat = [v for row in matrix for v in row]
        lo = min(flat) if vmin is None else vmin
        hi = max(flat) if vmax is None else vmax
        cw, ch = w / cols, h / rows
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if diverging:
                    m = max(abs(lo), abs(hi)) or 1
                    t = v / m
                    fill, op = (DIV_POS if t >= 0 else DIV_NEG), min(1, abs(t)) * 0.9 + 0.05
                    attrs = f'fill="{fill}" fill-opacity="{op:.2f}"'
                else:
                    t = (v - lo) / ((hi - lo) or 1)
                    attrs = f'fill="{SEQ[min(len(SEQ) - 1, int(t * (len(SEQ) - 1) + 0.5))]}"'
                rl = row_labels[i] if row_labels else i
                cl = col_labels[j] if col_labels else j
                tip = esc(tip_fmt.format(r=rl, c=cl, v=v))
                self.body.append(f'<rect x="{x + j * cw + 1:.1f}" y="{y + i * ch + 1:.1f}" width="{cw - 2:.1f}" '
                                 f'height="{ch - 2:.1f}" rx="2" {attrs}><title>{tip}</title></rect>')
                if fmt_cell:
                    dark = (t > 0.55) if not diverging else abs(t) > 0.55
                    fillc = '#ffffff' if dark else 'var(--ink)'
                    self.body.append(f'<text x="{x + j * cw + cw / 2:.1f}" y="{y + i * ch + ch / 2 + 4:.1f}" font-size="11" '
                                     f'text-anchor="middle" style="fill:{fillc}">{esc(fmt_cell.format(v))}</text>')
        for i, lab in enumerate(row_labels):
            self.text(x - 6, y + i * ch + ch / 2 + 4, lab, cls='t2', size=11, anchor='end')
        for j, lab in enumerate(col_labels):
            self.text(x + j * cw + cw / 2, y - 6, lab, cls='t2', size=11, anchor='middle')

    def legend_row(self, x, y, entries):
        """entries: list of (label, series_or_colour, kind) with kind in line|dot|box."""
        cx = x
        for label, c, kind in entries:
            c = color(c)
            if kind == 'line':
                self.body.append(f'<line x1="{cx}" x2="{cx + 18}" y1="{y - 4}" y2="{y - 4}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
            elif kind == 'dot':
                self.body.append(f'<circle cx="{cx + 9}" cy="{y - 4}" r="4" fill="{c}"/>')
            else:
                self.body.append(f'<rect x="{cx + 2}" y="{y - 10}" width="12" height="12" rx="3" fill="{c}"/>')
            self.text(cx + 24, y, label, cls='t2', size=12)
            cx += 24 + 7 * len(label) + 18

    # ---- output ---------------------------------------------------------
    def render(self):
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="{self.W}" height="{self.H}" '
                 f'role="img" aria-label="{esc(self.title)}">',
                 f'<style>{_css()}</style>',
                 '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker></defs>',
                 f'<rect class="bg" width="{self.W}" height="{self.H}" rx="8"/>']
        if self.title:
            parts.append(f'<text class="title" x="20" y="30">{esc(self.title)}</text>')
        if self.subtitle:
            parts.append(f'<text class="sub" x="20" y="{48 if self.title else 30}">{esc(self.subtitle)}</text>')
        for ax in self._axes:
            parts += ax.render()
        for ax, where in self._legend_axes:
            if len(ax.legend_items) >= 2:
                ly = (ax.y - 12 - (18 if ax.title else 0)) if where == 'top' else ax.y + ax.h + 50
                self.legend_row(ax.x, ly, ax.legend_items)
        parts += self.body
        parts.append('</svg>')
        return '\n'.join(parts)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f'{name}.svg'
        path.write_text(self.render())
        return path
