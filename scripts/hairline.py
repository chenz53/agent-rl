"""Hairline-style isometric line figures for the book's mechanism diagrams (pilot).

Visual language adapted from Lucas Marques' Hairline (MIT, github.com/lucasmarkes/hairline);
this is an independent re-implementation, not their code. The rules:
  * one isometric projection (azimuth 45 deg, elevation 30 deg) and one stroke weight;
  * rounded solids drawn as a bright silhouette with dim inner creases; filled plates hide
    whatever is behind them (painter's order, back to front);
  * a dotted floor so the scene has ground; no text inside the drawing, only numbered
    callouts outside it with hairline leaders, plus short readouts in a corner;
  * colour is restraint: exactly one stroke family changes colour (the accent), and it may
    carry a slow pulse that stops under prefers-reduced-motion.
Light and dark tokens live inside the SVG, as in svgplot.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'tutorial' / 'figures'
COS, SIN = math.cos(math.radians(30)), math.sin(math.radians(30))

TOKENS = {  # role: (light, dark)
    'plate': ('#fcfcfb', '#111113'),
    'hi': ('#18181b', '#fafafa'),      # silhouettes, primary text
    'mid': ('#71717a', '#a1a1aa'),     # creases, leaders, secondary text
    'lo': ('#d4d4d8', '#3f3f46'),      # floor, receding guides
    'accent': ('#2a78d6', '#3987e5'),  # the one stroke that changes colour
}


def css():
    light = ';'.join(f'--{k}:{v[0]}' for k, v in TOKENS.items())
    dark = ';'.join(f'--{k}:{v[1]}' for k, v in TOKENS.items())
    return f"""
svg{{{light};font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
@media (prefers-color-scheme: dark){{svg{{{dark}}}}}
.plate{{fill:var(--plate)}}
.sil{{fill:var(--plate);stroke:var(--hi);stroke-width:.9;stroke-linejoin:round;vector-effect:non-scaling-stroke}}
.crease{{fill:none;stroke:var(--mid);stroke-width:.9;stroke-linecap:round;opacity:.75;vector-effect:non-scaling-stroke}}
.floor{{fill:var(--lo)}}
.guide{{fill:none;stroke:var(--lo);stroke-width:.9;stroke-dasharray:2 4;vector-effect:non-scaling-stroke}}
.track{{fill:none;stroke:var(--mid);stroke-width:.9;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}}
.accent{{fill:none;stroke:var(--accent);stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round}}
.pulse{{fill:none;stroke:var(--accent);stroke-width:2.6;stroke-linecap:round;stroke-dasharray:10 400;
  animation:flow 3.2s linear infinite}}
@keyframes flow{{from{{stroke-dashoffset:410}}to{{stroke-dashoffset:0}}}}
@media (prefers-reduced-motion: reduce){{.pulse{{animation:none;stroke-dasharray:none;opacity:0}}}}
.leader{{fill:none;stroke:var(--mid);stroke-width:.7;vector-effect:non-scaling-stroke}}
.dot{{fill:var(--hi)}}
.num{{fill:var(--mid);font-size:10px;letter-spacing:.08em}}
.name{{fill:var(--hi);font-size:12px;font-weight:600;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.desc{{fill:var(--mid);font-size:11px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.cap{{fill:var(--mid);font-size:10px;letter-spacing:.14em}}
.read{{fill:var(--hi);font-size:11px}}
.tlabel{{fill:var(--mid);font-size:10px}}
.cell{{fill:var(--plate);stroke:var(--mid);stroke-width:.9}}
.cell-prompt{{fill:var(--plate);stroke:var(--mid);stroke-width:.9;stroke-dasharray:1.5 2.5}}
.cell-policy{{fill:var(--plate);stroke:var(--accent);stroke-width:1.4}}
.cell-reward{{fill:var(--plate);stroke:var(--hi);stroke-width:.9}}
.ctext{{fill:var(--hi);font-size:11px}}
.mask{{fill:var(--mid);font-size:10px}}
.mask1{{fill:var(--accent);font-size:10px;font-weight:600}}
.card{{fill:var(--plate);stroke:var(--hi);stroke-width:.9}}
.card-soft{{fill:var(--plate);stroke:var(--mid);stroke-width:.9}}
.card-accent{{fill:var(--plate);stroke:var(--accent);stroke-width:1.4}}
.card-dot{{fill:var(--plate);stroke:var(--mid);stroke-width:.9;stroke-dasharray:1.5 2.5}}
.ctitle{{fill:var(--hi);font-size:12px;font-weight:600;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.cline{{fill:var(--mid);font-size:11px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.cmono{{fill:var(--hi);font-size:11px}}
.chead{{fill:var(--mid);font-size:10px;letter-spacing:.12em}}
.vpos{{fill:var(--accent);font-size:13px;font-weight:600}}
.vneg{{fill:var(--hi);font-size:13px;font-weight:600}}
.vzero{{fill:var(--mid);font-size:13px}}
.ttemplate{{fill:var(--mid);font-size:10px}}
.readk{{fill:var(--mid);font-size:11px}}
"""


class Scene:
    """An isometric scene: world units (x right-down, y left-down, z up) -> screen pixels."""

    def __init__(self, width, height, scale=34, origin=(0, 0)):
        self.W, self.H, self.s, self.o = width, height, scale, origin
        self.layers = []  # (depth, svg) drawn back to front
        self.overlay = []  # text and callouts, always on top

    def p(self, x, y, z=0.0):
        return (self.o[0] + (x - y) * COS * self.s, self.o[1] + (x + y) * SIN * self.s - z * self.s)

    def add(self, depth, svg):
        self.layers.append((depth, svg))

    # ---- primitives -----------------------------------------------------
    @staticmethod
    def rounded(points, r):
        """Closed path through screen points with every corner rounded by r pixels."""
        n, d = len(points), []
        for i in range(n):
            a, b, c = points[i - 1], points[i], points[(i + 1) % n]
            ab, bc = math.dist(a, b), math.dist(b, c)
            k1, k2 = min(r, ab / 2) / ab, min(r, bc / 2) / bc
            p1 = (b[0] + (a[0] - b[0]) * k1, b[1] + (a[1] - b[1]) * k1)
            p2 = (b[0] + (c[0] - b[0]) * k2, b[1] + (c[1] - b[1]) * k2)
            d.append(f'{"M" if i == 0 else "L"}{p1[0]:.1f},{p1[1]:.1f} Q{b[0]:.1f},{b[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}')
        return ' '.join(d) + ' Z'

    @staticmethod
    def uv(u, v):
        """Screen-aligned grid -> world: u runs right on screen, v runs down."""
        return (u + v) / 2, (v - u) / 2

    def solid(self, u, v, a, z0, h, r=5, depth=None):
        x, y = self.uv(u, v)
        return self.box(x - a, y - a, x + a, y + a, z0, h, r=r, depth=depth)

    def path_uv(self, pts, **kw):
        return self.track([self.uv(u, v) for u, v in pts], **kw)

    def floor(self, x0, y0, x1, y1, step=1.0):
        dots = []
        y = y0
        while y <= y1 + 1e-9:
            x = x0
            while x <= x1 + 1e-9:
                sx, sy = self.p(x, y)
                dots.append(f'<circle class="floor" cx="{sx:.1f}" cy="{sy:.1f}" r=".9"/>')
                x += step
            y += step
        self.add(-1e9, '\n'.join(dots))

    def box(self, x0, y0, x1, y1, z0, h, r=5, inset=0.12, depth=None):
        """A rounded isometric solid: bright silhouette, three dim creases at the near corner."""
        z1 = z0 + h
        sil = [self.p(x0, y0, z1), self.p(x1, y0, z1), self.p(x1, y0, z0),
               self.p(x1, y1, z0), self.p(x0, y1, z0), self.p(x0, y1, z1)]
        near = self.p(x1, y1, z1)
        ends = [self.p(x1, y0 + inset, z1), self.p(x0 + inset, y1, z1), self.p(x1, y1, z0 + min(inset, h / 3))]
        creases = ' '.join(f'M{near[0]:.1f},{near[1]:.1f} L{e[0]:.1f},{e[1]:.1f}' for e in ends)
        depth = x1 + y1 + z0 * 0.01 if depth is None else depth
        self.add(depth, f'<path class="sil" d="{self.rounded(sil, r)}"/><path class="crease" d="{creases}"/>')
        return self.p((x0 + x1) / 2, (y0 + y1) / 2, z1)  # top-centre anchor for callouts

    def track(self, pts, z=0.02, cls='track', depth=-1e8, arrow=True):
        """A polyline on (or just above) the floor, with rounded joints and a chevron head."""
        s = [self.p(x, y, z) for x, y in pts]
        d = f'M{s[0][0]:.1f},{s[0][1]:.1f} ' + ' '.join(f'L{x:.1f},{y:.1f}' for x, y in s[1:])
        out = f'<path class="{cls}" d="{d}"/>'
        if arrow:
            (ax, ay), (bx, by) = s[-2], s[-1]
            ang = math.atan2(by - ay, bx - ax)
            w = [(bx - 7 * math.cos(ang - 0.45), by - 7 * math.sin(ang - 0.45)),
                 (bx - 7 * math.cos(ang + 0.45), by - 7 * math.sin(ang + 0.45))]
            out += f'<path class="{cls}" d="M{w[0][0]:.1f},{w[0][1]:.1f} L{bx:.1f},{by:.1f} L{w[1][0]:.1f},{w[1][1]:.1f}"/>'
        self.add(depth, out)
        return d

    def callout(self, anchor, label_xy, num, name, desc, align='start'):
        ax, ay = anchor
        lx, ly = label_xy
        elbow = (lx - (6 if align == 'start' else -6), ly - 4)
        self.overlay.append(
            f'<circle class="dot" cx="{ax:.1f}" cy="{ay:.1f}" r="1.8"/>'
            f'<path class="leader" d="M{ax:.1f},{ay:.1f} L{ax:.1f},{elbow[1]:.1f} L{elbow[0]:.1f},{elbow[1]:.1f}"/>'
            f'<text x="{lx}" y="{ly}" text-anchor="{align}"><tspan class="num">{num}  </tspan>'
            f'<tspan class="name">{html.escape(name)}</tspan></text>'
            f'<text class="desc" x="{lx}" y="{ly + 15}" text-anchor="{align}">{html.escape(desc)}</text>')

    def tlabel(self, x, y, z, s, dx=6, dy=-5, anchor='start'):
        px, py = self.p(x, y, z)
        self.overlay.append(f'<text class="tlabel" x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}">{html.escape(s)}</text>')

    def strip(self, x, y, cells, h=24, gap=4):
        """Flat token strip: cells are (text, kind) with kind in prompt|policy|tool|reward; mask row below."""
        cx = x
        for text, kind in cells:
            w = max(34, 6.7 * len(text) + 16)
            cls = {'prompt': 'cell-prompt', 'policy': 'cell-policy', 'tool': 'cell', 'reward': 'cell-reward'}[kind]
            self.overlay.append(f'<rect class="{cls}" x="{cx:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="6"/>'
                                f'<text class="ctext" x="{cx + w / 2:.1f}" y="{y + h / 2 + 4}" text-anchor="middle">{html.escape(text)}</text>')
            if kind != 'reward':
                m = '1' if kind == 'policy' else '0'
                self.overlay.append(f'<text class="{"mask1" if m == "1" else "mask"}" x="{cx + w / 2:.1f}" y="{y + h + 16}" text-anchor="middle">{m}</text>')
            cx += w + gap
        return cx

    # ---- ports: where a track meets a solid's visible footprint ------------
    @staticmethod
    def port(u, v, half, side, offset=0.0, gap=0.35):
        """Point just outside the footprint diamond |du|+|dv| <= 2*half of a solid centred at (u, v).
        side: 'left'/'right' (offset moves along v) or 'top'/'bottom' (offset moves along u)."""
        r = 2 * half - abs(offset) + gap
        return {'left': (u - r, v + offset), 'right': (u + r, v + offset),
                'top': (u + offset, v - r), 'bottom': (u + offset, v + r)}[side]

    # ---- flat (non-isometric) primitives ------------------------------------
    def head(self, label, subtitle):
        self.text(24, 34, html.escape(label))
        self.text(24, 50, html.escape(subtitle), cls='readk')

    def card(self, x, y, w, h, title=None, lines=(), num=None, style='card', mono=False, align='middle', tip=None):
        t = f'<title>{html.escape(tip)}</title>' if tip else ''
        out = [f'<rect class="{style}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8">{t}</rect>']
        n = (1 if title else 0) + len(lines)
        ty = y + h / 2 - (n - 1) * 7.5 + 4
        tx = {'middle': x + w / 2, 'start': x + 12}[align]
        if title:
            pre = f'<tspan class="num">{num}  </tspan>' if num else ''
            out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{align}">{pre}<tspan class="ctitle">{html.escape(title)}</tspan></text>')
            ty += 15
        for line in lines:
            out.append(f'<text class="{"cmono" if mono else "cline"}" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{align}">{html.escape(line)}</text>')
            ty += 15
        self.overlay.append(''.join(out))

    def farrow(self, pts, cls='track', arrow=True, r=8):
        """Screen-space polyline with rounded joints and a chevron head."""
        d = f'M{pts[0][0]:.1f},{pts[0][1]:.1f}'
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            k1 = min(r, math.dist(a, b) / 2) / math.dist(a, b)
            k2 = min(r, math.dist(b, c) / 2) / math.dist(b, c)
            p1 = (b[0] + (a[0] - b[0]) * k1, b[1] + (a[1] - b[1]) * k1)
            p2 = (b[0] + (c[0] - b[0]) * k2, b[1] + (c[1] - b[1]) * k2)
            d += f' L{p1[0]:.1f},{p1[1]:.1f} Q{b[0]:.1f},{b[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}'
        d += f' L{pts[-1][0]:.1f},{pts[-1][1]:.1f}'
        out = f'<path class="{cls}" d="{d}"/>'
        if arrow:
            (ax, ay), (bx, by) = pts[-2], pts[-1]
            ang = math.atan2(by - ay, bx - ax)
            w = [(bx - 7 * math.cos(ang - 0.45), by - 7 * math.sin(ang - 0.45)),
                 (bx - 7 * math.cos(ang + 0.45), by - 7 * math.sin(ang + 0.45))]
            out += f'<path class="{cls}" d="M{w[0][0]:.1f},{w[0][1]:.1f} L{bx:.1f},{by:.1f} L{w[1][0]:.1f},{w[1][1]:.1f}"/>'
        self.overlay.append(out)
        return d

    def label(self, x, y, s, cls='tlabel', anchor='start'):
        self.overlay.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{html.escape(s)}</text>')

    def wrapped_strip(self, x0, y0, width, cells, mask=None, h=22, gap=3, row_gap=10, size=10):
        """Token cells wrapped into rows. kinds: template|prompt|policy|tool. mask marks action tokens."""
        style = {'prompt': 'cell-prompt', 'policy': 'cell-policy', 'tool': 'cell', 'template': None}
        x, y = x0, y0
        extra = 9 if mask is not None else 0
        for k, (text, kind) in enumerate(cells):
            w = max(20, 6.3 * len(text) + 10)
            if x + w > x0 + width:
                x, y = x0, y + h + row_gap + extra
            cls = style[kind]
            tip = f'<title>{html.escape(kind)}: {html.escape(text)}</title>'
            if cls:
                self.overlay.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="5">{tip}</rect>')
            tcls = 'ttemplate' if kind == 'template' else 'ctext'
            self.overlay.append(f'<text class="{tcls}" x="{x + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" font-size="{size}" '
                                f'text-anchor="middle">{html.escape(text)}</text>')
            if mask is not None and mask[k]:
                self.overlay.append(f'<path class="accent" d="M{x + 3:.1f},{y + h + 4:.1f} H{x + w - 3:.1f}"/>')
            x += w + gap
        return y + h + extra

    def legend_cells(self, x, y, entries):
        """entries: (style, label); style in cell-policy|cell|cell-prompt|template."""
        for cls, label in entries:
            if cls == 'template':
                self.overlay.append(f'<text class="ttemplate" x="{x + 11}" y="{y + 10}" text-anchor="middle">&lt;|·|&gt;</text>')
            else:
                self.overlay.append(f'<rect class="{cls}" x="{x}" y="{y}" width="22" height="12" rx="4"/>')
            self.overlay.append(f'<text class="desc" x="{x + 30}" y="{y + 10}">{html.escape(label)}</text>')
            x += 30 + 7 * len(label) + 26

    def text(self, x, y, s, cls='cap', anchor='start'):
        self.overlay.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>')

    def svg(self, title):
        body = '\n'.join(s for _, s in sorted(self.layers, key=lambda t: t[0]))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="{self.W}" '
                f'height="{self.H}" role="img" aria-label="{title}"><style>{css()}</style>'
                f'<rect class="plate" width="{self.W}" height="{self.H}" rx="10"/>{body}\n{"".join(self.overlay)}</svg>')


def agent_loop():
    """Figure 1.1 in the hairline style, keeping every element of the original: the rollout loop
    (policy, parser, environment, context), the learning chain (trajectory, reward, advantage,
    update), evaluation, weight sync, and a token strip with ownership and the action mask.
    Solids sit on a screen-aligned grid, so flows read horizontally and vertically."""
    sc = Scene(1100, 690, scale=36, origin=(480, 78))
    sc.floor(0, 0, 14, 14)
    L = lambda u, v, z=0.0: sc.p(*sc.uv(u, v), z)
    # Rollout loop (2 x 2) on the left and the learning column on the right.
    policy = [sc.solid(-7, 4, 0.8, 0.3 * k, 0.18, r=4) for k in range(3)][-1]
    parser = sc.solid(0, 4, 0.6, 0.0, 0.4, r=3)
    env = sc.solid(0, 11, 1.0, 0.0, 0.7, r=6)
    sc.solid(0, 11, 0.4, 0.7, 0.25, r=3, depth=100)                    # the file under repair
    context = sc.solid(-7, 11, 0.65, 0.0, 0.35, r=3)
    traj = sc.solid(9, 2.5, 0.6, 0.0, 0.28, r=3)
    reward = sc.solid(9, 7.5, 0.6, 0.0, 0.28, r=3)
    adv = sc.solid(9, 12.5, 0.6, 0.0, 0.28, r=3)
    update = [sc.solid(9, 17.5, 0.8, 0.3 * k, 0.18, r=4) for k in range(3)][-1]
    evaluation = sc.solid(15.5, 17.5, 0.55, 0.0, 0.28, r=3)
    # Tracks. The policy's action tokens are the single accented stroke family.
    action = sc.path_uv([(-5.0, 4), (-1.6, 4)], cls='accent')
    sc.add(-1e8 + 1, f'<path class="pulse" d="{action}"/>')
    sc.path_uv([(0, 5.6), (0, 8.6)])                                    # parsed tool call
    sc.path_uv([(-2.4, 11), (-5.4, 11)])                                # observation tokens
    sc.path_uv([(-7, 9.4), (-7, 6.0)])                                  # next turn conditions on everything
    sc.path_uv([(2.4, 11), (5.0, 11), (5.0, 2.5), (7.4, 2.5)])          # finished trajectories
    sc.path_uv([(9, 4.1), (9, 5.7)])
    sc.path_uv([(9, 9.1), (9, 10.7)])
    sc.path_uv([(9, 14.1), (9, 15.5)])
    sc.path_uv([(11.0, 17.5), (14.0, 17.5)], cls='guide')               # evaluation branch
    sc.path_uv([(7.0, 17.5), (-9.5, 17.5), (-9.5, 4), (-9.0, 4)], cls='guide')  # weight sync
    # Track labels sit beside the lines, never on a solid.
    def tl(u, v, text, dx=0, dy=-8, anchor='middle'):
        x, y = L(u, v)
        sc.overlay.append(f'<text class="tlabel" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">{html.escape(text)}</text>')
    tl(-3.3, 4, 'action tokens')
    tl(0, 7.1, 'tool call:', dx=10, dy=-3, anchor='start')
    tl(0, 7.1, 'read / test / edit / submit', dx=10, dy=10, anchor='start')
    tl(-3.9, 11, 'observation tokens', dy=18)
    tl(-7, 7.7, 'next turn conditions', dx=-10, dy=-3, anchor='end')
    tl(-7, 7.7, 'on all earlier tokens', dx=-10, dy=10, anchor='end')
    tl(-3.5, 7.5, 'repeat until', dy=-2)
    tl(-3.5, 7.5, 'submit or budget', dy=11)
    tl(3.7, 11, 'finished trajectories', dy=18)
    tl(-1.0, 17.5, 'new weights θ → rollout engine', dy=18)
    # Callouts, with the original's wording.
    def at(pt, dx, dy, num, name, desc, align='start'):
        sc.callout(pt, (pt[0] + dx, pt[1] + dy), num, name, desc, align)
    at(policy, -12, -36, '01', 'Policy πθ', 'LLM samples tokens', 'end')
    at(parser, 12, -42, '02', 'Parser', '<call> tool [arg] </call>')
    at(L(0, 13.0), 14, 34, '03', 'Environment', 'repo, tools, sandbox')
    at(L(-7, 12.3), -14, 34, '04', 'Context', 'prompt + history', 'end')
    # In the column, leaders leave from each solid's right corner so they never ride on a flow arrow.
    at(L(10.2, 2.5, 0.28), 26, -10, '05', 'Trajectory', 'token ids, sampling log-probs, action mask')
    at(L(10.2, 7.5, 0.28), 26, -10, '06', 'Reward R(τ)', 'verifier: hidden tests')
    at(L(10.2, 12.5, 0.28), 26, -10, '07', 'Advantage Â', 'reward minus a baseline')
    at(L(10.6, 17.5, 0.78), 26, -30, '08', 'Policy update', '∇ log πθ · Â on masked tokens')
    at(L(15.5, 18.6), -10, 36, '09', 'Evaluation', 'held-out tasks, fixed harness, pass@k', 'end')
    sc.text(24, 34, 'FIG 1.1 · THE AGENTIC RL LOOP')
    sc.text(24, 50, 'rollout on the left · learning on the right', cls='readk')
    # Flat token strip: one trajectory, coloured by owner, with its action mask.
    y0 = 596
    sc.text(24, y0 - 14, 'ONE TRAJECTORY AS TOKENS · ONLY ACCENTED TOKENS CARRY POLICY GRADIENT')
    end = sc.strip(24, y0, [('issue', 'prompt'), ('<call> read </call>', 'policy'), ('source', 'tool'),
                            ('<call> edit swap_minmax </call>', 'policy'), ('edited', 'tool'),
                            ('<call> test </call>', 'policy'), ('PASS x3', 'tool'),
                            ('<call> submit </call>', 'policy'), ('R = 1', 'reward')])
    sc.text(end + 6, y0 + 40, 'action mask', cls='mask')
    lx = 24
    for cls, label in [('cell-policy', 'policy action'), ('cell', 'tool / environment'), ('cell-prompt', 'prompt / template')]:
        sc.overlay.append(f'<rect class="{cls}" x="{lx}" y="{y0 + 54}" width="22" height="12" rx="4"/>'
                          f'<text class="desc" x="{lx + 30}" y="{y0 + 64}">{label}</text>')
        lx += 30 + 7 * len(label) + 28
    return sc.svg('The agentic RL loop')


def repair_mdp(prob):
    """Figure 2.1: the tabular repair MDP. prob(state, action, next) reads make_repair_mdp's P.
    States are solids; transitions are floor tracks labelled action: probability. Only the two
    rewarded submissions are accented."""
    sc = Scene(1000, 540, scale=36, origin=(456, 100))
    sc.floor(-1, -1, 17, 17)
    a = 0.7
    at = {'unread': (-11, 8), 'read': (-5.5, 8), 'edited_ok': (1, 3), 'edited_bad': (1, 13),
          'passed': (7.5, 3), 'failed': (7.5, 13), 'done': (14, 3)}
    top = {name: sc.solid(u, v, a, 0.0, 0.34, r=4) for name, (u, v) in at.items()}
    L = lambda u, v: sc.p(*sc.uv(u, v), 0.0)
    port = lambda n, side, off=0.0, gap=0.35: sc.port(*at[n], a, side, off, gap)
    f = lambda x: f'{x:g}' if x in (0, 1) else f'{x:.1f}'

    def tl(u, v, text, dx=0, dy=-7, anchor='middle'):
        x, y = L(u, v)
        sc.label(x + dx, y + dy, text, anchor=anchor)

    sc.path_uv([port('unread', 'right', 0.3), port('read', 'left', 0.3)])
    tl(-8.25, 8.3, f'read: {f(prob("unread", "read", "read"))}')
    rp = port('read', 'right', 0.3)
    sc.path_uv([rp, (-2.2, rp[1]), (-2.2, 3.3), port('edited_ok', 'left', 0.3)])
    sc.path_uv([(-2.2, rp[1]), (-2.2, 13.3), port('edited_bad', 'left', 0.3)])
    tl(-2.2, 5.2, f'edit: {f(prob("read", "edit", "edited_ok"))}', dx=8, dy=2, anchor='start')
    tl(-2.2, 10.9, f'edit: {f(prob("read", "edit", "edited_bad"))}', dx=8, dy=2, anchor='start')
    sc.path_uv([port('unread', 'top', 0, 0.6), (-11, 0.4), (1, 0.4), port('edited_ok', 'top', 0, 0.9)], cls='guide')
    sc.path_uv([port('unread', 'bottom', 0, 0.35), (-11, 17.2), (-3.0, 17.2), (-3.0, 15.2), (0.3, 15.2), port('edited_bad', 'bottom', -0.7, 0.35)], cls='guide')
    tl(-5, 0.4, f'edit (guess): {f(prob("unread", "edit", "edited_ok"))}')
    tl(-7, 17.2, f'edit (guess): {f(prob("unread", "edit", "edited_bad"))}', dy=16)
    sc.path_uv([port('edited_ok', 'right', 0.3), port('passed', 'left', 0.3)])
    tl(4.25, 3.3, 'test: 1')
    sc.path_uv([port('edited_bad', 'right', 0.2), port('failed', 'left', 0.2)])
    tl(4.25, 13.2, 'test: 1')
    sc.path_uv([port('failed', 'left', 0.95), port('edited_bad', 'right', 0.95)])
    tl(4.25, 13.95, f'edit: {f(prob("failed", "edit", "edited_bad"))}', dy=16)
    sc.path_uv([port('failed', 'top', 0, 0.9), (7.5, 8.0), (0.6, 8.0), port('edited_ok', 'bottom', -0.4)])
    tl(4.0, 8.0, f'edit: {f(prob("failed", "edit", "edited_ok"))}', dy=16)
    sc.path_uv([port('passed', 'right', 0.3), port('done', 'left', 0.3)], cls='accent')
    tl(10.75, 3.3, 'submit: r = 1')
    sc.path_uv([port('edited_ok', 'bottom', 0.5), (1.5, 5.6), (14, 5.6), port('done', 'bottom')], cls='accent')
    tl(9.5, 5.6, 'submit: r = 1', dy=16)
    names = {'unread': 'start', 'read': 'file seen', 'edited_ok': 'file correct', 'edited_bad': 'file wrong',
             'passed': 'tests pass', 'failed': 'tests fail', 'done': 'terminal'}
    for k, name in enumerate(at):
        x, y = top[name]
        if at[name][1] > 10:          # bottom row: label below the solid, on its outer side
            u, v = at[name]
            side = -1 if name == 'edited_bad' else 1
            bx, by = L(u + side * 2 * a, v)
            sc.callout((bx, by), (bx + side * 12, by + 44), f'{k + 1:02d}', name, names[name],
                       'end' if side < 0 else 'start')
        else:
            sc.callout((x, y), (x + 10, y - (66 if name == 'edited_ok' else 48)), f'{k + 1:02d}', name, names[name])
    sc.head('FIG 2.1 · THE REPAIR MDP', 'states of the clamp episode; tracks are labelled action: probability (from make_repair_mdp)')
    for i, line in enumerate(['submit elsewhere → done, r = 0', 'read / test elsewhere → self-loop', 'accent: rewarded submissions']):
        sc.text(976, 470 + i * 17, html.escape(line), cls='readk', anchor='end')
    return sc.svg('The repair MDP')


def env_state_machine():
    """Figure 4.1: the BugFix environment as a state machine. The accent marks the one path
    on which reward is computed: submit, hidden-test grading, done."""
    sc = Scene(1040, 590, scale=31, origin=(410, 150))
    sc.floor(-3, -3, 18, 18)
    a = 0.65
    at = {'reset': (-12.5, 9), 'await': (-7, 9), 'parse': (-1.5, 9), 'bad': (5, 2.5), 'tools': (5, 9),
          'submit': (5, 16), 'budget': (11, 5.75), 'done_budget': (18, 5.75), 'done_submit': (11, 16)}
    top = {n: sc.solid(u, v, a, 0.0, 0.34, r=4) for n, (u, v) in at.items()}
    L = lambda u, v: sc.p(*sc.uv(u, v), 0.0)
    port = lambda n, side, off=0.0, gap=0.35: sc.port(*at[n], a, side, off, gap)

    def tl(u, v, text, dx=0, dy=-7, anchor='middle'):
        x, y = L(u, v)
        sc.label(x + dx, y + dy, text, anchor=anchor)

    sc.path_uv([port('reset', 'right', 0.3), port('await', 'left', 0.3)])
    sc.path_uv([port('await', 'right', 0.3), port('parse', 'left', 0.3)])
    tl(-4.25, 9.3, 'text')
    pr = port('parse', 'right', 0.3)
    sc.path_uv([pr, (1.6, pr[1]), (1.6, 2.8), port('bad', 'left', 0.3)])
    sc.path_uv([(1.6, pr[1]), port('tools', 'left', 0.3)])
    sc.path_uv([(1.6, pr[1]), (1.6, 16.3), port('submit', 'left', 0.3)], cls='accent')
    tl(1.6, 4.6, 'None', dx=8, dy=2, anchor='start')
    tl(1.6, 13.2, 'submit', dx=8, dy=2, anchor='start')
    sc.path_uv([port('bad', 'right', 0.3), (8.4, 2.8), (8.4, 6.05), port('budget', 'left', 0.3)])
    sc.path_uv([port('tools', 'right', 0.3), (8.4, 9.3), (8.4, 6.05)], arrow=False)
    sc.path_uv([port('budget', 'right', 0.3), port('done_budget', 'left', 0.3)])
    tl(14.5, 6.05, 'yes')
    bt = port('budget', 'top', 0.7, 0.9)
    sc.path_uv([bt, (bt[0], -3.2), (-7, -3.2), port('await', 'top', 0, 0.9)])
    tl(2, -3.2, 'no: next turn')
    sc.path_uv([port('submit', 'right', 0.3), port('done_submit', 'left', 0.3)], cls='accent')
    tl(8, 16.3, 'grade', dy=-7)
    sc.add(-1e8 + 1, f'<path class="pulse" d="{sc.path_uv([port("submit", "right", 0.3), port("done_submit", "left", 0.3)], cls="accent", arrow=False)}"/>')
    text = {'reset': ('reset(task)', 'issue text'), 'await': ('awaiting call', 'turn k'),
            'parse': ('parse_call', 'grammar check'), 'bad': ('malformed / unknown', 'error observation'),
            'tools': ('read / test / edit', 'tool observation'), 'submit': ('submit', 'grade on hidden tests'),
            'budget': ('turn == max_turns?', 'budget check'), 'done_budget': ('done: stop=budget', 'reward 0'),
            'done_submit': ('done: stop=submit', 'reward = reward_fn(info)')}
    below = {'tools', 'submit', 'done_submit', 'reset', 'parse', 'done_budget', 'budget'}
    for k, n in enumerate(at):
        name, desc = text[n]
        u, v = at[n]
        if n in below:
            bx, by = L(u, v + 2 * a)
            left = n == 'parse'
            sc.callout((bx, by), (bx - 10 if left else bx + 10, by + 38), f'{k + 1:02d}', name, desc,
                       'end' if left else 'start')
        else:
            x, y = top[n]
            sc.callout((x, y), (x + 10, y - 44), f'{k + 1:02d}', name, desc)
    sc.head('FIG 4.1 · THE BUGFIX ENVIRONMENT AS A STATE MACHINE', 'every call consumes one turn; only submit reaches the hidden-test grader')
    for i, line in enumerate(['info always records public_pass and hidden_pass,', 'so the reward can be swapped without touching the tools',
                              'accent: the one path that computes reward']):
        sc.text(1016, 530 + i * 17, html.escape(line), cls='readk', anchor='end')
    return sc.svg('The BugFix environment as a state machine')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'agentic-rl-loop.svg').write_text(agent_loop())
    print(OUT / 'agentic-rl-loop.svg')
