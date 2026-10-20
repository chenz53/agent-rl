"""Figures for Chapter 3: temperature, Qwen3 token log-prob strips, bf16 vs fp32 gap."""
import math

import torch

import agentrl as arl
from svgplot import Figure, SEQ, esc


def temperature():
    fig = Figure(760, 340, title='Temperature reshapes the behavior policy',
                 subtitle='Left: entropy vs temperature for logits of three scales (V = 8). Right: one distribution at three temperatures.')
    gen = torch.Generator().manual_seed(3)
    base = torch.randn(8, generator=gen).sort(descending=True).values
    temps = [0.05 * k for k in range(1, 81)]
    ax = fig.axes(0, 0, 0.5, 1, xlabel='temperature', ylabel='entropy (nats)', ylim=(0, 2.2))
    for scale, slot in [(1.0, 1), (3.0, 2), (8.0, 3)]:
        ent = [arl.entropy_from_logits(scale * base / t).item() for t in temps]
        ax.line(temps, ent, label=f'scale {scale:g}', series=slot, end_label=False)
    ax.hline(math.log(8), label='log V')
    ax.legend()
    ax2 = fig.axes(0.5, 0, 0.5, 1, xlabel='token rank', ylabel='probability', ylim=(0, 0.8))
    groups = []
    for t, slot in [(0.5, 1), (1.0, 2), (2.0, 3)]:
        groups.append((f'T = {t:g}', (3.0 * base / t).softmax(-1).tolist(), slot))
    ax2.bars([str(i + 1) for i in range(8)], groups)
    ax2.legend()
    fig.save('ch03-temperature')


def _strip(fig, x, y, pieces, values, vmax=8.0, h=30):
    cx = x
    for text, v in zip(pieces, values):
        w = max(30, 7.4 * len(text) + 12)
        level = min(len(SEQ) - 1, int(min(1.0, -v / vmax) * (len(SEQ) - 1) + 0.5))
        dark = level > 6
        fig.body.append(f'<rect x="{cx:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{SEQ[level]}">'
                        f'<title>{esc(text)}: log-prob {v:.2f}</title></rect>')
        fig.text(cx + w / 2, y + 13, text, anchor='middle', size=11, mono=True,
                 fill='#ffffff' if dark else '#0b0b0b')
        fig.text(cx + w / 2, y + 25, f'{v:.1f}', anchor='middle', size=9,
                 fill='#ffffff' if dark else '#52514e')
        cx += w + 2
    return cx


def qwen_logp_strip():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval()
    rows = {'buggy': 'def clamp(value, lower, upper):\n    return min(lower, max(value, upper))\n',
            'fixed': 'def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n'}
    fig = Figure(1000, 300, title='Qwen3-0.6B log-probability of each token of clamp',
                 subtitle='Cells show log pi(token | prefix); darker = less likely. Ġ (space) shown as ·, Ċ (newline) as ⏎.')
    y = 90
    for name, text in rows.items():
        ids = tok(text, return_tensors='pt').input_ids
        with torch.no_grad():
            lp = arl.selected_log_probs(model(ids).logits[:, :-1], ids[:, 1:])[0].tolist()
        pieces = [p.replace('Ġ', '·').replace('Ċ', '⏎') for p in tok.convert_ids_to_tokens(ids[0, 1:])]
        fig.text(20, y + 20, f'{name}', size=12, weight=600)
        fig.text(20, y + 36, f'sum {sum(lp):.1f}', cls='t2', size=11)
        # split the strip into signature and body lines for width
        _strip(fig, 90, y, pieces[:8], lp[:8])
        _strip(fig, 90, y + 36, pieces[8:], lp[8:])
        y += 92
    fig.text(90, 284, 'first token "def" has no prefix and is not scored', cls='t2', size=11)
    fig.save('ch03-qwen-logp-strip')


def bf16_gap():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    code = '\n'.join(f'def f{i}(x, y):\n    total = 0\n    for k in range(x, y + {i}):\n'
                     f'        total += k * {i}\n    return max(total, {i})\n' for i in range(12))
    ids = tok(code, return_tensors='pt').input_ids
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=dtype).eval()
        with torch.no_grad():
            out[dtype] = arl.selected_log_probs(model(ids).logits[:, :-1], ids[:, 1:])[0]
    lp32, diff = out[torch.float32], out[torch.bfloat16] - out[torch.float32]
    fig = Figure(720, 360, title='bf16 vs fp32 log-probabilities, same weights',
                 subtitle=f'Qwen3-0.6B on {len(diff)} code tokens; one dot per token')
    ax = fig.axes(0, 0, 1, 1, xlabel='fp32 log-probability of the token', ylabel='bf16 minus fp32 (nats)',
                  ylim=(-0.15, 0.15), xlim=(min(lp32.tolist()) - 0.2, 0.2))
    tips = [f'logp {a:.2f}, gap {b:+.3f}' for a, b in zip(lp32.tolist(), diff.tolist())]
    ax.scatter(lp32.tolist(), diff.tolist(), series=1, r=3, tips=tips)
    ax.hline(0)
    fig.save('ch03-bf16-gap')


FIGURES = {'ch03-temperature': temperature, 'ch03-qwen-logp-strip': qwen_logp_strip, 'ch03-bf16-gap': bf16_gap}
