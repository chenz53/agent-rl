"""Figures for Chapter 14: padding vs packing from real episodes, memory breakdown, pass-rate filtering.

Packing and filtering use the chapter's own settings and saved functions, so drawn layouts and curves
match the printed numbers. The memory breakdown is arithmetic under assumptions stated in its subtitle.
"""
import copy

import torch

torch.set_num_threads(4)

import agentrl as arl
from svgplot import Figure

_CACHE = {}


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        base = arl.make_rl_base(tok)
        cfg = arl.RLConfig()
        trajs = arl.sample_groups(base, lambda: arl.make_env(cfg), arl.make_tasks('train')[:8], tok, cfg,
                                  torch.Generator().manual_seed(0))
        _CACHE.update(tok=tok, base=base, cfg=cfg, trajs=trajs)
    return _CACHE


def _row(fig, x, y, owners, px, h):
    """Draw one row of tokens as runs: policy (blue), other real tokens (aqua), padding (neutral)."""
    run_start = 0
    for k in range(1, len(owners) + 1):
        if k == len(owners) or owners[k] != owners[run_start]:
            kind = owners[run_start]
            fill = {'policy': 'var(--s1)', 'other': 'var(--s3)', 'pad': 'var(--neutral)'}[kind]
            op = '1' if kind != 'other' else '0.55'
            fig.body.append(f'<rect x="{x + run_start * px:.1f}" y="{y:.1f}" width="{(k - run_start) * px:.1f}" '
                            f'height="{h - 1:.1f}" fill="{fill}" fill-opacity="{op}"/>')
            run_start = k


def packing():
    c = _setup()
    trajs = c['trajs']
    T = max(len(t.ids) for t in trajs)
    packed = arl.pack_sequences(trajs, 256, c['tok'].pad_id)
    px, h = 2.2, 6
    real = sum(len(t.ids) for t in trajs)
    fig = Figure(1000, 520, title='Padding vs packing for the same 64 TinyAgent episodes',
                 subtitle=f'Equal area = equal compute. Padded: {len(trajs)} x {T} = {len(trajs) * T:,} positions; '
                          f'packed: {len(packed["rows"])} x 256 = {len(packed["rows"]) * 256:,}; real tokens: {real:,}.')
    x0, y0 = 40, 110
    fig.text(x0, y0 - 14, f'padded batch [{len(trajs)}, {T}]', cls='t', size=13, weight=600)
    for r, t in enumerate(trajs):
        owners = ['policy' if o == arl.OWNER['policy'] else 'other' for o in t.owner.tolist()]
        _row(fig, x0, y0 + r * h, owners + ['pad'] * (T - len(owners)), px, h)
    x1 = x0 + T * px + 50
    fig.text(x1, y0 - 14, f'packed rows [{len(packed["rows"])}, 256], block-diagonal mask', cls='t', size=13,
             weight=600)
    for r, members in enumerate(packed['rows']):
        owners = []
        for i in members:
            owners += ['policy' if o == arl.OWNER['policy'] else 'other' for o in trajs[i].owner.tolist()]
        _row(fig, x1, y0 + r * h, owners + ['pad'] * (256 - len(owners)), px, h)
        offset = 0
        for i in members[:-1]:                                    # sequence boundaries inside a row
            offset += len(trajs[i].ids)
            fig.body.append(f'<line x1="{x1 + offset * px:.1f}" x2="{x1 + offset * px:.1f}" y1="{y0 + r * h - 1:.1f}" '
                            f'y2="{y0 + (r + 1) * h:.1f}" stroke="var(--ink)" stroke-width="1.5"/>')
    note_y = y0 + len(packed['rows']) * h + 30
    fig.text(x1, note_y, 'Dark ticks mark sequence boundaries. Each sequence restarts its', cls='t2', size=12)
    fig.text(x1, note_y + 17, 'position ids at 0 and attends only within its own block.', cls='t2', size=12)
    fig.legend_row(x1, note_y + 60, [('policy tokens', 1, 'box'), ('prompt, template, tool', 3, 'box'),
                                     ('padding', 'var(--neutral)', 'box')])
    fig.save('ch14-packing')


def memory():
    N, n_gpu, T, V, D, layers = 8.2e9, 8, 32_768, 151_936, 4096, 36
    base = {'parameters (bf16)': 2 * N / n_gpu, 'gradients (fp32)': 4 * N / n_gpu,
            'Adam + master (fp32)': 12 * N / n_gpu, 'activations (checkpointed)': 2 * D * T * layers + 16 * D * T}
    logits = {'full logits': 12 * T * V, 'chunked': 4.5 * T * V + 12 * 1024 * V, 'chunked + recompute': 12 * 1024 * V}
    fig = Figure(1000, 360, title='Per-GPU memory of one RL microbatch: Qwen3-8B, 32k tokens, FSDP over 8 GPUs',
                 subtitle='Arithmetic, not a measurement. Logit bytes per element (12 full, 4.5 stored when chunked) '
                          'are taken from the Section 14.3 measurement on Qwen3-0.6B.')
    x0, y0, scale, bh = 210, 110, 6.0, 34                      # px per GB
    parts = list(base) + ['logits']
    for r, (name, lg) in enumerate(logits.items()):
        y = y0 + r * (bh + 26)
        fig.text(x0 - 10, y + bh / 2 + 4, name, cls='t2', size=12, anchor='end')
        x = x0
        for k, part in enumerate(parts):
            gb = (base[part] if part != 'logits' else lg) / 1e9
            fig.body.append(f'<rect x="{x:.1f}" y="{y}" width="{gb * scale:.1f}" height="{bh}" fill="var(--s{k + 1})" '
                            f'stroke="var(--surface)" stroke-width="1"><title>{part}: {gb:.1f} GB</title></rect>')
            x += gb * scale
        total = sum(base.values()) / 1e9 + lg / 1e9
        fig.text(x + 8, y + bh / 2 + 4, f'{total:.0f} GB (logits {lg / 1e9:.1f})', cls='t', size=12)
    for gb in range(0, 121, 20):
        fig.body.append(f'<line class="grid" x1="{x0 + gb * scale}" x2="{x0 + gb * scale}" y1="{y0 - 8}" '
                        f'y2="{y0 + 3 * (bh + 26) - 20}"/>')
        fig.text(x0 + gb * scale, y0 + 3 * (bh + 26) - 4, f'{gb} GB', cls='m', size=10, anchor='middle')
    fig.legend_row(40, 330, [(p.split(' (')[0], k + 1, 'box') for k, p in enumerate(parts)])
    fig.save('ch14-memory')


def filtering():
    c = _setup()
    train, held = arl.make_tasks('train'), arl.make_tasks('heldout')
    arms = {'all tasks': (-1.0, 2.0), 'filter (0, 1)': (0.0, 1.0)}
    seeds = [0, 1, 2, 3]
    runs, finals = {}, {}
    for arm, (lo, hi) in arms.items():
        for s in seeds:
            model = copy.deepcopy(c['base'])
            runs[arm, s] = arl.train_curriculum(model, train, c['tok'], arl.RLConfig(seed=s), 60, lo, hi)
            finals[arm, s] = arl.evaluate(model, held, c['tok'], 16, generator=torch.Generator().manual_seed(s),
                                          cfg=c['cfg'])['pass_rate']
    steps = list(range(1, 61))
    fig = Figure(1000, 400, title='Pass-rate filtering: similar held-out pass rate for fewer rollouts',
                 subtitle='TinyAgent, default GRPO, 60 steps, 4 seeds per arm. Tiny-scale mechanism demonstration.')
    ax = fig.axes(0, 0, 0.5, 1, xlabel='step', ylabel='cumulative episodes sampled', xlim=(1, 60))
    bx = fig.axes(0.5, 0, 0.5, 1, xlabel='episodes sampled in 60 steps', ylabel='held-out pass rate',
                  ylim=(0, 1), xlim=(1500, 4200))
    for k, arm in enumerate(arms):
        cums = []
        for s in seeds:
            total, cum = 0, []
            for h in runs[arm, s]:
                total += h['n_episodes']
                cum.append(total)
            cums.append(cum)
        ax.band(steps, [min(x) for x in zip(*cums)], [max(x) for x in zip(*cums)], series=k + 1)
        ax.line(steps, [sum(x) / len(seeds) for x in zip(*cums)], label=arm, series=k + 1)
        bx.scatter([cums[i][-1] for i in range(len(seeds))], [finals[arm, s] for s in seeds], label=arm, series=k + 1,
                   r=6)
    bx.hline(0.2, label='blind single guess')
    ax.legend()
    bx.legend()
    fig.save('ch14-filtering')


FIGURES = {'packing': packing, 'memory': memory, 'filtering': filtering}
