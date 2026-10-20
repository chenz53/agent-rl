"""Figures for Chapter 17: pass@k curves, checkpoint and seed intervals, harness drift, interval coverage.

Runs reuse the chapter's seeds and generators, so plotted numbers match the printed ones.
"""
import copy
import dataclasses

import torch

import agentrl as arl
from svgplot import Figure

torch.set_num_threads(4)
_CACHE = {}
KS = list(range(1, 33))


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        train, held = arl.make_tasks('train'), arl.make_tasks('heldout')
        rl_base = arl.make_rl_base(tok)
        _CACHE.update(tok=tok, train=train, held=held, rl_base=rl_base, cfg=arl.RLConfig())
    return _CACHE


def _snapshots(seed, steps):
    c = _setup()
    key = ('run', seed, steps)
    if key not in c:
        model, snaps = copy.deepcopy(c['rl_base']), {0: c['rl_base']}
        for chunk in range(steps // 10):
            arl.train_rl(model, c['train'], c['tok'], arl.RLConfig(seed=100 * seed + chunk), steps=10)
            snaps[10 * (chunk + 1)] = copy.deepcopy(model)
        c[key] = snaps
    return c[key]


def _counts(model, tasks, n, seed, harness=None):
    c = _setup()
    return arl.sample_counts(model, tasks, c['tok'], n, harness or c['cfg'], torch.Generator().manual_seed(seed))


def _whisker(ax, x, y, lo, hi, series, label=None):
    ax.line([x, x], [lo, hi], series=series, end_label=False, width=2)
    ax.scatter([x], [y], series=series, label=label, r=5,
               tips=[f'{label or ""} {y:.3f} [{lo:.3f}, {hi:.3f}]'])


def passk():
    c = _setup()
    run = _snapshots(0, 80)
    models = [('RL base', c['rl_base'], 2), ('GRPO, 20 steps', run[20], 3), ('GRPO, 80 steps', run[80], 1)]
    counts = {name: _counts(m, c['held'], 64, 6) for name, m, _ in models}
    fig = Figure(1000, 420, title='pass@k on held-out tasks: the base agent catches up by k = 32',
                 subtitle='Unbiased estimator from 64 episodes per task, 10 tasks. Dashed: a uniform guess among the '
                          '5 repair ops. Band: task-level bootstrap 95% interval.')
    left = fig.axes(0, 0, 0.55, 1, xlabel='k (attempts)', ylabel='pass@k', xlog=True, xlim=(1, 32), ylim=(0, 1.02),
                    xticks=[1, 2, 4, 8, 16, 32], xticklabels=['1', '2', '4', '8', '16', '32'], title='mean over tasks')
    left.line(KS, [1 - 0.8 ** k for k in KS], label='uniform guess', series='var(--muted)', dashed=True,
              end_label=False)
    for name, _, s in models:
        per_task = [arl.pass_at_k(64, counts[name], k) for k in KS]            # each [10]
        if name != 'GRPO, 20 steps':
            ci = [arl.bootstrap_ci(v) for v in per_task]
            left.band(KS, [a for a, _ in ci], [b for _, b in ci], series=s)
        left.line(KS, [v.mean().item() for v in per_task], label=name, series=s, end_label=False)
    left.legend()
    hard = int(counts['GRPO, 80 steps'].argmin())
    name = c['held'][hard].name
    right = fig.axes(0.55, 0, 0.45, 1, xlabel='k (attempts)', ylabel='pass@k', xlog=True, xlim=(1, 32),
                     ylim=(0, 1.02), xticks=[1, 2, 4, 8, 16, 32], xticklabels=['1', '2', '4', '8', '16', '32'],
                     title=f'{name}: successes of 64 = '
                           + ', '.join(str(int(counts[m][hard])) for m, _, _ in models))
    for (model_name, _, s), short in zip(models, ['base', 'step 20', 'step 80']):
        right.line(KS, [arl.pass_at_k(64, int(counts[model_name][hard]), k) for k in KS], label=short, series=s,
                   end_label=False)
    right.legend()
    fig.save('ch17-passk')


def checkpoints():
    c = _setup()
    run = _snapshots(0, 80)
    val, test = c['held'][0::2], c['held'][1::2]
    fig = Figure(1000, 440, title='Intervals for checkpoints and seeds (held-out pass@1)',
                 subtitle='Left: one run, 5 validation and 5 test tasks, 32 episodes each, task-bootstrap 95% intervals. '
                          'Right: 4 seeds at step 30, naive vs task-level interval.')
    steps = list(range(10, 90, 10))
    left = fig.axes(0, 0, 0.6, 1, xlabel='checkpoint (training step)', ylabel='pass@1', ylim=(0, 1.05),
                    xlim=(5, 85), xticks=steps, title='checkpoints of seed 0')
    for offset, tasks, s, label, seed in [(-1.2, val, 3, 'validation', 7), (1.2, test, 2, 'test', 7)]:
        for i, step in enumerate(steps):
            rates = _counts(run[step], tasks, 32, seed).double() / 32
            lo, hi = arl.bootstrap_ci(rates)
            _whisker(left, step + offset, rates.mean().item(), lo, hi, s, label if i == 0 else None)
    left.legend()
    seeds = [run[30]] + [_snapshots(s, 30)[30] for s in [1, 2, 3]]
    right = fig.axes(0.6, 0, 0.4, 1, xlabel='training seed', ylabel='pass@1', ylim=(0, 1.05), xlim=(-0.6, 3.6),
                     xticks=[0, 1, 2, 3], title='step 30, four seeds')
    for i, m in enumerate(seeds):
        cnt = _counts(m, c['held'], 32, 1000 * i)
        episodes = (torch.arange(32) < cnt.unsqueeze(1)).flatten()
        mean = cnt.sum().item() / 320
        lo, hi = arl.bootstrap_ci(episodes)
        _whisker(right, i - 0.15, mean, lo, hi, 2, 'resample episodes' if i == 0 else None)
        lo, hi = arl.bootstrap_ci(cnt.double() / 32)
        _whisker(right, i + 0.15, mean, lo, hi, 1, 'resample tasks' if i == 0 else None)
    right.legend()
    fig.save('ch17-checkpoints')


SETTINGS = {'training': {}, 'greedy': dict(temperature=0.0), 'T = 0.7': dict(temperature=0.7),
            'T = 1.3': dict(temperature=1.3), '4 turns': dict(max_turns=4), '3 turns': dict(max_turns=3),
            'no rename': dict(rename=False), 'any edits': dict(max_edits=None)}


def drift():
    c = _setup()
    run = _snapshots(0, 80)
    fig = Figure(1000, 420, title='Harness drift: one checkpoint, eight harnesses',
                 subtitle='Held-out pass@1, 32 episodes per task; whiskers: task-bootstrap 95% interval. Only the '
                          'harness changes between columns.')
    labels = list(SETTINGS)
    ax = fig.axes(0, 0, 1, 1, ylabel='held-out pass@1', ylim=(0, 1.05), xlim=(-0.5, len(labels) - 0.5),
                  xticks=list(range(len(labels))), xticklabels=labels)
    for i, label in enumerate(labels):
        harness = dataclasses.replace(c['cfg'], **SETTINGS[label])
        for dx, (name, m, s) in zip([-0.12, 0.12], [('RL base', c['rl_base'], 2), ('GRPO, 80 steps', run[80], 1)]):
            rates = _counts(m, c['held'], 32, 3, harness).double() / 32
            lo, hi = arl.bootstrap_ci(rates)
            _whisker(ax, i + dx, rates.mean().item(), lo, hi, s, name if i == 0 else None)
    ax.hline(0.2, label='blind single guess')
    ax.legend()
    fig.save('ch17-drift')


def coverage():
    torch.manual_seed(0)
    N, sims, mu = 10, 300, 0.4 / 0.7
    ns = [1, 2, 4, 8, 16, 32, 64]
    cover = {'episodes': [], 'tasks': []}
    for n in ns:
        hits = {'episodes': 0, 'tasks': 0}
        for s in range(sims):
            p_task = torch.distributions.Beta(0.4, 0.3).sample((N,))
            outcomes = torch.bernoulli(p_task.unsqueeze(1).expand(N, n))
            gen = torch.Generator().manual_seed(s)
            for unit, values in [('episodes', outcomes.flatten()), ('tasks', outcomes.mean(1))]:
                lo, hi = arl.bootstrap_ci(values, n_boot=500, generator=gen)
                hits[unit] += lo <= mu <= hi
        for unit in cover:
            cover[unit].append(hits[unit] / sims)
    fig = Figure(760, 400, title='Coverage of a nominal 95% interval for the population pass rate',
                 subtitle='Simulation: 10 tasks with pass rates from Beta(0.4, 0.3), n episodes each, 300 evaluations '
                          'per point.')
    ax = fig.axes(0, 0, 1, 1, xlabel='episodes per task (n)', ylabel='coverage', xlog=True, xlim=(1, 64),
                  ylim=(0, 1), xticks=ns, xticklabels=[str(n) for n in ns])
    ax.hline(0.95, label='nominal 0.95')
    ax.line(ns, cover['tasks'], label='resample tasks', series=1, markers=True, end_label=False)
    ax.line(ns, cover['episodes'], label='resample episodes', series=2, markers=True, end_label=False)
    ax.legend()
    fig.save('ch17-coverage')


FIGURES = {'ch17-passk': passk, 'ch17-checkpoints': checkpoints, 'ch17-drift': drift, 'ch17-coverage': coverage}
