"""Figures for Chapter 9: shortcut diagnosis, single-field ablations with seed bands, dynamic-sampling cost.

Every run uses the chapter's configurations and seeds, so plotted numbers match the printed ones.
"""
import copy

import torch

import agentrl as arl
from svgplot import Figure

_CACHE = {}
ARMS = {'default': {}, 'no std': dict(adv_mode='mean'), 'seq_mean_token_mean': dict(loss_mode='seq_mean_token_mean'),
        'clip-higher 0.28': dict(clip_high=0.28), 'dynamic sampling': dict(dynamic_sampling=True)}


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        _CACHE.update(tok=tok, train=arl.make_tasks('train'), heldout=arl.make_tasks('heldout'),
                      base=arl.make_base_agent(tok), rl_base=arl.make_rl_base(tok))
    return _CACHE


def _ablation():
    c = _setup()
    if 'runs' not in c:
        runs = {}
        for arm, fields in ARMS.items():
            for seed in [0, 1, 2]:
                cfg, model = arl.RLConfig(seed=seed, **fields), copy.deepcopy(c['rl_base'])
                runs[arm, seed] = arl.train_rl(model, c['train'], c['tok'], cfg, steps=40)
        c['runs'] = runs
    return c['runs']


def shortcuts():
    c = _setup()
    original = arl.RLConfig(rename=False, max_edits=None)
    settings = [('base, original', c['base'], original), ('base, renamed', c['base'], arl.RLConfig(max_edits=None)),
                ('base, one edit', c['base'], arl.RLConfig()), ('RL base, default', c['rl_base'], arl.RLConfig()),
                ('RL base, original', c['rl_base'], original)]
    held, reads, right = [], [], []
    for _, start, cfg in settings:
        model = copy.deepcopy(start)
        arl.train_rl(model, c['train'], c['tok'], cfg, steps=80)
        s = arl.evaluate(model, c['heldout'], c['tok'], 16, generator=torch.Generator().manual_seed(7), cfg=cfg)
        held.append(s['pass_rate'])
        reads.append(s['reads'])
        right.append(s['first_edit_right'])
    fig = Figure(1000, 420, title='What GRPO learned in five settings (held-out tasks, 80 steps)',
                 subtitle='Starting policy x environment; dashed line: a blind single guess (0.20). Only the RL base agent '
                          'learns to read.')
    ax = fig.axes(0, 0, 1, 1, ylabel='per held-out episode', ylim=(0, 1.1))
    ax.bars([n for n, _, _ in settings], [('pass rate', held, 1), ('reads', reads, 2),
                                         ('first edit right', right, 3)], value_labels=True, fmt_value='{:.2f}')
    ax.hline(0.2)
    ax.legend()
    fig.save('ch09-shortcuts')


def ablations():
    runs = _ablation()
    steps = list(range(40))
    fig = Figure(1000, 620, title='Single-field ablations of the default GRPO (3 seeds, 40 steps)',
                 subtitle='Training batch reward; line = mean of seeds, band = min to max. Tiny-scale mechanism demo.')
    others = [a for a in ARMS if a != 'default']

    def stats(arm):
        curves = [[h['reward'] for h in runs[arm, s]] for s in range(3)]
        smooth = [[sum(cv[max(0, i - 4):i + 1]) / len(cv[max(0, i - 4):i + 1]) for i in steps] for cv in curves]
        return ([sum(x) / 3 for x in zip(*smooth)], [min(x) for x in zip(*smooth)], [max(x) for x in zip(*smooth)])
    base_mean, base_lo, base_hi = stats('default')
    for k, arm in enumerate(others):
        ax = fig.axes(0.5 * (k % 2), 0.5 * (k // 2), 0.5, 0.5, xlabel='step', ylabel='batch reward (5-step mean)',
                      ylim=(0, 1), xlim=(0, 39), title=arm)
        mean, lo, hi = stats(arm)
        ax.band(steps, base_lo, base_hi, series=2)
        ax.line(steps, base_mean, label='default', series=2, end_label=False)
        ax.band(steps, lo, hi, series=1)
        ax.line(steps, mean, label=arm, series=1, end_label=False)
        ax.legend()
    fig.save('ch09-ablations')


def dynamic_sampling():
    runs = _ablation()
    steps = list(range(40))
    zero = [sum(runs['default', s][i]['zero_adv_frac'] for s in range(3)) / 3 for i in steps]
    fig = Figure(1000, 400, title='Dynamic sampling: a fuller batch, paid for in rollouts',
                 subtitle='Mean of 3 seeds. Right: training batch reward (5-step mean) against episodes sampled so far.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='step', ylabel='share of zero-advantage groups', ylim=(0, 1), xlim=(0, 39))
    left.line(steps, zero, label='default', series=1, end_label=False)
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='episodes sampled (thousands)', ylabel='batch reward', ylim=(0, 1))
    for series, arm in [(2, 'default'), (1, 'dynamic sampling')]:
        hists = [runs[arm, s] for s in range(3)]
        cum = [sum(sum(h[j]['n_episodes'] for j in range(i + 1)) for h in hists) / 3 / 1000 for i in steps]
        rew = [sum(h[i]['reward'] for h in hists) / 3 for i in steps]
        smooth = [sum(rew[max(0, i - 4):i + 1]) / len(rew[max(0, i - 4):i + 1]) for i in steps]
        right.line(cum, smooth, label=arm, series=series, end_label=False)
    right.legend()
    fig.save('ch09-dynamic-sampling')


FIGURES = {'ch09-shortcuts': shortcuts, 'ch09-ablations': ablations, 'ch09-dynamic-sampling': dynamic_sampling}
