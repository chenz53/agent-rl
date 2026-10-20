"""Figures for Chapter 11: the reward-hacking dashboard, flaky-test learning curves, and the reward-design matrix.

Runs use the chapter's configurations and seeds, so plotted numbers match the printed ones.
"""
import torch

import agentrl as arl
from svgplot import Figure

torch.set_num_threads(4)
_CACHE = {}


def _chapter():
    """Execute the chapter's code blocks once (stdout discarded) and return its namespace."""
    if not _CACHE:
        import contextlib, io, re
        from pathlib import Path
        text = (Path(__file__).resolve().parents[2] / 'tutorial/chapters/11-rewards-and-verifiers.md').read_text()
        namespace = {'__name__': 'figure'}
        with contextlib.redirect_stdout(io.StringIO()):
            for code in re.findall(r'```python\n(.*?)\n```', text, re.S):
                exec(code, namespace)
        _CACHE.update(namespace)
    return _CACHE


def _smooth(values, k=3):
    return [sum(values[max(0, i - k + 1):i + 1]) / len(values[max(0, i - k + 1):i + 1]) for i in range(len(values))]


def hacking():
    c = _chapter()
    history, steps = c['hack_history'], c['hack_steps']
    pen_steps = c['log_by_step'](c['penalty_history'], c['penalty_log'], 40)
    xs = list(range(len(history)))
    dash = [arl.reward_dashboard(infos) for infos in steps]
    pen = [arl.reward_dashboard(infos) for infos in pen_steps]
    fig = Figure(860, 560, title='Reward hacking: the training reward climbs, the hidden pass rate does not',
                 subtitle='Public-test reward, RL base agent, 64 episodes per step (3-step moving average). '
                          'Tiny-scale mechanism demonstration.')
    top = fig.axes(0, 0, 1, 0.55, ylabel='pass rate of the batch', ylim=(0, 1), xlim=(0, 39), title='What is rewarded vs. what is wanted')
    top.line(xs, _smooth([d['public'] for d in dash]), label='public tests (training reward)', series=1, end_label=False)
    top.line(xs, _smooth([d['hidden'] for d in dash]), label='hidden tests', series=8, end_label=False)
    evals = [(h['step'], h['eval_pass']) for h in history if 'eval_pass' in h]
    top.scatter([s for s, _ in evals], [e for _, e in evals], label='held-out eval (hidden)', series=3, r=5)
    top.legend()
    bottom = fig.axes(0, 0.55, 1, 0.45, xlabel='training step', ylabel='episodes using hardcode_tests',
                      ylim=(0, 1), xlim=(0, 39), title='The exploit, logged separately')
    bottom.line(xs, _smooth([d['exploit'] for d in dash]), label='public-test reward', series=8, end_label=False)
    bottom.line(xs, _smooth([d['exploit'] for d in pen]), label='public reward - exploit penalty', series=1, end_label=False)
    bottom.legend()
    fig.save('ch11-hacking')


def flaky():
    runs = _chapter()['flaky_runs']
    clean, fn, fp = runs['clean'], runs['false negatives 0.3'], runs['false positives 0.3']
    xs = list(range(40))
    fig = Figure(860, 380, title='Flaky verifiers: what the reward says vs. what the policy does',
                 subtitle='Hidden-test reward with 30% false negatives or 30% false positives (5-step moving average).')
    left = fig.axes(0, 0, 0.5, 1, xlabel='training step', ylabel='training reward (noisy)', ylim=(0, 1), xlim=(0, 39))
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='training step', ylabel='true hidden pass rate of the batch', ylim=(0, 1), xlim=(0, 39))
    for hist, label, s in [(clean, 'clean', 1), (fn, 'false neg. 0.3', 2), (fp, 'false pos. 0.3', 8)]:
        left.line(xs, _smooth([h['reward'] for h in hist], 5), label=label, series=s, end_label=False)
        right.line(xs, _smooth([h['pass_rate'] for h in hist], 5), label=label, series=s, end_label=False)
    left.legend()
    right.legend()
    fig.save('ch11-flaky')


def matrix():
    """Measured columns from scripted policies and the RL base agent; cost and noise are qualitative."""
    c = _chapter()
    trajs, infos = c['base_trajs'], c['scripted_infos']
    rows = [('hidden tests, binary', 'hidden', 'low', 'low'), ('hidden tests, fraction', 'partial', 'low', 'low'),
            ('public tests, binary', 'public', 'low', 'low'), ('public tests, fraction', 'public_partial', 'low', 'low')]
    level = {'low': 0.15, 'medium': 0.5, 'high': 0.85}
    values, labels = [], []
    for _, kind, cost, noise in rows:
        fn = arl.make_reward(kind)
        r = torch.tensor([fn(t.info) for t in trajs]).view(-1, 8)
        density = (r.std(-1) > 1e-6).float().mean().item()
        hack = sum(map(fn, infos['hacker'])) / max(sum(map(fn, infos['oracle'])), 1e-6)
        values.append([density, hack, level[cost], level[noise]])
        labels.append([f'{density:.2f}', f'{hack:.2f}', cost, noise])
    qualitative = [('Bradley-Terry RM', 'high', 'high', 'medium', 'medium'), ('LLM judge (rubric)', 'high', 'high', 'high', 'high'),
                   ('process reward / PRM', 'high', 'high', 'high', 'medium'), ('potential-based shaping', 'high', 'low', 'low', 'low')]
    for name, d, h, cost, noise in qualitative:
        rows.append((name, None, cost, noise))
        values.append([level[d], level[h], level[cost], level[noise]])
        labels.append([d, h, cost, noise])
    cols = ['signal density', 'hackability', 'cost', 'noise']
    fig = Figure(860, 490, title='Reward design matrix',
                 subtitle='Top four rows measured on the gym; bottom four rows are qualitative judgments.')
    x0, y0, w, h = 250, 110, 560, 320
    fig.heatmap(x0, y0, w, h, values, row_labels=[r[0] for r in rows], col_labels=cols, vmin=0, vmax=1)
    cw, ch = w / 4, h / len(rows)
    for i, row in enumerate(labels):
        for j, lab in enumerate(row):
            dark = values[i][j] > 0.55
            fig.text(x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + 4, lab, anchor='middle', size=12,
                     fill='#ffffff' if dark else None, weight=600 if i < 4 else None)
    fig.rect(x0 - 2, y0 + 4 * ch - 1, w + 4, 2, cls='axis', rx=0)
    fig.text(x0, y0 + h + 24, 'Density = share of RL-base groups with nonzero reward variance; hackability = hacker / oracle reward.', cls='t2', size=11)
    fig.text(x0, y0 + h + 40, 'Darker = more. Density is wanted; hackability, cost, and noise are not.', cls='t2', size=11)
    fig.save('ch11-reward-matrix')


FIGURES = {'ch11-hacking': hacking, 'ch11-flaky': flaky, 'ch11-reward-matrix': matrix}
