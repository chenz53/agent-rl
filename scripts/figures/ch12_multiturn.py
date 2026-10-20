"""Figures for Chapter 12: behavior under budgets and penalties, turn-level credit on a real episode, context policies.

The chapter's code blocks are executed once and their results reused, so plotted numbers match the printed ones.
"""
import torch

import agentrl as arl
from svgplot import Figure
import hairline

torch.set_num_threads(4)
_CACHE = {}


def _chapter():
    """Execute the chapter's code blocks once (stdout discarded) and return its namespace."""
    if not _CACHE:
        import contextlib, io, re
        from pathlib import Path
        text = (Path(__file__).resolve().parents[2] / 'tutorial/chapters/12-multi-turn-and-long-horizons.md').read_text()
        namespace = {'__name__': 'figure'}
        with contextlib.redirect_stdout(io.StringIO()):
            for code in re.findall(r'```python\n(.*?)\n```', text, re.S):
                exec(code, namespace)
        _CACHE.update(namespace)
    return _CACHE


def _smooth(values, k=5):
    return [sum(values[max(0, i - k + 1):i + 1]) / len(values[max(0, i - k + 1):i + 1]) for i in range(len(values))]


def behavior():
    c = _chapter()
    hist = c['histories']
    arms = [('default', 1), ('penalty 0.05', 3), ('penalty 0.3', 8), ('max_turns 3', 2)]
    xs = list(range(40))
    fig = Figure(980, 650, title='What the agent does depends on what a call costs and how many it gets',
                 subtitle='Training batch, 5-step moving average. RL base agent, hidden-test verdict, 64 episodes per step, one seed.')
    panels = [('pass_rate', 'hidden-test pass rate', (0, 1)), ('reads', 'reads per episode', (0, 2)),
              ('tests', 'tests per episode', (0, 1.2)), ('turns', 'calls per episode', (0, 6))]
    for k, (key, label, ylim) in enumerate(panels):
        ax = fig.axes(0.5 * (k % 2), 0.5 * (k // 2), 0.5, 0.5, xlabel='training step', ylabel=label, ylim=ylim,
                      xlim=(0, 39), title=label)
        for name, series in arms:
            ax.line(xs, _smooth([h[key] for h in hist[name]]), series=series, end_label=False)
    fig.legend_row(64, fig.top + 6, [(name, series, 'line') for name, series in arms])
    fig.save('ch12-behavior')


def turn_credit():
    """One successful held-out episode of the turn-GAE policy: per-call rewards, values, TD errors, advantages."""
    c = _chapter()
    policy, critic, _ = c['credit_runs']['turn GAE']
    cfg, tok = c['cost_cfg'], c['tok']
    trajs = arl.sample_groups(policy, lambda: arl.make_env(cfg), c['heldout_tasks'], tok, arl.RLConfig(group_size=4),
                              torch.Generator().manual_seed(5))
    episode = next(t for t in trajs if t.reward > 0 and t.info['n_turns'] == 4 and t.info['n_invalid'] == 0)
    batch = arl.collate([episode], tok.pad_id)
    mask = batch['action_mask']
    rewards = c['per_call_rewards']([episode], mask, c['STEP_COST'])
    with torch.no_grad():
        values = arl.critic_values(critic, batch['ids'])
    turn_adv = arl.turn_advantages(rewards, values, mask, batch['turn_ids'], 1.0, 0.95)[0]
    token_adv = arl.gae_advantages(rewards, values, mask, 1.0, 0.95)[0][0]
    turn = batch['turn_ids'][0]
    K = episode.info['n_turns']
    first = [int(((turn == k) & (mask[0] > 0)).nonzero()[0]) for k in range(K)]
    r_k = [float(rewards[0][(turn == k)].sum()) for k in range(K)]
    v_k = [float(values[0][i]) for i in first] + [0.0]
    delta = [r_k[k] + v_k[k + 1] - v_k[k] for k in range(K)]
    a_turn = [float(turn_adv[i]) for i in first]
    a_tok = [[float(token_adv[i]) for i in range(len(turn)) if turn[i] == k and mask[0][i] > 0] for k in range(K)]
    calls = [a.replace('<call> ', '').replace(' </call>', '') for a in episode.info['actions']]
    sc = hairline.Scene(980, 470)
    sc.head('FIG 12.1 · TURN-LEVEL CREDIT ON ONE REAL EPISODE',
            f'turn-GAE policy and its critic after 40 steps; reward = verdict − {c["STEP_COST"]} per call; '
            'γ = 1, λ = 0.95; unwhitened')
    x0, w, gap, y0 = 190, 150, 34, 84
    rows = [('reward r_k', r_k), ('value V(s_k)', v_k[:K]), ('TD error δ_k', delta), ('turn advantage A_k', a_turn)]
    sc.label(x0 - 18, y0 + 25, 'CALL a_k', cls='chead', anchor='end')
    for row, (name, _) in enumerate(rows, start=1):
        sc.label(x0 - 18, y0 + row * 54 + 23, name, cls='cline', anchor='end')
    for k in range(K):
        x = x0 + k * (w + gap)
        sc.card(x, y0, w, 40, lines=[calls[k]], style='card', mono=True)
        for row, (_, vals) in enumerate(rows, start=1):
            v = vals[k]
            cls = 'vpos' if v > 0.005 else ('vneg' if v < -0.005 else 'vzero')
            y = y0 + row * 54
            sc.card(x + 22, y, w - 44, 36, style='card-soft')
            sc.label(x + w / 2, y + 23, f'{v:+.3f}'.replace('-', '−'), cls=cls, anchor='middle')
        if k < K - 1:
            sc.farrow([(x + w + 4, y0 + 20), (x + w + gap - 6, y0 + 20)])
    sc.label(x0 + K * (w + gap) - gap + 10, y0 + 2 * 54 + 23, 'V(end) = 0', cls='desc')
    spread = [f'{min(v):+.2f}..{max(v):+.2f}'.replace('-', '−') for v in a_tok]
    sc.label(x0, y0 + 5 * 54 + 26, 'Token-level GAE on the same episode varies within each call: ' + ',  '.join(spread), cls='desc')
    sc.label(x0, y0 + 5 * 54 + 46, 'δ_k = r_k + V(s_k+1) − V(s_k);  A_k = δ_k + λ · A_k+1.  Accent: positive; ink: negative.',
             cls='desc')
    (hairline.OUT / 'ch12-turn-credit.svg').write_text(sc.svg('Turn-level credit on one real episode'))


def context():
    c = _chapter()
    table = c['context_table']
    rows, cols = ['full', 'keep_obs=1'], [None, 1, 0]
    col_labels = ['full', 'keep_obs=1', 'keep_obs=0']
    fig = Figure(940, 330, title='Train and evaluate under the same context policy',
                 subtitle='Held-out tasks, 16 episodes each. Rows: training context; columns: evaluation context.')
    for k, (key, name, vmax) in enumerate([('pass_rate', 'pass rate', 1.0), ('tests', 'tests per episode', 1.0)]):
        x0 = 150 + k * 460
        fig.text(x0, 104, name, cls='t', size=13, weight=600)
        fig.heatmap(x0, 140, 300, 140, [[table[r, e][key] for e in cols] for r in rows],
                    row_labels=[f'trained {r}' for r in rows], col_labels=col_labels, vmin=0, vmax=vmax,
                    fmt_cell='{:.2f}')
    fig.save('ch12-context')


FIGURES = {'ch12-behavior': behavior, 'ch12-turn-credit': turn_credit, 'ch12-context': context}
