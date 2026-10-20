"""Figures for Chapter 18: the failure dashboard, the invariant matrix, and the triage map.

Runs use the chapter's configurations, seeds, and step counts, so plotted numbers match the printed ones.
"""
import copy
import dataclasses

import torch

import agentrl as arl
from svgplot import Figure
import hairline

torch.set_num_threads(4)
_CACHE = {}

# (panel title, run name, config overrides, bug, steps, signature metric, label, control run)
FAILURES = [
    ('(1) mask off by one', 'mask_shift', {}, 'mask_shift', 24, 'logp_gap', 'log10 logp_gap', 'healthy'),
    ('(2) stale old_logp', 'stale_old_logp', {}, 'stale_old_logp', 24, 'clip_frac', 'clip_frac', 'healthy'),
    ('(3) entropy collapse', 'entropy_sign', {}, 'entropy_sign', 24, 'entropy', 'entropy (nats)', 'healthy'),
    ('(4) length bias, 12 turns', 'length bias', dict(max_turns=12, loss_mode='seq_mean_token_mean'), None, 16,
     'length_fail', 'tokens per failure', 'healthy, 12 turns'),
    ('(5) advantage sign error', 'sign_flip', {}, 'sign_flip', 8, 'turns', 'turns per episode', 'healthy'),
    ('(6) reward hacking', 'public reward', dict(reward='public'), None, 24, 'pass_rate', 'hidden-test pass',
     'healthy'),
    ('(7) learning rate 1e-2', 'lr 1e-2', dict(lr=1e-2), None, 8, 'malformed', 'malformed calls', 'healthy'),
    ('(8) accumulation normalizer', 'accum_local', {}, 'accum_local', 24, 'grad_norm', 'grad norm', 'healthy'),
    ('(9) sampler at T = 0.7', 'sampler_temp', {}, 'sampler_temp', 24, 'ratio_max', 'max ratio', 'healthy'),
    ('(10) reference = policy', 'ref_is_policy', dict(kl_coef=0.1), 'ref_is_policy', 8, 'kl_ref', 'KL to reference',
     'KL 0.1'),
]
CONTROLS = {'healthy': ({}, 24), 'healthy, 12 turns': (dict(max_turns=12), 16), 'KL 0.1': (dict(kl_coef=0.1), 8)}


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        _CACHE.update(tok=tok, train=arl.make_tasks('train'), heldout=arl.make_tasks('heldout'),
                      base=arl.make_rl_base(tok), runs={})
    return _CACHE


def _run(name, fields, bug, steps):
    c = _setup()
    if name not in c['runs']:
        c['runs'][name] = arl.train_logged(copy.deepcopy(c['base']), c['train'], c['tok'], arl.RLConfig(**fields),
                                           steps, bug, c['heldout'], eval_every=8)
    return c['runs'][name]


def dashboard():
    import math
    fig = Figure(1250, 1010, title='Ten failures and their dashboard signatures (TinyAgent, 64 episodes per step)',
                 subtitle='Top of each pair: training batch reward. Bottom: the metric that exposes the fault. '
                          'Blue: matched healthy run; red: the run with the fault.')
    fig.legend_row(20, fig.top + 6, [('healthy (matched control)', 1, 'line'), ('with the fault', 8, 'line')])
    for k, (title, name, fields, bug, steps, key, label, control) in enumerate(FAILURES):
        col, pair = k % 5, k // 5
        bad = _run(name, fields, bug, steps)
        good = _run(control, *CONTROLS[control][:1], None, CONTROLS[control][1])
        top = fig.axes(col / 5, 0.03 + pair * 0.485, 0.2, 0.2, margin=(56, 26, 14, 26), title=title,
                       ylabel='reward', ylim=(0, 1), xlim=(0, 23))
        low = fig.axes(col / 5, 0.03 + pair * 0.485 + 0.2, 0.2, 0.26, margin=(56, 14, 14, 44), xlabel='step',
                       ylabel=label, xlim=(0, 23))
        for ax, metric in [(top, 'reward'), (low, key)]:
            for h, series in [(good, 1), (bad, 8)]:
                ys = [r.get(metric, 0.0) for r in h]
                if metric == 'logp_gap':
                    ys = [math.log10(max(y, 1e-12)) for y in ys]
                ax.line([r['step'] for r in h], ys, series=series, end_label=False)
    fig.save('ch18-dashboard')


CHECK_LABELS = ['mask =\npolicy tokens', 'sampler =\ntrainer logp', 'ratio = 1\nat step 0', 'group adv\nmean = 0',
                'loss invariant\nto padding', 'accumulated =\none-pass grad', 'old_logp\nfrozen', 'update follows\nsign of A',
                'KL > 0 after\na step']


def checks():
    c = _setup()
    check_cfg = arl.RLConfig(kl_coef=0.1)

    def probe(bug):
        cfg = dataclasses.replace(check_cfg, temperature=arl.SAMPLER_TEMP) if bug == 'sampler_temp' else check_cfg
        return arl.sample_groups(c['base'], lambda: arl.make_env(cfg), c['train'][:8], c['tok'], cfg,
                                 torch.Generator().manual_seed(1))
    rows = [None, *arl.BUGS]
    matrix = {bug: arl.sanity_checks(c['base'], probe(bug), c['tok'], check_cfg, bug) for bug in rows}
    x0, y0, cw, ch = 170, 120, 106, 34
    fig = Figure(x0 + cw * 9 + 20, y0 + ch * len(rows) + 40, title='Which invariant catches which fault',
                 subtitle='sanity_checks on 64 base-agent episodes before training; cell text: the measured value. '
                          'Red: the check fails.')
    for j, label in enumerate(CHECK_LABELS):
        fig.text(x0 + j * cw + cw / 2, y0 - 26, label, cls='t2', size=11, anchor='middle')
    for i, bug in enumerate(rows):
        fig.text(x0 - 10, y0 + i * ch + ch / 2 + 4, 'no fault' if bug is None else bug, cls='t2', size=12,
                 anchor='end', mono=True)
        for j, (value, ok) in enumerate(matrix[bug].values()):
            x, y = x0 + j * cw + 2, y0 + i * ch + 2
            text = f'{value:.2g}' if isinstance(value, float) else str(value)
            if ok:
                fig.rect(x, y, cw - 4, ch - 4, rx=4, tip=f'{bug}: passes ({text})')
                fig.text(x + (cw - 4) / 2, y + ch / 2 + 2, text, cls='t2', size=11, anchor='middle', mono=True)
            else:
                fig.rect(x, y, cw - 4, ch - 4, fill=8, cls='', rx=4, tip=f'{bug}: FAILS ({text})')
                fig.text(x + (cw - 4) / 2, y + ch / 2 + 2, text, size=11, anchor='middle', mono=True,
                         fill='#ffffff', weight=600)
    fig.save('ch18-checks')


TRIAGE = [
    ('logp_gap above ~1e-4', 'sampler vs trainer log-probs on the loss mask', 'sampler = trainer logp; mask = policy',
     '(1) mask off by one, (9) temperature', 'retokenization drift (Ch. 5), engine mismatch (Ch. 13)'),
    ('clip_frac = 0, ratio exactly 1', 'with minibatches > 1 or epochs > 1', 'old_logp frozen in the update',
     '(2) stale old_logp', 'old log-probs recomputed inside the loop'),
    ('reward falls from step 0', 'turns rise, exploit and junk calls rise', 'update follows the sign of A',
     '(5) sign error', 'in the advantage, the loss, or the reward'),
    ('approx_kl >> 0.01, entropy jumps', 'malformed calls rise within steps', 'replay the step at lr / 10',
     '(7) learning rate too high', 'or too many epochs per batch'),
    ('entropy -> 0 at low reward', 'zero_adv_frac -> 1, reads vanish', 'regularizer signs; pass@k; temperature',
     '(3) entropy collapse', 'exploration is gone before the skill'),
    ('failing episodes get longer', 'turns rise, reward unchanged', 'length by outcome; aggregation mode',
     '(4) length bias', 'seq_mean_token_mean, step rewards'),
    ('reward up, held-out hidden down', 'behavior shifts to one op', 'independent verifier; exploit share',
     '(6) reward hacking', 'reward measures the wrong thing (Ch. 11)'),
    ('kl_ref = 0 with kl_coef > 0', 'run identical to one without KL', 'KL to reference > 0 after a step',
     '(10) reference = policy', 'missing deep copy or frozen copy'),
    ('nothing visible on the dashboard', 'results change with microbatching', 'accumulated = one-pass gradient',
     '(8) accumulation normalizer', 'and loss invariant to padding'),
]


def triage():
    """Figure 18.3 in the hairline style: symptom -> first test (accent) -> likely fault."""
    W, top, rh = 1180, 112, 62
    sc = hairline.Scene(W, top + rh * len(TRIAGE) + 24)
    sc.head('FIG 18.3 · TRIAGE: FROM A DASHBOARD SYMPTOM TO THE TEST THAT DECIDES IT',
            "run the cheap invariants of Section 18.6 first; numbers refer to the chapter's catalogue of faults")
    cols = [(24, 360, 'SYMPTOM'), (434, 330, 'FIRST TEST'), (834, 322, 'LIKELY FAULT')]
    for x, w, head in cols:
        sc.label(x + 14, top - 12, head, cls='chead')
    for i, (symptom, detail, test, fault, note) in enumerate(TRIAGE):
        y = top + i * rh
        h = rh - 14
        sc.card(cols[0][0], y, cols[0][1], h, symptom, [detail], style='card-soft')
        sc.card(cols[1][0], y, cols[1][1], h, test, style='card-accent')
        sc.card(cols[2][0], y, cols[2][1], h, fault, [note], style='card')
        mid = y + h / 2
        sc.farrow([(cols[0][0] + cols[0][1] + 4, mid), (cols[1][0] - 6, mid)])
        sc.farrow([(cols[1][0] + cols[1][1] + 4, mid), (cols[2][0] - 6, mid)])
    (hairline.OUT / 'ch18-triage.svg').write_text(sc.svg('Triage map'))


FIGURES = {'ch18-dashboard': dashboard, 'ch18-checks': checks, 'ch18-triage': triage}
