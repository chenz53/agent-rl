"""Figures for Chapter 15: weighted-likelihood view, DPO diagnostics, recipe curves, hacked RFT.

Data figures re-execute the chapter's own code blocks (stdout suppressed), so every plotted
number is the number the chapter prints.
"""
import contextlib
import io
from svgplot import Figure
import hairline

CHAPTER = '15-sft-and-rejection-sampling.md'
_NS = {}


def chapter_namespace():
    if not _NS:
        from check_tutorial_snippets import BLOCK, CHAPTERS
        source = (CHAPTERS / CHAPTER).read_text()
        _NS['__name__'] = 'figure_ch15'
        with contextlib.redirect_stdout(io.StringIO()):
            for match in BLOCK.finditer(source):
                exec(compile(match['code'], CHAPTER, 'exec'), _NS)
    return _NS


def weights():
    """Figure 15.1 in the hairline style: one row per objective, columns for data source and weight."""
    sc = hairline.Scene(980, 360)
    sc.head('FIG 15.1 · FOUR OBJECTIVES, ONE FORM: Σ w_i ∇ log π(τ_i) OVER ACTION TOKENS',
            'they differ in where the trajectories come from and in the weight each one gets')
    cols = [(24, 'METHOD'), (230, 'TRAJECTORIES τ_i FROM'), (500, 'WEIGHT w_i'), (730, 'SIGNAL ON FAILURES')]
    rows = [('SFT', 'an expert (demonstrations)', '1', 'none (no failures in data)'),
            ('RFT / expert iteration', 'current policy, filtered by R = 1', 'R_i / p̂  (kept only)', 'none: discarded'),
            ('policy gradient', 'current policy (fresh samples)', 'R_i − mean R of the group', 'pushed down (w < 0)'),
            ('DPO', 'fixed dataset of pairs', '± β · σ(−margin)', 'rejected pushed down, offline')]
    for x, head in cols:
        sc.label(x + 14, 92, head, cls='chead')
    for r, (name, src, wt, neg) in enumerate(rows):
        y = 104 + r * 52
        sc.card(24, y, 932, 40, style='card-accent' if name == 'policy gradient' else 'card-soft')
        sc.label(38, y + 25, name, cls='ctitle')
        for x, text, cls in [(230, src, 'cline'), (500, wt, 'cmono'), (730, neg, 'cline')]:
            sc.label(x + 14, y + 25, text, cls=cls)
        for x in (230, 500, 730):
            sc.overlay.append(f'<path class="crease" d="M{x},{y + 9} V{y + 31}"/>')
    sc.label(24, 330, 'Action mask everywhere: environment terms of log p(τ) carry no parameters (SFT, PG) or cancel in π/π_ref (DPO). '
             'Accent: the estimator the rest of the book builds on.', cls='desc')
    (hairline.OUT / 'ch15-weights.svg').write_text(sc.svg('Four objectives, one form'))


def dpo():
    ns = chapter_namespace()
    base_train, trajs = ns['pass_rate'](ns['weak'], ns['train'], ns['tok'], 4)
    base_invalid = sum(t.info['n_invalid'] for t in trajs) / len(trajs)
    fig = Figure(1000, 360, title='Offline DPO on weak-base pairs: the loss improves, the policy degrades',
                 subtitle='Plain DPO (solid) and DPO + NLL on chosen episodes (dashed); beta = 0.1, 55 pairs.')
    a = fig.axes(0, 0, 0.36, 1, xlabel='DPO step', ylabel='log pi - log pi_ref (nats)', title='sequence log-ratio')
    b = fig.axes(0.36, 0, 0.32, 1, xlabel='DPO step', ylabel='hidden-test pass rate', title='train pass rate', ylim=(0, 1))
    c = fig.axes(0.68, 0, 0.32, 1, xlabel='DPO step', ylabel='malformed calls / episode', title='malformed calls')
    for log, dashed, tag in [(ns['dpo_log'], False, 'DPO'), (ns['nll_log'], True, 'DPO+NLL')]:
        steps = [0] + [r['step'] for r in log]
        a.line(steps, [0] + [r['d_chosen'] for r in log], label=None if dashed else 'chosen', series=1, dashed=dashed, markers=True, end_label=False)
        a.line(steps, [0] + [r['d_rejected'] for r in log], label=None if dashed else 'rejected', series=8, dashed=dashed, markers=True, end_label=False)
        b.line(steps, [base_train] + [r['train'] for r in log], label=tag, series=1 if not dashed else 3, dashed=dashed, markers=True, end_label=False)
        c.line(steps, [base_invalid] + [r['invalid'] for r in log], label=tag, series=8 if not dashed else 3, dashed=dashed, markers=True, end_label=False)
    for ax in (a, b, c):
        ax.legend()
    fig.save('ch15-dpo')


def recipes():
    ns = chapter_namespace()
    fig = Figure(1000, 380, title='Four recipes from the same weak base (BugFix gym, one seed)',
                 subtitle='Hidden-test pass rate; x = gradient steps (RFT: 15 SFT steps per round; SFT -> RL branches at step 20).')
    left = fig.axes(0, 0, 0.5, 1, xlabel='gradient step', ylabel='pass rate', title='training tasks', ylim=(0, 1.05))
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='gradient step', ylabel='pass rate', title='held-out variants', ylim=(0, 1.05))
    series = {'SFT (oracle)': 2, 'RFT': 4, 'SFT -> RL': 1, 'RL only': 3}
    for name, hist in ns['recipes'].items():
        xs = [s for s, _, _ in hist]
        left.line(xs, [a for _, a, _ in hist], label=name, series=series[name], markers=True, end_label=False)
        right.line(xs, [b for _, _, b in hist], label=name, series=series[name], markers=True, end_label=False)
    left.vline(20, label='cold start')
    left.legend()
    right.legend()
    fig.save('ch15-recipes')


def hacked():
    ns = chapter_namespace()
    log = ns['hack_log']
    rounds = [f'round {r["round"]}' for r in log]
    fig = Figure(940, 360, title='RFT with a public-test filter keeps hacks',
                 subtitle='Left: kept episodes per round. Right: the policy after each round, public vs hidden tests.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='RFT round', ylabel='kept episodes')
    left.bars(rounds, [('genuine fix', [r['kept'] - r['hacks'] for r in log], 1),
                       ('hack (fails hidden)', [r['hacks'] for r in log], 8)], value_labels=True, fmt_value='{:.0f}')
    left.legend()
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='RFT round', ylabel='pass rate', ylim=(0, 1.05), xticks=[1, 2, 3])
    xs = [r['round'] for r in log]
    right.line(xs, [r['public'] for r in log], label='public tests (filter)', series=2, markers=True, end_label=False)
    right.line(xs, [r['hidden'] for r in log], label='hidden tests', series=1, markers=True, end_label=False)
    right.legend()
    fig.save('ch15-hacked-rft')


FIGURES = {'ch15-weights': weights, 'ch15-dpo': dpo, 'ch15-recipes': recipes, 'ch15-hacked-rft': hacked}
