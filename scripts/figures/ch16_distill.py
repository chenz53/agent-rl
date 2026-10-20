"""Figures for Chapter 16: distillation learning curves, recovery after mistakes, top-k bias, wrong teacher.

Data figures re-execute the chapter's own code blocks (stdout suppressed), so every plotted
number is the number the chapter prints.
"""
import contextlib
import io
from svgplot import Figure

CHAPTER = '16-distillation.md'
_NS = {}


def chapter_namespace():
    if not _NS:
        from check_tutorial_snippets import BLOCK, CHAPTERS
        source = (CHAPTERS / CHAPTER).read_text()
        _NS['__name__'] = 'figure_ch16'
        with contextlib.redirect_stdout(io.StringIO()):
            for match in BLOCK.finditer(source):
                exec(compile(match['code'], CHAPTER, 'exec'), _NS)
    return _NS


def curves():
    ns = chapter_namespace()
    fig = Figure(1000, 380, title='Distilling a 157k-parameter teacher into a 41.5k-parameter student',
                 subtitle='Hidden-test pass rate; every step uses 32 episodes (4 tasks x 8). One seed.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='student step', ylabel='pass rate', title='training tasks', ylim=(0, 1.05))
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='student step', ylabel='pass rate', title='held-out variants', ylim=(0, 1.05))
    slots = {'on-policy': 1, 'off-policy': 2, 'outcome RL': 3}
    for name, (_, hist, _) in ns['runs'].items():
        xs = [s for s, _, _ in hist]
        left.line(xs, [a for _, a, _ in hist], label=name, series=slots[name], markers=True, end_label=False)
        right.line(xs, [b for _, _, b in hist], label=name, series=slots[name], markers=True, end_label=False)
    for ax in (left, right):
        ax.legend()
    fig.save('ch16-curves')


def recovery():
    ns = chapter_namespace()
    rec = ns['recovery']
    fig = Figure(940, 360, title='Success after an injected mistake',
                 subtitle='Episodes start with the injected actions, then the model continues; 40 episodes per bar.')
    ax = fig.axes(0, 0, 1, 1, xlabel='injected prefix', ylabel='success rate', ylim=(0, 1.1))
    names = list(next(iter(rec.values())))
    slots = {'clean teacher': 4, 'teacher': 3, 'off-policy': 2, 'on-policy': 1}
    ax.bars(list(rec), [(n, [rec[i][n] for i in rec], slots[n]) for n in names], value_labels=True, fmt_value='{:.2f}')
    ax.legend()
    fig.save('ch16-recovery')


def topk():
    ns = chapter_namespace()
    rows = ns['topk_rows']
    full = ns['full'].item()
    ks = [r[0] for r in rows]
    fig = Figure(1000, 360, title='Top-k teacher distributions on student prefixes',
                 subtitle='Renormalized top-k drops the student mass outside the teacher top-k; a tail bucket keeps it.')
    a = fig.axes(0, 0, 0.5, 1, xlabel='k (log scale)', ylabel='loss / full-vocabulary loss', title='loss value',
                 xlog=True, ylim=(0, 1.05))
    b = fig.axes(0.5, 0, 0.5, 1, xlabel='k (log scale)', ylabel='cosine with full gradient', title='gradient direction',
                 xlog=True, ylim=(0, 1.05))
    a.line(ks, [r[1] / full for r in rows], label='renormalized top-k', series=8, markers=True, end_label=False)
    a.line(ks, [r[2] / full for r in rows], label='top-k + tail bucket', series=1, markers=True, end_label=False)
    b.line(ks, [r[3] for r in rows], label='renormalized top-k', series=8, markers=True, end_label=False)
    b.line(ks, [r[4] for r in rows], label='top-k + tail bucket', series=1, markers=True, end_label=False)
    a.legend()
    b.legend()
    fig.save('ch16-topk')


def wrong_teacher():
    ns = chapter_namespace()
    fams = list(ns['arl'].FAMILIES)
    models = [('bad teacher', ns['bad_teacher'], 8)] + [(k, v, s) for (k, v), s in zip(ns['wrong'].items(), [2, 3, 1])]
    fig = Figure(980, 380, title='A teacher that is wrong on one family, and three students distilled from it',
                 subtitle='Per-family hidden-test pass rate on training tasks (8 samples per task).')
    ax = fig.axes(0, 0, 1, 1, xlabel='bug family', ylabel='hidden-test pass rate', ylim=(0, 1.1))
    groups = []
    for name, model, slot in models:
        rates = ns['by_family'](model)
        groups.append((name, [rates[f] for f in fams], slot))
    ax.bars(fams, groups)
    ax.text(3.5, 0.06, 'zero: 0.00 for all four', anchor='middle', cls='note')
    ax.legend()
    fig.save('ch16-wrong-teacher')


FIGURES = {'ch16-curves': curves, 'ch16-recovery': recovery, 'ch16-topk': topk, 'ch16-wrong-teacher': wrong_teacher}
