"""Figures for Chapter 4: environment state machine and scripted baselines."""
import agentrl as arl
from svgplot import Figure
import hairline


def state_machine():
    """Figure 4.1 in the hairline style (scripts/hairline.py)."""
    (hairline.OUT / 'ch04-env-state-machine.svg').write_text(hairline.env_state_machine())


def baselines():
    tasks = arl.make_tasks('train')
    families = list(arl.FAMILIES)
    kinds = [('oracle', 1, 1), ('blind', 40, 3), ('random', 40, 2), ('hacker', 1, 8)]
    fig = Figure(940, 380, title='Scripted baselines on the BugFix gym (train split)',
                 subtitle='Left: hidden-test reward by family. Right: public pass rate vs hidden reward.')
    left = fig.axes(0, 0, 0.6, 1, xlabel='bug family', ylabel='hidden-test reward', ylim=(0, 1.05))
    groups, public, hidden = [], [], []
    for kind, repeats, slot in kinds:
        policy, env = arl.scripted_policy(kind, seed=0), arl.BugFixEnv()
        runs = [arl.run_episode(policy, env, t) for t in tasks for _ in range(repeats)]
        groups.append((kind, [sum(r for r, i in runs if i['family'] == f) / (4 * repeats) for f in families], slot))
        public.append(sum(i['submitted'] and i['public_pass'] for _, i in runs) / len(runs))
        hidden.append(sum(r for r, _ in runs) / len(runs))
    left.bars(families, groups)
    left.hline(0.2, label='guess 1 of 5 ops')
    left.legend()
    right = fig.axes(0.6, 0, 0.4, 1, xlabel='policy', ylabel='rate', ylim=(0, 1.05))
    right.bars([k for k, _, _ in kinds], [('public pass', public, 2), ('hidden reward', hidden, 1)])
    right.legend()
    fig.save('ch04-baselines')


FIGURES = {'ch04-env-state-machine': state_machine, 'ch04-baselines': baselines}
