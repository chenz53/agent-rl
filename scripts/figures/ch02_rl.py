"""Figures for Chapter 2 (RL fundamentals): the repair MDP and its values."""
import torch

import agentrl as arl
from svgplot import Figure
import hairline


def _policy(names):
    pol = torch.zeros(len(arl.REPAIR_STATES), len(arl.REPAIR_ACTIONS), dtype=torch.float64)
    for s, name in enumerate(names):
        pol[s, arl.REPAIR_ACTIONS.index(name)] = 1.0
    return pol


POLICIES = [
    ('careful', ['read', 'edit', 'test', 'test', 'submit', 'edit', 'submit']),
    ('read-edit-submit', ['read', 'edit', 'submit', 'submit', 'submit', 'edit', 'submit']),
    ('blind', ['edit', 'edit', 'submit', 'submit', 'submit', 'edit', 'submit']),
]


def repair_mdp():
    """Figure 2.1 in the hairline style; every probability is read from make_repair_mdp."""
    P, _ = arl.make_repair_mdp()
    S, A = arl.REPAIR_STATES.index, arl.REPAIR_ACTIONS.index
    prob = lambda state, action, nxt: P[S(state), A(action), S(nxt)].item()
    (hairline.OUT / 'repair-mdp.svg').write_text(hairline.repair_mdp(prob))


def repair_values():
    fig = Figure(860, 380, title='Discounting and step costs change the preferred policy',
                 subtitle='V(unread) computed exactly with policy_evaluation on the repair MDP')
    gammas = [0.5 + 0.01 * i for i in range(51)]
    costs = [0.005 * i for i in range(61)]
    ax1 = fig.axes(0, 0, 0.5, 1, xlabel='discount gamma (step cost 0)', ylabel='V(unread)', ylim=(0, 1.05))
    ax2 = fig.axes(0.5, 0, 0.5, 1, xlabel='step cost per action (gamma = 1)', ylabel='V(unread)', ylim=(-0.4, 1.05))
    P0, R0 = arl.make_repair_mdp()
    for i, (name, names) in enumerate(POLICIES):
        pol = _policy(names)
        v1 = [arl.policy_evaluation(P0, R0, pol, g)[0].item() for g in gammas]
        v2 = [arl.policy_evaluation(*arl.make_repair_mdp(c), pol, 1.0)[0].item() for c in costs]
        ax1.line(gammas, v1, label=name, series=i + 1, end_label=False)
        ax2.line(costs, v2, label=name, series=i + 1, end_label=False)
    ax2.hline(0, style='axis')
    ax1.legend()
    ax2.legend()
    fig.save('repair-values')


FIGURES = {'repair-mdp': repair_mdp, 'repair-values': repair_values}
