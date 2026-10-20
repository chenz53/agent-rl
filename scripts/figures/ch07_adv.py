"""Figures for Chapter 7: GAE bias/variance over lambda, std-normalization difficulty bias, advantage strips.

Computations mirror the chapter's code with the same seeds, so plotted numbers match the printed ones.
"""
import torch
import torch.nn.functional as F

import agentrl as arl
from svgplot import Figure, DIV_NEG, DIV_POS, esc


def _sample_steps(P, R, policy, n, generator, max_steps=40):
    S = R.shape[0]
    states = torch.zeros(n, max_steps + 1, dtype=torch.long)
    actions = torch.zeros(n, max_steps + 1, dtype=torch.long)
    rewards = torch.zeros(n, max_steps, dtype=P.dtype)
    alive = torch.zeros(n, max_steps, dtype=P.dtype)
    for t in range(max_steps):
        s = states[:, t]
        a = torch.multinomial(policy[s], 1, generator=generator).squeeze(-1)
        live = s != S - 1
        actions[:, t], rewards[:, t], alive[:, t] = a, R[s, a] * live, live.to(P.dtype)
        probs = P[s, a].clone()
        probs[~live, -1] = 1.0
        states[:, t + 1] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return states, actions, rewards, alive


def gae_lambda():
    P, R = arl.make_repair_mdp(step_cost=0.05)
    torch.manual_seed(0)
    policy = torch.softmax(0.5 * torch.randn(7, 4, dtype=torch.float64), -1)
    theta = policy.log().requires_grad_()
    pi = torch.softmax(theta, -1)
    exact = torch.autograd.grad(arl.policy_evaluation(P, R, pi, 1.0)[0], theta)[0]
    pi = pi.detach()
    states, actions, rewards, alive = _sample_steps(P, R, pi, 40_000, torch.Generator().manual_seed(1))
    score = F.one_hot(actions[:, :-1], 4).double() - pi[states[:, :-1]]
    rows = (torch.arange(len(states))[:, None].expand(-1, states.shape[1] - 1), states[:, :-1])

    def stats(critic, lam):
        adv, _ = arl.gae_advantages(rewards, critic[states[:, :-1]], alive, 1.0, lam)
        g = torch.zeros(len(states), 7, 4, dtype=torch.float64)
        g.index_put_(rows, adv[..., None] * score * alive[..., None], accumulate=True)
        return ((g.mean(0) - exact).norm() / exact.norm()).item(), g.var(0).sum().item()

    tables = [('exact V', arl.policy_evaluation(P, R, pi, 1.0), 1),
              ('stale V', arl.policy_evaluation(P, R, torch.full((7, 4), 0.25, dtype=torch.float64), 1.0), 2),
              ('V = 0', torch.zeros(7, dtype=torch.float64), 3)]
    lams = [k / 20 for k in range(21)]
    fig = Figure(980, 400, title='GAE on the repair MDP: policy-gradient bias and variance vs lambda',
                 subtitle='40,000 episodes; exact gradient from autograd through policy_evaluation.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='lambda', ylabel='relative bias of the gradient', xlim=(0, 1))
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='lambda', ylabel='variance per episode', xlim=(0, 1))
    for name, table, slot in tables:
        b, v = zip(*[stats(table, lam) for lam in lams])
        left.line(lams, b, label=name, series=slot, end_label=False)
        right.line(lams, v, label=name, series=slot, end_label=False)
    noise = (stats(tables[0][1], 1.0)[1] / len(states)) ** 0.5 / exact.norm().item()
    left.hline(noise)
    left.text(0.02, noise, f'Monte Carlo noise floor {noise:.3f}', dy=-6)
    left.legend()
    right.legend()
    fig.save('ch07-gae-lambda')


def difficulty_bias():
    G = 8
    k = torch.arange(G + 1, dtype=torch.float64)
    groups = (torch.arange(G)[None] < k[:, None]).double().flatten()
    adv = {norm: arl.group_advantages(groups, G, normalize_std=norm).view(G + 1, G).abs().sum(-1)
           for norm in (False, True)}
    ps = [0.02 + 0.96 * j / 48 for j in range(49)]
    curves = {False: [], True: []}
    for p in ps:
        prob = torch.distributions.Binomial(G, torch.tensor(p, dtype=torch.float64)).log_prob(k).exp()
        for norm in curves:
            curves[norm].append((prob * adv[norm]).sum().item())
    fig = Figure(980, 400, title='Standard-deviation normalization and task difficulty (G = 8, binary reward)',
                 subtitle='Expected total |advantage| of a group, computed exactly from the binomial distribution.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='task pass rate p', ylabel='E[ sum_i |A_i| ] per group', xlim=(0, 1))
    left.line(ps, curves[False], label='mean only (Dr. GRPO, RLOO-like)', series=1, end_label=False)
    left.line(ps, curves[True], label='mean / std (GRPO)', series=2, end_label=False)
    left.legend()
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='task pass rate p', ylabel='ratio: normalized / mean only',
                     xlim=(0, 1), ylim=(0, 3))
    right.line(ps, [b / a for a, b in zip(curves[False], curves[True])], label='ratio', series=2, end_label=False)
    right.hline(2.0, label='2.0 at p = 0.5')
    fig.save('ch07-difficulty-bias')


def advantage_strip():
    tok = arl.build_tokenizer(arl.make_tasks('all'))
    train = arl.make_tasks('train')
    base = arl.make_base_agent(tok)

    def sample(n, seed):
        ts = [t for t in train for _ in range(n)]
        return arl.rollout_batch(base, [arl.BugFixEnv() for _ in ts], ts, tok,
                                 generator=torch.Generator().manual_seed(seed))

    train_batch = arl.collate(sample(24, 1), tok.pad_id)
    torch.manual_seed(0)
    critic = arl.make_critic(base)
    y = arl.returns_to_go(arl.outcome_rewards(train_batch['rewards'], train_batch['action_mask']),
                          train_batch['action_mask'])
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(150):
        idx = torch.randint(len(y), (32,), generator=gen)
        loss = arl.masked_mean((arl.critic_values(critic, train_batch['ids'][idx]) - y[idx]) ** 2,
                               train_batch['action_mask'][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    fresh = sample(8, 2)
    index = next(i for i, t in enumerate(fresh) if t.reward > 0 and t.info['n_turns'] >= 4)
    group = fresh[index - index % 8: index - index % 8 + 8]
    episode = fresh[index]
    row = arl.collate([episode], tok.pad_id)
    m = row['action_mask']
    r1 = arl.outcome_rewards(row['rewards'], m)
    with torch.no_grad():
        v = arl.critic_values(critic, row['ids'])
    schemes = [('outcome (RLOO)', (arl.rloo_advantages(torch.tensor([t.reward for t in group]), 8)[index % 8] * m)[0]),
               ('token GAE, lam 0.95', arl.gae_advantages(r1, v, m, 1.0, 0.95)[0][0]),
               ('turn GAE, lam 0.95', arl.turn_advantages(r1, v, m, row['turn_ids'], 1.0, 0.95)[0])]
    # Cells: prompt collapsed, each observation span collapsed, policy tokens one by one.
    names = {c: n for n, c in arl.OWNER.items()}
    cells, j = [], 1
    ids, owner = episode.ids.tolist(), episode.owner.tolist()
    while j < len(ids):
        o = names[owner[j]]
        if o == 'policy':
            cells.append((tok.itos[ids[j]], 'policy', j - 1))
            j += 1
            continue
        k = j
        while k < len(ids) and names[owner[k]] != 'policy':
            k += 1
        label = 'issue' if o == 'prompt' else f'obs ({k - j})'
        cells.append((label, 'prompt' if o == 'prompt' else 'tool', None))
        j = k
    vmax = max(float(a.abs().max()) for _, a in schemes)
    widths = [max(36, 6.6 * len(t) + 10) for t, _, _ in cells]
    fig = Figure(int(170 + sum(w + 2 for w in widths) + 20), 300, title=f'Per-token advantages on one successful episode ({episode.info["task"]})',
                 subtitle='Blue positive, red negative, intensity = magnitude; gray = no advantage (prompt, template, tool).')
    x0, y = 170, 84
    slot = {'policy': 1, 'tool': 2, 'prompt': 3}
    x = x0
    for (text, kind, _), w in zip(cells, widths):
        fig.tokens(x, y, [(text, slot[kind], f'{kind}: {text}')], h=24, size=10, cell=w)
        x += w + 2
    fig.text(x0 - 10, y + 16, 'tokens', cls='t2', size=11, anchor='end')
    for r, (name, adv) in enumerate(schemes):
        yy = y + 40 + r * 40
        fig.text(x0 - 10, yy + 17, name, cls='t2', size=11, anchor='end')
        x = x0
        for (text, kind, col), w in zip(cells, widths):
            if col is None:
                fig.body.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="26" rx="3" '
                                f'style="fill:var(--neutral)"><title>{esc(kind)}: no advantage</title></rect>')
            else:
                a = float(adv[col])
                op = 0.1 + 0.8 * abs(a) / vmax
                fill = DIV_POS if a >= 0 else DIV_NEG
                fig.body.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="26" rx="3" fill="{fill}" '
                                f'fill-opacity="{op:.2f}"><title>{esc(text)}: {a:+.3f}</title></rect>')
                fig.text(x + w / 2, yy + 17, f'{a:+.2f}', size=9, anchor='middle')
            x += w + 2
    fig.text(20, 286, f'Critic: TinyAgent value model from the base agent, 150 steps on 480 episodes. '
             f'Episode reward {episode.reward:g}; group of 8 attempts at the same task.', cls='t2', size=11)
    fig.save('ch07-advantage-strip')


FIGURES = {'ch07-gae-lambda': gae_lambda, 'ch07-difficulty-bias': difficulty_bias,
           'ch07-advantage-strip': advantage_strip}
