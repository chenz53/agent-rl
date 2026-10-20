"""Figures for Chapter 8: importance weights, the clipped-objective landscape, and per-epoch PPO diagnostics.

Computations mirror the chapter's code with the same seeds, so plotted numbers match the printed ones.
"""
import copy

import torch

import agentrl as arl
from svgplot import Figure

_CACHE = {}


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        _CACHE.update(tok=tok, train=arl.make_tasks('train'), base=arl.make_base_agent(tok))
    return _CACHE


def _sample(model, tasks, n, seed):
    c = _setup()
    bt = [t for t in tasks for _ in range(n)]
    return arl.rollout_batch(model, [arl.BugFixEnv() for _ in bt], bt, c['tok'],
                             generator=torch.Generator().manual_seed(seed))


def _old_batch():
    c = _setup()
    if 'old' not in c:
        batch = arl.collate(_sample(c['base'], c['train'], 16, seed=0), c['tok'].pad_id)
        with torch.no_grad():
            c['old'] = (batch, arl.sequence_log_probs(c['base'], batch['ids']))
    return c['old']


def _reinforce_steps(model, opt, k, gen):
    c = _setup()
    for _ in range(k):
        tasks = [c['train'][i] for i in torch.randint(len(c['train']), (64,), generator=gen).tolist()]
        trajs = arl.rollout_batch(model, [arl.BugFixEnv() for _ in tasks], tasks, c['tok'], generator=gen)
        batch = arl.collate(trajs, c['tok'].pad_id)
        arl.policy_gradient_step(model, opt, batch, batch['rewards'] - batch['rewards'].mean())


def is_weights():
    c = _setup()
    batch, old_logp = _old_batch()
    mask = batch['action_mask']
    moved, gen = copy.deepcopy(c['base']), torch.Generator().manual_seed(1)
    opt = torch.optim.Adam(moved.parameters(), lr=2e-3)
    ks, ess, done, log_w10 = [0, 1, 3, 10, 30], [], 0, None
    for k in ks:
        _reinforce_steps(moved, opt, k - done, gen)
        done = k
        with torch.no_grad():
            log_w = ((arl.sequence_log_probs(moved, batch['ids']) - old_logp) * mask).sum(-1)
        w = log_w.exp()
        ess.append((w.sum() ** 2 / (w ** 2).sum()).item() / len(w))
        if k == 10:
            log_w10 = log_w
    fig = Figure(1000, 400, title='Exact trajectory importance weights degrade fast',
                 subtitle='320 base-agent episodes reweighted to a policy after k REINFORCE updates (Eq. 8.1).')
    left = fig.axes(0, 0, 0.5, 1, xlabel='REINFORCE updates k', ylabel='effective sample size / N', ylim=(0, 1.05))
    left.line(ks, ess, label='ESS / N', series=1, markers=True, end_label=False)
    for k, e in zip(ks, ess):
        left.text(k, e, f'{e:.2f}', dx=6, dy=-8)
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='policy tokens in the episode', ylabel='log w after 10 updates')
    right.scatter(mask.sum(-1).tolist(), log_w10.tolist(), series=1, r=3)
    right.hline(0.0)
    fig.save('ch08-is-weights')


def clip_landscape():
    ratios = [0.3 + 0.01 * i for i in range(471)]  # 0.3 .. 5.0
    fig = Figure(1000, 400, title='The clipped surrogate, per token (eps = 0.2)',
                 subtitle='Objective min(rA, clip(r, 0.8, 1.2) A) against the ratio r; flat pieces have zero gradient.')
    for col, sign in enumerate([1.0, -1.0]):
        ax = fig.axes(0.5 * col, 0, 0.5, 1, xlabel='ratio r = pi_theta / pi_old', ylabel='per-token objective',
                      title=f'A = {sign:+.0f}', xlim=(0.3, 5.0))
        r = torch.tensor(ratios).log()[None]
        adv = torch.tensor([sign])
        clipped = []
        dual = []
        for i in range(len(ratios)):
            loss, _ = arl.ppo_policy_loss(r[:, i:i + 1], torch.zeros(1, 1), adv, torch.ones(1, 1))
            clipped.append(-loss.item())
            loss, _ = arl.ppo_policy_loss(r[:, i:i + 1], torch.zeros(1, 1), adv, torch.ones(1, 1), dual_clip=3.0)
            dual.append(-loss.item())
        ax.line(ratios, [sign * x for x in ratios], label='unclipped rA', series=2, dashed=True, end_label=False)
        ax.line(ratios, clipped, label='clipped objective', series=1, end_label=False)
        if sign < 0:
            ax.line(ratios, dual, label='with dual clip c = 3', series=3, end_label=False)
        ax.vline(0.8)
        ax.vline(1.2)
        ax.legend()
    fig.save('ch08-clip-landscape')


def _mc_returns(batch):
    return arl.returns_to_go(arl.outcome_rewards(batch['rewards'], batch['action_mask']), batch['action_mask'])


def _warm_critic(policy, steps=100, seed=0):
    c = _setup()
    critic, data = arl.make_critic(policy), arl.collate(_sample(policy, c['train'], 24, seed=1), c['tok'].pad_id)
    opt, gen, targets = torch.optim.Adam(critic.parameters(), lr=1e-3), torch.Generator().manual_seed(seed), _mc_returns(data)
    for _ in range(steps):
        idx = torch.randint(len(targets), (32,), generator=gen)
        loss = arl.masked_mean((arl.critic_values(critic, data['ids'][idx]) - targets[idx]) ** 2,
                               data['action_mask'][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return critic


def _prepare(policy, critic, batch, lam):
    m = batch['action_mask']
    with torch.no_grad():
        old_logp = arl.sequence_log_probs(policy, batch['ids'])
        old_values = arl.critic_values(critic, batch['ids'])
    adv, returns = arl.gae_advantages(arl.outcome_rewards(batch['rewards'], m), old_values, m, 1.0, lam)
    on = m > 0
    adv = (adv - adv[on].mean()) / (adv[on].std() + 1e-8) * m
    return old_logp, old_values, adv, returns


def _epochs(policy, critic, popt, vopt, batch, prepared, epochs, minibatches, gen):
    old_logp, old_values, adv, returns = prepared
    ids, m, per_epoch = batch['ids'], batch['action_mask'], []
    for _ in range(epochs):
        logs = []
        for rows in torch.randperm(len(ids), generator=gen).chunk(minibatches):
            loss, stats = arl.ppo_policy_loss(arl.sequence_log_probs(policy, ids[rows]), old_logp[rows], adv[rows],
                                              m[rows])
            popt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            popt.step()
            vloss = arl.value_loss(arl.critic_values(critic, ids[rows]), old_values[rows], returns[rows], m[rows])
            vopt.zero_grad()
            vloss.backward()
            vopt.step()
            logs.append(stats)
        per_epoch.append({k: sum(s[k] for s in logs) / len(logs) for k in logs[0]})
    return per_epoch


def _run_ppo(lam=1.0, iters=30, seed=0):
    c = _setup()
    torch.manual_seed(seed)
    policy, critic, gen = copy.deepcopy(c['base']), _warm_critic(c['base']), torch.Generator().manual_seed(seed)
    popt, vopt = torch.optim.Adam(policy.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    out = []
    for _ in range(iters):
        tasks = [c['train'][i] for i in torch.randint(len(c['train']), (64,), generator=gen).tolist()]
        batch = arl.collate(arl.rollout_batch(policy, [arl.BugFixEnv() for _ in tasks], tasks, c['tok'],
                                              generator=gen), c['tok'].pad_id)
        out.append(_epochs(policy, critic, popt, vopt, batch, _prepare(policy, critic, batch, lam), 4, 4, gen))
    return out


def epochs():
    c = _setup()
    run = _run_ppo()
    run_clip = [sum(it[e]['clip_frac'] for it in run) / len(run) for e in range(4)]
    run_kl = [sum(it[e]['approx_kl'] for it in run) / len(run) for e in range(4)]
    batch, _ = _old_batch()
    torch.manual_seed(0)
    policy, critic = copy.deepcopy(c['base']), _warm_critic(c['base'])
    popt, vopt = torch.optim.Adam(policy.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    many = _epochs(policy, critic, popt, vopt, batch, _prepare(policy, critic, batch, 1.0), 16, 4,
                   torch.Generator().manual_seed(0))
    fig = Figure(1000, 400, title='PPO diagnostics across epochs',
                 subtitle='4 minibatches per epoch. Training run: mean of 30 iterations (lambda = 1). One batch: 320 episodes. '
                          'Dashed: KL target 0.02.')
    xs16, xs4 = list(range(1, 17)), list(range(1, 5))
    left = fig.axes(0, 0, 0.5, 1, xlabel='epoch', ylabel='clip fraction', xlim=(1, 16))
    left.line(xs4, run_clip, label='training run (4 epochs)', series=1, markers=True, end_label=False)
    left.line(xs16, [e['clip_frac'] for e in many], label='one batch, 16 epochs', series=2, markers=True,
              end_label=False)
    left.legend()
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='epoch', ylabel='approximate KL(old || new)', xlim=(1, 16))
    right.line(xs4, run_kl, label='training run (4 epochs)', series=1, markers=True, end_label=False)
    right.line(xs16, [e['approx_kl'] for e in many], label='one batch, 16 epochs', series=2, markers=True,
               end_label=False)
    right.hline(0.02)
    right.legend()
    fig.save('ch08-epochs')


FIGURES = {'ch08-is-weights': is_weights, 'ch08-clip-landscape': clip_landscape, 'ch08-epochs': epochs}
