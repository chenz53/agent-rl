"""Figures for Chapter 6: baseline variance, loss-aggregation weights, and the first REINFORCE curve.

Every computation mirrors the chapter's code (same seeds), so the plotted numbers match the printed ones.
"""
import copy

import torch

import agentrl as arl
from svgplot import Figure

_CACHE = {}


def _setup():
    if not _CACHE:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        _CACHE.update(tok=tok, train=arl.make_tasks('train'), heldout=arl.make_tasks('heldout'),
                      base=arl.make_base_agent(tok))
    return _CACHE


def _sample(model, tasks, n_per_task, seed):
    c = _setup()
    batch_tasks = [t for t in tasks for _ in range(n_per_task)]
    return arl.rollout_batch(model, [arl.BugFixEnv() for _ in batch_tasks], batch_tasks, c['tok'],
                             generator=torch.Generator().manual_seed(seed))


def _scores(model, trajs):
    rows = []
    for t in trajs:
        logp = arl.sequence_log_probs(model, t.ids[None])[0]
        grads = torch.autograd.grad((t.action_mask * logp).sum(), list(model.parameters()))
        rows.append(torch.cat([g.flatten() for g in grads]))
    return torch.stack(rows)


def baseline_variance():
    c = _setup()
    probe = _sample(c['base'], c['train'], 12, seed=3)
    scores = _scores(c['base'], probe)
    rewards = torch.tensor([t.reward for t in probe])
    var = lambda b: ((rewards - b)[:, None] * scores).var(0).sum().item()
    sq = scores.pow(2).sum(-1)
    b_star = ((rewards * sq).sum() / sq.sum()).item()
    other = _sample(c['base'], c['train'], 12, seed=4)
    task_mean = {t.name: sum(o.reward for o in other if o.info['task'] == t.name) / 12 for t in c['train']}
    per_task = torch.tensor([task_mean[t.info['task']] for t in probe])
    bs = [k / 40 for k in range(41)]
    fig = Figure(760, 400, title='REINFORCE estimator variance vs baseline (base agent, 240 episodes)',
                 subtitle='Trace of the covariance of per-episode gradients over all TinyAgent parameters.')
    ax = fig.axes(0, 0, 1, 1, xlabel='constant baseline b', ylabel='total variance', xlim=(0, 1))
    ax.line(bs, [var(b) for b in bs], label='constant b', series=1, end_label=False)
    ax.hline(var(per_task), label=f'per-task mean (independent batch): {var(per_task):.1f}')
    named = [('none', 0.0), ('b*', b_star), ('mean reward', rewards.mean().item())]
    ax.scatter([b for _, b in named], [var(b) for _, b in named], series=2, r=5,
               tips=[f'{n}: b={b:.3f}, variance {var(b):.1f}' for n, b in named])
    for (name, b), dy in zip(named, [-12, 22, -12]):
        ax.text(b, var(b), f'{name} ({var(b):.1f})', dx=6 if name != 'none' else 6, dy=dy)
    fig.save('ch06-baseline-variance')


def aggregation():
    c = _setup()
    probe = _sample(c['base'], c['train'], 12, seed=3)
    batch = arl.collate(probe, c['tok'].pad_id)
    mask = batch['action_mask']
    length = mask.sum(-1)
    per_token = torch.zeros_like(mask, requires_grad=True)
    totals = {}
    for mode in ['token_mean', 'seq_mean_token_mean']:
        w = torch.autograd.grad(arl.aggregate_loss(per_token, mask, mode), per_token)[0].sum(-1)
        totals[mode] = (100 * w / w.sum()).tolist()
    fig = Figure(1000, 400, title='How loss aggregation weights trajectories',
                 subtitle='Left: own length (base-agent probe batch). Right: the rest of the batch gets longer.')
    left = fig.axes(0, 0, 0.5, 1, xlabel='policy tokens in the trajectory', ylabel='share of batch weight (%)')
    left.scatter(length.tolist(), totals['token_mean'], label='length-proportional modes', series=1, r=4)
    left.scatter(length.tolist(), totals['seq_mean_token_mean'], label='seq_mean_token_mean', series=2, r=4)
    left.legend()
    # Right: a 10-token trajectory in a batch of 64 whose other 63 trajectories have mean length L.
    others = list(range(10, 41, 2))
    rel = {m: [] for m in ['token_mean', 'seq_mean_token_mean']}
    for L in others:
        lengths = torch.tensor([10] + [L] * 63)
        m = (torch.arange(40)[None] < lengths[:, None]).float()
        x = torch.zeros_like(m, requires_grad=True)
        for mode in rel:
            w = torch.autograd.grad(arl.aggregate_loss(x, m, mode), x)[0][0, 0].item()
            rel[mode].append(w)
    right = fig.axes(0.5, 0, 0.5, 1, xlabel='mean policy tokens of the other 63 trajectories',
                     ylabel='weight of one token (rel. to L = 10)', ylim=(0, 1.15))
    right.line(others, [w / rel['token_mean'][0] for w in rel['token_mean']], label='token_mean', series=1,
               end_label=False)
    right.line(others, [w / rel['seq_mean_token_mean'][0] for w in rel['seq_mean_token_mean']],
               label='other modes', series=2, end_label=False)
    right.legend()
    fig.save('ch06-aggregation')


def reinforce_curve():
    c = _setup()
    tok, train, heldout, base = c['tok'], c['train'], c['heldout'], c['base']
    base_pass = arl.rollout_stats(_sample(base, train, 16, seed=0))['pass_rate']
    policy = copy.deepcopy(base)
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    baseline, rewards, evals = base_pass, [], []
    for step in range(80):
        if step % 10 == 0:
            evals.append((step, arl.rollout_stats(_sample(policy, heldout, 8, seed=100 + step))['pass_rate']))
        tasks = [train[i] for i in torch.randint(len(train), (64,), generator=gen).tolist()]
        trajs = arl.rollout_batch(policy, [arl.BugFixEnv() for _ in tasks], tasks, tok, generator=gen)
        batch = arl.collate(trajs, tok.pad_id)
        arl.policy_gradient_step(policy, opt, batch, batch['rewards'] - baseline)
        baseline = 0.9 * baseline + 0.1 * batch['rewards'].mean().item()
        rewards.append(batch['rewards'].mean().item())
    evals.append((80, arl.rollout_stats(_sample(policy, heldout, 8, seed=180))['pass_rate']))
    fig = Figure(820, 400, title='First REINFORCE run on the BugFix gym',
                 subtitle='Base agent + EMA baseline, 64 episodes per step, token_mean, Adam lr 2e-3, one seed.')
    ax = fig.axes(0, 0, 1, 1, xlabel='update step', ylabel='reward / pass rate', ylim=(0, 1.0), xlim=(0, 80))
    ax.line(list(range(80)), rewards, label='train batch reward', series=1, width=1.5,
            end_label=False)
    ax.line([s for s, _ in evals], [p for _, p in evals], label='held-out pass rate', series=2, markers=True,
            end_label=False)
    ax.hline(0.21, label='blind guesser 0.21')
    ax.hline(base_pass, label=f'base agent (train) {base_pass:.2f}')
    ax.legend()
    fig.save('ch06-reinforce-curve')


FIGURES = {'ch06-baseline-variance': baseline_variance, 'ch06-aggregation': aggregation,
           'ch06-reinforce-curve': reinforce_curve}
