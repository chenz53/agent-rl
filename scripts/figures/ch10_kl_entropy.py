"""Figures for Chapter 10: KL estimator bias/variance/gradient error, forward vs reverse KL fits, entropy curves.

The estimator and fit figures enumerate exactly; the training figure reruns the chapter's arms with the same seeds.
"""
import copy
import math

import torch

import agentrl as arl
from svgplot import Figure

_CACHE = {}


def _smooth(xs, w=5):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def kl_estimators():
    torch.manual_seed(0)
    logits = torch.randn(50)
    noise = torch.randn(50)
    kinds = ['k1', 'k2', 'k3']
    kl_x, bias, rel_std, grad_err = [], {k: [] for k in kinds}, {k: [] for k in kinds}, {k: [] for k in kinds}
    for scale in torch.logspace(-1.5, 0.5, 30).tolist():
        theta = logits.clone().requires_grad_(True)
        ref_logp = torch.log_softmax(logits + scale * noise, -1)
        logp = torch.log_softmax(theta, -1)
        p = logp.exp()
        true_kl = (p * (logp - ref_logp)).sum()
        g_rev = torch.autograd.grad(true_kl, theta)[0]
        kl_x.append(true_kl.item())
        for k in kinds:
            logp = torch.log_softmax(theta, -1)
            values = arl.kl_estimate(logp, ref_logp, k)                  # [K]
            w = logp.exp().detach()
            mean = (w * values).sum()
            g = torch.autograd.grad(mean, theta)[0]
            std = (w * (values.detach() - mean.detach()) ** 2).sum().sqrt()
            bias[k].append(mean.item() / true_kl.item() - 1)
            rel_std[k].append(std.item() / true_kl.item())
            grad_err[k].append(((g - g_rev).norm() / g_rev.norm()).item())
    fig = Figure(1080, 400, title='KL estimators k1, k2, k3 as the policy moves away from the reference',
                 subtitle='Exact moments over 50 outcomes (no sampling). Right: error of the expected gradient of each '
                          'estimator used as a loss, against the true gradient of KL(pi || ref).')
    lo, hi = min(kl_x), max(kl_x)
    a = fig.axes(0, 0, 1 / 3, 1, xlabel='true KL(pi || ref)', ylabel='relative bias  E[k]/KL - 1', xlog=True,
                 xlim=(lo, hi), title='Bias')
    b = fig.axes(1 / 3, 0, 1 / 3, 1, xlabel='true KL(pi || ref)', ylabel='per-sample std / KL', xlog=True, ylog=True,
                 xlim=(lo, hi), ylim=(1, 300), title='Spread of one sample')
    c = fig.axes(2 / 3, 0, 1 / 3, 1, xlabel='true KL(pi || ref)', ylabel='relative gradient error', xlog=True,
                 xlim=(lo, hi), ylim=(0, 1.4), title='Gradient as a loss')
    for s, k in enumerate(kinds, start=1):
        a.line(kl_x, bias[k], label=k, series=s, end_label=False)
        b.line(kl_x, rel_std[k], label=k, series=s, end_label=False)
        c.line(kl_x, grad_err[k], label=k, series=s, end_label=False)
    c.text(lo * 1.3, 0.06, 'k2 (= k3+): exact', cls='note')
    a.text(lo * 1.3, 0.06, 'k1 and k3: exactly 0', cls='note')
    a.legend()
    b.legend()
    c.legend()
    fig.save('ch10-kl-estimators')


def forward_reverse():
    x = torch.linspace(-6, 6, 121)

    def gauss(mu, sigma):
        return torch.log_softmax(-0.5 * ((x - mu) / sigma) ** 2, -1)
    target = torch.logsumexp(torch.stack([gauss(-2.5, 0.6), gauss(2.5, 0.6)]), 0) - math.log(2)
    fits = {}
    for direction in ['forward', 'reverse']:
        mu, log_s = torch.tensor(1.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
        opt = torch.optim.Adam([mu, log_s], lr=0.05)
        for _ in range(600):
            q = gauss(mu, log_s.exp())
            p, u, v = (target, target, q) if direction == 'forward' else (q, q, target)
            kl = (p.exp() * (u - v)).sum()
            opt.zero_grad()
            kl.backward()
            opt.step()
        fits[direction] = (gauss(mu, log_s.exp()).detach().exp(), mu.item(), log_s.exp().item())
    fig = Figure(860, 380, title='Fitting one Gaussian to two modes',
                 subtitle='Forward KL(target || q) covers both modes and puts 30% of its mass in the valley; '
                          'reverse KL(q || target) commits to one mode.')
    ax = fig.axes(0, 0, 1, 1, xlabel='outcome x (121-point grid)', ylabel='probability', xlim=(-6, 6))
    xs = x.tolist()
    ax.band(xs, [0.0] * len(xs), target.exp().tolist(), series=3, opacity=0.25)
    ax.line(xs, target.exp().tolist(), label='target (bimodal)', series=3, end_label=False)
    q, mu, s = fits['forward']
    ax.line(xs, q.tolist(), label=f'forward-KL fit (mu {mu:.1f}, sigma {s:.2f})', series=2, end_label=False)
    q, mu, s = fits['reverse']
    ax.line(xs, q.tolist(), label=f'reverse-KL fit (mu {mu:.1f}, sigma {s:.2f})', series=1, end_label=False, dashed=True)
    ax.legend()
    fig.save('ch10-forward-reverse')


ARMS = {
    'none': ({}, {}),
    'KL k3+ 0.05': (dict(kl_coef=0.05, kl_kind='k3+'), {}),
    'KL k3+ 0.3': (dict(kl_coef=0.3, kl_kind='k3+'), {}),
    'KL k3+ 1.0': (dict(kl_coef=1.0, kl_kind='k3+'), {}),
    'KL k3 0.3': (dict(kl_coef=0.3, kl_kind='k3'), {}),
    'KL reward 0.3': (dict(kl_coef=0.3, kl_kind='k1'), dict(kl_in_reward=True)),
    'entropy 0.03': ({}, dict(entropy_coef=0.03)),
}
SLOT = {'none': 2, 'KL k3+ 0.05': 1, 'KL k3+ 0.3': 3, 'KL k3+ 1.0': 4,
        'entropy 0.03': 1, 'KL k3 0.3': 3, 'KL reward 0.3': 4}


def entropy_curves():
    torch.set_num_threads(4)                      # same setting as the chapter, so the runs match
    tok = arl.build_tokenizer(arl.make_tasks('all'))
    train = arl.make_tasks('train')
    rl_base = arl.make_rl_base(tok)
    hist = {}
    for name, (fields, kwargs) in ARMS.items():
        model = copy.deepcopy(rl_base)
        hist[name] = arl.train_regularized(model, train, tok, arl.RLConfig(**fields), 40, **kwargs)
    steps = list(range(40))
    fig = Figure(1080, 640, title='Entropy and reward under different regularizers (TinyAgent, 40 GRPO steps, seed 0)',
                 subtitle='Top: mean policy-token entropy during the update. Bottom: training batch reward. '
                          '5-step moving averages; one seed, a mechanism demonstration.')
    left = ['none', 'KL k3+ 0.05', 'KL k3+ 0.3', 'KL k3+ 1.0']
    right = ['none', 'entropy 0.03', 'KL k3 0.3', 'KL reward 0.3']
    for col, names in enumerate([left, right]):
        top = fig.axes(col / 2, 0, 0.5, 0.5, xlabel='step', ylabel='entropy (nats)', ylim=(0, 0.6), xlim=(0, 39),
                       title='KL in the loss, k3+ gradient' if col == 0 else 'Entropy bonus, k3 loss, KL in the reward')
        bottom = fig.axes(col / 2, 0.5, 0.5, 0.5, xlabel='step', ylabel='batch reward', ylim=(0, 1), xlim=(0, 39))
        for name in names:
            series = SLOT[name]
            top.line(steps, _smooth([h['entropy'] for h in hist[name]]), label=name, series=series, end_label=False)
            bottom.line(steps, _smooth([h['reward'] for h in hist[name]]), label=name, series=series, end_label=False)
        top.legend()
    fig.save('ch10-entropy-curves')


FIGURES = {'ch10-kl-estimators': kl_estimators, 'ch10-forward-reverse': forward_reverse,
           'ch10-entropy-curves': entropy_curves}
