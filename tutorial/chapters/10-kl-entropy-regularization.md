# 10. KL, Entropy, and Regularization

The default GRPO run of [Chapter 9](09-grpo-and-variants.md#94-shortcut-features-in-environment-design) succeeded, and its logs contain two numbers that a post-training engineer would ask about before calling it healthy. The entropy of the policy tokens fell from 0.57 to 0.08 nats in 80 steps, and by the end four of every five groups had identical rewards and carried no gradient. The policy had become nearly deterministic. On this gym that is what success looks like, because each task has one right procedure. On a real coding distribution, a policy that has stopped exploring cannot discover a better procedure later, and a collapsed sampler makes pass@$k$ no better than pass@1. The opposite worry is also real: an RL objective with nothing holding the policy near its starting point is free to drift into regions where the reward is wrong, the language degrades, or skills that the reward does not measure are forgotten.

This chapter studies the two standard regularizers that address these worries. A **KL penalty** pulls the policy toward a frozen **reference policy**, usually the starting checkpoint. An **entropy bonus** pushes it away from determinism. Both look like one extra line in a loss, and both hide subtleties that change the gradient a trainer actually computes: the three common KL estimators produce three different gradients when used as losses, one of them zero in expectation and one the gradient of a different divergence; putting the KL in the reward and putting it in the loss differ for sequences; and an entropy bonus over a full vocabulary pushes probability onto tokens the policy should never emit. We derive each fact, check it numerically on distributions small enough to enumerate, then compare the regularizers on TinyAgent: no regularizer, KL penalties of several strengths and placements, entropy bonuses, and a higher temperature, measured by entropy, held-out pass rate, and pass@$k$; clip-higher gets a one-batch experiment of its own. Two functions are saved, `kl_estimate` and `entropy_bonus`, together with the update, training loop, and sequence-KL helper that use them.

```python
#@save
import copy
import math
import torch
import agentrl as arl
```

The TinyAgent experiments use the book's default setting from Chapter 9: the anonymized one-edit gym, the RL base agent, and the training and held-out splits.

```python
torch.set_num_threads(4)              # fixed CPU threads: fast and repeatable on a shared check host
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
rl_base = arl.make_rl_base(tok)
```

## 10.1 The reference policy

The **reference policy** $\pi_{\text{ref}}$ is a fixed policy, in practice a frozen copy of the model at the start of RL (the SFT or base checkpoint). The KL-regularized objective is

$$
J_\beta(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

with the sequence KL decomposed into per-token terms along the policy's own prefixes ([Appendix A.6](A-math.md#a6-kl-divergence)). Three reasons for it recur in the literature, and they are different reasons. First, in RLHF the reward is a learned model that is accurate only near the data it was trained on; the KL keeps the policy where the reward can be trusted. Second, the reference carries capabilities the reward does not measure (fluency, formatting, knowledge), and the KL limits how much of them the policy may trade away. Third, the KL is a *cumulative* trust region: PPO's clip (Chapter 8) limits how far one update moves from $\pi_{\text{old}}$, while the KL limits how far the whole run moves from $\pi_{\text{ref}}$. With a verifiable reward such as hidden tests, the first reason disappears, which is the core of the case for dropping the KL (Section 10.7).

What does the regularized objective want? For a single decision (a bandit) the answer is in closed form. Write $\pi$ for a distribution over $K$ actions with rewards $r(a)$. Then

$$
\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) = -\beta \sum_a \pi(a) \log \frac{\pi(a)}{\pi_{\text{ref}}(a)\, e^{r(a)/\beta}} = \beta \log Z - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*),
$$

where $\pi^*(a) = \pi_{\text{ref}}(a)\, e^{r(a)/\beta} / Z$ and $Z = \sum_a \pi_{\text{ref}}(a)\, e^{r(a)/\beta}$. Since the KL is nonnegative and zero only at $\pi = \pi^*$, the optimum is $\pi^*$ and the optimal value is $\beta \log Z$. The optimum *tilts* the reference by the exponentiated reward: $\beta$ is a temperature that interpolates between $\pi_{\text{ref}}$ ($\beta \to \infty$) and the best action **in the reference's support** ($\beta \to 0$). An action the reference never takes stays at probability zero for every $\beta$. The same derivation with $\pi_{\text{ref}}$ uniform gives the entropy bonus, because $\mathrm{KL}(\pi \,\|\, \mathrm{Uniform}) = \log K - H(\pi)$: maximizing $\mathbb{E}[r] + \alpha H(\pi)$ is a KL penalty toward the uniform distribution, with optimum $\mathrm{softmax}(r/\alpha)$. [Chapter 15](15-sft-and-rejection-sampling.md) uses the same closed form to derive DPO.

```python
def kl_regularized_optimum(reward, ref_logp, beta):
    """argmax over pi of E_pi[r] - beta * KL(pi || ref): log-softmax(ref_logp + r / beta), [K] -> [K]."""
    return torch.log_softmax(ref_logp + reward / beta, dim=-1)

def bandit_report(logp, reward, ref_logp):
    """Expected reward, KL(pi || ref), and entropy of a categorical policy given as log-probs [K]."""
    p = logp.exp()
    kl = torch.where(p > 0, p * (logp - ref_logp), 0.0).sum()     # 0 log 0 := 0
    entropy = -torch.where(p > 0, p * logp, 0.0).sum()
    return (p * reward).sum().item(), kl.item(), entropy.item()
```

The bandit below has five actions. Action 4 has the highest reward, but the reference never takes it; actions 0 and 1 are good, and the reference prefers the weaker one.

```python
reward = torch.tensor([1.0, 0.6, 0.0, 0.0, 1.2])                  # [K]: action 4 is the best
ref_logp = torch.tensor([0.05, 0.45, 0.30, 0.20, 0.0]).log()      # [K]: the reference never takes action 4
for beta in [10.0, 1.0, 0.3, 0.1, 0.01]:
    logp = kl_regularized_optimum(reward, ref_logp, beta)
    r, kl, h = bandit_report(logp, reward, ref_logp)
    print(f'beta {beta:5.2f} | E[r] {r:.3f} | KL {kl:.3f} | entropy {h:.3f} | pi* {[round(x, 2) for x in logp.exp().tolist()]}')
alpha = 0.3                                                       # entropy bonus = KL to a uniform reference
logp = kl_regularized_optimum(reward, torch.full((5,), 0.2).log(), alpha)
print(f'entropy bonus 0.3 | E[r] {bandit_report(logp, reward, ref_logp)[0]:.3f} | pi* {[round(x, 2) for x in logp.exp().tolist()]}')
```

Output:

```text
beta 10.00 | E[r] 0.331 | KL 0.001 | entropy 1.190 | pi* [0.05, 0.46, 0.29, 0.19, 0.0]
beta  1.00 | E[r] 0.431 | KL 0.056 | entropy 1.143 | pi* [0.09, 0.56, 0.21, 0.14, 0.0]
beta  0.30 | E[r] 0.650 | KL 0.512 | entropy 0.930 | pi* [0.27, 0.64, 0.06, 0.04, 0.0]
beta  0.10 | E[r] 0.943 | KL 2.273 | entropy 0.411 | pi* [0.86, 0.14, 0.0, 0.0, 0.0]
beta  0.01 | E[r] 1.000 | KL 2.996 | entropy 0.000 | pi* [1.0, 0.0, 0.0, 0.0, 0.0]
entropy bonus 0.3 | E[r] 1.065 | pi* [0.3, 0.08, 0.01, 0.01, 0.59]
```

At $\beta = 10$ the optimum is the reference, with KL 0.001. As $\beta$ falls, mass moves first to action 1, which the reference already likes, and then to action 0, the better action the reference rarely takes; at $\beta = 0.01$ the policy is deterministic on action 0 with expected reward 1.0 and entropy 0. The KL saturates at $\log(1/0.05) = 2.996$, the cost of putting all mass on an action the reference takes 5% of the time. Action 4, worth 1.2, never receives any mass. The entropy bonus with $\alpha = 0.3$ behaves differently: its implicit reference is uniform, so it puts 0.59 on action 4 and reaches a higher expected reward, 1.065, than any KL setting. In a language model the "actions outside the reference's support" are mostly garbage tokens, and the uniform reference's indifference to them is the entropy bonus's main hazard (Section 10.6). We check the closed form against random perturbations of $\pi^*$:

```python
beta = 0.3
best = kl_regularized_optimum(reward, ref_logp, beta)
def objective(logp):
    r, kl, _ = bandit_report(logp, reward, ref_logp)
    return r - beta * kl
torch.manual_seed(0)
others = [objective(torch.log_softmax(best + 0.3 * torch.randn(5), -1)) for _ in range(1000)]
log_z = torch.logsumexp(ref_logp + reward / beta, -1).item()
print(f'J(pi*) {objective(best):.4f} | beta log Z {beta * log_z:.4f} | J(pi*) - best of 1000 perturbations {objective(best) - max(others):.1e}')
```

Output:

```text
J(pi*) 0.4961 | beta log Z 0.4961 | J(pi*) - best of 1000 perturbations 3.3e-05
```

The value at $\pi^*$ equals $\beta \log Z$, and no perturbation does better, as the identity above requires.

## 10.2 Estimating KL from samples

The trainer never sees full distributions over sequences. It sees sampled tokens $y \sim \pi_\theta(\cdot \mid c)$ and two numbers per token, $\log \pi_\theta(y \mid c)$ and $\log \pi_{\text{ref}}(y \mid c)$. Write their difference as $\ell = \log \pi_{\text{ref}}(y \mid c) - \log \pi_\theta(y \mid c)$. John Schulman's note [*Approximating KL Divergence*](http://joschu.net/blog/kl-approx.html) (2020) compares three per-sample estimators of $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$:

$$
k_1 = -\ell, \qquad k_2 = \tfrac{1}{2}\ell^2, \qquad k_3 = e^{\ell} - 1 - \ell .
$$

$k_1$ is unbiased by definition and takes negative values on many samples. $k_3$ adds $e^\ell - 1$, which has mean zero under $\pi_\theta$ because $\mathbb{E}_{\pi_\theta}[\pi_{\text{ref}}/\pi_\theta] = 1$; it is $k_1$ with a control variate ([Appendix A.3](A-math.md)), unbiased and never negative since $e^\ell \ge 1 + \ell$. $k_2$ is biased: a Taylor expansion shows that it matches the KL to second order in $\ell$ and overestimates beyond. Chapter 9 saved all three as `kl_penalty`. [Appendix A.7](A-math.md#a7-estimating-kl-from-samples) sampled them at two distances; here we compute their exact mean and per-sample standard deviation by enumerating every outcome of a 50-way distribution, at four distances between policy and reference.

```python
def estimator_moments(logp, ref_logp):
    """Exact mean and std of k1, k2, k3 under x ~ pi, by enumerating the K outcomes: [K] -> dict."""
    p = logp.exp()
    moments = {}
    for kind in ['k1', 'k2', 'k3']:
        k = arl.kl_penalty(logp, ref_logp, kind)                    # [K], the estimate if outcome x is drawn
        mean = (p * k).sum()
        moments[kind] = (mean.item(), (p * (k - mean) ** 2).sum().sqrt().item())
    return moments

torch.manual_seed(0)
logp = torch.log_softmax(torch.randn(50), -1)                       # pi_theta over K = 50 outcomes
noise = torch.randn(50)
for scale in [0.05, 0.3, 1.0, 2.0]:
    ref_logp = torch.log_softmax(logp + scale * noise, -1)           # pi_ref, further away as scale grows
    true_kl = (logp.exp() * (logp - ref_logp)).sum().item()
    cells = ' | '.join(f'{k} mean/KL {m / true_kl:5.3f} std/KL {s / true_kl:6.2f}'
                       for k, (m, s) in estimator_moments(logp, ref_logp).items())
    print(f'KL {true_kl:.4f} | {cells}')
```

Output:

```text
KL 0.0010 | k1 mean/KL 1.000 std/KL  44.84 | k2 mean/KL 1.011 std/KL   1.29 | k3 mean/KL 1.000 std/KL   1.25
KL 0.0343 | k1 mean/KL 1.000 std/KL   7.89 | k2 mean/KL 1.084 std/KL   1.50 | k3 mean/KL 1.000 std/KL   1.27
KL 0.3247 | k1 mean/KL 1.000 std/KL   2.78 | k2 mean/KL 1.414 std/KL   2.19 | k3 mean/KL 1.000 std/KL   1.38
KL 1.0841 | k1 mean/KL 1.000 std/KL   1.66 | k2 mean/KL 2.041 std/KL   3.23 | k3 mean/KL 1.000 std/KL   1.84
```

The means are exact, so $k_1$ and $k_3$ show a ratio of 1.000 at every distance, and $k_2$'s bias grows with the distance: 1% at KL 0.001, 41% at KL 0.32, and a factor of 2 at KL 1.08. The spread tells the other half. At KL 0.001, typical of a per-token KL between adjacent checkpoints, one $k_1$ sample has a standard deviation 45 times the quantity it estimates, against 1.25 times for $k_3$: averaging $k_1$ to a given precision takes about $(44.84/1.25)^2 \approx 1300$ times as many tokens. The reason is that $k_1$ is first order in $\ell$, so its spread scales like $|\ell|$, while the KL itself is second order. The advantage of $k_3$ shrinks as the distributions separate, and at KL 1.08 its spread is comparable to $k_1$'s. For *logging* a KL, $k_3$ is the right default: unbiased, nonnegative, and low-variance in the regime where training lives. Using an estimator as a *loss* is a separate question.

## 10.3 The gradient each estimator produces as a loss

A KL term in the loss is implemented by evaluating an estimator on the batch's sampled tokens and adding $\beta$ times its mean to the loss. Autograd then differentiates the estimator through $\log \pi_\theta(y \mid c)$, with the sampled tokens held fixed. It does *not* differentiate through the fact that the tokens were sampled from $\pi_\theta$. The expected gradient of such a loss is therefore $\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta k(y)]$, which is not $\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[k(y)]$: the score-function term $\mathbb{E}[k(y)\, \nabla_\theta \log \pi_\theta(y)]$ of [Chapter 6](06-policy-gradient.md) is missing.

Compute each one, with $\nabla \ell = -\nabla \log \pi_\theta(y)$ and $\mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta] = 0$. For $k_1 = -\ell$, the per-sample gradient is $\nabla \log \pi_\theta(y)$, whose expectation is zero: a $k_1$ loss contributes only noise. For $k_2 = \ell^2/2$, the per-sample gradient is $-\ell\, \nabla \log\pi_\theta(y) = \big(\log\pi_\theta(y) - \log\pi_{\text{ref}}(y)\big)\nabla\log\pi_\theta(y)$, and its expectation is the true gradient of the reverse KL,

$$
\nabla_\theta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \sum_y \nabla \pi_\theta(y) \big(\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)\big) + \sum_y \pi_\theta(y)\, \nabla \log \pi_\theta(y) = \mathbb{E}_{\pi_\theta}\big[(\log \pi_\theta - \log \pi_{\text{ref}})\, \nabla \log \pi_\theta\big].
$$

The biased estimator gives the unbiased gradient. For $k_3 = e^\ell - 1 - \ell$, the per-sample gradient is $(1 - e^{\ell})\nabla\log\pi_\theta(y)$, and

$$
\mathbb{E}_{\pi_\theta}\big[(1 - \tfrac{\pi_{\text{ref}}}{\pi_\theta})\nabla \log \pi_\theta\big] = 0 - \sum_y \pi_{\text{ref}}(y)\, \nabla \log \pi_\theta(y) = \nabla_\theta\, \mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta).
$$

The unbiased, low-variance estimator gives the gradient of the **forward** KL, the other direction. Near $\pi_\theta = \pi_{\text{ref}}$ the two directions agree to second order, so their gradients agree to first order, and the difference grows with the distance. We check all of this exactly by weighting each outcome by its (detached) sampling probability, which is the expectation a large batch of samples would approach.

```python
def expected_loss_grad(theta, ref_logp, kind):
    """Gradient of E_{x~pi}[k(x)] with x's sampling weights held fixed, as autograd sees a loss on samples."""
    logp = torch.log_softmax(theta, -1)                             # [K]
    weights = logp.exp().detach()                                   # the sampler's probabilities
    return torch.autograd.grad((weights * kl_estimate(logp, ref_logp, kind)).sum(), theta)[0]

def kl_grads(theta, ref_logp):
    """True gradients of reverse KL(pi || ref) and forward KL(ref || pi) with respect to the logits."""
    logp = torch.log_softmax(theta, -1)
    reverse = (logp.exp() * (logp - ref_logp)).sum()
    forward = (ref_logp.exp() * (ref_logp - logp)).sum()
    return torch.autograd.grad(reverse, theta, retain_graph=True)[0], torch.autograd.grad(forward, theta)[0]
```

The saved function is a thin layer over Chapter 9's `kl_penalty`, which it calls for the values. It adds one thing: kinds ending in `+` use the **straight-through** construction `g - g.detach() + v.detach()`, whose forward value is the estimator $v$ (so the logged number is, say, $k_3$) and whose gradient is that of $g = k_2$ (so the loss pushes along the reverse-KL gradient). verl implements the same option under the same `+` suffix ([`core_algos.kl_penalty`](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py), commit `00094bd`, inspected 2026-10-13), with a comment that $k_1$ and $k_3$ have the right expected value but not the right expected gradient.

```python
#@save
def kl_estimate(logp, ref_logp, kind='k3'):
    """Per-token KL(pi_theta || pi_ref) estimate [B, T]; kind 'k1'|'k2'|'k3', or 'k1+'|'k3+' = value of k, gradient of k2."""
    if kind.endswith('+'):
        value = arl.kl_penalty(logp, ref_logp, kind[:-1])
        grad = arl.kl_penalty(logp, ref_logp, 'k2')
        return grad - grad.detach() + value.detach()                  # straight-through: forward value, k2 gradient
    return arl.kl_penalty(logp, ref_logp, kind)
```

```python
torch.manual_seed(0)
theta = torch.randn(6, requires_grad=True)                          # logits of pi_theta, [K]
ref_logp = torch.log_softmax(torch.randn(6), -1)                    # pi_ref, [K]
reverse, forward = kl_grads(theta, ref_logp)
def rel(g, target):
    return ((g - target).norm() / target.norm()).item()            # relative error of a gradient
for kind in ['k1', 'k2', 'k3', 'k3+']:
    g = expected_loss_grad(theta, ref_logp, kind)
    print(f'{kind:<3} | |E grad| {g.norm():.4f} | error vs reverse-KL grad {rel(g, reverse):.4f}'
          f' | error vs forward-KL grad {rel(g, forward):.4f}')
```

Output:

```text
k1  | |E grad| 0.0000 | error vs reverse-KL grad 1.0000 | error vs forward-KL grad 1.0000
k2  | |E grad| 0.3341 | error vs reverse-KL grad 0.0000 | error vs forward-KL grad 0.3394
k3  | |E grad| 0.4911 | error vs reverse-KL grad 0.4989 | error vs forward-KL grad 0.0000
k3+ | |E grad| 0.3341 | error vs reverse-KL grad 0.0000 | error vs forward-KL grad 0.3394
```

Each line is the derivation in numbers. The expected $k_1$ gradient is zero, so its relative error against either target is exactly 1. The $k_2$ gradient matches the reverse-KL gradient to every printed digit, and the $k_3$ gradient matches the *forward*-KL gradient while missing the reverse one by 50%. The straight-through `k3+` has $k_2$'s gradient exactly. How large is $k_3$'s error at the distances training actually reaches?

```python
noise = torch.randn(6)
for scale in [0.1, 0.5, 1.0, 2.0]:
    ref_logp = torch.log_softmax(theta.detach() + scale * noise, -1)
    true_kl = (torch.softmax(theta, -1) * (torch.log_softmax(theta, -1) - ref_logp)).sum().item()
    reverse, _ = kl_grads(theta, ref_logp)
    errors = ' | '.join(f'{k} {rel(expected_loss_grad(theta, ref_logp, k), reverse):.3f}' for k in ['k2', 'k3', 'k3+'])
    print(f'KL {true_kl:.4f} | gradient error vs reverse KL: {errors}')
```

Output:

```text
KL 0.0022 | gradient error vs reverse KL: k2 0.000 | k3 0.034 | k3+ 0.000
KL 0.0593 | gradient error vs reverse KL: k2 0.000 | k3 0.172 | k3+ 0.000
KL 0.2582 | gradient error vs reverse KL: k2 0.000 | k3 0.336 | k3+ 0.000
KL 1.0933 | gradient error vs reverse KL: k2 0.000 | k3 0.554 | k3+ 0.000
```

At a per-token KL of 0.002, the $k_3$ loss is within 3.4% of the reverse-KL gradient, and at 0.06 it is 17% off. Chapter 9's GRPO preset averaged a per-token $k_3$ of 0.17 against its reference, between the second and third rows, where this table puts the error at 17% to 34%. In that regime, a "reverse-KL penalty" implemented as a $k_3$ loss, as in DeepSeekMath's GRPO and verl's default `low_var_kl`, is closer to a forward-KL penalty. Section 10.5 shows why the direction matters, and Section 10.7 measures the difference on TinyAgent.

![Figure 10.1: KL estimators as the policy moves away from the reference, computed exactly on a 50-outcome distribution. Left: relative bias of each estimator's value. Middle: per-sample standard deviation relative to the true KL (log scale). Right: relative error of each estimator's expected loss gradient against the true reverse-KL gradient.](../figures/ch10-kl-estimators.svg)

Figure 10.1 puts the three properties side by side over two decades of true KL. Read the panels from left to right as three different questions. Is the logged value right? $k_1$ and $k_3$, yes. Is one sample informative? $k_2$ and $k_3$, yes, by an order of magnitude at small KL. Does the loss push in the right direction? Only $k_2$ (and therefore `k3+`), at any distance. No single estimator wins all three, which is why the straight-through combination exists.

> **Pitfall.** Rows of a batch in a PPO-style update come from $\pi_{\text{old}}$, not from the current $\pi_\theta$, after the first minibatch. The derivations above assume on-policy samples. With the small ratios of Chapter 9's default (clip fraction about 0.1%), this is a minor additional bias; with many epochs it is another reason the KL term's gradient is only approximately what its name says.

## 10.4 KL in the reward versus KL in the loss

There are two places to put the penalty. **KL in the reward** subtracts $\beta$ times a per-token $k_1$ (the log-ratio) from the reward and lets the policy gradient handle it, as RLHF-style PPO does. **KL in the loss** adds $\beta$ times a per-token estimator to the loss, as GRPO does. For one token, the reward version is REINFORCE on the reward $-k_1$, whose gradient is $-\mathbb{E}[k_1 \nabla\log\pi_\theta] $; this is exactly the negative reverse-KL gradient derived above, so for a single token, "$k_1$ in the reward" and "$k_2$ in the loss" agree.

For sequences they do not. By Appendix A.6, the sequence KL is $\mathbb{E}\big[\sum_t \mathrm{KL}_t(c_t)\big]$, where $\mathrm{KL}_t(c_t)$ is the per-token KL at prefix $c_t$. Its gradient has two parts: how each $\mathrm{KL}_t$ changes with the parameters at a fixed prefix, and how the choice of token $y_t$ changes which prefixes, and therefore which future KLs, are visited. A per-token loss captures only the first part, because each token's loss term is a function of that token's own log-probability. The policy gradient on the reward captures both, because each token's log-probability is weighted by the KL of everything that follows it (the reward-to-go of [Section 6.5](06-policy-gradient.md)). The difference is credit for *future* divergence: a token that steers the episode into a region where the policy and reference disagree is penalized by the reward version and not by the loss version.

We check this on a two-token sequence model with a vocabulary of 4, small enough to enumerate all 16 sequences: the first token has its own logits, and the second token's logits depend on the first.

```python
torch.manual_seed(0)
V = 4
first = torch.randn(V, requires_grad=True)                          # logits of y1, [V]
second = torch.randn(V, V, requires_grad=True)                      # logits of y2 given y1, [V, V]
ref_first, ref_second = torch.log_softmax(torch.randn(V), -1), torch.log_softmax(torch.randn(V, V), -1)

def token_logps():
    """log pi(y1) broadcast to [V, V] and log pi(y2 | y1) [V, V]; row = y1, column = y2."""
    return torch.log_softmax(first, -1)[:, None].expand(V, V), torch.log_softmax(second, -1)

lp1, lp2 = token_logps()
seq_logp, seq_ref = lp1 + lp2, ref_first[:, None] + ref_second      # [V, V] each
exact = torch.autograd.grad((seq_logp.exp() * (seq_logp - seq_ref)).sum(), [first, second])
weights = seq_logp.exp().detach()                                   # probability of each sampled sequence
```

`surrogate` builds the expected loss that autograd would see for each placement. For the loss placements, each sequence contributes the estimator at both tokens. For the reward placement, each token's log-probability is multiplied by the detached sum of $k_1$ from that token to the end.

```python
def surrogate(mode):
    """Expected surrogate over all V*V sequences: per-token KL in the loss, or KL as a reward."""
    lp1, lp2 = token_logps()
    r1, r2 = ref_first[:, None].expand(V, V), ref_second
    if mode == 'reward':                                           # REINFORCE on reward -KL, reward-to-go
        k1, k2 = (lp1 - r1).detach(), (lp2 - r2).detach()
        return (weights * ((k1 + k2) * lp1 + k2 * lp2)).sum()
    return (weights * (kl_estimate(lp1, r1, mode) + kl_estimate(lp2, r2, mode))).sum()

for mode in ['k1', 'k2', 'k3', 'reward']:
    grads = torch.autograd.grad(surrogate(mode), [first, second])
    print(f'{mode:<6} | error on first-token logits {rel(grads[0], exact[0]):.3f}'
          f' | on second-token logits {rel(grads[1], exact[1]):.3f}')
```

Output:

```text
k1     | error on first-token logits 1.000 | on second-token logits 1.000
k2     | error on first-token logits 0.681 | on second-token logits 0.000
k3     | error on first-token logits 0.550 | on second-token logits 0.414
reward | error on first-token logits 0.000 | on second-token logits 0.000
```

The reward placement reproduces the exact gradient of the sequence KL for both tokens. The $k_2$ loss is exact for the second token, the last one, where there is no future, and 68% wrong for the first token, whose effect on the second token's KL it ignores. $k_3$ is wrong on both, and $k_1$ is zero on both. The per-token $k_2$ loss is the exact gradient of a different objective: $\sum_t \mathrm{KL}_t$ with the distribution over prefixes held fixed.

Which placement is better is a design question rather than a correctness one. The reward placement penalizes the right quantity, but it enters the advantage and so inherits the policy gradient's variance; with GRPO's group normalization, it is also rescaled by each group's reward spread, so its strength varies from task to task. The loss placement is a low-variance, local regularizer that keeps every token's distribution near the reference without accounting for where the episode goes. verl exposes both: `algorithm.use_kl_in_reward` with a $k_1$ penalty by default and an optional adaptive controller, and `actor.use_kl_loss` with `kl_loss_type: low_var_kl` ($k_3$, clamped to $[-10, 10]$) and a default coefficient of 0.001 (commit `00094bd`, inspected 2026-10-13). For agents, the future-divergence term matters more than for single-turn tasks, because the tokens of an early call decide which observations, and therefore which later contexts, the episode reaches.

## 10.5 Forward versus reverse KL

The reverse KL $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb{E}_{\pi_\theta}[\log \pi_\theta - \log \pi_{\text{ref}}]$ is an expectation under the policy. It is large wherever the policy puts mass that the reference does not, and indifferent to reference mass that the policy ignores. The forward KL $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$ is an expectation under the reference, and it is large wherever the reference has mass that the policy lacks. The first is called **mode-seeking** (or zero-forcing), the second **mass-covering**. Maximum likelihood and SFT minimize a forward KL from the data; on-policy distillation in [Chapter 16](16-distillation.md) minimizes a reverse KL to the teacher. The difference is clearest when a simple family must approximate a distribution it cannot represent. We fit one discretized Gaussian to a mixture of two by gradient descent in each direction.

```python
x = torch.linspace(-6, 6, 121)                                      # a 1-D grid of K = 121 outcomes
def gaussian_logp(mu, sigma):
    """A discretized Gaussian on the grid, as log-probs [K]."""
    return torch.log_softmax(-0.5 * ((x - mu) / sigma) ** 2, -1)
target = torch.logsumexp(torch.stack([gaussian_logp(-2.5, 0.6), gaussian_logp(2.5, 0.6)]), 0) - math.log(2)

def fit(direction, steps=600):
    """Fit one Gaussian to the bimodal target by minimizing KL in the given direction."""
    mu, log_sigma = torch.tensor(1.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=0.05)
    for _ in range(steps):
        q = gaussian_logp(mu, log_sigma.exp())
        p, a, b = (target, target, q) if direction == 'forward' else (q, q, target)
        kl = (p.exp() * (a - b)).sum()                              # forward: KL(target || q); reverse: KL(q || target)
        opt.zero_grad(); kl.backward(); opt.step()
    return mu.item(), log_sigma.exp().item(), kl.item(), q.detach().exp()
```

```python
gap = x.abs() < 1                                                    # the valley between the modes
print(f'target    | mass in the valley {target.exp()[gap].sum():.3f}')
for direction in ['forward', 'reverse']:
    mu, sigma, kl, q = fit(direction)
    print(f'{direction:<9} | mu {mu:5.2f} sigma {sigma:.2f} | KL {kl:.3f} | mass left {q[x < 0].sum():.2f}'
          f' right {q[x > 0].sum():.2f} | mass in the valley {q[gap].sum():.3f}')
```

Output:

```text
target    | mass in the valley 0.008
forward   | mu  0.00 sigma 2.84 | KL 0.738 | mass left 0.51 right 0.49 | mass in the valley 0.298
reverse   | mu  2.50 sigma 0.60 | KL 0.693 | mass left 0.00 right 1.00 | mass in the valley 0.008
```

The forward fit centers between the modes with a standard deviation of 2.84, splitting its mass evenly and putting 30% of it in the valley, where the target has 0.8%. The reverse fit sits exactly on the right mode (mean 2.50, standard deviation 0.60), puts nothing on the left, and reaches a KL of 0.693, which is $\log 2$: the cost of covering one of two equal modes perfectly. Which mode it picks depends on the initialization (we started it at 1.0, to the right of center).

![Figure 10.2: A single Gaussian fitted to a bimodal target on a 121-point grid. The forward-KL fit spreads across both modes and the valley; the reverse-KL fit commits to one mode.](../figures/ch10-forward-reverse.svg)

Figure 10.2 shows the two fits over the target. For RL the consequences are direct. A reverse-KL penalty to $\pi_{\text{ref}}$ lets the policy abandon reference modes (strategies the starting model sometimes used) at no cost, so it does not by itself preserve diversity, and it forbids mass where the reference has none, as the bandit's action 4 showed. A forward-KL pull, which is what a $k_3$ loss implements at moderate distances (Section 10.3), penalizes abandoning reference modes and so acts more like a diversity-preserving regularizer. That is a side effect of an estimator's gradient, not a design choice anyone wrote down, and it is worth knowing which one a configuration actually computes.

## 10.6 Entropy collapse and the levers against it

**Entropy collapse** is the steady fall of the policy's per-token entropy during RL, toward a policy that samples the same tokens every time. It is not a bug in any one component: it is what a policy gradient does to a softmax policy when its high-probability choices are rewarded. The mechanism fits in one line. For logits $z$ over $K$ tokens with $\pi = \mathrm{softmax}(z)$ and $H = -\sum_j \pi_j \log \pi_j$, differentiating gives $\partial H / \partial z_j = -\pi_j(\log \pi_j + H)$, so a small logit change $\Delta z$ changes the entropy by

$$
\Delta H \approx -\sum_j \pi_j\, (\log \pi_j + H)\, \Delta z_j = -\mathrm{Cov}_{j \sim \pi}\big(\log \pi_j,\ \Delta z_j\big),
$$

using $\mathbb{E}_\pi[\log \pi] = -H$. Entropy falls when the update raises the logits of tokens that are already likely, and rises when it raises the logits of unlikely ones. Cui et al., [*The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*](https://arxiv.org/abs/2505.22617) (2025-05-28), derive this covariance form for policy-gradient and natural-gradient updates and report that, without intervention, entropy and validation performance trade off along a fitted curve. One policy-gradient step on a sampled token $a$ with advantage $A$ changes the logits by $\Delta z = \eta A (e_a - \pi)$. Four cases cover the possibilities.

```python
def entropy_of(z):
    """Entropy of softmax(z) for logits z [K]."""
    return arl.entropy_from_logits(z).item()

z = torch.tensor([3.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])           # logits [K]; action 0 is likely, 7 rare
p, logp, H = torch.softmax(z, -1), torch.log_softmax(z, -1), entropy_of(z)
eta = 0.05
print(f'entropy {H:.4f} | pi(likely) {p[0]:.3f} | pi(rare) {p[7]:.4f}')
for name, a, adv in [('likely, A > 0', 0, 1.0), ('likely, A < 0', 0, -1.0), ('rare, A > 0', 7, 1.0), ('rare, A < 0', 7, -1.0)]:
    dz = eta * adv * (torch.eye(8)[a] - p)                                # one policy-gradient step on token a
    predicted = -(p * (logp + H) * dz).sum().item()                       # -Cov_pi(log pi, dz)
    print(f'{name:<14} | predicted dH {predicted:+.5f} | actual dH {entropy_of(z + dz) - H:+.5f}')
```

Output:

```text
entropy 1.1941 | pi(likely) 0.647 | pi(rare) 0.0044
likely, A > 0  | predicted dH -0.01035 | actual dH -0.01038
likely, A < 0  | predicted dH +0.01035 | actual dH +0.01032
rare, A > 0    | predicted dH +0.01511 | actual dH +0.01512
rare, A < 0    | predicted dH -0.01511 | actual dH -0.01510
```

The first-order prediction matches the actual change to within 1% in every case, and the signs follow the covariance. Rewarding the likely token (probability 0.647) lowers the entropy; rewarding the rare token (probability 0.0044) raises it; penalizing either does the reverse. Now consider a GRPO batch late in training. Within a group, the successful attempts are mostly made of the policy's likely tokens, so most positive-advantage updates are of the first kind. The failures are made of a mix, and penalizing a rare token (the fourth row) also lowers entropy. Three of the four cases reduce entropy whenever the policy is already right more often than not, which is why collapse is the default dynamics of on-policy RL, not an accident of one run.

The levers against collapse act on this covariance in different ways.

- An **entropy bonus** adds $-\alpha H$ to the loss, whose gradient on the logits is $\alpha\,\pi_j(\log\pi_j + H)$ per token: it raises every token whose log-probability is below $-H$ and lowers the likely ones. Over a vocabulary of 316 words (or 151k tokens for Qwen3), "every unlikely token" is mostly tokens the policy should never emit. Section 10.1 showed the same thing in closed form: the entropy bonus is a KL penalty toward the uniform distribution.
- A **KL penalty** pulls each token's distribution toward the reference's, which has the reference's entropy. It resists collapse only as far as the reference is uncertain, and, as a reverse KL, it does not penalize dropping alternatives the reference had (Section 10.5).
- **Clip-higher** (DAPO) raises $\epsilon_{\text{high}}$. With $A > 0$, PPO's clip removes a token's gradient once $\pi_\theta(a) > (1 + \epsilon_{\text{high}})\,\pi_{\text{old}}(a)$, so the absolute increase a token can receive before the clip binds is $\epsilon_{\text{high}}\,\pi_{\text{old}}(a)$: 0.18 for a token at 0.9, and 0.002 for a token at 0.01. The upper clip therefore binds on rare tokens with positive advantage, which are exactly the updates that raise entropy in the table above. DAPO's stated reason is the same: "the probability of an unlikely 'exploration' token is too tightly bounded to be uplifted", and it raises $\epsilon_{\text{high}}$ from 0.2 to 0.28. The lever works only if the clip binds, which requires the policy to move within a batch (several minibatches or epochs).
- **Sampling temperature** $T$ samples from $\mathrm{softmax}(z/T)$. Because the trainer computes log-probabilities at the same $T$ (Chapter 9's `rl_update` divides the logits by `cfg.temperature`), the optimized policy *is* the tempered one, and the network can sharpen $z$ to undo any fixed $T$. A higher temperature delays collapse rather than preventing it; ProRL reports that with raised rollout temperature "entropy continues to decline steadily as training progresses". Evaluating at a lower temperature than training is then a separate, deliberate choice ([Chapter 17](17-evaluation.md)).

`train_rl` has no hook for an extra loss term, and Chapter 9's functions must not change. We therefore save `entropy_bonus` and a regularized version of Chapter 9's update and loop, built from the same saved pieces (`sample_step`, `behavior_stats`, `compute_advantages`, `policy_loss`). `entropy_bonus` is the masked mean of the full-vocabulary entropy at policy positions; unlike the KL estimators, it is exact at each visited prefix rather than a one-sample estimate, because the logits for the whole vocabulary are already computed.

```python
#@save
def entropy_bonus(logits, mask):
    """Mean full-vocabulary entropy over policy positions: logits [B, T, V], mask [B, T] -> differentiable scalar."""
    return arl.masked_mean(arl.entropy_from_logits(logits), mask)
```

`regularized_update` is Chapter 9's `rl_update` with two differences: the loss subtracts `entropy_coef` times the entropy bonus, and the KL term uses `kl_estimate`, so `cfg.kl_kind` may be `'k3+'`. The logged `entropy` is the bonus's value at each minibatch, computed before the step.

```python
#@save
def regularized_update(model, optimizer, batch, advantages, cfg, ref_model=None, generator=None, entropy_coef=0.0):
    """arl.rl_update with an entropy bonus and the KL term computed by kl_estimate(cfg.kl_kind) -> dict of stats."""
    ids, mask = batch['ids'], batch['action_mask']                          # [B, T], [B, T-1]
    with torch.no_grad():
        old_logp = arl.selected_log_probs(model(ids)[:, :-1] / cfg.temperature, ids[:, 1:])
        ref_logp = None if ref_model is None else \
            arl.selected_log_probs(ref_model(ids)[:, :-1] / cfg.temperature, ids[:, 1:])
    logs = []
    for _ in range(cfg.epochs):
        for rows in torch.randperm(len(ids), generator=generator).chunk(cfg.minibatches):
            logits = model(ids[rows])[:, :-1] / cfg.temperature             # [b, T-1, V]
            logp = arl.selected_log_probs(logits, ids[rows, 1:])
            loss, stats = arl.policy_loss(logp, old_logp[rows], advantages[rows], mask[rows], cfg)
            entropy = entropy_bonus(logits, mask[rows])
            loss = loss - entropy_coef * entropy
            if ref_logp is not None and cfg.kl_coef:
                kl = arl.masked_mean(kl_estimate(logp, ref_logp[rows], cfg.kl_kind), mask[rows])
                loss, stats['kl_ref'] = loss + cfg.kl_coef * kl, kl.item()
            optimizer.zero_grad()
            loss.backward()
            stats['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            optimizer.step()
            stats['entropy'] = entropy.item()
            logs.append(stats)
    return {k: sum(s[k] for s in logs) / len(logs) for k in logs[0]}
```

`train_regularized` is `train_rl` without periodic evaluation, plus the reward placement of Section 10.4. With `kl_in_reward=True`, each episode's reward becomes $r_i - \beta \sum_t k_{1,t}$, computed from the sampler's log-probabilities and the reference before the advantages, and the loss carries no KL term. `sequence_kl` computes that sum.

```python
#@save
def sequence_kl(batch, ref_model, temperature=1.0):
    """Sum over policy tokens of k1 = log pi_old - log pi_ref, from the sampler's log-probs -> [B]."""
    ids, mask = batch['ids'], batch['action_mask']
    with torch.no_grad():
        ref_logp = arl.selected_log_probs(ref_model(ids)[:, :-1] / temperature, ids[:, 1:])
    return ((batch['sample_logp'] - ref_logp) * mask).sum(-1)


def train_regularized(model, tasks, tok, cfg, steps, entropy_coef=0.0, kl_in_reward=False, ref_model=None):
    """arl.train_rl with an entropy bonus and KL in the loss or (kl_in_reward) in the reward -> history."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env_factory = lambda: arl.make_env(cfg)
    if cfg.kl_coef and ref_model is None:
        ref_model = copy.deepcopy(model).eval()                             # frozen pi_ref = the starting policy
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for step in range(steps):
        kept, sampled = arl.sample_step(model, env_factory, tasks, tok, cfg, gen)
        record = dict(step=step, **arl.behavior_stats(sampled))
        batch = arl.collate(kept, tok.pad_id)
        batch['n_turns'] = torch.tensor([float(t.info['n_turns']) for t in kept])
        if kl_in_reward and cfg.kl_coef:                                    # r_i - beta * sum_t k1_t, then advantages
            seq_kl = sequence_kl(batch, ref_model, cfg.temperature)
            batch['rewards'], record['kl_seq'] = batch['rewards'] - cfg.kl_coef * seq_kl, seq_kl.mean().item()
        advantages = arl.compute_advantages(batch, cfg)
        record.update(regularized_update(model, optimizer, batch, advantages, cfg,
                                         None if kl_in_reward else ref_model, gen, entropy_coef))
        history.append(record)
    return history
```

With both regularizers off, the loop must be Chapter 9's. It consumes the random generator in the same order as `train_rl`, so three steps of each from the same start should give identical parameters.

```python
cfg = arl.RLConfig()
model_a, model_b = copy.deepcopy(rl_base), copy.deepcopy(rl_base)
arl.train_rl(model_a, train_tasks, tok, cfg, steps=3)
train_regularized(model_b, train_tasks, tok, cfg, steps=3)
print('largest parameter difference after 3 steps:',
      max((a - b).abs().max().item() for a, b in zip(model_a.parameters(), model_b.parameters())))
```

Output:

```text
largest parameter difference after 3 steps: 0.0
```

The difference is exactly zero, so every regularized run below differs from Chapter 9's `train_rl` only by the regularizer it switches on.

### Clip-higher on one batch

Whether the upper clip binds depends on how far the policy moves within a batch, and Chapter 9's default (one epoch, two minibatches) clipped 0.13% of tokens, too few for $\epsilon_{\text{high}}$ to matter. To see what the clip does when it binds, we take one batch of 64 episodes from the RL base agent and run 8 full-batch steps on it with $\epsilon_{\text{high}}$ at 0.2, at DAPO's 0.28, and effectively unbounded. After the 8 steps we count the positive-advantage policy tokens whose ratio exceeds $1 + \epsilon_{\text{high}}$, and how many of them started below probability 0.5.

```python
cfg = arl.RLConfig()
trajs = arl.sample_groups(rl_base, lambda: arl.make_env(cfg), train_tasks[:8], tok, cfg, torch.Generator().manual_seed(0))
batch = arl.collate(trajs, tok.pad_id)
ids, mask, advantages = batch['ids'], batch['action_mask'], arl.compute_advantages(batch, cfg)
with torch.no_grad():
    old_logp = arl.selected_log_probs(rl_base(ids)[:, :-1], ids[:, 1:])                # [B, T-1]
positive = (advantages[:, None] > 0) & (mask > 0)                                    # [B, T-1]
rare = positive & (old_logp.exp() < 0.5)                                             # [B, T-1]
print(f'{int(positive.sum())} policy tokens with A > 0, {int(rare.sum())} of them with pi_old < 0.5')
for clip_high in [0.2, 0.28, 100.0]:
    model, step_cfg = copy.deepcopy(rl_base), arl.RLConfig(clip_high=clip_high)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(8):
        logits = model(ids)[:, :-1]
        loss, stats = arl.policy_loss(arl.selected_log_probs(logits, ids[:, 1:]), old_logp, advantages, mask, step_cfg)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        logits = model(ids)[:, :-1]
        above = positive & ((arl.selected_log_probs(logits, ids[:, 1:]) - old_logp).exp() > 1 + clip_high)
        entropy = entropy_bonus(logits, mask).item()
    print(f'clip_high {clip_high:6.2f} | entropy {entropy_bonus(rl_base(ids)[:, :-1], mask).item():.3f} -> {entropy:.3f}'
          f' | last-step clip_frac {stats["clip_frac"]:.3f} | A>0 tokens above the upper clip: {int(above.sum())},'
          f' {int((above & rare).sum())} of them with pi_old < 0.5')
```

Output:

```text
121 policy tokens with A > 0, 17 of them with pi_old < 0.5
clip_high   0.20 | entropy 0.547 -> 0.538 | last-step clip_frac 0.114 | A>0 tokens above the upper clip: 20, 15 of them with pi_old < 0.5
clip_high   0.28 | entropy 0.547 -> 0.531 | last-step clip_frac 0.104 | A>0 tokens above the upper clip: 15, 14 of them with pi_old < 0.5
clip_high 100.00 | entropy 0.547 -> 0.508 | last-step clip_frac 0.151 | A>0 tokens above the upper clip: 0, 0 of them with pi_old < 0.5
```

In both clipped settings, most of the positive-advantage tokens that ended above the upper clip had started below probability 0.5 (15 of 20 at $\epsilon_{\text{high}} = 0.2$, 14 of 15 at 0.28), although only 17 of the 121 positive-advantage tokens started there. That part of DAPO's argument holds: the upper clip is a constraint on rare tokens. The entropy column does not follow the argument's conclusion. After 8 steps, entropy fell least with the tightest clip (0.547 to 0.538) and most with no upper clip at all (to 0.508). Removing the cap frees every positive-advantage token, including the mid-probability ones whose growth sharpens the distribution, and on this batch that effect outweighs the extra room for rare tokens. DAPO reports that clip-higher raises entropy in long-reasoning training at 32B scale; on TinyAgent we do not see it, either here or in Chapter 9's ablation, so the comparison below leaves clip-higher out and the claim stays with its source.

## 10.7 Comparing the regularizers on TinyAgent

We now train the default configuration for 40 steps (where Chapter 9's run still had headroom) under nine settings: no regularizer; KL penalties in the loss with the correct gradient (`k3+`) at $\beta$ of 0.05, 0.3, and 1.0; the GRPO-style $k_3$ loss at 0.3; the KL in the reward at 0.3 (Section 10.4); entropy bonuses of 0.03 and 0.1; and a sampling temperature of 1.3. Each run starts from the same RL base agent with seed 0. Besides held-out pass@1 we measure diversity. For a task with $c$ passing attempts out of $n$, the probability that $k$ attempts drawn without replacement contain a pass is $1 - \binom{n-c}{k}/\binom{n}{k}$; [Chapter 17](17-evaluation.md) derives this estimator and saves it, so here it is a local helper. We also count distinct action sequences among the 16 attempts at each task, and estimate the per-token KL from the starting policy with $k_3$.

```python
def pass_at_k(n, c, k):
    """Chance that k of n attempts (c correct), drawn without replacement, contain a pass; see Chapter 17."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def diversity(model, n=16, seed=0):
    """Held-out pass@1, pass@8, distinct action sequences per task, invalid calls, and k3 KL to rl_base."""
    cfg = arl.RLConfig(group_size=n)
    trajs = arl.sample_groups(model, lambda: arl.make_env(cfg), heldout_tasks, tok, cfg,
                              torch.Generator().manual_seed(seed))
    groups = [trajs[i:i + n] for i in range(0, len(trajs), n)]
    solved = [sum(bool(t.info['submitted'] and t.info['hidden_pass']) for t in g) for g in groups]
    batch = arl.collate(trajs, tok.pad_id)
    with torch.no_grad():
        start_logp = arl.selected_log_probs(rl_base(batch['ids'])[:, :-1], batch['ids'][:, 1:])
    return dict(pass1=sum(pass_at_k(n, c, 1) for c in solved) / len(groups),
                pass8=sum(pass_at_k(n, c, 8) for c in solved) / len(groups),
                distinct=sum(len({' '.join(t.info['actions']) for t in g}) for g in groups) / len(groups),
                invalid=sum(t.info['n_invalid'] for t in trajs) / len(trajs),
                kl_start=arl.masked_mean(kl_estimate(batch['sample_logp'], start_logp), batch['action_mask']).item())
```

```python
ARMS = {'none': ({}, {}),
        'KL k3+ 0.05': (dict(kl_coef=0.05, kl_kind='k3+'), {}),
        'KL k3+ 0.3': (dict(kl_coef=0.3, kl_kind='k3+'), {}),
        'KL k3+ 1.0': (dict(kl_coef=1.0, kl_kind='k3+'), {}),
        'KL k3 0.3': (dict(kl_coef=0.3, kl_kind='k3'), {}),
        'KL reward 0.3': (dict(kl_coef=0.3, kl_kind='k1'), dict(kl_in_reward=True)),
        'entropy 0.03': ({}, dict(entropy_coef=0.03)),
        'entropy 0.1': ({}, dict(entropy_coef=0.1)),
        'temperature 1.3': (dict(temperature=1.3), {})}
runs = {}
for name, (fields, options) in ARMS.items():
    model = copy.deepcopy(rl_base)
    history = train_regularized(model, train_tasks, tok, arl.RLConfig(**fields), 40, **options)
    runs[name] = dict(model=model, history=history, **diversity(model))
```

```python
print(f'{"setting":<16}| entropy 0-4 -> 35-39 | reads | KL from start | pass@1 | pass@8 | distinct | invalid')
for name, run in runs.items():
    entropy = [h['entropy'] for h in run['history']]
    print(f'{name:<16}| {sum(entropy[:5]) / 5:.3f} -> {sum(entropy[-5:]) / 5:.3f}       |'
          f' {run["history"][-1]["reads"]:5.2f} | {run["kl_start"]:13.3f} | {run["pass1"]:6.3f} |'
          f' {run["pass8"]:6.3f} | {run["distinct"]:8.1f} | {run["invalid"]:7.2f}')
```

Output:

```text
setting         | entropy 0-4 -> 35-39 | reads | KL from start | pass@1 | pass@8 | distinct | invalid
none            | 0.520 -> 0.120       |  0.98 |         0.237 |  0.725 |  0.800 |      4.3 |    0.04
KL k3+ 0.05     | 0.525 -> 0.124       |  1.16 |         0.235 |  0.800 |  1.000 |      4.7 |    0.06
KL k3+ 0.3      | 0.539 -> 0.284       |  0.81 |         0.146 |  0.588 |  0.992 |      8.3 |    0.07
KL k3+ 1.0      | 0.532 -> 0.536       |  0.53 |         0.031 |  0.231 |  0.896 |     11.5 |    0.19
KL k3 0.3       | 0.532 -> 0.294       |  0.97 |         0.117 |  0.594 |  0.999 |      8.0 |    0.05
KL reward 0.3   | 0.541 -> 0.424       |  0.48 |         0.059 |  0.344 |  0.973 |     10.1 |    0.07
entropy 0.03    | 0.533 -> 0.312       |  0.94 |         0.325 |  0.656 |  0.946 |      8.7 |    0.28
entropy 0.1     | 0.540 -> 5.707       |  0.00 |         6.964 |  0.000 |  0.000 |     16.0 |    6.00
temperature 1.3 | 0.782 -> 0.474       |  0.92 |         0.517 |  0.731 |  0.986 |      5.2 |    0.25
```

Read the table by column, and remember that it is one seed on 10 held-out tasks with 16 attempts each, so pass@8 moves in steps of about 0.1 when a task is never solved.

- **Entropy.** Without a regularizer the policy-token entropy fell from 0.52 to 0.12 in 40 steps. Every regularizer except the weakest KL slowed the fall, roughly in proportion to its strength: a KL coefficient of 0.3 held it at 0.28, 1.0 held it where it started (0.54), and the entropy bonus of 0.03 at 0.31. The bonus of 0.1 drove the policy to the uniform distribution: 5.707 nats against $\log 316 = 5.756$, six malformed calls per episode, and zero pass rate.
- **pass@1 against pass@8.** The unregularized run had a pass@1 of 0.725 and a pass@8 of only 0.80: two held-out tasks were never solved in 16 attempts, because the collapsed policy repeated the same wrong procedure on them. The weakest KL (0.05) left entropy unchanged at 0.124 and still reached pass@8 of 1.000, with the best pass@1 of the table, 0.80; at one seed we cannot separate that from noise. Stronger KL bought diversity with pass@1: at $\beta = 0.3$ the distinct action sequences per task doubled (8.3 against 4.3) and pass@8 reached 0.99, while pass@1 fell to 0.59. At $\beta = 1.0$, reading did not take over (0.53 reads per episode) and pass@1 was 0.23, close to the blind ceiling of 0.2, with pass@8 still 0.90 because the near-reference policy samples many different guesses.
- **Estimator.** The $k_3$ loss and the `k3+` loss at $\beta = 0.3$ ended at nearly the same entropy (0.294 and 0.284), pass@1 (0.594 and 0.588), and distance from the start. At these per-token KLs (about 0.1), Section 10.3 puts the gradient difference near 25%, and one seed does not resolve its effect.
- **Placement.** The KL in the reward at the same $\beta = 0.3$ was much stronger than either loss placement: pass@1 0.34, reads 0.48, and the smallest distance from the start apart from $\beta = 1.0$ (0.059). This is the scale difference of Section 10.4: the reward version sums the log-ratio over the episode's 12 to 14 policy tokens instead of averaging it, and then enters an advantage normalized by each group's spread.
- **Entropy bonus and temperature.** Both kept entropy higher while moving the policy *further* from the start than no regularizer (KL 0.325 and 0.517 against 0.237), and both raised malformed calls (0.28 and 0.25 per episode against 0.04). The entropy bonus spreads mass onto tokens no policy should emit; the temperature run's network sharpened its logits to compensate for $T = 1.3$, so evaluated at $T = 1$ it is further from the start policy than any setting except the collapsed entropy bonus. Its training-time entropy, measured at $T = 1.3$, still fell from 0.78 to 0.47.

![Figure 10.3: Policy-token entropy (top) and training batch reward (bottom) over 40 GRPO steps from the RL base agent, one seed. Left: KL penalties with the reverse-KL gradient (k3+) at three coefficients. Right: an entropy bonus, the k3 loss, and the KL in the reward. The unregularized run is the orange reference line in every panel.](../figures/ch10-entropy-curves.svg)

Figure 10.3 shows the dynamics behind the table. The left column orders the KL runs by coefficient: the stronger the penalty, the later and shallower the entropy fall, and the lower the reward curve. The right column shows the other placements. The entropy bonus tracks the unregularized reward curve closely while keeping entropy near 0.3; the $k_3$ loss follows nearly the same path as the `k3+` loss at the same coefficient (the two green curves, one in each column); and the KL in the reward holds entropy highest and reward lowest. No regularizer improved the training reward at 40 steps on this gym, which has one correct procedure per task: what they buy is diversity in the samples, at a cost in pass@1 that grows with their strength.

> **Evidence.** Measured here: one seed per setting, 40 steps, 64 episodes per step, evaluation on 10 held-out tasks with 16 attempts each. Differences in pass@1 smaller than about 0.08 are within the noise of this evaluation (Chapter 9 measured a spread of 0.71 to 0.81 across three seeds for the default run). The qualitative ordering by strength (entropy, distinct sequences, distance from the start) is the mechanism being demonstrated.

### The case for dropping the KL

DAPO removes the KL term, arguing that when training long reasoning, "the model distribution can diverge significantly from the initial model, thus this restriction is not necessary" ([DAPO](https://arxiv.org/abs/2503.14476), Section 2.3, 2025-03-18). The public coding-agent recipes in [Chapter 20](20-frontier.md) (DeepSWE, SkyRL-Agent) also report no KL loss and no entropy loss. The argument from Section 10.1 supports this when the reward is a verifier: there is no learned reward to exploit, so the first reason for a reference disappears, and the remaining reasons (preserving unmeasured capabilities, a cumulative trust region) can be handled by evaluation on other tasks and by PPO's per-step clip. The counter-position is ProRL ([2505.24864](https://arxiv.org/abs/2505.24864), 2025-05-30), which trains for more than 2,000 steps and reports that "explicit regularization via a KL divergence penalty provides a stronger and more stable solution", helping "maintain entropy", while noting that over time "the KL term may increasingly dominate the loss, leading to diminishing policy updates". Its answer is to reset the reference, which Section 10.8 reproduces. Both are reports from single-turn reasoning at large scale, not controlled evidence for agents.

## 10.8 Design problems in practice

> **Problem: entropy collapse early in training.** *Symptom:* the policy-token entropy halves within the first tenth of the run while the reward is still low, the share of zero-advantage groups climbs, and pass@$k$ on held-out tasks approaches pass@1. *Cause:* the covariance of Section 10.6: in each group, the successful attempts are mostly the policy's already-likely tokens, so most positive-advantage updates sharpen the distribution, and the step size (learning rate, epochs) sets how fast. *Options:* (a) lower the learning rate or the number of epochs, which slows learning too; (b) a KL penalty to the starting policy, which holds entropy near the reference's at a cost in pass@1; (c) an entropy bonus, with a small coefficient and a watch on malformed outputs; (d) clip-higher, which helps only if the clip binds; (e) a higher sampling temperature, which delays collapse. *Default:* the book's recommendation is to log entropy and pass@$k$ from the first step, and to treat collapse as a problem only when pass@$k$ falls or a known-good behavior disappears; try (b) with the `k3+` gradient before (c). The histories show how fast the default run collapses and what (b) and (c) change.

```python
for name in ['none', 'KL k3+ 0.3', 'entropy 0.03']:
    hist = runs[name]['history']
    cells = ' | '.join(f'step {s:2d}: H {hist[s]["entropy"]:.2f} r {hist[s]["reward"]:.2f}' for s in [0, 5, 10, 20, 39])
    print(f'{name:<13} | {cells}')
```

Output:

```text
none          | step  0: H 0.56 r 0.17 | step  5: H 0.49 r 0.22 | step 10: H 0.37 r 0.41 | step 20: H 0.18 r 0.72 | step 39: H 0.11 r 0.86
KL k3+ 0.3    | step  0: H 0.56 r 0.17 | step  5: H 0.51 r 0.23 | step 10: H 0.41 r 0.41 | step 20: H 0.30 r 0.59 | step 39: H 0.26 r 0.73
entropy 0.03  | step  0: H 0.57 r 0.17 | step  5: H 0.47 r 0.31 | step 10: H 0.42 r 0.41 | step 20: H 0.30 r 0.64 | step 39: H 0.28 r 0.86
```

Without a regularizer, entropy fell by a third in the first 10 steps (0.56 to 0.37) while the batch reward was still 0.41, and to 0.18 by step 20; Chapter 9's 80-step run continued to 0.08. The KL penalty and the entropy bonus both slowed the fall to about 0.30 at step 20. The entropy bonus did so with the same batch reward at step 39 (0.86), and the KL at a cost (0.73). What the default trajectory cost on held-out tasks is in the table above: the two tasks it never solved.

> **Problem: a KL coefficient that is too strong or too weak.** *Symptom:* too strong, the logged KL stays near zero, reward plateaus early, and a behavior the reference rarely shows (here, reading) never takes over; too weak, the KL grows without bound and the term does nothing measurable. *Cause:* at the optimum of Section 10.1, the policy tilts the reference by $e^{r/\beta}$, so $\beta$ must be compared with reward differences, and the per-token KL must be compared with per-token advantages. The scale also depends on placement: a per-token mean in the loss and a per-episode sum in the reward differ by the number of policy tokens (about 13 here, thousands for agents). *Options:* (a) sweep $\beta$ on a log grid while logging the KL; (b) an adaptive controller that adjusts $\beta$ toward a target KL (verl's `AdaptiveKLController`); (c) no KL, with a held-out regression suite. *Default:* (c) for verifiable rewards, following the public coding-agent recipes; (b) when a learned reward or judge is in the loop. The KL logs of the three `k3+` runs show the regimes.

```python
for name in ['KL k3+ 0.05', 'KL k3+ 0.3', 'KL k3+ 1.0']:
    kl = [h['kl_ref'] for h in runs[name]['history']]
    print(f'{name:<12} | KL to reference at steps 0, 10, 20, 39: {kl[0]:.4f} {kl[10]:.4f} {kl[20]:.4f} {kl[39]:.4f}'
          f' | reads {runs[name]["history"][-1]["reads"]:.2f} | pass@1 {runs[name]["pass1"]:.3f}')
```

Output:

```text
KL k3+ 0.05  | KL to reference at steps 0, 10, 20, 39: 0.0036 0.0657 0.2036 0.1708 | reads 1.16 | pass@1 0.800
KL k3+ 0.3   | KL to reference at steps 0, 10, 20, 39: 0.0036 0.0344 0.0921 0.1115 | reads 0.81 | pass@1 0.588
KL k3+ 1.0   | KL to reference at steps 0, 10, 20, 39: 0.0036 0.0105 0.0089 0.0311 | reads 0.53 | pass@1 0.231
```

With $\beta = 0.05$, the KL to the reference climbed to 0.20 by step 20, close to the unregularized run's final distance from the start (0.237 on held-out episodes in the table), so the term changed little. With $\beta = 1.0$, it held the policy within a per-token KL of 0.03 for the whole run, and reading, which the reference does about once in every four or five episodes (Chapter 9 measured 0.21 to 0.24 reads per episode), never took over (0.53 reads per episode, pass@1 0.23). $\beta = 0.3$ is the intermediate regime: the KL plateaued near 0.1 while the reward kept rising slowly.

> **Problem: the memory and compute cost of the reference model.** *Symptom:* enabling a KL term runs the trainer out of memory or adds a large share to step time. *Cause:* the reference is a second full model that must produce log-probabilities for every training token. It needs only its weights (no gradients or optimizer state), but it adds one forward pass to a step that otherwise consists of the trainer's log-probability pass for $\pi_{\text{old}}$ (Chapter 9 recomputes it) and one forward and backward pass. *Options:* (a) compute reference log-probs on the inference fleet during rollout and ship them with the batch; (b) offload or shard the reference separately; (c) with LoRA training, use the base model with adapters disabled as the reference, at no extra memory; (d) drop the KL. *Default:* (d) when the reward is verifiable, otherwise (c) or (a). The arithmetic for a few model sizes, assuming bf16 weights and gradients with fp32 master weights and Adam moments (16 bytes per trained parameter):

```python
print(f'TinyAgent: {sum(p.numel() for p in rl_base.parameters()):,} parameters')
for name, params in [('0.6B', 0.6e9), ('8B', 8e9), ('32B', 32e9)]:
    policy_state = 16 * params                    # 2 (bf16 weights) + 2 (grads) + 4 (fp32 master) + 8 (Adam m, v)
    reference = 2 * params                        # bf16 weights only
    # compute in forward-pass units: old_logp 1 + forward 1 + backward ~2 = 4; the reference adds 1
    print(f'{name:>4} model | policy training state {policy_state / 2**30:5.0f} GiB | reference weights'
          f' {reference / 2**30:4.0f} GiB (+{reference / policy_state:.1%}) | trainer compute per step +25%')
```

Output:

```text
TinyAgent: 157,244 parameters
0.6B model | policy training state     9 GiB | reference weights    1 GiB (+12.5%) | trainer compute per step +25%
  8B model | policy training state   119 GiB | reference weights   15 GiB (+12.5%) | trainer compute per step +25%
 32B model | policy training state   477 GiB | reference weights   60 GiB (+12.5%) | trainer compute per step +25%
```

The ratio is the same 12.5% at every size, because it is two bytes against sixteen per parameter. The absolute numbers decide feasibility: for a 32B policy, the reference adds 60 GiB that must fit beside the sharded training state on every node that computes reference log-probs, and one more forward pass over every training token: counting a backward pass as about two forwards, a step without a reference costs about four forward-pass units (the $\pi_{\text{old}}$ pass, the forward, the backward) and the reference adds a fifth, 25% more trainer compute. TinyAgent's 157k parameters make the reference free here, which is why this chapter can afford one in every run; the decision at scale is not free.

> **Problem: resetting the reference.** *Symptom:* with a KL penalty, the logged KL settles at a plateau while the reward stops improving; the penalty, not the task, has become the binding constraint. *Cause:* the reference is the starting policy, so the further training has to go from the start, the more of each update the KL gradient cancels. *Options:* (a) lower $\beta$; (b) hard-reset the reference to the current policy (and, as ProRL does, the optimizer state), which keeps the KL as a local trust region rather than an anchor to the start; (c) drop the KL. *Default:* if a KL is used at all, reset on a schedule tied to a validation plateau, as ProRL reports doing. We continue the $\beta = 0.3$ run for 20 steps twice, once anchored to the start and once with the reference reset to the current policy; both restart the optimizer.

```python
anchored = runs['KL k3+ 0.3']['model']
cfg = arl.RLConfig(kl_coef=0.3, kl_kind='k3+', seed=1)
for name, reference in [('keep the start as reference', rl_base), ('reset reference to current', copy.deepcopy(anchored).eval())]:
    model = copy.deepcopy(anchored)
    history = train_regularized(model, train_tasks, tok, cfg, 20, ref_model=reference)
    d = diversity(model)
    print(f'{name:<28} | KL to reference {sum(h["kl_ref"] for h in history[-5:]) / 5:.3f} | entropy'
          f' {sum(h["entropy"] for h in history[-5:]) / 5:.3f} | pass@1 {d["pass1"]:.3f} | pass@8 {d["pass8"]:.3f}'
          f' | distinct {d["distinct"]:.1f}')
```

Output:

```text
keep the start as reference  | KL to reference 0.184 | entropy 0.214 | pass@1 0.731 | pass@8 0.990 | distinct 6.0
reset reference to current   | KL to reference 0.029 | entropy 0.135 | pass@1 0.787 | pass@8 0.996 | distinct 4.6
```

Kept anchored to the start, the policy's KL to the reference sat at 0.18 and its pass@1 rose to 0.731. With the reference reset to the current policy, the KL to the (new) reference was only 0.029, entropy fell to 0.135, and pass@1 rose further, to 0.787, with fewer distinct sequences (4.6 against 6.0). The reset does what ProRL describes: it removes the accumulated pull toward the start and lets the policy keep improving, and with it goes part of the diversity the KL had preserved. One seed and 160 evaluation episodes cannot rank 0.731 against 0.787; the direction of every column is the mechanism.

## Summary

- The KL-regularized objective has the closed-form optimum $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$: the reference tilted by the exponentiated reward, never leaving the reference's support. An entropy bonus is the same objective with a uniform reference.
- Of Schulman's estimators, $k_1$ and $k_3$ are unbiased values and $k_3$ has by far the lowest spread at small KL. As losses, they give different gradients: $k_1$ gives zero in expectation, $k_2$ gives the reverse-KL gradient, and $k_3$ gives the forward-KL gradient, with an error against the reverse direction that grows with the distance. `kl_estimate(..., 'k3+')` logs $k_3$ and differentiates as $k_2$.
- For sequences, a per-token KL loss ignores how a token changes the KL of the tokens after it; KL in the reward, through the policy gradient, gives the exact sequence gradient. On TinyAgent the reward placement at the same $\beta$ was a much stronger penalty, because it sums over tokens and enters the advantage.
- Reverse KL is mode-seeking and forward KL mass-covering; a reverse-KL penalty does not by itself preserve the reference's alternatives.
- Entropy falls when updates raise already-likely tokens, $\Delta H \approx -\mathrm{Cov}(\log\pi, \Delta z)$. On TinyAgent the unregularized run went from 0.52 to 0.12 nats in 40 steps and never solved two of ten held-out tasks in 16 attempts; KL penalties and an entropy bonus slowed the collapse and raised pass@8 at a cost in pass@1, an entropy bonus of 0.1 destroyed the action format, and clip-higher had no measurable effect on entropy. These are one-seed mechanism demonstrations.
- With verifiable rewards, dropping the KL (DAPO, DeepSWE) removes a reference model and a tuning knob; keeping it (ProRL) trades pass@1 for entropy and needs reference resets to stop anchoring training to its start.

## Exercises

1. Show that adding the missing score-function term to a $k_1$ loss, $\mathbb{E}[\nabla k_1 + \mathrm{sg}(k_1)\nabla\log\pi_\theta]$, gives the reverse-KL gradient, and that the same correction applied to $k_3$ also does. Check both with `expected_loss_grad`-style enumeration. Why does nobody implement the corrected $k_3$ instead of `k3+`?
2. Expand $\mathbb{E}_{\pi_\theta}[k_2]$ to third order in $\ell$ and predict the bias column of Section 10.2 for small KL. At what KL does your approximation fail?
3. Change an assumption: rerun Section 10.7 for 80 steps instead of 40, with the unregularized and `KL k3+ 0.05` settings only. Does the unregularized run's pass@8 fall below the regularized one's once its entropy reaches Chapter 9's 0.08? Report the difference against the spread of three seeds.
4. Extend the two-token model of Section 10.4 to three tokens and implement a loss placement that is exact: the per-token $k_2$ loss plus a REINFORCE term carrying each token's future KL. Verify it against the exact gradient, then explain which part a GRPO trainer would have to add to its loss to implement it.
5. Implement an adaptive KL coefficient ($\beta \leftarrow \beta(1 + \mathrm{clip}(\mathrm{KL}/\mathrm{KL}_{\text{target}} - 1, \pm 0.2)/10)$ per step) in a copy of `train_regularized`. Which target KL lets TinyAgent learn to read while keeping pass@8 at least as high as the unregularized run?
6. Restrict the entropy bonus to the first policy token after each tool observation (the positions ARPO identifies as high-entropy; [Chapter 20](20-frontier.md)). Compare malformed calls and pass@8 with the uniform bonus at a matched mean entropy.
7. Design review: a team trains a SWE agent with verl's defaults, `use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001`, on a test-execution reward. The logged `kl_loss` is 0.02 and policy entropy has fallen by 70% in 200 steps. Using this chapter, argue whether the KL term is doing anything, which gradient it computes, what you would log next, and what you would change if the reward were an LLM judge instead of tests.
