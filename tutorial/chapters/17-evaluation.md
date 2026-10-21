# 17. Evaluation

Every chapter so far has ended with a number: a held-out pass rate of 0.84, an ablation arm at 0.71 against a default at 0.76, a variant "within seed noise". [Chapter 9](09-grpo-and-variants.md#96-controlled-ablations-at-tiny-scale) could only say that last phrase informally. This chapter makes evaluation a measurement with a definition, an error bar, and a protocol. It asks five questions that a post-training team answers every week. What quantity does a pass rate estimate, and with what estimator? How uncertain is it, given that ten tasks and thirty-two samples are not 320 independent coin flips? Is checkpoint B better than checkpoint A, and how many tasks or seeds would it take to know? Which choices of the harness define the number, so that changing one of them changes the result without changing the model? And which checkpoint may we report without having tuned on the test set?

The chapter answers them on TinyAgent with GRPO checkpoints from the default setting of Chapter 9, and it uses the same machinery for the most debated measurement in RL for reasoning models: whether RL improves pass@$k$ for large $k$ or only sharpens the distribution the base model already had. Every number here is a mechanism demonstration on a 158k-parameter model with ten held-out tasks. The ten tasks are the point: they make the statistical problems large enough to see.

```python
import copy
import dataclasses
import math
import random
import torch
import agentrl as arl

torch.set_num_threads(4)                                   # the shared check host is busy
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
rl_base = arl.make_rl_base(tok)
cfg = arl.RLConfig()                                       # the training harness, reused for evaluation
```

The checkpoints come from one default GRPO run of 80 steps. `train_snapshots` calls `arl.train_rl` in chunks of ten steps and keeps a frozen copy after each chunk. Each call restarts the Adam optimizer and uses its own seed, a small departure from one uninterrupted 80-step run that does not matter for what this chapter measures.

```python
def train_snapshots(seed, steps, every=10):
    """Default GRPO from rl_base in chunks of `every` steps -> {step: frozen copy of the policy}."""
    model, snapshots = copy.deepcopy(rl_base), {0: rl_base}
    for chunk in range(steps // every):
        arl.train_rl(model, train_tasks, tok, arl.RLConfig(seed=100 * seed + chunk), steps=every)
        snapshots[every * (chunk + 1)] = copy.deepcopy(model)
    return snapshots

run = train_snapshots(seed=0, steps=80)
trained = run[80]
```

## 17.1 What a pass rate estimates

An evaluation estimates a property of a policy *under a harness*. The harness $H$ is everything except the weights: the task set, the prompt template, the environment and its budgets (turns, edits, context length), the sampling temperature, and the success criterion. For a task $j$, write

$$
p_j(\pi, H) = \Pr\big[\text{an episode of } \pi \text{ on task } j \text{ under } H \text{ succeeds}\big].
$$

The benchmark's **pass@1** is the task average $\bar p = \frac{1}{N}\sum_{j=1}^{N} p_j$. It is not "the fraction of episodes that passed" unless every task gets the same number of episodes, and it is not a property of the weights alone: change $H$ and every $p_j$ changes. The success criterion here is the one Chapter 4 fixed for evaluation, independent of any training reward: the agent submitted and every hidden test passed.

The raw data of an evaluation are therefore **per-task success counts**: draw $n$ episodes on each of $N$ tasks and record $c_j$, the number that succeeded. Everything else in this chapter (pass@1, pass@$k$, intervals, paired tests) is a function of the vector $c$. `sample_counts` produces it with the training harness `cfg`, through the same `sample_groups` that training uses, so the template, renaming, edit budget, turn budget, and temperature cannot silently differ.

```python
#@save
import dataclasses
import math
import torch
import agentrl as arl


def sample_counts(model, tasks, tok, n, cfg=None, generator=None):
    """n episodes per task under cfg's harness -> successes per task [N] (submitted and hidden tests pass)."""
    cfg = dataclasses.replace(cfg or arl.RLConfig(), group_size=n)
    generator = generator or torch.Generator().manual_seed(0)
    trajs = arl.sample_groups(model, lambda: arl.make_env(cfg), tasks, tok, cfg, generator)
    passed = torch.tensor([bool(t.info['submitted'] and t.info['hidden_pass']) for t in trajs])
    return passed.view(len(tasks), n).sum(-1)                                   # [N]
```

We draw 32 episodes on each of the ten held-out tasks for the RL base agent and for the checkpoint after 80 steps.

```python
print('tasks:', ' '.join(t.name.split('/')[1] for t in heldout_tasks))
counts = {}
for name, model in [('RL base', rl_base), ('GRPO 80', trained)]:
    counts[name] = sample_counts(model, heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(1))  # [10]
    print(f'{name:<8} pass@1 {counts[name].sum().item() / 320:.3f} | successes of 32:', counts[name].tolist())
```

Output:

```text
tasks: pin saturate series step_total qualifies under_cap share speed product_upto add_threes
RL base  pass@1 0.144 | successes of 32: [3, 8, 1, 6, 4, 4, 6, 2, 7, 5]
GRPO 80  pass@1 0.819 | successes of 32: [31, 32, 31, 30, 29, 17, 31, 1, 29, 31]
```

The two pass rates, 0.144 and 0.819, are task averages of very different vectors. The base agent's successes are spread almost evenly, 1 to 8 of 32 on every task: it guesses, and every task is about equally guessable. The trained agent solves eight tasks in nearly every episode, `under_cap` in 17 of 32, and `speed` in 1 of 32. A single number hides that one task is essentially unsolved, and the statistics below will show that this heterogeneity, not the number of episodes, sets the uncertainty of the trained agent's score.

## 17.2 pass@$k$ and its unbiased estimator

Coding systems are often allowed several attempts: sample $k$ patches and keep one that passes, given a verifier. **pass@$k$** is the probability that at least one of $k$ independent attempts succeeds, averaged over tasks:

$$
\text{pass@}k = \frac{1}{N}\sum_{j=1}^{N} \big(1 - (1 - p_j)^k\big).
$$

It needs an oracle to pick the passing attempt, so it measures *coverage*, an upper bound on what any selector can achieve, not a deployable success rate ([Chapter 20](20-frontier.md#201-how-to-read-a-frontier-entry) distinguishes it from Best@$k$).

**The plug-in estimator is biased.** The obvious estimate replaces $p_j$ by $\hat p_j = c_j/n$. For $k \ge 2$ the map $f(p) = 1 - (1-p)^k$ is strictly concave on $(0, 1)$, so by Jensen's inequality $\mathbb{E}[f(\hat p_j)] < f(\mathbb{E}[\hat p_j]) = f(p_j)$ whenever $0 < p_j < 1$: the plug-in systematically *underestimates* pass@$k$, most for small $n$ and large $k$.

**The unbiased estimator.** Chen et al. (2021) draw $n \ge k$ samples and ask a different question of them: if we picked $k$ of the $n$ samples uniformly at random without replacement, what is the probability that all $k$ fail? With $c$ successes, it is $\binom{n-c}{k}\big/\binom{n}{k}$, so

$$
\widehat{\text{pass@}k}_j = 1 - \frac{\binom{n-c_j}{k}}{\binom{n}{k}}.
$$

Unbiasedness takes one line. For any *fixed* set $S$ of $k$ sample indices, the event "every sample in $S$ fails" has probability $(1-p_j)^k$, because samples are independent. The estimator's failure term is the average of that event's indicator over all $\binom{n}{k}$ sets $S$. Expectation is linear, so its expected value is $(1-p_j)^k$ exactly. (It is a U-statistic, the minimum-variance unbiased estimator built from the sample.) Unbiasedness holds per task, and averaging over tasks keeps it.

**A stable form.** The binomial coefficients overflow floating point long before $n$ is large: $\binom{2000}{1000}$ has 600 digits. Writing both coefficients as factorials and cancelling,

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \frac{(n-c)!\,(n-k)!}{n!\,(n-c-k)!} = \prod_{i=n-c+1}^{n} \frac{i-k}{i} = \prod_{i=n-c+1}^{n}\Big(1 - \frac{k}{i}\Big),
$$

a product of $c$ factors in $(0, 1)$ when $n - c \ge k$. When $n - c < k$, every $k$-subset contains a success and the estimate is exactly 1.

```python
#@save
def pass_at_k(n, c, k):
    """Unbiased pass@k from c successes out of n samples; c is an int or a tensor [N] -> float or [N]."""
    if not 1 <= k <= n:
        raise ValueError(f'need 1 <= k <= n, got k={k}, n={n}')
    c = torch.as_tensor(c, dtype=torch.float64)
    i = torch.arange(1, n + 1, dtype=torch.float64)                            # [n]
    factors = torch.where(i > n - c.unsqueeze(-1), 1 - k / i, torch.ones(()))  # [..., n]
    estimate = torch.where(n - c < k, torch.ones(()), 1 - factors.prod(-1))
    return estimate.item() if estimate.dim() == 0 else estimate
```

The check below draws 100,000 evaluations of $n = 10$ samples for one task at several true pass rates and compares the average of each estimator of pass@5 with the truth.

```python
torch.manual_seed(0)
n, k = 10, 5
print(' p   | true pass@5 | plug-in mean | unbiased mean')
for p in [0.05, 0.2, 0.5]:
    c = torch.distributions.Binomial(n, torch.full((100_000,), p)).sample()     # [100000]
    plug_in = 1 - (1 - c / n) ** k
    print(f'{p:.2f} | {1 - (1 - p) ** k:11.4f} | {plug_in.mean():12.4f} | {pass_at_k(n, c, k).mean():13.4f}')
```

Output:

```text
 p   | true pass@5 | plug-in mean | unbiased mean
0.05 |      0.2262 |       0.1892 |        0.2265
0.20 |      0.6723 |       0.5926 |        0.6713
0.50 |      0.9688 |       0.9326 |        0.9683
```

The plug-in estimate is low at every $p$, by 0.04 to 0.08, which is larger than most of the differences Chapter 9 tried to read. The unbiased estimator matches the truth to within Monte Carlo error (about 0.001 here). The bias is largest at intermediate $p$, where $f$ is most curved over the spread of $\hat p$.

The product form agrees with exact rational arithmetic, and it still works where the direct ratio of binomials cannot be represented.

```python
from fractions import Fraction
exact = 1 - Fraction(math.comb(7, 5), math.comb(10, 5))
print(f'n=10, c=3, k=5: product form {pass_at_k(10, 3, 5):.6f} | exact {float(exact):.6f}')
try:
    direct = 1 - math.comb(1995, 1000) / float(math.comb(2000, 1000))
except OverflowError as err:
    print(f'n=2000, c=5, k=1000: direct binomials -> {err} | product form {pass_at_k(2000, 5, 1000):.6f}')
```

Output:

```text
n=10, c=3, k=5: product form 0.916667 | exact 0.916667
n=2000, c=5, k=1000: direct binomials -> int too large to convert to float | product form 0.968906
```

For $n = 10, c = 3, k = 5$ the product form gives $1 - 21/252 = 0.9167$, the value worked out in [Appendix B, Problem 21](B-design-review.md#problem-21-compute-passk). For $n = 2000$, $\binom{2000}{1000}$ has more than 600 digits and cannot be converted to a float, while the product of five factors near $1/2$ gives $0.9689 \approx 1 - 2^{-5}$. Chen et al. use $n = 200$ and $k \le 100$ and note the same instability. Choose $n$ well above the largest $k$ you report: the estimate at $k$ close to $n$ is nearly a single indicator per task and has high variance.

## 17.3 Confidence intervals

A pass rate without an interval cannot be compared with anything. Three intervals cover the cases this book meets.

### Binomial intervals for one task

For $n$ independent episodes with success probability $p$, $\hat p = c/n$ has standard error $\sqrt{p(1-p)/n}$. The **Wald interval** $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ plugs in $\hat p$, and it fails exactly where agent evaluations live: at $\hat p = 0$ or $1$ its width is zero, and near the edges it undercovers. The **Wilson interval** inverts the test instead of plugging in. It collects every $p$ that the score test would not reject, $|\hat p - p| \le z\sqrt{p(1-p)/n}$. Squaring gives a quadratic in $p$ whose roots are

$$
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.
$$

The center is pulled toward 1/2 and the width never collapses, which is what a small $n$ needs.

```python
#@save
def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for c successes of n Bernoulli trials; successes int or tensor -> (lo, hi)."""
    p = torch.as_tensor(successes, dtype=torch.float64) / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z / denom * torch.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return center - half, center + half
```

We measure the coverage of both intervals over 20,000 simulated evaluations of 40 episodes each.

```python
torch.manual_seed(0)
n, z = 40, 1.96
print(' p   | Wald coverage | Wilson coverage')
for p in [0.03, 0.2, 0.5, 0.9, 0.97]:
    c = torch.distributions.Binomial(n, torch.full((20_000,), p)).sample()     # [20000]
    p_hat = c / n
    half = z * (p_hat * (1 - p_hat) / n).sqrt()
    wald = ((p_hat - half <= p) & (p <= p_hat + half)).double().mean()
    lo, hi = wilson_interval(c, n)
    print(f'{p:.2f} | {wald:13.3f} | {((lo <= p) & (p <= hi)).double().mean():15.3f}')
```

Output:

```text
 p   | Wald coverage | Wilson coverage
0.03 |         0.699 |           0.968
0.20 |         0.902 |           0.926
0.50 |         0.918 |           0.960
0.90 |         0.912 |           0.945
0.97 |         0.701 |           0.972
```

At $p = 0.03$ and $p = 0.97$, the Wald interval covers the truth in only 70% of evaluations, because about 30% of 40-episode evaluations see zero failures (or zero successes) and report an interval of width zero. Wilson stays between 0.93 and 0.97 at every $p$. Use Wilson, or an exact interval, for one task or for a small number of episodes; the Wald interval is acceptable only when $n\hat p$ and $n(1-\hat p)$ are both comfortably large.

### The bootstrap, and which unit to resample

The **bootstrap** estimates the sampling distribution of a statistic by recomputing it on data resampled with replacement from the observed data. The percentile interval takes the $\alpha/2$ and $1-\alpha/2$ quantiles of the recomputed values. The one assumption that matters is that the resampled units are independent draws from the population of interest. `bootstrap_ci` takes one value per unit; `paired_bootstrap_diff` resamples the *same* units for two systems.

```python
#@save
def bootstrap_ci(values, n_boot=1000, alpha=0.05, generator=None):
    """Percentile bootstrap interval for the mean of values [N], one value per independent unit -> (lo, hi)."""
    values = torch.as_tensor(values, dtype=torch.float64)
    generator = generator or torch.Generator().manual_seed(0)
    idx = torch.randint(len(values), (n_boot, len(values)), generator=generator)   # [n_boot, N]
    means = values[idx].mean(-1)                                                    # [n_boot]
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64))
    return lo.item(), hi.item()


def paired_bootstrap_diff(a, b, n_boot=1000, alpha=0.05, generator=None):
    """Mean of a - b over paired units [N] with a percentile bootstrap interval -> (diff, lo, hi)."""
    diff = torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)
    return (diff.mean().item(), *bootstrap_ci(diff, n_boot, alpha, generator))
```

In an evaluation of $N$ tasks with $n$ episodes each, the episodes are not independent: episodes of the same task share its difficulty. Treat the tasks as drawn from a population with mean $\mu$ and between-task variance $\sigma^2_{\text{task}} = \mathrm{Var}(p_j)$. By the law of total variance, the estimate $\hat\mu = \frac{1}{N}\sum_j c_j/n$ has

$$
\mathrm{Var}(\hat\mu) = \frac{1}{N}\Big(\sigma^2_{\text{task}} + \frac{\mathbb{E}[p_j(1-p_j)]}{n}\Big),
$$

whereas treating all $Nn$ episodes as independent assumes $\mu(1-\mu)/(Nn)$. Since $\mu(1-\mu) = \sigma^2_{\text{task}} + \mathbb{E}[p_j(1-p_j)]$, the ratio of the true variance to the naive one is the **design effect** $1 + (n-1)\,\rho$, with the intra-task correlation $\rho = \sigma^2_{\text{task}}/(\mu(1-\mu))$. Two consequences follow. When tasks differ ($\rho > 0$), the naive interval is too narrow by a factor $\sqrt{1 + (n-1)\rho}$. And as $n \to \infty$ the variance does not go to zero but to $\sigma^2_{\text{task}}/N$: more samples per task cannot compensate for too few tasks. The fix is the **cluster bootstrap**: resample tasks, carrying each task's episodes along, which is `bootstrap_ci` applied to per-task rates.

Which interval is right depends on the question, as Miller (2024) emphasizes for language-model evals. "How well does this checkpoint do on *these* ten tasks?" conditions on the tasks, and only episode sampling is random. "How well will it do on tasks *like* these?" treats the tasks as a sample. The second is almost always the question. A simulation with a known population separates the two: task pass rates drawn from a U-shaped Beta(0.4, 0.3) distribution (most tasks nearly always or nearly never solved, as in agent benchmarks), ten tasks, 32 episodes each.

```python
torch.manual_seed(0)
N, n, sims, mu = 10, 32, 400, 0.4 / 0.7
hits = {('episodes', 'population'): 0, ('episodes', 'these tasks'): 0, ('tasks', 'population'): 0}
for s in range(sims):
    p_task = torch.distributions.Beta(0.4, 0.3).sample((N,))                     # [N]
    outcomes = torch.bernoulli(p_task.unsqueeze(1).expand(N, n))                  # [N, n]
    gen = torch.Generator().manual_seed(s)
    for unit, values in [('episodes', outcomes.flatten()), ('tasks', outcomes.mean(1))]:
        lo, hi = bootstrap_ci(values, n_boot=500, generator=gen)
        for target, truth in [('population', mu), ('these tasks', p_task.mean().item())]:
            if (unit, target) in hits:
                hits[unit, target] += lo <= truth <= hi
for (unit, target), h in hits.items():
    print(f'resample {unit:<8} | covers the {target:<11} mean in {h / sims:.3f} of evaluations')
```

Output:

```text
resample episodes | covers the population  mean in 0.378 of evaluations
resample episodes | covers the these tasks mean in 0.998 of evaluations
resample tasks    | covers the population  mean in 0.932 of evaluations
```

Resampling episodes covers the population mean in 38% of evaluations: its intervals are so narrow that they usually miss. They cover the mean of *these ten tasks* in 99.8% of evaluations, which is the question they answer (and they are conservative even for that, because the binomial variance $\mu(1-\mu)$ overstates the within-task variance). Resampling tasks covers the population mean in 93% of evaluations, close to nominal; the remaining shortfall is the percentile bootstrap's known undercoverage at $N = 10$. Figure 17.1 repeats the experiment for $n$ from 1 to 64 episodes per task. With one episode per task the two intervals coincide; as $n$ grows, the episode-level interval shrinks toward zero width while the true uncertainty stays at $\sigma^2_{\text{task}}/N$, and its coverage falls below 0.3.

![Figure 17.1: Coverage of nominal 95% bootstrap intervals for the population pass rate, resampling episodes versus resampling tasks, as the number of episodes per task grows. Simulation with 10 tasks whose pass rates are drawn from Beta(0.4, 0.3), 300 evaluations per point.](../figures/ch17-coverage.svg)

The same comparison on the measured counts of Section 17.1, with the moment estimate of the intra-task correlation:

```python
for name, c in counts.items():
    rates = c.double() / 32                                                     # [10] per-task pass rates
    episodes = (torch.arange(32) < c.unsqueeze(1)).flatten()                   # [320] one outcome per episode
    lo_e, hi_e = bootstrap_ci(episodes)
    lo_t, hi_t = bootstrap_ci(rates)
    task_var = (rates.var() - (rates * (1 - rates)).mean() / 31).clamp(min=0)   # sigma^2_task estimate
    rho = (task_var / (rates.mean() * (1 - rates.mean()))).item()
    print(f'{name:<8} episodes [{lo_e:.3f}, {hi_e:.3f}] width {hi_e - lo_e:.3f} | tasks [{lo_t:.3f}, {hi_t:.3f}]'
          f' width {hi_t - lo_t:.3f} | rho {rho:.2f}, design effect {1 + 31 * rho:.1f}')
```

Output:

```text
RL base  episodes [0.103, 0.178] width 0.075 | tasks [0.106, 0.188] width 0.081 | rho 0.01, design effect 1.2
GRPO 80  episodes [0.775, 0.856] width 0.081 | tasks [0.603, 0.959] width 0.356 | rho 0.63, design effect 20.4
```

For the base agent, the two intervals agree (widths 0.075 and 0.081): its tasks are equally hard, $\rho \approx 0.01$, and its episodes behave like independent coin flips. For the trained agent, $\rho = 0.63$ and the design effect is 20: the task-level interval, $[0.60, 0.96]$, is 4.4 times as wide as the episode-level one. With ten tasks and one of them nearly unsolved, the honest statement is that the trained agent's held-out pass rate on tasks like these is somewhere between about 0.6 and 0.96. More episodes per task would not narrow that; more tasks would.

## 17.4 Comparing two runs

Two kinds of randomness separate two numbers. **Evaluation noise** comes from sampling episodes from a fixed checkpoint, and more episodes shrink it. **Training noise** comes from the seed of the run (task order, sampled groups, minibatch order), and it is a property of the recipe: two runs of the same recipe end at different checkpoints. Whether training noise belongs in the error bar depends on the claim. "Checkpoint B is better than checkpoint A" is a claim about two fixed checkpoints. "Recipe B is better than recipe A" is a claim about distributions of checkpoints, and its unit of replication is the seed.

We train three more seeds of the default recipe for 30 steps, a point at which the runs have not saturated, and evaluate each of the four step-30 checkpoints twice, with independent episodes (320 per evaluation). The duplicate evaluations measure the evaluation noise; the spread across seeds contains both kinds.

```python
seed_ckpts = [run[30]] + [train_snapshots(seed, 30)[30] for seed in [1, 2, 3]]
evals = torch.tensor([[sample_counts(m, heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(1000 * s + e))
                       .sum().item() / 320 for e in range(2)] for s, m in enumerate(seed_ckpts)])   # [4 seeds, 2]
seed_rates = evals.mean(1)                                                       # [4]
eval_sd = ((evals[:, 0] - evals[:, 1]) ** 2 / 2).mean().sqrt()                  # sd of one evaluation
train_var = (seed_rates.var() - eval_sd ** 2 / 2).clamp(min=0)                   # seed variance net of eval noise
print('pass@1 at step 30, two evaluations per seed:', [[round(x, 3) for x in row] for row in evals.tolist()])
print(f'sd of one evaluation {eval_sd:.3f} | sd across seeds (mean of 2 evals) {seed_rates.std():.3f}'
      f' | training-noise sd {train_var.sqrt():.3f}')
```

Output:

```text
pass@1 at step 30, two evaluations per seed: [[0.728, 0.709], [0.694, 0.722], [0.697, 0.716], [0.731, 0.709]]
sd of one evaluation 0.016 | sd across seeds (mean of 2 evals) 0.007 | training-noise sd 0.000
```

The two evaluations of the same checkpoint differ by up to 0.028, and the standard deviation of one 320-episode evaluation is 0.016, close to the binomial $\sqrt{0.71 \cdot 0.29/320} = 0.025$ (it estimates the conditional, these-ten-tasks noise, which Section 17.3 showed is smaller). The four seeds' means, 0.706 to 0.720, spread *less* than two evaluations of one seed, and the estimated training-noise standard deviation is zero. In this setting, at step 30, the default recipe is highly reproducible across seeds, and almost all of the visible spread is evaluation noise. This also re-reads [Chapter 9's ablation table](09-grpo-and-variants.md#96-controlled-ablations-at-tiny-scale): its default arm's three held-out values, 0.76 to 0.81, came from 80 episodes each, whose evaluation standard error is about 0.045, so that spread was mostly evaluation noise too. Neither result says seeds do not matter in general. Seed variance depends on the recipe, the scale, and the point in training, and it has to be measured with independent evaluations, as here, before it can be ignored.

### Paired comparisons

When two systems run on the same tasks, compare them task by task. The per-task difference $d_j = \hat p^{A}_j - \hat p^{B}_j$ removes the task's difficulty, which both systems share, so

$$
\mathrm{Var}(\bar d) = \frac{1}{N}\big(\mathrm{Var}(\hat p^A_j) + \mathrm{Var}(\hat p^B_j) - 2\,\mathrm{Cov}(\hat p^A_j, \hat p^B_j)\big),
$$

and the covariance term, large when tasks differ in difficulty, is what an unpaired comparison throws away. We compare the checkpoints at steps 60 and 80 on the same ten tasks, and we also give both the same random generator. Two similar policies driven by the same random numbers tend to sample the same episodes, which correlates their outcomes further. This is the **common random numbers** technique from simulation; it is legitimate for a paired comparison, and it is why the same generator must *not* be reused when the goal is to measure evaluation noise, as in the block above.

```python
rates_60 = sample_counts(run[60], heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(9)).double() / 32
rates_80 = sample_counts(run[80], heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(9)).double() / 32
diff, lo, hi = paired_bootstrap_diff(rates_80, rates_60)
unpaired_se = ((rates_80.var() + rates_60.var()) / 10).sqrt().item()
paired_se = ((rates_80 - rates_60).std() / 10 ** 0.5).item()
print(f'step 80 - step 60: {diff:+.3f}, paired 95% interval [{lo:+.3f}, {hi:+.3f}]')
print(f'standard error: unpaired {unpaired_se:.3f} | paired {paired_se:.3f}')
print(f'correlation of per-task rates: {torch.corrcoef(torch.stack([rates_60, rates_80]))[0, 1]:.2f}')
```

Output:

```text
step 80 - step 60: +0.041, paired 95% interval [+0.006, +0.087]
standard error: unpaired 0.132 | paired 0.023
correlation of per-task rates: 0.97
```

The step-80 checkpoint is 0.041 better than the step-60 one, with a paired interval of $[+0.006, +0.087]$ that excludes zero. Treated as two unpaired samples of ten tasks, the same difference has a standard error of 0.132, six times the paired 0.023, and would be invisible. The per-task rates of the two checkpoints correlate at 0.97: the tasks that are hard for one are hard for the other, and pairing subtracts that shared difficulty. The paired interval answers a question about *these* tasks and these two checkpoints; it says nothing about the recipe, and with ten tasks it is itself a rough interval.

### How many tasks or seeds?

A **power calculation** turns the comparison around: given an effect $\delta$ we want to detect, how many units do we need? For a two-sided test at level $\alpha$ with power $1-\beta$, the mean of $N$ paired differences with per-unit standard deviation $\sigma_d$ must satisfy $\delta/(\sigma_d/\sqrt N) \ge z_{1-\alpha/2} + z_{1-\beta}$, so

$$
N \ge \Big(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\delta}\Big)^2.
$$

For two independent groups (unpaired tasks, or two recipes with separate seeds), the difference of two means has per-unit standard deviation $\sqrt2\,\sigma$, and the formula gives the size of *each* group.

```python
#@save
def required_units(sd, effect, alpha=0.05, power=0.8):
    """Units (tasks, seeds) for a two-sided z-test to detect a mean difference `effect` given per-unit sd."""
    normal = torch.distributions.Normal(0.0, 1.0)
    z = normal.icdf(torch.tensor(1 - alpha / 2)) + normal.icdf(torch.tensor(power))
    return math.ceil((z.item() * sd / effect) ** 2)
```

We plug in the measured spreads: the per-task difference between two checkpoints, the per-task spread of one checkpoint (for an unpaired design), and the spread across seeds.

```python
sd_paired, sd_task, sd_seed = (rates_80 - rates_60).std().item(), rates_80.std().item(), seed_rates.std().item()
print(f'measured sd: paired task diff {sd_paired:.3f} | per-task rate {sd_task:.3f} | across seeds {sd_seed:.3f}')
print(' effect | tasks, paired | tasks per arm, unpaired | seeds per arm')
for effect in [0.02, 0.05, 0.10]:
    print(f'  {effect:.2f} | {required_units(sd_paired, effect):13d} | '
          f'{required_units(2 ** 0.5 * sd_task, effect):23d} | {required_units(2 ** 0.5 * sd_seed, effect):13d}')
```

Output:

```text
measured sd: paired task diff 0.072 | per-task rate 0.291 | across seeds 0.007
 effect | tasks, paired | tasks per arm, unpaired | seeds per arm
  0.02 |           103 |                    3326 |             3
  0.05 |            17 |                     533 |             1
  0.10 |             5 |                     134 |             1
```

With paired tasks, detecting a 2-point difference between two checkpoints of this kind needs about 103 tasks, and a 5-point difference about 17. An unpaired design (for example, comparing a number from a paper with a number from your harness on different task samples) needs about 30 times as many: 533 tasks per arm for 5 points. The seed column is small because the measured spread across seeds was small; the formula's answer of one seed is a statement about power, not about practice. At least three seeds per arm are needed just to estimate $\sigma_{\text{seed}}$, and a recipe comparison should never rest on fewer.

> **In production.** The same arithmetic sets the size of real benchmarks. On SWE-bench Verified's 500 tasks, the binomial standard error of a pass rate near 0.4 is about 0.022 for one attempt per task, so the difference of two independent single-attempt runs has a standard error of about 0.031, and a 2-point gap is well inside it. Per-task pairing, several attempts per task (as DeepSWE's average of 16 runs does, [Chapter 20](20-frontier.md)), and multiple training seeds are the levers. This is arithmetic from the binomial formula, not a measurement.

## 17.5 The harness defines the number

A harness choice is any setting outside the weights that changes $p_j(\pi, H)$. If training and evaluation, or two evaluations being compared, differ in one, the comparison measures the harness. We evaluate the same two checkpoints under the training harness and under seven one-field changes.

```python
SETTINGS = {'training harness': {}, 'greedy (T = 0)': dict(temperature=0.0), 'T = 0.7': dict(temperature=0.7),
            'T = 1.3': dict(temperature=1.3), 'max_turns = 4': dict(max_turns=4), 'max_turns = 3': dict(max_turns=3),
            'no renaming': dict(rename=False), 'unlimited edits': dict(max_edits=None)}
drift = {}
for label, fields in SETTINGS.items():
    harness = dataclasses.replace(cfg, **fields)
    drift[label] = [sample_counts(m, heldout_tasks, tok, 32, harness, torch.Generator().manual_seed(3))
                    for m in (rl_base, trained)]
    base_rate, rl_rate = (c.sum().item() / 320 for c in drift[label])
    print(f'{label:<17} RL base {base_rate:.3f} | GRPO 80 {rl_rate:.3f}')
```

Output:

```text
training harness  RL base 0.184 | GRPO 80 0.856
greedy (T = 0)    RL base 0.166 | GRPO 80 0.894
T = 0.7           RL base 0.181 | GRPO 80 0.884
T = 1.3           RL base 0.119 | GRPO 80 0.756
max_turns = 4     RL base 0.169 | GRPO 80 0.822
max_turns = 3     RL base 0.116 | GRPO 80 0.722
no renaming       RL base 0.178 | GRPO 80 0.863
unlimited edits   RL base 0.200 | GRPO 80 0.856
```

Changing one field of the harness moves the trained agent's number by up to 0.13. A temperature of 1.3 instead of the training temperature 1.0 costs 0.10; lowering it to 0.7, or decoding greedily, *adds* 0.03 to 0.04, because a sharper distribution samples the policy's mode, which is right on most tasks. A three-turn budget costs 0.13: the learned procedure is read, edit, test, submit, so episodes that test before submitting run out of turns, and only the episodes that skip the test still pass. The settings that the agent does not use leave its number unchanged: it never stacks edits, so unlimited edits change nothing, and it ignores names, so renaming changes nothing. For the base agent the same settings move the number between 0.116 and 0.200, but that agent guesses, so a harness that allows more attempts or more turns pays it more. Figure 17.2 shows these numbers with task-level intervals. Each harness change is a few lines of configuration in a real system; each is larger than most effects a team tries to measure.

![Figure 17.2: Held-out pass@1 of the RL base agent and the 80-step GRPO checkpoint under the training harness and seven one-field changes, with task-bootstrap 95% intervals. Dashed: the 0.2 pass rate of a blind single guess.](../figures/ch17-drift.svg)

### Greedy versus sampled evaluation

Greedy decoding (temperature 0) evaluates the mode of the policy; sampled evaluation at the training temperature evaluates the distribution that training optimized and that a best-of-$k$ system would draw from. They answer different questions, and the per-task counts show how.

```python
for i, name in enumerate(['RL base', 'GRPO 80']):
    print(f'{name:<8} sampled', drift['training harness'][i].tolist())
    print(f'{name:<8} greedy ', drift['greedy (T = 0)'][i].tolist())
```

Output:

```text
RL base  sampled [6, 7, 4, 8, 6, 6, 5, 3, 9, 5]
RL base  greedy  [0, 0, 0, 0, 16, 16, 11, 10, 0, 0]
GRPO 80  sampled [32, 32, 32, 30, 28, 25, 31, 1, 31, 32]
GRPO 80  greedy  [32, 32, 32, 32, 32, 30, 32, 0, 32, 32]
```

Greedy decoding is deterministic given the prompt, so its per-task counts are 0 or 32 except where the prompt itself varies. Here it does vary: renaming draws a random function name for every episode, which is why the base agent's greedy counts are 10 to 16 on four tasks and 0 on the other six. Its greedy behavior is essentially one fixed guess that is right only on the `compare` and `zero` tasks, and only for some names, while its sampled behavior spreads over all five repair ops. For the trained agent, greedy decoding removes the residual mistakes on most tasks (`under_cap` rises from 25 to 30) and turns `speed`, where the trained agent occasionally succeeds, into a hard zero. Greedy pass@1 is the right number for a deployment that decodes greedily. It is the wrong number for a sampled deployment or a best-of-$k$ system, and it hides exactly the low-probability successes that the pass@$k$ debate of Section 17.7 is about. The book's default is sampled evaluation at the training temperature, with the temperature reported.

## 17.6 Held-out splits and leakage

A held-out set answers a generalization question, and the question depends on what is held out. The default split of Chapter 4 holds out **variants**: two of the six variants of every bug family, with different function names, constants, and tests. It measures whether the agent generalizes across surface forms of bugs it was trained on. Holding out a **family** asks whether it generalizes to a kind of bug it has never been rewarded for fixing. We train for 30 steps on the four families other than `zero` (missing zero guard) and evaluate on all six `zero` tasks and on the eight held-out variants of the other families.

```python
FAMILY = 'zero'
four_families = copy.deepcopy(rl_base)
arl.train_rl(four_families, [t for t in train_tasks if t.family != FAMILY], tok, cfg, steps=30)
splits = {f'held-out family ({FAMILY}, 6 tasks)': [t for t in arl.make_tasks('all') if t.family == FAMILY],
          'held-out variants (8 tasks)': [t for t in heldout_tasks if t.family != FAMILY]}
for split, tasks in splits.items():
    for name, model in [('RL base', rl_base), ('GRPO, 4 families', four_families)]:
        s = arl.evaluate(model, tasks, tok, 16, generator=torch.Generator().manual_seed(2), cfg=cfg)
        print(f'{split:<34} {name:<17} pass {s["pass_rate"]:.3f} | reads {s["reads"]:.2f}'
              f' | first edit right {s["first_edit_right"]:.2f}')
```

Output:

```text
held-out family (zero, 6 tasks)    RL base           pass 0.125 | reads 0.35 | first edit right 0.21
held-out family (zero, 6 tasks)    GRPO, 4 families  pass 0.031 | reads 1.41 | first edit right 0.03
held-out variants (8 tasks)        RL base           pass 0.148 | reads 0.22 | first edit right 0.23
held-out variants (8 tasks)        GRPO, 4 families  pass 0.930 | reads 1.06 | first edit right 0.93
```

The two splits give opposite verdicts on the same checkpoint. On held-out variants of trained families, the agent reaches 0.930 from 0.148. On the unseen family it falls to 0.031, *below* the base agent's 0.125. It still reads (1.41 calls per episode) but its first edit is right in 3% of episodes: training made it confident in the four ops it was rewarded for and removed the guessing that gave the base agent its 0.125. A variant split would have reported a strong agent; a family split reports one that has learned four mappings and nothing about missing zero guards. Real coding benchmarks face the same choice at larger scale: held-out issues from training repositories, held-out repositories, or held-out languages measure different things, and a report should name the split.

**Contamination and leakage.** A held-out number is only held out if nothing about the held-out tasks reached training. In the gym, three channels exist, and it is worth checking each. The *tokenizer* is built from all 30 tasks (`build_tokenizer(make_tasks('all'))`), so held-out words have embeddings; a word that never occurs in training keeps its random initialization, which is harmless for the answer but is a leak of vocabulary. The *base agent* was behavior-cloned on training tasks only. The third channel is the one [Chapter 9's Exercise 3](09-grpo-and-variants.md#exercises) points out: `randomize_names` draws from all 30 function names, so held-out names appear in training episodes as random names. The block measures how often, and then tests whether it matters by evaluating with the renaming pool restricted to training names or to held-out names.

```python
heldout_names, train_names = {t.fn_name for t in heldout_tasks}, {t.fn_name for t in train_tasks}
rng = random.Random(0)
renamed = [arl.randomize_names(t, rng).fn_name for t in train_tasks for _ in range(50)]
print(f'training episodes renamed to a held-out function name: {sum(x in heldout_names for x in renamed) / len(renamed):.2f}')

def heldout_pass_with_pool(model, pool, seed=4):
    """Held-out pass@1 when episodes are renamed only to names in pool (temporarily replaces arl.grpo.FN_NAMES)."""
    saved, arl.grpo.FN_NAMES = arl.grpo.FN_NAMES, sorted(pool)
    try:
        return sample_counts(model, heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(seed)).sum().item() / 320
    finally:
        arl.grpo.FN_NAMES = saved

for name, pool in [('training names', train_names), ('held-out names', heldout_names)]:
    print(f'held-out tasks renamed to {name:<15} GRPO 80 pass@1 {heldout_pass_with_pool(trained, pool):.3f}')
```

Output:

```text
training episodes renamed to a held-out function name: 0.34
held-out tasks renamed to training names  GRPO 80 pass@1 0.844
held-out tasks renamed to held-out names  GRPO 80 pass@1 0.863
```

About a third of training episodes (0.34, ten of the 29 candidate names) carry a held-out function name. Restricting the held-out evaluation to training names or to held-out names gives 0.844 and 0.863, a difference within the evaluation noise of Section 17.4 (a standard deviation of about 0.016 per evaluation). The trained policy does not use the name, because renaming made the name uninformative during training, so this leak is real but inert for this metric. That is the general shape of a leakage audit: list the channels, then measure whether each one can move the number. The leaks that move numbers on real benchmarks carry *answers*, not names: an issue text that contains the fix, a benchmark repository in the pretraining or RL data, or test files visible to the agent ([Chapter 9](09-grpo-and-variants.md#94-shortcut-features-in-environment-design) and [Chapter 20](20-frontier.md) cite measured cases). Decontaminate training tasks against every evaluation set by repository, issue, and patch, and keep a fresh evaluation set built after the training data was frozen.

## 17.7 Does RL narrow the distribution? pass@$k$ on TinyAgent

[Chapter 20](20-frontier.md#does-rl-expand-capability-beyond-the-base-models-passk) summarizes a public debate. Yue et al. (2025) report that on math benchmarks RLVR-trained models win at small $k$ but "base models achieve a higher pass@k score when k is large", and read this as RL sharpening the base model's distribution rather than adding capability; ProRL (2025) reports the opposite after longer training with reference resets. Both are reported by their authors on math with large models, not reproduced here. The measurement itself is cheap with the tools of this chapter: unbiased pass@$k$ on held-out tasks for the base agent and for RL checkpoints, from the same 64 episodes per task. We add a reference curve, a uniform guess among the five repair ops, which passes with probability 0.2 per attempt.

```python
ks = [1, 2, 4, 8, 16, 32]
print('model     | ' + ' | '.join(f'k={k:<3d}' for k in ks))
passk_counts = {}
for name, model in [('RL base', rl_base), ('GRPO 20', run[20]), ('GRPO 80', trained)]:
    passk_counts[name] = sample_counts(model, heldout_tasks, tok, 64, cfg, torch.Generator().manual_seed(6))
    print(f'{name:<9} | ' + ' | '.join(f'{pass_at_k(64, passk_counts[name], k).mean():.3f}' for k in ks))
print(f'{"guess":<9} | ' + ' | '.join(f'{1 - 0.8 ** k:.3f}' for k in ks))
for name, c in passk_counts.items():
    print(f'{name:<9} successes of 64 per task:', c.tolist())
```

Output:

```text
model     | k=1   | k=2   | k=4   | k=8   | k=16  | k=32 
RL base   | 0.150 | 0.279 | 0.484 | 0.742 | 0.941 | 0.998
GRPO 20   | 0.616 | 0.791 | 0.897 | 0.963 | 0.995 | 1.000
GRPO 80   | 0.853 | 0.900 | 0.917 | 0.933 | 0.958 | 0.988
guess     | 0.200 | 0.360 | 0.590 | 0.832 | 0.972 | 0.999
RL base   successes of 64 per task: [12, 10, 9, 12, 8, 10, 10, 10, 6, 9]
GRPO 20   successes of 64 per task: [52, 48, 47, 45, 23, 11, 49, 13, 48, 58]
GRPO 80   successes of 64 per task: [63, 63, 63, 62, 60, 45, 62, 3, 62, 63]
```

The curves cross. At $k = 1$ the 80-step checkpoint leads the base agent by 0.70; at $k = 16$ it still leads, 0.958 against 0.941; at $k = 32$ the base agent is ahead, 0.998 against 0.988. The 20-step checkpoint is at or above both at every $k$. The per-task counts locate the crossing in one task, `speed`. The base agent solves it in 10 of 64 episodes, the 20-step checkpoint in 13, and the 80-step checkpoint in 3. Its pass@32 on `speed` is therefore about 0.88 against essentially 1 for the base agent, and that one task produces the whole deficit at large $k$. What did training do on that task?

```python
from collections import Counter
speed = [t for t in heldout_tasks if t.name == 'zero/speed']
print('zero/speed source:', arl.compact_source(speed[0].source).replace('\n', ' / '), '| fix:', speed[0].fix_op)
for name, model in [('RL base', rl_base), ('GRPO 80', trained)]:
    trajs = arl.sample_groups(model, lambda: arl.make_env(cfg), speed, tok, dataclasses.replace(cfg, group_size=64),
                              torch.Generator().manual_seed(6))
    edits = Counter(t.info['ops'][0] if t.info['ops'] else 'no edit' for t in trajs)
    print(f'{name:<8} first edit on zero/speed:', dict(edits.most_common()))
```

Output:

```text
zero/speed source: def speed ( miles , hours ) : / return miles * 2 // hours | fix: add_zero_guard
RL base  first edit on zero/speed: {'fix_comparison': 18, 'add_zero_guard': 13, 'fix_range_end': 9, 'hardcode_tests': 8, 'swap_minmax': 7, 'no edit': 6, 'fix_init': 3}
GRPO 80  first edit on zero/speed: {'fix_comparison': 59, 'add_zero_guard': 4, 'no edit': 1}
```

`speed` contains both a multiplication and an integer division; its bug is the missing zero guard. The base agent's first edit on it is spread over all six edit ops, `add_zero_guard` in 13 of 64 episodes. The trained agent applies `fix_comparison` in 59 of 64 episodes and the right op in 4. Somewhere between steps 20 and 80, the policy's mapping from source to op became confident and wrong for this task, and the probability of the right op fell from about 0.2 to 0.06. This is narrowing in the precise sense of the debate: RL concentrated probability on the behavior that its other experiences rewarded and removed most of the mass from an alternative that occasionally worked. Figure 17.3 shows the mean curves and the `speed` task.

![Figure 17.3: Left: unbiased pass@k on the ten held-out tasks for the RL base agent and the GRPO checkpoints after 20 and 80 steps, from 64 episodes per task, with task-bootstrap bands for the base agent and the 80-step checkpoint; dashed, a uniform guess among five ops. Right: the same curves for zero/speed alone.](../figures/ch17-passk.svg)

So, does RL narrow the distribution on TinyAgent? Measured here, yes on one task of ten, and the narrowing appeared late: the 20-step checkpoint dominated the base agent at every $k$. Two caveats bound the reading. First, the base agent's large-$k$ coverage is not a capability: its curve sits just below the uniform-guess curve, because in a five-op action space 32 guesses almost always include the right op. pass@$k$ at large $k$ measures how many attempts a guesser needs whenever the answer space is small, which is also a standing criticism of large-$k$ pass rates on benchmarks with short answers. Second, the crossing rests on a single task and one training run; with ten tasks, the task-level band of the 80-step curve spans the base agent's curve at $k \ge 16$. The mechanism is nevertheless the one the debate is about, and the per-task view is how to find it: look for tasks whose success count fell during training, and inspect what the policy now does instead.

> **Evidence.** Measured here: one 80-step GRPO run of a 158k-parameter model, ten held-out tasks, 64 episodes per task per checkpoint. Reported by sources: Yue et al. and ProRL (math, large models; Chapter 20). The tiny-scale result illustrates the measurement and one mechanism of narrowing; it is not evidence for either side of the debate at scale.

## 17.8 Choosing a checkpoint without touching the test set

Every choice made by looking at a number turns that number into a training signal. Choosing the checkpoint with the highest test score, then reporting that score, reports the maximum of several noisy estimates, which is biased upward: the **winner's curse**. With $m$ checkpoints of equal true skill and evaluation noise $\sigma$, the expected maximum is about $\mu + \sigma\,\mathbb{E}[\max_{i \le m} Z_i]$, which is $\mu + 1.42\sigma$ for $m = 8$. The fix is the one used for hyperparameters: split the held-out tasks into a **validation** set used for every decision and a **test** set evaluated once, for the final report. With ten held-out tasks we can afford only five and five, one variant of each family in each.

```python
val_tasks, test_tasks = heldout_tasks[0::2], heldout_tasks[1::2]          # one variant per family in each
print('validation:', [t.name for t in val_tasks])
print('test:      ', [t.name for t in test_tasks])

def rate(model, tasks, seed, n=32):
    """Pass@1 of model on tasks from n episodes per task drawn with generator seed."""
    return sample_counts(model, tasks, tok, n, cfg, torch.Generator().manual_seed(seed)).sum().item() / (n * len(tasks))

steps = list(range(10, 90, 10))
table = {step: (rate(run[step], val_tasks, 7), rate(run[step], test_tasks, 7)) for step in steps}
print('step | validation | test')
for step, (val, test) in table.items():
    print(f'{step:4d} | {val:10.3f} | {test:.3f}')
by_val, by_test = max(steps, key=lambda s: table[s][0]), max(steps, key=lambda s: table[s][1])
print(f'chosen by validation: step {by_val}, test {table[by_val][1]:.3f} | chosen by test: step {by_test},'
      f' test {table[by_test][1]:.3f}')
```

Output:

```text
validation: ['minmax/pin', 'range/series', 'compare/qualifies', 'zero/share', 'init/product_upto']
test:       ['minmax/saturate', 'range/step_total', 'compare/under_cap', 'zero/speed', 'init/add_threes']
step | validation | test
  10 |      0.450 | 0.394
  20 |      0.769 | 0.606
  30 |      0.781 | 0.675
  40 |      0.794 | 0.744
  50 |      0.762 | 0.744
  60 |      0.925 | 0.719
  70 |      0.850 | 0.762
  80 |      0.975 | 0.744
chosen by validation: step 80, test 0.744 | chosen by test: step 70, test 0.762
```

The validation and test columns disagree in level more than in shape. Validation reaches 0.975 at step 80 while test stays near 0.74 from step 40 on, because the test half contains `speed` and the validation half does not: with five tasks per split, the composition of the split decides the level. Choosing by validation picks step 80, whose test score, 0.744, is an unbiased estimate for the chosen checkpoint because the test set played no part in the choice. Choosing by test picks step 70 and reports 0.762. Whether that particular number is inflated cannot be read from one evaluation. The winner's curse is a statement about the expected value of the procedure, so we measure it by repeating the procedure: ten small test evaluations (8 episodes per task), each used to pick a checkpoint, and each pick scored on the nine evaluations that did not choose it.

```python
repeats = torch.tensor([[rate(run[step], test_tasks, 100 + r, n=8) for step in steps] for r in range(10)])  # [10, 8]
best = repeats.argmax(-1)                                                     # [10] the pick of each evaluation
reported = repeats[torch.arange(10), best]
held_back = torch.stack([(repeats[:, b].sum() - repeats[r, b]) / 9 for r, b in enumerate(best.tolist())])
print('checkpoint picked by each of 10 small test evaluations:', [steps[b] for b in best.tolist()])
print(f'score of the pick on its own evaluation {reported.mean():.3f} | on the other 9 evaluations {held_back.mean():.3f}')
torch.manual_seed(0)
se = (0.85 * 0.15 / 160) ** 0.5
best_of_8 = (0.85 + se * torch.randn(100_000, 8)).max(-1).values               # [100000]
print(f'simulation, 8 equally good checkpoints (true 0.850, se {se:.3f}): best reported averages {best_of_8.mean():.3f}')
```

Output:

```text
checkpoint picked by each of 10 small test evaluations: [50, 50, 50, 50, 80, 70, 30, 40, 50, 50]
score of the pick on its own evaluation 0.795 | on the other 9 evaluations 0.766
simulation, 8 equally good checkpoints (true 0.850, se 0.028): best reported averages 0.890
```

Different small evaluations pick different checkpoints (steps 30 to 80, most often 50), which already shows that the ranking among the later checkpoints is noise. The pick scores 0.795 on the evaluation that chose it and 0.766 on the evaluations that did not: selecting on the test set inflated the reported number by 0.03, about half a standard error of one of these 40-episode evaluations. The simulation with a 160-episode test set and eight equally good checkpoints gives the same order, 0.04 above the truth. The inflation grows with the number of checkpoints, configurations, and prompts tried against the same test set, which is why a test set that a team evaluates against every week stops being a test set. Figure 17.4 shows the checkpoint table with task-level intervals, and the four seeds of Section 17.4 with both kinds of interval.

![Figure 17.4: Left: validation and test pass@1 of the checkpoints of one GRPO run (5 tasks each, 32 episodes per task) with task-bootstrap 95% intervals. Right: the four seeds at step 30, each with an episode-level and a task-level bootstrap interval over all ten held-out tasks.](../figures/ch17-checkpoints.svg)

The intervals in the left panel overlap almost entirely from step 40 on. With five test tasks, no ranking of those checkpoints is supported, and the defensible report is the validation-chosen checkpoint's test score with its interval. In a larger evaluation, the same protocol scales: decide on validation (including early stopping, checkpoint averaging, and decoding settings), evaluate on test once per reported model, and report how many models were evaluated on that test set.

## 17.9 Design problems in practice

> **Problem: evaluation noise larger than the effect.** *Symptom:* an ablation table where arms differ by a few points, a periodic eval curve that zigzags, or a "win" that disappears on rerun. *Cause:* the evaluation's standard error, computed at the right unit (Section 17.3), is as large as the effect, and the comparison is unpaired or unreplicated. `train_rl`'s periodic `eval_pass` uses 4 episodes per task, 40 on the held-out set; the block re-evaluates one checkpoint eight times at that size. *Options:* (a) more episodes per task, which helps until the task term $\sigma^2_{\text{task}}/N$ dominates; (b) more tasks, the only fix for that term; (c) paired comparisons on the same tasks, with common random numbers when comparing checkpoints; (d) more training seeds when the claim is about a recipe. *Default:* before running an ablation, compute the paired standard deviation from a pilot evaluation and use `required_units` to size it; report intervals at the task level.

```python
small = [sample_counts(trained, heldout_tasks, tok, 4, cfg, torch.Generator().manual_seed(s)).sum().item() / 40
         for s in range(8)]
print('GRPO 80, 4 episodes per task, 8 evaluations:', [round(x, 3) for x in small])
print(f'range {max(small) - min(small):.3f}')
```

Output:

```text
GRPO 80, 4 episodes per task, 8 evaluations: [0.875, 0.9, 0.85, 0.8, 0.9, 0.9, 0.8, 0.825]
range 0.100
```

The same checkpoint scores between 0.80 and 0.90 across eight 40-episode evaluations, a range of 0.10. Chapter 9's preset and ablation tables compared arms evaluated with 80 episodes each, at a comparable noise level, and most of their differences were below this range; that is why Chapter 9 could not rank its variants.

> **Problem: harness drift.** *Symptom:* a checkpoint's number changes between two dashboards, a reproduction misses a paper's number by several points, or an improvement appears exactly when the eval code changed. *Cause:* $p_j(\pi, H)$ depends on the harness, and some harness field differs: temperature, budgets, template, the environment's rules, or the success criterion (Section 17.5). Drift is most dangerous when the changed field is one a weak policy can exploit. The block runs a skill-free policy that applies three fixed ops and submits, under the training harness and under the plain `BugFixEnv` that allows unlimited edits. Chapter 15's `pass_rate` helper evaluates in the plain environment, which matches Chapter 15's own training setup but not the default setting of Chapter 9 onward, so numbers from the two are not directly comparable. *Options:* (a) build evaluation from the training configuration object, as `sample_counts(..., cfg)` does, so that every field is shared by construction; (b) version the harness and record it with every number; (c) run a fixed reference policy (scripted or a frozen checkpoint) through every harness change and require its number to stay put. *Default:* (a) and (c). A canary policy whose score is known catches drift that no amount of looking at the evaluated model will.

```python
def stacker(observations):
    """A skill-free policy: three repair ops in a fixed order, then submit."""
    turn = len(observations) - 1
    ops = ['swap_minmax', 'fix_range_end', 'fix_comparison']
    return f'<call> edit {ops[turn]} </call>' if turn < 3 else '<call> submit </call>'

for name, make in [('training harness (one edit)', lambda: arl.make_env(cfg)), ('plain BugFixEnv', arl.BugFixEnv)]:
    results = [arl.run_episode(stacker, make(), t) for t in heldout_tasks]
    print(f'{name:<28} held-out pass {sum(r for r, _ in results) / len(results):.2f}')
```

Output:

```text
training harness (one edit)  held-out pass 0.20
plain BugFixEnv              held-out pass 0.60
```

The same skill-free policy passes 0.20 of held-out tasks under the training harness and 0.60 under the plain environment, because the plain environment lets it stack three guesses ([Section 9.4](09-grpo-and-variants.md#the-second-shortcut)). A model compared across those two harnesses would be credited with a 40-point improvement it did not make.

> **Problem: an evaluation that reuses the training reward.** *Symptom:* the training reward and the evaluation number rise together, and the behavior behind them is not the intended skill. *Cause:* when the evaluation computes success with the same function the policy was optimized against, it inherits every flaw the optimizer found: public tests that can be special-cased, a judge that can be flattered, a format check that can be satisfied without solving anything ([Chapter 11](11-rewards-and-verifiers.md)). The block runs Chapter 4's scripted hacker, which special-cases the public test inputs, through the gym with the public-test reward and the hidden-test reward. *Options:* (a) evaluate with a stronger, held-out verifier than the training reward (hidden tests, more tests, a different test generator); (b) audit samples of passing trajectories by hand or with behavior statistics; (c) track the gap between the training reward and the held-out verifier as a dashboard metric. *Default:* (a) always, and (c) as an alarm. `sample_counts` scores with the hidden tests whatever `cfg.reward` says, so an RL run trained on a hackable reward is still evaluated honestly.

```python
hacker = arl.scripted_policy('hacker')
for reward in ['public', 'hidden']:
    results = [arl.run_episode(hacker, arl.EditBudgetEnv(reward=reward), t) for t in heldout_tasks]
    hidden = sum(info['submitted'] and info['hidden_pass'] for _, info in results) / len(results)
    print(f'reward={reward:<7} mean reward {sum(r for r, _ in results) / len(results):.2f} | hidden pass {hidden:.2f}')
```

Output:

```text
reward=public  mean reward 1.00 | hidden pass 0.00
reward=hidden  mean reward 0.00 | hidden pass 0.00
```

Under its own training reward, the hacker scores a perfect 1.00; under the hidden tests, 0.00. An evaluation that reused the public-test reward would have reported a perfect agent.

> **Problem: choosing the checkpoint by the test set.** *Symptom:* the reported test score is never matched again, by a rerun, by a fresh test set, or by users. *Cause:* every decision made by looking at test numbers (the checkpoint, the decoding temperature, the prompt, which of several runs to report) selects on noise, and the reported maximum is biased upward by roughly $\sigma\,\mathbb{E}[\max_i Z_i]$ (Section 17.8, where it was 0.03 to 0.04). *Options:* (a) a validation split for all decisions and a test split used once; (b) when tasks are too few to split, report the pre-registered final checkpoint rather than the best one; (c) report the number of models evaluated on the test set, so readers can discount. *Default:* (a), with the selection rule written down before the test evaluation. For long-lived benchmarks, refresh the test set periodically, since a team's accumulated decisions leak into it even when each one is careful.

## Summary

- An evaluation estimates $p_j(\pi, H)$, a per-task success probability under a harness, and its raw data are per-task success counts. `sample_counts` produces them with the training configuration, scoring with hidden tests whatever the training reward was.
- pass@$k$ needs the unbiased estimator $1 - \binom{n-c}{k}/\binom{n}{k}$, computed as a product; the plug-in $1 - (1-\hat p)^k$ underestimated pass@5 by 0.04 to 0.08 at $n = 10$.
- Episodes of one task are correlated. With the trained agent's heterogeneous tasks the design effect was 20, and the task-level interval was 4.4 times as wide as the episode-level one; resampling episodes covered the population mean in 38% of simulated evaluations. Use Wilson for single tasks and the task-level bootstrap for benchmarks.
- Compare on the same tasks: pairing cut the standard error of a checkpoint difference from 0.132 to 0.023. Size comparisons with a power calculation (17 paired tasks for 5 points, 533 per arm unpaired) and measure seed variance with independent evaluations; at step 30 it was indistinguishable from evaluation noise here.
- One harness field moved the trained agent's held-out pass rate by up to 0.13, and a looser environment tripled a skill-free policy's score. Held-out variants (0.93) and a held-out family (0.03) gave opposite verdicts on one checkpoint.
- The base agent overtook the 80-step checkpoint at $k = 32$ (0.998 against 0.988) because RL drove the right op on one task from 13 to 4 of 64 episodes; at $k$ large enough, a guesser in a five-op space covers everything. Selecting on the test set inflated a reported score by about 0.03; decide on validation and report test once.

## Exercises

1. Prove that $1 - \binom{n-c}{k}/\binom{n}{k}$ is the minimum-variance unbiased estimator of $1-(1-p)^k$ from $n$ Bernoulli samples, using the fact that $c$ is a complete sufficient statistic. Then compute its variance for $n = 64$, $k = 32$, $p = 0.05$ by simulation and compare it with $k = 4$.
2. Change an assumption: rerun Section 17.7 with $n = 256$ episodes per task and $k$ up to 128, and with the 40-step checkpoint added. Does the crossing move? Does any task other than `speed` lose coverage? Plot per-task success counts across all checkpoints and find the step at which `speed` starts to fall.
3. Implement a Bayesian alternative to the task-level bootstrap: a Beta prior per task and a hierarchical model for task pass rates, sampled with simple Gibbs or importance sampling in torch. Compare its interval with `bootstrap_ci` on the counts of Section 17.1.
4. Repeat the held-out family experiment of Section 17.6 for each of the five families. Is the collapse below the base agent's guessing rate universal? Relate the result to the entropy of the first edit at the start of an unseen-family episode ([Chapter 10](10-kl-entropy-regularization.md)).
5. Write a harness canary: a function that runs a fixed set of scripted policies (`oracle`, `blind`, the `stacker` of Section 17.9) through `make_env(cfg)` and asserts their pass rates. Which fields of `RLConfig` does it detect a change in, and which can it miss?
6. Common random numbers: compare the paired standard error of the step-60 versus step-80 difference with the same generator for both checkpoints and with different generators. Explain the difference with the variance formula of Section 17.4.
7. Design review: a teammate reports that a new advantage estimator improves SWE-bench Verified pass@1 from 41.0% to 43.4%, from one training run per arm and one sample per task, and that the checkpoint was "the best of the last five". Using this chapter, list what you would ask for before accepting the claim, estimate the standard error of the reported difference under an unpaired and a paired analysis, and say how many seeds and samples you would request.
