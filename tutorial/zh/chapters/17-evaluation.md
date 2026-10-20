# 17. 评测

前面出现了不少通过率：留出任务上的 0.84、某次消融的 0.71 与默认设置的 0.76。怎样判断这些差异是否可信？本章从测量本身入手：通过率估计哪个量、误差有多大、如何比较检查点，以及该怎样使用验证集和测试集。尤其要分清，10 个任务各采样 32 次，不能直接当作 320 次针对不同任务的独立试验。

本章在 TinyAgent 上回答这些问题，使用的是第 9 章默认设置下的 GRPO 检查点；它还用同一套工具处理推理模型强化学习中争议最大的一项测量：强化学习究竟提升了大 $k$ 下的 pass@$k$，还是只是让基座模型原本就有的分布变得更尖锐。这里的每个数字都是一个 158k 参数模型在 10 个留出任务上的机制演示（mechanism demonstration）。任务只有 10 个，正好能让样本量不足带来的不确定性清楚地显现。

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

检查点来自一次 80 步的默认 GRPO 训练。`train_snapshots` 以每 10 步为一段调用 `arl.train_rl`，每段结束后保存一份冻结副本。每次调用都会重启 Adam 优化器并使用自己的随机种子，这与一次不间断的 80 步训练略有不同，但对本章要测量的东西没有影响。

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

## 17.1 通过率估计的是什么

评测估计的是策略*在某个评测框架下*的性质。评测框架 $H$ 指权重之外的一切：任务集、提示模板、环境及其预算（轮数、编辑次数、上下文长度）、采样温度，以及成功判据。对任务 $j$，记

$$
p_j(\pi, H) = \Pr\big[\text{an episode of } \pi \text{ on task } j \text{ under } H \text{ succeeds}\big].
$$

基准的 **pass@1** 是任务平均 $\bar p = \frac{1}{N}\sum_{j=1}^{N} p_j$。除非每个任务的回合（episode）数相同，否则它并不是“通过的回合所占比例”；它也不是权重单独具有的性质：改变 $H$，每个 $p_j$ 都会改变。这里的成功判据是第 4 章为评测固定下来的那一个，与任何训练奖励无关：智能体提交了答案，并且所有隐藏测试都通过。

因此，一次评测的原始数据是**每个任务的成功次数**：在 $N$ 个任务上各采样 $n$ 个回合，记录成功的次数 $c_j$。本章其余内容（pass@1、pass@$k$、区间、配对检验）都是向量 $c$ 的函数。`sample_counts` 用训练时的评测框架 `cfg`、通过训练所用的同一个 `sample_groups` 来产生它，这样模板、重命名、编辑预算、轮数预算和温度就不会悄无声息地出现差异。

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

我们在 10 个留出任务上，分别为 RL 基座智能体和第 80 步的检查点各采样 32 个回合。

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

两个通过率 0.144 和 0.819，是两个差别很大的向量的任务平均。基座智能体的成功几乎均匀分布，每个任务都是 32 次中成功 1 到 8 次：它在猜，而每个任务大约同样好猜。训练后的智能体在 8 个任务上几乎每个回合都成功，`under_cap` 是 32 次中 17 次，`speed` 是 32 次中 1 次。单个数字掩盖了一个任务基本未被解决的事实；下面的统计会表明，决定训练后智能体得分不确定性的，正是这种异质性，而不是回合数。

## 17.2 pass@$k$ 及其无偏估计量

编码系统通常允许多次尝试：在有验证器（verifier）的前提下，采样 $k$ 个补丁并保留一个通过的。**pass@$k$** 是 $k$ 次独立尝试中至少一次成功的概率，再对任务取平均：

$$
\text{pass@}k = \frac{1}{N}\sum_{j=1}^{N} \big(1 - (1 - p_j)^k\big).
$$

它需要一个能挑出通过尝试的预言机，因此衡量的是*覆盖率*，即任何选择器所能达到的上界，而不是可部署的成功率（[第 20 章](20-frontier.md#201-如何阅读一条前沿条目)将它与 Best@$k$ 区分开）。

**代入估计量是有偏的。** 最直接的估计是用 $\hat p_j = c_j/n$ 替代 $p_j$。当 $k \ge 2$ 时，映射 $f(p) = 1 - (1-p)^k$ 在 $(0, 1)$ 上严格凹，因此由 Jensen 不等式，只要 $0 < p_j < 1$，就有 $\mathbb{E}[f(\hat p_j)] < f(\mathbb{E}[\hat p_j]) = f(p_j)$：代入估计会系统性地*低估* pass@$k$，$n$ 越小、$k$ 越大，低估越严重。

**无偏估计量。** Chen et al. (2021) 采样 $n \ge k$ 个样本，并对它们提出另一个问题：如果从 $n$ 个样本中无放回地均匀随机挑出 $k$ 个，它们全部失败的概率是多少？若有 $c$ 个成功，该概率为 $\binom{n-c}{k}\big/\binom{n}{k}$，于是

$$
\widehat{\text{pass@}k}_j = 1 - \frac{\binom{n-c_j}{k}}{\binom{n}{k}}.
$$

无偏性一行即可证明。对任意*固定的* $k$ 个样本下标集合 $S$，事件“$S$ 中每个样本都失败”的概率为 $(1-p_j)^k$，因为样本相互独立。估计量中的失败项是该事件指示函数在全部 $\binom{n}{k}$ 个集合 $S$ 上的平均。期望是线性的，所以它的期望恰好是 $(1-p_j)^k$。（它是一个 U 统计量，即由样本构造的最小方差无偏估计量。）无偏性对每个任务成立，对任务取平均后依然成立。

**数值稳定的形式。** 远在 $n$ 变大之前，二项式系数就会让浮点数溢出：$\binom{2000}{1000}$ 有 600 位数字。把两个系数写成阶乘并约分，

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}} = \frac{(n-c)!\,(n-k)!}{n!\,(n-c-k)!} = \prod_{i=n-c+1}^{n} \frac{i-k}{i} = \prod_{i=n-c+1}^{n}\Big(1 - \frac{k}{i}\Big),
$$

当 $n - c \ge k$ 时，这是 $c$ 个落在 $(0, 1)$ 内的因子之积。当 $n - c < k$ 时，每个 $k$ 元子集都包含一个成功，估计值恰好为 1。

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

下面的检查对一个任务、在几个不同的真实通过率下，各抽取 100,000 次 $n = 10$ 个样本的评测，并把每个 pass@5 估计量的平均值与真值比较。

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

代入估计在每个 $p$ 下都偏低，低 0.04 到 0.08，这比第 9 章试图解读的大多数差异都大。无偏估计量在蒙特卡洛误差（这里约 0.001）之内与真值一致。偏差在中间的 $p$ 处最大，因为在 $\hat p$ 的分布范围内，$f$ 在那里弯曲得最厉害。

乘积形式与精确的有理数运算一致，并且在二项式系数之比无法表示的地方依然可用。

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

对 $n = 10, c = 3, k = 5$，乘积形式给出 $1 - 21/252 = 0.9167$，即[附录 B 问题 21](B-design-review.md#问题-21计算-passk) 中算出的值。对 $n = 2000$，$\binom{2000}{1000}$ 超过 600 位，无法转换为浮点数，而 5 个接近 $1/2$ 的因子之积给出 $0.9689 \approx 1 - 2^{-5}$。Chen et al. 使用 $n = 200$、$k \le 100$，并指出了同样的不稳定问题。$n$ 要选得远大于你报告的最大 $k$：当 $k$ 接近 $n$ 时，估计值几乎就是每个任务的单个指示变量，方差很大。

## 17.3 置信区间

没有区间的通过率无法与任何东西比较。三种区间覆盖了本书遇到的情形。

### 单个任务的二项区间

对 $n$ 个成功概率为 $p$ 的独立回合，$\hat p = c/n$ 的标准误为 $\sqrt{p(1-p)/n}$。**Wald 区间** $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ 代入了 $\hat p$，而它恰恰在智能体评测所处的区域失效：当 $\hat p = 0$ 或 $1$ 时宽度为零，靠近边界时覆盖不足。**Wilson 区间**不做代入，而是反转检验：它收集得分检验不会拒绝的所有 $p$，即 $|\hat p - p| \le z\sqrt{p(1-p)/n}$。两边平方得到关于 $p$ 的二次方程，其根为

$$
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.
$$

区间中心被拉向 1/2，宽度永不坍缩，这正是小 $n$ 所需要的。

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

我们在 20,000 次模拟评测（每次 40 个回合）上测量两种区间的覆盖率。

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

在 $p = 0.03$ 和 $p = 0.97$ 处，Wald 区间只在 70% 的评测中覆盖真值，因为大约 30% 的 40 回合评测一次失败（或一次成功）都没有观察到，于是报告宽度为零的区间。Wilson 区间在每个 $p$ 下都保持在 0.93 到 0.97 之间。对单个任务或少量回合，请使用 Wilson 区间或精确区间；只有当 $n\hat p$ 和 $n(1-\hat p)$ 都足够大时，Wald 区间才可以接受。

### 自助法，以及该重采样哪个单元

**自助法（bootstrap）**通过在从观测数据中有放回重采样得到的数据上重新计算统计量，来估计该统计量的抽样分布。百分位区间取重算值的 $\alpha/2$ 和 $1-\alpha/2$ 分位数。这里首先要检查：被重采样的单元是否可以看作从目标总体中独立抽取。样本量过小等问题仍会影响区间的可靠性。`bootstrap_ci` 接收每个单元一个值；`paired_bootstrap_diff` 为两个系统重采样*相同*的单元。

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

在一次包含 $N$ 个任务、每个任务 $n$ 个回合的评测中，回合之间并不独立：同一任务的回合共享该任务的难度。把任务视为从一个均值为 $\mu$、任务间方差为 $\sigma^2_{\text{task}} = \mathrm{Var}(p_j)$ 的总体中抽取。由全方差公式，估计值 $\hat\mu = \frac{1}{N}\sum_j c_j/n$ 满足

$$
\mathrm{Var}(\hat\mu) = \frac{1}{N}\Big(\sigma^2_{\text{task}} + \frac{\mathbb{E}[p_j(1-p_j)]}{n}\Big),
$$

而把全部 $Nn$ 个回合当作独立样本，则假设方差为 $\mu(1-\mu)/(Nn)$。由于 $\mu(1-\mu) = \sigma^2_{\text{task}} + \mathbb{E}[p_j(1-p_j)]$，真实方差与朴素方差之比就是**设计效应**（design effect）$1 + (n-1)\,\rho$，其中任务内相关系数 $\rho = \sigma^2_{\text{task}}/(\mu(1-\mu))$。由此得到两个推论。当任务之间存在差异（$\rho > 0$）时，朴素区间偏窄 $\sqrt{1 + (n-1)\rho}$ 倍。并且当 $n \to \infty$ 时，方差并不趋于零，而是趋于 $\sigma^2_{\text{task}}/N$：每个任务采样再多，也弥补不了任务太少。解决办法是**聚类自助法**（cluster bootstrap）：对任务做重采样，每个任务的回合随之一起带走，也就是把 `bootstrap_ci` 应用于每个任务的通过率。

哪种区间正确取决于问题，Miller (2024) 针对语言模型评测强调了这一点。“这个检查点在*这 10 个*任务上表现如何？”是以任务为条件的，只有回合采样是随机的。“它在*类似的*任务上会表现如何？”则把任务视为一个样本。讨论对新任务的泛化时，应回答第二个问题。一个总体已知的模拟可以把两者分开：任务通过率从 U 形的 Beta(0.4, 0.3) 分布中抽取（大多数任务几乎总是或几乎从不被解决，与智能体基准中的情形一样），共 10 个任务，每个任务 32 个回合。

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

重采样回合只在 38% 的评测中覆盖总体均值：它的区间太窄，通常会错过。它们在 99.8% 的评测中覆盖*这 10 个任务*的均值，这才是它们回答的问题（而且即使对这个问题也偏保守，因为二项方差 $\mu(1-\mu)$ 高估了任务内方差）。重采样任务在 93% 的评测中覆盖总体均值，接近名义水平；剩下的不足是百分位自助法在 $N = 10$ 时已知的覆盖不足。图 17.1 对每任务回合数 $n$ 从 1 到 64 重复了这个实验。每个任务只有一个回合时，两种区间重合；随着 $n$ 增大，回合级区间的宽度缩向零，而真实的不确定性停留在 $\sigma^2_{\text{task}}/N$，其覆盖率降到 0.3 以下。

![图 17.1：随着每任务回合数增加，名义 95% 自助区间对总体通过率的覆盖率，分别对回合重采样和对任务重采样。模拟设置为 10 个任务，其通过率从 Beta(0.4, 0.3) 中抽取，每个点 300 次评测。](../../figures/ch17-coverage.svg)

在 17.1 节实测的计数上做同样的比较，并给出任务内相关系数的矩估计：

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

对基座智能体，两种区间一致（宽度 0.075 和 0.081）：它的各个任务同样难，$\rho \approx 0.01$，它的回合表现得像独立的抛硬币。对训练后的智能体，$\rho = 0.63$，设计效应为 20：任务级区间 $[0.60, 0.96]$ 的宽度是回合级区间的 4.4 倍。只有 10 个任务、其中一个几乎没被解决，诚实的说法是：训练后的智能体在类似任务上的留出通过率大约介于 0.6 和 0.96 之间。继续增加每个任务的回合数，只能减少任务内的采样噪声；这里更大的不确定性来自任务差异，需要更多独立任务才能有效缩小。

## 17.4 比较两次训练

把两个数字区分开的随机性有两种。**评测噪声**来自从固定检查点采样回合，更多回合能缩小它。**训练噪声**来自训练的随机种子（任务顺序、采样的组、小批次顺序），它是训练配方的性质：同一配方的两次训练会停在不同的检查点。训练噪声是否应计入误差棒，取决于所做的论断。“检查点 B 优于检查点 A”是关于两个固定检查点的论断。“配方 B 优于配方 A”是关于检查点分布的论断，它的重复单元是随机种子。

我们用默认配方再训练三个随机种子，各 30 步——此时训练尚未饱和——并对四个第 30 步检查点各做两次相互独立的评测（每次 320 个回合）。重复评测衡量评测噪声；跨种子的离散则包含两种噪声。

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

同一检查点的两次评测相差最多 0.028，一次 320 回合评测的标准差为 0.016，接近二项式的 $\sqrt{0.71 \cdot 0.29/320} = 0.025$（它估计的是条件噪声，即“这 10 个任务”上的噪声，17.3 节已表明它更小）。四个种子的均值从 0.706 到 0.720，离散程度*小于*同一种子的两次评测，估计出的训练噪声标准差为零。在这个设置下、在第 30 步，默认配方在不同种子之间高度可复现，几乎所有可见的离散都是评测噪声。这也让我们重新解读[第 9 章的消融表](09-grpo-and-variants.md#96-小规模下的受控消融)：其默认组的三个留出值 0.76 到 0.81 各来自 80 个回合，其评测标准误约为 0.045，所以那里的离散也主要是评测噪声。这两个结果都不表示随机种子在一般情况下无关紧要。种子方差取决于配方、规模和训练所处的阶段，必须像这里一样用相互独立的评测测量过，才能忽略它。

### 配对比较

当两个系统在相同任务上运行时，应逐任务比较。每个任务的差 $d_j = \hat p^{A}_j - \hat p^{B}_j$ 消去了两个系统共享的任务难度，因此

$$
\mathrm{Var}(\bar d) = \frac{1}{N}\big(\mathrm{Var}(\hat p^A_j) + \mathrm{Var}(\hat p^B_j) - 2\,\mathrm{Cov}(\hat p^A_j, \hat p^B_j)\big),
$$

协方差项在任务难度差异大时很大，而非配对比较恰恰丢掉了它。我们在相同的 10 个任务上比较第 60 步和第 80 步的检查点，并让两者使用同一个随机数生成器。两个相似的策略在相同随机数驱动下往往采样出相同的回合，这进一步提高了它们结果之间的相关性。这就是仿真中的**公共随机数**（common random numbers）技术；它可以降低配对差值的方差，也正因如此，当目标是测量评测噪声时（如上一个代码块），*不能*重复使用同一个生成器。

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

第 80 步检查点比第 60 步检查点高 0.041，配对区间 $[+0.006, +0.087]$ 不包含零。若当作两个非配对的 10 任务样本，同样的差异的标准误是 0.132，是配对标准误 0.023 的六倍，差异将完全看不出来。两个检查点的逐任务通过率相关系数为 0.97：对一个难的任务对另一个也难，配对把这部分共享难度减掉了。配对区间回答的是关于*这些*任务和这两个检查点的问题；它对配方一无所知，而且只有 10 个任务，它本身也只是一个粗略的区间。

### 需要多少任务或随机种子？

**功效分析**（power calculation）把比较反过来：给定我们想检测的效应 $\delta$，需要多少个单元？对显著性水平 $\alpha$、功效 $1-\beta$ 的双侧检验，$N$ 个配对差（每单元标准差为 $\sigma_d$）的均值必须满足 $\delta/(\sigma_d/\sqrt N) \ge z_{1-\alpha/2} + z_{1-\beta}$，因此

$$
N \ge \Big(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\delta}\Big)^2.
$$

对两个独立组（非配对的任务，或各用一组种子的两个配方），两均值之差的每单元标准差为 $\sqrt2\,\sigma$，公式给出的是*每一组*的大小。

```python
#@save
def required_units(sd, effect, alpha=0.05, power=0.8):
    """Units (tasks, seeds) for a two-sided z-test to detect a mean difference `effect` given per-unit sd."""
    normal = torch.distributions.Normal(0.0, 1.0)
    z = normal.icdf(torch.tensor(1 - alpha / 2)) + normal.icdf(torch.tensor(power))
    return math.ceil((z.item() * sd / effect) ** 2)
```

我们代入实测的离散度：两个检查点之间的逐任务差、单个检查点的逐任务离散（用于非配对设计），以及跨种子的离散。

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

在配对任务下，要检测这类两个检查点之间 2 个百分点的差异，大约需要 103 个任务，5 个百分点大约需要 17 个。非配对设计（例如，把论文里的数字与你自己评测框架下、在不同任务样本上得到的数字比较）需要约 30 倍的数量：5 个百分点每组需要 533 个任务。种子一列很小，因为实测的跨种子离散很小；公式给出的“一个种子”是关于功效的陈述，而不是关于实践的建议。光是估计 $\sigma_{\text{seed}}$ 每组就至少需要三个种子，配方比较绝不应建立在更少的种子上。

> **生产实践。** 同样的算术决定了真实基准的规模。在 SWE-bench Verified 的 500 个任务上，若每个任务尝试一次，通过率在 0.4 附近时的二项标准误约为 0.022，因此两次独立的单次尝试运行之差的标准误约为 0.031，2 个百分点的差距完全落在其内。逐任务配对、每任务多次尝试（如 DeepSWE 取 16 次运行的平均，见[第 20 章](20-frontier.md)），以及多个训练种子，是可用的手段。这是由二项公式算出的结果，不是测量。

## 17.5 评测框架定义了这个数字

评测框架选择是指权重之外任何会改变 $p_j(\pi, H)$ 的设置。如果训练与评测之间、或被比较的两次评测之间有一项不同，那么比较测到的就是评测框架。我们在训练评测框架以及七种单字段改动下评测同样的两个检查点。

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

只改动评测框架的一个字段，就能让训练后智能体的数字最多变化 0.13。温度取 1.3 而非训练温度 1.0 会损失 0.10；降到 0.7 或贪心解码则会*增加* 0.03 到 0.04，因为这些设置更偏向当前策略的高概率选择，而这些选择在大多数任务上是正确的。三轮预算损失 0.13：学到的流程是读取、编辑、测试、提交，所以在提交之前先测试的回合会用完轮数，只有跳过测试的回合仍能通过。智能体不用的设置则不会改变它的数字：它从不叠加编辑，所以不限编辑次数没有影响；它忽略函数名，所以重命名没有影响。对基座智能体，同样的设置使数字在 0.116 到 0.200 之间变动，但那个智能体是在猜，所以允许更多尝试或更多轮数的评测框架会给它更多回报。图 17.2 展示了这些数字及其任务级区间。在真实系统中，每一处评测框架改动都只是几行配置；而每一处带来的变化都大于团队试图测量的大多数效应。

![图 17.2：RL 基座智能体与 80 步 GRPO 检查点在训练评测框架及七种单字段改动下的留出 pass@1，附任务级自助法 95% 区间。虚线：盲猜一次的通过率 0.2。](../../figures/ch17-drift.svg)

### 贪心评测与采样评测

贪心解码（温度 0）评测的是每一步都选择最高概率 token 时的表现；这不保证整段序列的概率全局最高。在训练温度下采样，评测的才是训练所优化的随机策略，也是使用相同采样配置的 best-of-$k$ 系统会采样的分布。二者回答不同的问题，逐任务计数显示了区别所在。

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

给定提示，贪心解码是确定性的，所以除非提示本身发生变化，其逐任务计数只能是 0 或 32。这里提示确实会变：重命名为每个回合随机抽取一个函数名，这就是为什么基座智能体的贪心计数在四个任务上是 10 到 16，在其余六个任务上是 0。它的贪心行为本质上是一个固定的猜测，只在 `compare` 和 `zero` 任务上、且只对部分函数名是对的；而它的采样行为则分散在全部五种修复操作上。对训练后的智能体，贪心解码消除了大多数任务上的残余错误（`under_cap` 从 25 升到 30），并把训练后智能体偶尔能成功的 `speed` 变成了彻底的零。对于贪心解码的部署，贪心 pass@1 是正确的数字。对于采样部署或 best-of-$k$ 系统，它是错误的数字，而且它恰好掩盖了 17.7 节 pass@$k$ 之争所关注的那些低概率成功。本书的默认做法是在训练温度下做采样评测，并报告所用温度。

## 17.6 留出划分与泄漏

留出集回答的是一个泛化问题，而这个问题取决于留出了什么。第 4 章的默认划分留出的是**变体**：每种缺陷类型六个变体中的两个，函数名、常数和测试都不同。它衡量的是智能体能否在训练过的 bug 的不同表面形式之间泛化。留出一种**缺陷类型**，问的则是它能否泛化到一类从未因修复而获得奖励的 bug。我们在 `zero`（缺少除零保护）以外的四种缺陷类型上训练 30 步，然后在全部六个 `zero` 任务以及其他缺陷类型的八个留出变体上评测。

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

两种划分对同一个检查点给出了相反的结论。在训练过的缺陷类型的留出变体上，智能体从 0.148 提升到 0.930。在未见过的缺陷类型上，它降到 0.031，*低于*基座智能体的 0.125。它仍然会读取（每回合 1.41 次调用），但第一次编辑只在 3% 的回合中正确：训练让它对那四种获得过奖励的操作充满信心，并消除了让基座智能体得到 0.125 的猜测行为。变体划分会报告一个强大的智能体；按缺陷类型划分报告的则是一个学会了四种映射、却对缺少除零保护一无所知的智能体。真实编程基准在更大规模上面临同样的选择：来自训练仓库的留出 issue、留出仓库、留出编程语言衡量的是不同的东西，报告中应说明所用的划分。

**污染与泄漏。** 只有当留出任务的任何信息都没有进入训练时，留出数字才真正是留出的。在 gym 中存在三条渠道，值得逐一检查。*分词器*是用全部 30 个任务构建的（`build_tokenizer(make_tasks('all'))`），因此留出任务中的词都有嵌入；一个从未在训练中出现的词保持随机初始化，这对答案无害，但属于词表泄漏。*基座智能体*只在训练任务上做了行为克隆。第三条渠道就是[第 9 章练习 3](09-grpo-and-variants.md#练习) 指出的那一条：`randomize_names` 从全部 30 个函数名中抽取，所以留出任务的函数名会作为随机名字出现在训练回合中。下面的代码块测量这种情况出现的频率，然后把重命名池限制为只含训练函数名或只含留出函数名进行评测，以检验它是否有影响。

```python
heldout_names, train_names = {t.fn_name for t in heldout_tasks}, {t.fn_name for t in train_tasks}
rng = random.Random(0)
renamed = [arl.randomize_names(t, rng).fn_name for t in train_tasks for _ in range(50)]
print(f'training episodes renamed to a held-out function name: {sum(x in heldout_names for x in renamed) / len(renamed):.2f}')

def heldout_pass_with_pool(model, pool, seed=4):
    """Held-out pass@1 when episodes are renamed only to names in pool (temporarily replaces arl.FN_NAMES)."""
    saved, arl.FN_NAMES = arl.FN_NAMES, sorted(pool)
    try:
        return sample_counts(model, heldout_tasks, tok, 32, cfg, torch.Generator().manual_seed(seed)).sum().item() / 320
    finally:
        arl.FN_NAMES = saved

for name, pool in [('training names', train_names), ('held-out names', heldout_names)]:
    print(f'held-out tasks renamed to {name:<15} GRPO 80 pass@1 {heldout_pass_with_pool(trained, pool):.3f}')
```

Output:

```text
training episodes renamed to a held-out function name: 0.34
held-out tasks renamed to training names  GRPO 80 pass@1 0.844
held-out tasks renamed to held-out names  GRPO 80 pass@1 0.863
```

大约三分之一的训练回合（0.34，29 个候选名字中的 10 个）带有留出任务的函数名。把留出评测限制为训练函数名或留出函数名，分别得到 0.844 和 0.863，二者之差在 17.4 节的评测噪声之内（每次评测的标准差约 0.016）。训练后的策略不使用函数名，因为重命名在训练中让名字失去了信息量，所以这个泄漏是真实存在的，但对这个指标没有作用。这就是泄漏审计的一般形态：列出各条渠道，然后测量每一条能否移动这个数字。在真实基准上能移动数字的泄漏携带的是*答案*，而不是名字：包含修复方法的 issue 文本、出现在预训练或强化学习数据中的基准仓库，或智能体可见的测试文件（[第 9 章](09-grpo-and-variants.md#94-环境设计中的捷径特征)和[第 20 章](20-frontier.md)引用了实测案例）。要按仓库、issue 和补丁对训练任务与每个评测集做去污染，并保留一个在训练数据冻结之后才构建的新评测集。

## 17.7 强化学习是否让分布变窄？TinyAgent 上的 pass@$k$

[第 20 章](20-frontier.md#强化学习能把能力扩展到基座模型的-passk-之外吗)总结了一场公开争论。Yue et al. (2025) 报告，在数学基准上，经 RLVR 训练的模型在小 $k$ 时占优，但“base models achieve a higher pass@k score when k is large”（大 $k$ 时基座模型的 pass@k 更高），并将此解读为强化学习只是让基座模型的分布更尖锐，而非增加了能力；ProRL (2025) 则报告，在加入参考策略重置、训练更长时间后得到了相反的结论。两者都是作者在数学任务和大模型上报告的结果，本书没有复现。用本章的工具，这项测量本身并不昂贵：用每个任务相同的 64 个回合，在留出任务上计算基座智能体和强化学习检查点的无偏 pass@$k$。我们加上一条参考曲线：在五种修复操作中均匀随机猜测，每次尝试的通过概率为 0.2。

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

曲线发生了交叉。在 $k = 1$ 时，80 步检查点领先基座智能体 0.70；在 $k = 16$ 时仍然领先，0.958 对 0.941；在 $k = 32$ 时，基座智能体反超，0.998 对 0.988。20 步检查点在每个 $k$ 上都不低于两者。逐任务计数把交叉定位到一个任务上：`speed`。基座智能体在 64 个回合中解决它 10 次，20 步检查点 13 次，80 步检查点 3 次。因此 80 步检查点在 `speed` 上的 pass@32 约为 0.88，而基座智能体基本为 1，大 $k$ 处的全部差距都来自这一个任务。训练在这个任务上做了什么？

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

`speed` 同时包含乘法和整数除法；它的 bug 是缺少除零保护。基座智能体在它上面的第一次编辑分散在全部六种编辑操作上，`add_zero_guard` 出现在 64 个回合中的 13 个。训练后的智能体在 64 个回合中有 59 次使用 `fix_comparison`，用对操作的只有 4 次。在第 20 步到第 80 步之间的某个时候，策略从源码到操作的映射对这个任务变得自信而错误，正确操作的概率从约 0.2 降到了 0.06。这正是那场争论中精确意义上的变窄：强化学习把概率集中到其他经验所奖励的行为上，并从一个偶尔有效的替代方案中拿走了大部分概率质量。图 17.3 展示了平均曲线和 `speed` 任务。

![图 17.3：左：RL 基座智能体以及第 20 步和第 80 步 GRPO 检查点在 10 个留出任务上的无偏 pass@k，每个任务 64 个回合，基座智能体和 80 步检查点附任务级自助法区间带；虚线为在五种操作中均匀猜测。右：仅 zero/speed 一个任务上的相同曲线。](../../figures/ch17-passk.svg)

那么，强化学习在 TinyAgent 上是否让分布变窄了？本书测得：在 10 个任务中的 1 个上是的，而且变窄出现得很晚：20 步检查点在每个 $k$ 上都优于基座智能体。有两点注意事项限定了这一解读。第一，基座智能体在大 $k$ 下的覆盖率并不是一种能力：它的曲线紧贴在均匀猜测曲线之下，因为在五种操作的动作空间中，32 次猜测几乎总会包含正确的操作。只要答案空间很小，大 $k$ 下的 pass@$k$ 衡量的就是一个猜测者需要多少次尝试——这也是对短答案基准上大 $k$ 通过率的一项长期批评。第二，交叉只建立在一个任务和一次训练之上；只有 10 个任务时，80 步曲线的任务级区间带在 $k \ge 16$ 处覆盖了基座智能体的曲线。尽管如此，这里的机制正是那场争论所讨论的机制，而逐任务视角就是找到它的方法：寻找在训练过程中成功次数下降的任务，并检查策略现在转而在做什么。

> **证据。** 本书测得：一个 158k 参数模型的一次 80 步 GRPO 训练，10 个留出任务，每个检查点每个任务 64 个回合。文献报告：Yue et al. 与 ProRL（数学任务、大模型；见第 20 章）。这个小规模结果说明了测量方法和一种变窄机制；它不能作为这场争论中任何一方在大规模上的证据。

## 17.8 不碰测试集地选择检查点

每一个通过查看数字做出的选择，都会把那个数字变成训练信号。选择测试分数最高的检查点、再报告这个分数，报告的是多个带噪声估计中的最大值，它是向上有偏的：这就是**赢家诅咒**（winner's curse）。若有 $m$ 个真实水平相同的检查点、评测噪声为 $\sigma$，期望最大值约为 $\mu + \sigma\,\mathbb{E}[\max_{i \le m} Z_i]$，当 $m = 8$ 时为 $\mu + 1.42\sigma$。解决办法与超参数相同：把留出任务划分为用于每一个决策的**验证集**，以及只在最终报告时评测一次的**测试集**。只有 10 个留出任务，我们只能五五分，每一侧包含每种缺陷类型的一个变体。

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

验证列与测试列在水平上的分歧大于在形状上的分歧。验证分数在第 80 步达到 0.975，而测试分数从第 40 步起一直停留在 0.74 附近，因为测试那一半包含 `speed`，验证那一半不包含：每个划分只有五个任务时，划分的构成决定了水平。按验证选择会选中第 80 步，其测试分数 0.744 对所选检查点是无偏估计，因为测试集没有参与选择。按测试选择会选中第 70 步，报告 0.762。从一次评测无法看出这个具体数字是否被高估。赢家诅咒是关于这一选择流程期望值的陈述，所以我们通过重复这个流程来测量它：做 10 次小规模测试评测（每个任务 8 个回合），每次用来挑选一个检查点，再用另外 9 次没有参与挑选的评测给所选检查点打分。

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

不同的小规模评测挑出了不同的检查点（第 30 步到第 80 步，最常见的是第 50 步），这说明现有评测精度不足以稳定区分后期检查点。被选中的检查点在挑选它的评测上得 0.795，在其余评测上得 0.766：在测试集上做选择把报告的数字抬高了 0.03，大约是一次这种 40 回合评测标准误的一半。用 160 回合测试集和八个同样好的检查点做的模拟给出了相同的量级，比真值高 0.04。针对同一测试集尝试的检查点、配置和提示越多，高估就越大；这就是为什么一个团队每周都拿来评测的测试集，会不再是测试集。图 17.4 展示了附任务级区间的检查点表，以及 17.4 节的四个随机种子及其两种区间。

![图 17.4：左：一次 GRPO 训练各检查点的验证与测试 pass@1（各 5 个任务，每个任务 32 个回合），附任务级自助法 95% 区间。右：第 30 步的四个随机种子，每个都附有在全部 10 个留出任务上的回合级与任务级自助区间。](../../figures/ch17-checkpoints.svg)

左图中的区间从第 40 步起几乎完全重叠。只有五个测试任务时，这些检查点的任何排序都没有依据，站得住脚的报告是按验证集选出的检查点的测试分数及其区间。在更大的评测中，同样的规程可以照搬：在验证集上做决策（包括早停、检查点平均和解码设置），对每个要报告的模型只在测试集上评测一次，并报告有多少个模型在该测试集上评测过。

## 17.9 实践中的设计问题

> **问题：评测噪声大于效应。**
>
> **现象：** 消融表中各组相差几个百分点，周期性评测曲线呈锯齿状，或者一次“胜出”在重跑后消失。
>
> **原因：** 在正确单元上计算的评测标准误（17.3 节）与效应一样大，而比较是非配对或没有重复的。`train_rl` 的周期性 `eval_pass` 每个任务用 4 个回合，在留出集上共 40 个；下面的代码块以这个规模对同一个检查点重复评测八次。
>
> **可选方案：** (a) 增加每个任务的回合数，这在任务项 $\sigma^2_{\text{task}}/N$ 占主导之前有效；(b) 增加任务数，这是减小该项的唯一办法；(c) 在相同任务上做配对比较，比较检查点时使用公共随机数；(d) 当论断针对配方时，增加训练种子数。
>
> **建议做法：** 在做消融之前，先从一次试点评测计算配对标准差，并用 `required_units` 确定规模；在任务级别报告区间。

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

同一个检查点在八次 40 回合评测中得分介于 0.80 和 0.90 之间，极差 0.10。第 9 章的预设表和消融表比较的各组都是用 80 个回合评测的，噪声水平相当，而它们的大多数差异都低于这个极差；这就是第 9 章无法对其变体排序的原因。

> **问题：评测框架漂移。**
>
> **现象：** 同一检查点的数字在两个监控面板之间不一致，复现论文时差了几个百分点，或者改进恰好出现在评测代码改动之时。
>
> **原因：** $p_j(\pi, H)$ 依赖于评测框架，而某个评测框架字段不同了：温度、预算、模板、环境规则或成功判据（17.5 节）。当被改动的字段是弱策略可以利用的字段时，漂移最危险。下面的代码块运行一个没有技能的策略——它按固定顺序施加三种操作然后提交——分别在训练评测框架下和允许无限编辑的普通 `BugFixEnv` 下运行。第 15 章的 `pass_rate` 辅助函数在普通环境中评测，这与第 15 章自己的训练设置一致，但与第 9 章起的默认设置不一致，所以两者的数字不能直接比较。
>
> **可选方案：** (a) 从训练配置对象构建评测，就像 `sample_counts(..., cfg)` 那样，使每个字段在构造上就是共享的；(b) 为评测框架加版本号，并随每个数字一起记录；(c) 让一个固定的参考策略（脚本策略或一个冻结的检查点）经过每一次评测框架改动，并要求它的数字保持不变。
>
> **建议做法：** (a) 和 (c)。一个分数已知的金丝雀策略能捕捉到的漂移，无论怎样审视被评测的模型都发现不了。

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

同一个没有技能的策略，在训练评测框架下通过 0.20 的留出任务，在普通环境下通过 0.60，因为普通环境允许它叠加三次猜测（[9.4 节](09-grpo-and-variants.md#第二个捷径)）。一个在这两种评测框架之间比较的模型，会被记上一个它并未取得的 40 个百分点的提升。

> **问题：评测复用了训练奖励。**
>
> **现象：** 训练奖励和评测数字一起上升，而背后的行为并不是预期的技能。
>
> **原因：** 当评测用策略被优化所针对的同一个函数来计算成功时，它会继承优化器找到的每一个缺陷：可以被特判的公开测试、可以被讨好的评判模型、不解决任何问题也能满足的格式检查（[第 11 章](11-rewards-and-verifiers.md)）。下面的代码块让第 4 章的脚本化投机策略（它对公开测试的输入做特判）分别在公开测试奖励和隐藏测试奖励下运行 gym。
>
> **可选方案：** (a) 用比训练奖励更强的留出验证器评测（隐藏测试、更多测试、不同的测试生成器）；(b) 人工或借助行为统计抽查通过的轨迹；(c) 把训练奖励与留出验证器之间的差距作为监控面板指标来跟踪。
>
> **建议做法：** 始终采用 (a)，并以 (c) 作为警报。无论 `cfg.reward` 是什么，`sample_counts` 都用隐藏测试打分，所以即使强化学习是在可投机的奖励上训练的，评测仍然是诚实的。

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

在自己的训练奖励下，投机策略得到满分 1.00；在隐藏测试下得 0.00。复用公开测试奖励的评测会报告出一个完美的智能体。

> **问题：用测试集选择检查点。**
>
> **现象：** 报告的测试分数再也没有被重现过，无论是重跑、新的测试集还是用户。
>
> **原因：** 每一个通过查看测试数字做出的决策（检查点、解码温度、提示、报告哪一次训练）都是在噪声上做选择，报告的最大值大约向上偏 $\sigma\,\mathbb{E}[\max_i Z_i]$（17.8 节中为 0.03 到 0.04）。
>
> **可选方案：** (a) 用验证集做所有决策，测试集只用一次；(b) 任务太少无法划分时，报告预先登记的最终检查点，而不是最好的那个；(c) 报告在测试集上评测过的模型数量，让读者判断选择偏差可能有多大。
>
> **建议做法：** (a)，并在测试评测之前写下选择规则。对于长期使用的基准，要定期更新测试集，因为即使每个决策都很谨慎，团队累积的决策仍会泄漏进去。

## 小结

- 评测估计的是 $p_j(\pi, H)$，即某个评测框架下每个任务的成功概率，其原始数据是每个任务的成功次数。`sample_counts` 用训练配置生成这些数据，并且无论训练奖励是什么，都用隐藏测试打分。
- pass@$k$ 需要无偏估计量 $1 - \binom{n-c}{k}/\binom{n}{k}$，以乘积形式计算；在 $n = 10$ 时，代入估计 $1 - (1-\hat p)^k$ 把 pass@5 低估了 0.04 到 0.08。
- 同一任务的回合是相关的。对训练后智能体这样异质的任务，设计效应为 20，任务级区间的宽度是回合级区间的 4.4 倍；在模拟评测中，对回合重采样只在 38% 的情况下覆盖总体均值。单个任务用 Wilson 区间，基准用任务级自助法。
- 在相同任务上比较：配对把检查点差异的标准误从 0.132 降到 0.023。用功效分析确定比较的规模（检测 5 个百分点需要 17 个配对任务，非配对时每组 533 个），并用相互独立的评测测量种子方差；在这里第 30 步时，它与评测噪声无法区分。
- 评测框架的一个字段就让训练后智能体的留出通过率最多变化 0.13，一个更宽松的环境让无技能策略的分数变为三倍。留出变体（0.93）和留出类型（0.03）对同一个检查点给出了相反的结论。
- 基座智能体在 $k = 32$ 时反超 80 步检查点（0.998 对 0.988），原因是强化学习把一个任务上正确操作的出现次数从 64 次中的 13 次压到了 4 次；当 $k$ 足够大时，五种操作空间中的猜测者能覆盖一切。在测试集上做选择把报告分数抬高了约 0.03；应在验证集上决策，在测试集上只报告一次。

## 练习

1. 利用 $c$ 是完备充分统计量这一事实，证明 $1 - \binom{n-c}{k}/\binom{n}{k}$ 是基于 $n$ 个伯努利样本的 $1-(1-p)^k$ 的最小方差无偏估计量。然后通过模拟计算 $n = 64$、$k = 32$、$p = 0.05$ 时它的方差，并与 $k = 4$ 时比较。
2. 改变一个假设：以每个任务 $n = 256$ 个回合、$k$ 最大到 128 重跑 17.7 节，并加入第 40 步检查点。交叉点会移动吗？除 `speed` 之外，是否还有其他任务失去覆盖？绘制所有检查点的逐任务成功次数，找出 `speed` 开始下降的步数。
3. 实现任务级自助法的贝叶斯替代方案：每个任务一个 Beta 先验，任务通过率采用层次模型，在 torch 中用简单的 Gibbs 采样或重要性采样求解。在 17.1 节的计数上，将其区间与 `bootstrap_ci` 比较。
4. 对五种缺陷类型中的每一个，重复 17.6 节的留出类型实验。跌到基座智能体猜测率以下的现象是否普遍？把结果与未见过的缺陷类型回合开始时第一次编辑的熵联系起来（[第 10 章](10-kl-entropy-regularization.md)）。
5. 编写一个评测框架金丝雀：一个函数，让一组固定的脚本策略（`oracle`、`blind`、17.9 节的 `stacker`）通过 `make_env(cfg)` 运行，并断言它们的通过率。它能检测出 `RLConfig` 中哪些字段的改动，又会漏掉哪些？
6. 公共随机数：比较第 60 步与第 80 步差异的配对标准误，一种是两个检查点使用同一个生成器，另一种是使用不同的生成器。用 17.4 节的方差公式解释其差别。
7. 设计评审：一位同事报告，一种新的优势估计量把 SWE-bench Verified 的 pass@1 从 41.0% 提升到 43.4%，每组只有一次训练、每个任务只采样一次，而且检查点是“最后五个中最好的那个”。运用本章内容，列出你在接受这一论断之前会要求补充的内容，分别在非配对和配对分析下估计所报告差异的标准误，并说明你会要求多少个随机种子和多少次采样。
