# 8. PPO：安全地复用样本

第 6、7 章每次采样 64 个回合，只更新一步就丢弃这批数据。对 TinyAgent 来说，采样只需约五分之一秒；对真实编程智能体来说，一个回合可能占用几分钟的容器时间，并生成数万个 token，反而是梯度更新耗时较少。我们能否在同一批轨迹上多更新几次，让昂贵的 rollout 得到更充分的利用？

障碍就在推导本身。式 (6.1) 是*当前*策略下的期望。第一次更新之后，当前策略就不同于生成这批数据的策略。若仍按 on-policy 公式使用这些样本，梯度估计便不再对应当前策略的目标。本章构建相应的修正。重要性采样（importance sampling）能精确地修正期望，却在数值上失效，我们会测量这种失效。逐 token 比率是一个易于处理的近似，我们会说明它的误差。**近端策略优化**（Proximal Policy Optimization，PPO；Schulman et al., 2017）对这个比率做裁剪（clipping），限制代理目标对继续增大更新幅度的激励。随后我们加入裁剪的价值损失、小批次和轮次，用 PPO 和 critic（价值模型）训练 TinyAgent，并解读那些告诉工程师复用是否安全的诊断指标。本章保存两个函数：裁剪的策略损失和裁剪的价值损失。[第 9 章](09-grpo-and-variants.md)在它们之上构建本书的主训练循环。

```python
#@save
import torch
import agentrl as arl
```

设置与第 6 章和第 7 章相同：词级分词器、两个任务划分、基座智能体，以及一个对每个任务同步采样固定数量回合的辅助函数。

```python
import copy
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
base = arl.make_base_agent(tok)

def sample(model, tasks, n_per_task, seed):
    """n_per_task episodes of each task, sampled in lockstep at temperature 1."""
    batch_tasks = [t for t in tasks for _ in range(n_per_task)]
    return arl.rollout_batch(model, [arl.BugFixEnv() for _ in batch_tasks], batch_tasks, tok,
                             generator=torch.Generator().manual_seed(seed))
```

## 8.1 off-policy 复用与重要性采样

设 $\pi_{\text{old}}$ 为采样一个批次的策略，$\pi_\theta$ 为经过若干次更新之后的策略。目标函数是 $\pi_\theta$ 下的期望，而数据来自 $\pi_{\text{old}}$。**重要性采样**把一个期望改写成另一个（[附录 A.1](A-math.md)）：

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\big[R(\tau)\big] = \mathbb{E}_{\tau \sim p_{\text{old}}}\Big[\frac{p_\theta(\tau)}{p_{\text{old}}(\tau)}\, R(\tau)\Big].
$$

轨迹权重 $w(\tau) = p_\theta(\tau)/p_{\text{old}}(\tau)$ 看起来需要环境的转移概率，其实不需要。两个轨迹概率都可以按式 (6.2) 分解为策略 token 部分和其余部分，而“其余部分”（提示、模板、工具输出）在两个策略下的概率相同，因为环境没有变。它被约掉了：

$$
w(\tau) = \prod_{i:\, m_i = 1} \frac{\pi_\theta(y_i \mid c_i)}{\pi_{\text{old}}(y_i \mid c_i)} = \prod_{i:\, m_i = 1} \rho_i, \qquad \log w(\tau) = \sum_{i:\, m_i = 1} \big(\log \pi_\theta(y_i \mid c_i) - \log \pi_{\text{old}}(y_i \mid c_i)\big).
$$

这是式 (8.1)，即精确的轨迹比率，$\rho_i$ 是**逐 token 比率**（per-token ratio）。对 $\mathbb{E}_{\text{old}}[w R]$ 求导得到 $\mathbb{E}_{\text{old}}[w R\, \nabla \log p_\theta(\tau)]$，在 $\theta = \theta_{\text{old}}$ 处它又回到 REINFORCE。所以精确的修正是存在的，而且无偏。问题在于它的方差。

我们来测量它。基座智能体采样 320 个回合，作为“旧”批次。然后基座智能体的一个副本在*新*批次上走 $k$ 步 REINFORCE，以训练会造成的方式移动策略。对每个 $k$，我们为每个旧回合计算 $\log w$、**有效样本量**（effective sample size）$\mathrm{ESS} = (\sum_n w_n)^2 / \sum_n w_n^2$（加权批次相当于多少个等权样本）、新策略通过率的 IS 估计，以及在新样本上测得的通过率。

```python
old_batch = arl.collate(sample(base, train_tasks, 16, seed=0), tok.pad_id)   # 320 episodes from pi_old
mask = old_batch['action_mask']                                              # [B, T-1]
with torch.no_grad():
    old_logp = arl.sequence_log_probs(base, old_batch['ids'])                # [B, T-1]

def reinforce_steps(model, optimizer, k, gen):
    """k REINFORCE updates of model in place, each on 64 fresh episodes of random train tasks."""
    for _ in range(k):
        tasks = [train_tasks[i] for i in torch.randint(len(train_tasks), (64,), generator=gen).tolist()]
        trajs = arl.rollout_batch(model, [arl.BugFixEnv() for _ in tasks], tasks, tok, generator=gen)
        batch = arl.collate(trajs, tok.pad_id)
        arl.policy_gradient_step(model, optimizer, batch, batch['rewards'] - batch['rewards'].mean())
```

```python
moved, gen = copy.deepcopy(base), torch.Generator().manual_seed(1)
opt = torch.optim.Adam(moved.parameters(), lr=2e-3)
log_weights, done = {}, 0
print(' k | sd log w | max w  | ESS / N | IS estimate | fresh samples')
for k in [0, 1, 3, 10, 30]:
    reinforce_steps(moved, opt, k - done, gen)
    done = k
    with torch.no_grad():
        log_w = ((arl.sequence_log_probs(moved, old_batch['ids']) - old_logp) * mask).sum(-1)   # [B], Eq. (8.1)
    w, log_weights[k] = log_w.exp(), log_w
    ess = (w.sum() ** 2 / (w ** 2).sum()).item() / len(w)
    fresh = arl.rollout_stats(sample(moved, train_tasks, 16, seed=9))['pass_rate']
    print(f'{k:2d} | {log_w.std():8.2f} | {w.max():6.1f} | {ess:7.3f} | {(w * old_batch["rewards"]).mean():11.3f} |'
          f' {fresh:.3f}')
```

Output:

```text
 k | sd log w | max w  | ESS / N | IS estimate | fresh samples
 0 |     0.00 |    1.0 |   1.000 |       0.119 | 0.159
 1 |     0.80 |    3.5 |   0.776 |       0.144 | 0.197
 3 |     2.12 |   20.9 |   0.343 |       0.178 | 0.263
10 |     3.46 |    5.7 |   0.302 |       0.256 | 0.469
30 |    17.05 |   18.8 |   0.101 |       0.431 | 0.700
```

在 $k = 0$ 时，权重恰好为 1，IS 估计就是旧批次的通过率 0.119（同一策略的新样本给出 0.159；两者都是 320 个回合的估计）。一次更新后，对数权重散开，标准差为 0.8，批次相当于其规模的 78%。三次更新后只相当于三分之一，30 次后相当于十分之一。IS 估计在期望上是无偏的，但随着策略移动，它越来越低于真实值：10 次更新后是 0.256 对 0.469，30 次后是 0.431 对 0.700。无偏性是关于无穷多个批次的平均的陈述。在单个批次中，新策略最偏好的回合恰恰是旧策略很少采样到的那些，因此它们缺失了，估计值大概率偏低。$\log w$ 的标准差从 0.8 增长到 17，因为它是回合中所有策略 token 的求和。下一个代码块直接检验这一点。

```python
lengths = mask.sum(-1)                                                       # policy tokens per episode
for lo, hi in [(0, 8), (8, 16), (16, 64)]:
    rows = (lengths >= lo) & (lengths < hi)
    print(f'{lo:2d}-{hi:<2d} policy tokens: {int(rows.sum()):3d} episodes | sd log w after 3 updates'
          f' {log_weights[3][rows].std():.2f} | after 10: {log_weights[10][rows].std():.2f}')
```

Output:

```text
 0-8  policy tokens: 198 episodes | sd log w after 3 updates 0.50 | after 10: 1.42
 8-16 policy tokens:  35 episodes | sd log w after 3 updates 1.53 | after 10: 2.50
16-64 policy tokens:  87 episodes | sd log w after 3 updates 2.29 | after 10: 3.46
```

对数权重的离散程度随策略 token 数增长：十次更新后，策略 token 少于 8 个的回合标准差为 1.42，16 个及以上的回合为 3.46。如果每个 token 贡献一个大致独立、标准差为 $\sigma$ 的对数比率，那么 $L$ 个 token 之和的标准差为 $\sigma\sqrt{L}$，权重 $e^{\log w}$ 服从对数正态分布，方差像 $e^{\sigma^2 L}$ 那样增长。一个 BugFix 回合约有 12 个策略 token，一个 SWE 智能体回合有数千个。在这样的长度上，精确轨迹权重通常会有很大的方差，有限批次难以稳定估计。上面的独立近似说明了一种常见机制：比率乘积的方差可能随时间跨度指数增长。

![图 8.1：左：320 个基座智能体回合重新加权到已走 k 步 REINFORCE 更新的策略后的有效样本量。右：10 次更新后每个回合的对数权重与其策略 token 数的关系。](../../figures/ch08-is-weights.svg)

在图 8.1 中，请看右图的离散程度，而不是它的中心。短回合的对数权重保持在 0 附近，长回合则向下散布到 $-20$，即权重 $e^{-20}$：最长的那些回合是移动后的策略几乎不会再生成的，而它现在偏好的少数长回合则根本不在批次中。

### 逐 token 近似

PPO 不使用 $w(\tau)$。它使用每个 token 自己的比率 $\rho_i$，这只涉及一个 token 的概率，在策略变化不大时保持在 1 附近。其依据是**性能差分恒等式**（performance-difference identity；Kakade and Langford, 2002），这里写成 token 级形式：

$$
J(\theta) - J(\theta_{\text{old}}) = \mathbb{E}_{\tau \sim p_\theta}\Big[\sum_{i:\, m_i = 1} A^{\text{old}}(c_i, y_i)\Big] = \sum_i \mathbb{E}_{c_i \sim d_\theta}\, \mathbb{E}_{y_i \sim \pi_{\text{old}}(\cdot \mid c_i)}\big[\rho_i\, A^{\text{old}}(c_i, y_i)\big].
$$

内层期望在每个前缀上对*动作*重新加权，单个比率 $\rho_i$ 恰好能做到这一点。外层期望针对的是*新*策略访问的前缀分布 $d_\theta$，正是这部分需要之前所有比率的乘积。**代理目标**（surrogate）用 $d_{\text{old}}$（批次中的前缀）代替 $d_\theta$：

$$
L_{\text{old}}(\theta) = \mathbb{E}_{\tau \sim p_{\text{old}}}\Big[\sum_{i:\, m_i = 1} \rho_i\, \hat A_i\Big].
$$

这是式 (8.2)。它是一个近似，本书在它出现的每一处都如此说明：逐 token 比率修正了每个动作概率的变化，却忽略了所访问前缀的变化。$L_{\text{old}}$ 在 $\theta - \theta_{\text{old}}$ 的一阶近似下与 $J(\theta) - J(\theta_{\text{old}})$ 一致（相差一个常数），因此在 $\theta_{\text{old}}$ 处梯度相同，其误差随两个策略之间的距离增长。Schulman et al. (2015) 用策略之间最大 KL 散度的若干倍来界定这个误差，8.3 节的信赖域（trust region）就来源于此。我们在同一个旧批次上检验这个近似：比较代理目标对改进量的估计 $\frac{1}{N}\sum_n \sum_i m_i(\rho_{n,i} - 1)(R_n - \bar R)$ 与在新样本上测得的改进量。

```python
moved, gen = copy.deepcopy(base), torch.Generator().manual_seed(1)
opt = torch.optim.Adam(moved.parameters(), lr=2e-3)
advantage = (old_batch['rewards'] - old_batch['rewards'].mean())[:, None]     # [B, 1], outcome advantage
done, j_old = 0, arl.rollout_stats(sample(base, train_tasks, 16, seed=9))['pass_rate']
for k in [1, 3, 10]:
    reinforce_steps(moved, opt, k - done, gen)
    done = k
    with torch.no_grad():
        rho = (arl.sequence_log_probs(moved, old_batch['ids']) - old_logp).exp()   # [B, T-1]
    surrogate = (((rho - 1) * advantage * mask).sum(-1)).mean().item()
    fresh = arl.rollout_stats(sample(moved, train_tasks, 16, seed=9))['pass_rate'] - j_old
    print(f'k = {k:2d}: surrogate improvement {surrogate:+.3f} | measured improvement {fresh:+.3f}')
```

Output:

```text
k =  1: surrogate improvement +0.037 | measured improvement +0.038
k =  3: surrogate improvement +0.108 | measured improvement +0.103
k = 10: surrogate improvement +0.258 | measured improvement +0.309
```

对于一次和三次更新，代理目标对改进量的预测误差在半个百分点以内。十次更新后，它少估了六分之一，0.258 对 0.309。缺失的部分是前缀的偏移：移动后的策略会到达一些状态（例如正确的编辑后接提交），旧批次访问这些状态的次数太少，任何对单个 token 的重新加权都无法弥补。代理目标在 $\theta_{\text{old}}$ 附近是目标函数的好模型，远离它时则不是。这就引出了 PPO 所实现的规则：优化代理目标，但只在策略保持接近时这样做。

## 8.2 裁剪的代理目标

用许多步梯度更新去最大化式 (8.2)，会把比率推到尽可能远的地方，进入代理目标出错的区域。PPO 的裁剪目标消除了去那里的动力：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_{\text{old}}\Big[\sum_{i:\, m_i = 1} \min\big(\rho_i \hat A_i,\ \mathrm{clip}(\rho_i,\, 1 - \epsilon_{\text{low}},\, 1 + \epsilon_{\text{high}})\, \hat A_i\big)\Big].
$$

这是式 (8.3)。取最小值使它成为未裁剪代理目标的一个悲观下界，其行为取决于优势的符号：

- **$\hat A > 0$**（该 token 应变得更可能）。在 $\rho$ 达到 $1 + \epsilon_{\text{high}}$ 之前，该项是 $\rho \hat A$，之后保持常数。一旦好 token 的概率增长到 $1 + \epsilon_{\text{high}}$ 倍，它的梯度就为零。在 1 以下，最小值保留未裁剪项，因此一个概率下降了的好 token 总会被推回上去。
- **$\hat A < 0$**（该 token 应变得更不可能）。在 $\rho$ 降到 $1 - \epsilon_{\text{low}}$ 之前，该项是 $\rho \hat A$，之后保持常数。在 1 以上，保留更负的未裁剪项：一个概率*上升*了的坏 token 总会被推回下来，梯度随 $\rho$ 增大。

因此，裁剪消除了把一个 token 朝其优势所要求的方向推过 $1 \pm \epsilon$ 倍的动力。它从不消除用于撤销错误方向移动的梯度。下面的函数计算逐 token 目标，用第 6 章的 `aggregate_loss` 聚合，并返回 8.5 节用作诊断的统计量。被裁剪的 token 是裁剪分支严格更小的 token，这恰好是梯度为零的地方。

```python
#@save
def ppo_policy_loss(logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.2, mode='token_mean',
                    dual_clip=None, norm_const=None):
    """Clipped surrogate on policy tokens: logp, old_logp, mask [B, T]; advantages [B] or [B, T] -> (loss, stats)."""
    if advantages.dim() == 1:
        advantages = advantages[:, None]                          # one advantage per trajectory -> [B, 1]
    advantages = advantages.detach()
    log_ratio = logp - old_logp.detach()
    ratio = log_ratio.exp()                                        # rho = pi_theta / pi_old, [B, T]
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high) * advantages
    objective = torch.minimum(unclipped, clipped)                  # Eq. (8.3), per token
    if dual_clip is not None:                                      # A < 0: bound the objective below by c * A
        objective = torch.where(advantages < 0, torch.maximum(objective, dual_clip * advantages), objective)
    loss = arl.aggregate_loss(-objective, mask, mode, norm_const)
    with torch.no_grad():
        on = mask > 0
        r = ratio[on] if on.any() else torch.ones(1)
        stats = dict(clip_frac=((clipped < unclipped) & on).sum().item() / max(int(on.sum()), 1),
                     approx_kl=arl.masked_mean(ratio - 1 - log_ratio, mask).item(),   # k3 estimate of KL(old || new)
                     ratio_min=r.min().item(), ratio_max=r.max().item())
    return loss, stats
```

`approx_kl` 统计量是用 $\pi_{\text{old}}$ 的样本估计 $\mathrm{KL}(\pi_{\text{old}} \,\|\, \pi_\theta)$ 的 $k_3$ 估计量：$\mathbb{E}_{\text{old}}[\rho - 1 - \log \rho]$，它逐 token 非负（[附录 A](A-math.md)；[第 10 章](10-kl-entropy-regularization.md)将它与其他估计量比较）。损失对每个 token 对数概率的梯度展示了按符号区分的行为。取比率从 0.5 到 2 的六个 token，分别在 $\hat A = +1$ 和 $\hat A = -1$ 下计算；在单行上使用 `seq_mean_token_sum`，使每个 token 的梯度等于它的原始权重。

```python
ratios = torch.tensor([[0.5, 0.79, 1.0, 1.1, 1.3, 2.0]])                     # [1, 6]
old = torch.zeros(1, 6)
for sign in [+1.0, -1.0]:
    logp = ratios.log().requires_grad_()
    loss, stats = ppo_policy_loss(logp, old, torch.tensor([sign]), torch.ones(1, 6), mode='seq_mean_token_sum')
    grad = torch.autograd.grad(loss, logp)[0][0]                             # d loss / d log pi per token
    print(f'A = {sign:+.0f}: -d loss/d logp =', [f'{-g + 0.0:+.2f}' for g in grad.tolist()],
          f'| clip_frac {stats["clip_frac"]:.2f}')
```

Output:

```text
A = +1: -d loss/d logp = ['+0.50', '+0.79', '+1.00', '+1.10', '+0.00', '+0.00'] | clip_frac 0.33
A = -1: -d loss/d logp = ['+0.00', '+0.00', '-1.00', '-1.10', '-1.30', '-2.00'] | clip_frac 0.33
```

在未裁剪分支起作用的地方，每个对数概率上的上升方向是 $\rho \hat A$，因为 $\partial \rho / \partial \log \pi = \rho$。当 $\hat A = +1$ 时，比率为 1.3 和 2.0 的 token 已超过 $1 + \epsilon$，什么也得不到，而比率为 0.5 和 0.79 的 token 仍被向上推。当 $\hat A = -1$ 时，比率为 0.5 和 0.79 的 token 已低于 $1 - \epsilon = 0.8$，不再受影响；而比率为 2.0 的 token（一个变得两倍可能的坏动作）被推得最狠。比率为 0.79 的 token 用一个数字展示了这种不对称：同一个比率，在优势为负时被裁剪，在优势为正时仍起作用。两行中都是六个 token 里有两个被裁剪，裁剪比例为 0.33。

![图 8.2：A = +1（左）和 A = -1（右）时逐 token 裁剪目标随比率的变化，虚线为未裁剪的代理目标。平坦段是梯度为零的地方；c = 3 的双重裁剪为 A < 0 增加了一个下限。](../../figures/ch08-clip-landscape.svg)

图 8.2 画出了两种符号下的式 (8.3)。每幅图都是未裁剪的直线 $\rho \hat A$ 加上一段平坦部分：优势为正时在右侧，为负时在左侧。平坦部分是优化器不再关心的地方，而 1 另一侧的斜线部分是它始终关心的地方。

### 双重裁剪

$\hat A < 0$ 时无界的那一段是有意为之的，但它有一种失效模式。如果一个坏 token 的比率非常大，例如因为批次陈旧或采样器的概率有误（[第 13 章](13-rollout-systems.md)），$\rho \hat A$ 这一项就可能主导整个批次的梯度。**双重裁剪**（dual clipping；Ye et al., 2020）用 $c \hat A$ 为目标设置下界，常数 $c > 1$，通常取 3：

```python
logp = torch.tensor([[2.0, 5.0, 20.0]]).log().requires_grad_()               # three bad tokens, ratios 2, 5, 20
for c in [None, 3.0]:
    loss, _ = ppo_policy_loss(logp, torch.zeros(1, 3), torch.tensor([-1.0]), torch.ones(1, 3),
                              mode='seq_mean_token_sum', dual_clip=c)
    grad = torch.autograd.grad(loss, logp)[0][0]
    print(f'dual_clip={c}: loss {loss.item():5.1f} | -d loss/d logp {[round(-g, 1) for g in grad.tolist()]}')
```

Output:

```text
dual_clip=None: loss  27.0 | -d loss/d logp [-2.0, -5.0, -20.0]
dual_clip=3.0: loss   8.0 | -d loss/d logp [-2.0, -0.0, -0.0]
```

没有双重裁剪时，比率为 20 的 token 贡献的梯度是比率为 2 的 token 的十倍，损失为 $2 + 5 + 20 = 27$。当 $c = 3$ 时，比率超过 3 的两个 token 各贡献常数 3、没有梯度，所以损失为 8，于是单个严重 off-policy 的 token 无法接管整次更新。代价是这样的 token 在这次更新中完全得不到修正；当大比率更多是陈旧或不匹配数据的症状、而非策略真实变化时，这是可以接受的。

## 8.3 信赖域视角

8.1 节表明代理目标在 $\theta_{\text{old}}$ 附近是准确的。**信赖域策略优化**（Trust Region Policy Optimization，TRPO；Schulman et al., 2015）把这一点变成了算法。它的分析给出如下形式的下界：

$$
J(\theta) \ge J(\theta_{\text{old}}) + L_{\text{old}}(\theta) - C \max_c \mathrm{KL}\big(\pi_{\text{old}}(\cdot \mid c) \,\|\, \pi_\theta(\cdot \mid c)\big),
$$

其中这里的 $L_{\text{old}}$ 是代理目标给出的改进量，$C$ 取决于折扣和最大的优势。最大化右边可以保证改进。实践中，TRPO 在*平均* KL 约束 $\bar{\mathrm{KL}} \le \delta$ 下最大化 $L_{\text{old}}(\theta)$，并使用自然梯度：它借助 Fisher 向量积用共轭梯度法求解 $F x = g$（$F$ 为 Fisher 信息矩阵），把步长缩放到约束边界，再用线搜索回退，直到代理目标得到改进且 KL 位于区域之内。对语言模型来说，每个 Fisher 向量积都是对整个网络的一次二阶计算，线搜索还需要额外的前向计算。这就是 TRPO 本身很少用于 LLM，而它的思想却无处不在的原因。

PPO 用裁剪代替约束，除了一阶优化器之外没有额外开销。两者并不等价。裁剪消除了把 token 比率推过 $1 \pm \epsilon$ 的动力，但参数在 token 之间共享，所以一步有利于其他 token 的更新可能把一个已被裁剪的 token 推得更远，而没有任何东西把它拉回来。下面的代码块取旧批次的一个小批次，在裁剪损失上运行 40 步 Adam，并作为对照在未裁剪的代理目标（裁剪设为无穷大）上同样运行，然后报告 KL 以及比率落在 $[0.8, 1.2]$ 之外的比例。

```python
rows = torch.arange(64)
adv = (old_batch['rewards'][rows] - old_batch['rewards'][rows].mean())       # [64] outcome advantages
for name, eps in [('clipped, eps = 0.2', 0.2), ('unclipped', float('inf'))]:
    policy = copy.deepcopy(base)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for step in range(40):
        logp = arl.sequence_log_probs(policy, old_batch['ids'][rows])
        loss, stats = ppo_policy_loss(logp, old_logp[rows], adv, mask[rows], eps, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ratio = (logp.detach() - old_logp[rows]).exp()[mask[rows] > 0]
    outside = ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()
    print(f'{name:<18}: approx KL {stats["approx_kl"]:.3f} | ratios outside [0.8, 1.2] {outside:.2f}'
          f' | max ratio {ratio.max():.1f}')
```

Output:

```text
clipped, eps = 0.2: approx KL 0.237 | ratios outside [0.8, 1.2] 0.41 | max ratio 12.2
unclipped         : approx KL 5.885 | ratios outside [0.8, 1.2] 0.47 | max ratio 2896.7
```

这 64 个回合是旧批次中属于前四个训练任务的部分。40 步之后，裁剪把 KL 降低了 25 倍（从 5.9 到 0.24），把最大比率从约 2,900 降到 12。但它没有把比率留在区间内：41% 的比率最终落在区间外，而不裁剪时是 47%。裁剪是一个软信赖域。它通过把动力归零来减缓漂移，剩下的漂移取决于学习率、步数以及 token 之间共享参数的程度。这就是为什么 PPO 实现要限制在一个批次上的遍历次数并直接监控 KL（8.5 节），也是为什么有些实现会加入显式的 KL 惩罚，或在 KL 超过目标时提前停止。

## 8.4 价值损失与 PPO 循环

原始论文中的 PPO 与策略一起训练一个 critic，并使用 GAE 优势（[7.3 节](07-advantages-and-credit.md#73-广义优势估计)）。critic 在 `gae_advantages` 返回的价值目标 $\hat V^{\text{targ}}_i = \hat A_i + V_{\text{old}}(c_i)$ 上做回归。由于 critic 也在同一个批次上更新多次，同样的信赖域问题也会出现，PPO2 风格的实现以同样的方式对其裁剪：先把新旧价值之差裁剪到 $\epsilon_v$ 范围内，再取裁剪前后较大的回归损失；这限制了改善损失的激励，并非把价值预测硬性限制在区间内，

$$
\mathcal{L}^V = \tfrac{1}{2}\, \mathbb{E}\Big[\max\big((V_\phi - \hat V^{\text{targ}})^2,\ (V_{\text{old}} + \mathrm{clip}(V_\phi - V_{\text{old}}, -\epsilon_v, \epsilon_v) - \hat V^{\text{targ}})^2\big)\Big].
$$

取最大值使它成为悲观的，就像策略的取最小值一样。价值裁剪是否有帮助存在争议。Andrychowicz et al. (2021) 在一项针对连续控制任务上 on-policy 设计选择的大规模研究中发现，“PPO-style value loss clipping hurts the performance regardless of the clipping threshold”（无论裁剪阈值如何，PPO 风格的价值损失裁剪都会损害性能）；这一证据并非来自语言模型。我们实现它是因为它是许多 LLM PPO 框架的默认设置，用 `clip=None` 可以关闭它。

```python
#@save
def value_loss(values, old_values, returns, mask, clip=0.2):
    """Clipped value regression on policy tokens: values, old_values, returns, mask [B, T] -> scalar."""
    old_values, returns = old_values.detach(), returns.detach()
    per_token = (values - returns) ** 2
    if clip is not None:                                           # the move from old_values is limited to +-clip
        moved = old_values + (values - old_values).clamp(-clip, clip)
        per_token = torch.maximum(per_token, (moved - returns) ** 2)
    return 0.5 * arl.masked_mean(per_token, mask)
```

### critic 需要先行一步

[7.4 节](07-advantages-and-credit.md#74-训练-critic)测量了 TinyAgent critic 有多弱：在新回合上解释方差只有 0.02 到 0.08。一个新输出头从随机权重开始的 critic 更糟，并且在 $\lambda < 1$ 时，它的误差会直接进入策略梯度（图 7.1）。有两种标准的缓解办法，我们都会使用：在第一次策略更新之前，在初始策略的 rollout 上**预热**critic；并从 $\lambda = 1$ 开始，此时 critic 只是一个基线，不会使梯度有偏。`warm_critic` 就是第 7 章中在蒙特卡洛回报上的回归。

```python
def mc_returns(batch):
    """Monte Carlo value targets: the outcome on every policy token [B, T-1]."""
    return arl.returns_to_go(arl.outcome_rewards(batch['rewards'], batch['action_mask']), batch['action_mask'])

def warm_critic(policy, steps=100, seed=0):
    """A critic copied from the policy and regressed on 480 of its episodes before PPO starts."""
    critic, data = arl.make_critic(policy), arl.collate(sample(policy, train_tasks, 24, seed=1), tok.pad_id)
    opt, gen, targets = torch.optim.Adam(critic.parameters(), lr=1e-3), torch.Generator().manual_seed(seed), mc_returns(data)
    for _ in range(steps):
        idx = torch.randint(len(targets), (32,), generator=gen)
        values = arl.critic_values(critic, data['ids'][idx])
        loss = arl.masked_mean((values - targets[idx]) ** 2, data['action_mask'][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return critic
```

### 一次 PPO 迭代

一次迭代分两个阶段。第一阶段冻结定义 $\pi_{\text{old}}$ 的一切：采样 token 的对数概率、critic 的价值，以及由它们计算出的 GAE 优势和目标。然后像大多数 PPO 实现那样，在批次的所有策略 token 上对优势做**白化**（whitening，均值 0，标准差 1）。第二阶段把批次打乱并拆成小批次，重复训练 `epochs` 遍，每个小批次做一步策略更新和一步 critic 更新，并记录每一轮的统计量。

```python
def ppo_prepare(policy, critic, batch, lam):
    """Freeze pi_old: old log-probs, old values, whitened GAE advantages, value targets; all [B, T-1]."""
    m = batch['action_mask']
    with torch.no_grad():
        old_logp = arl.sequence_log_probs(policy, batch['ids'])
        old_values = arl.critic_values(critic, batch['ids'])
    adv, returns = arl.gae_advantages(arl.outcome_rewards(batch['rewards'], m), old_values, m, 1.0, lam)
    on = m > 0
    adv = (adv - adv[on].mean()) / (adv[on].std() + 1e-8) * m
    return old_logp, old_values, adv, returns
```

```python
def ppo_epochs(policy, critic, popt, vopt, batch, prepared, epochs=4, minibatches=4, gen=None, target_kl=None):
    """Clipped policy and value updates over shuffled minibatches -> list of per-epoch mean stats."""
    old_logp, old_values, adv, returns = prepared
    ids, m, per_epoch = batch['ids'], batch['action_mask'], []
    for _ in range(epochs):
        logs = []
        for rows in torch.randperm(len(ids), generator=gen).chunk(minibatches):
            loss, stats = ppo_policy_loss(arl.sequence_log_probs(policy, ids[rows]), old_logp[rows], adv[rows], m[rows])
            popt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            popt.step()
            vloss = value_loss(arl.critic_values(critic, ids[rows]), old_values[rows], returns[rows], m[rows])
            vopt.zero_grad()
            vloss.backward()
            vopt.step()
            logs.append(stats)
        per_epoch.append({k: sum(s[k] for s in logs) / len(logs) for k in logs[0]})
        if target_kl is not None and per_epoch[-1]['approx_kl'] > target_kl:
            break                                                 # early stopping (Section 8.5)
    return per_epoch
```

循环每次迭代像第 6 章那样对随机训练任务采样 64 个回合，记录批次奖励，以及 critic 相对于蒙特卡洛回报的解释方差（在 critic 见到该批次之前测量）。我们运行两次，分别使用 $\lambda = 1$ 和 $\lambda = 0.95$，每次 30 次迭代，每次迭代 4 个轮次（epoch）、4 个小批次：每个批次 16 步策略更新。

```python
def run_ppo(lam, iters=30, epochs=4, minibatches=4, lr=1e-3, target_kl=None, seed=0):
    """PPO with a warmed-up critic on the train tasks -> (policy, history of per-iteration dicts)."""
    torch.manual_seed(seed)
    policy, critic, gen = copy.deepcopy(base), warm_critic(base), torch.Generator().manual_seed(seed)
    popt, vopt = torch.optim.Adam(policy.parameters(), lr=lr), torch.optim.Adam(critic.parameters(), lr=1e-3)
    history = []
    for it in range(iters):
        tasks = [train_tasks[i] for i in torch.randint(len(train_tasks), (64,), generator=gen).tolist()]
        batch = arl.collate(arl.rollout_batch(policy, [arl.BugFixEnv() for _ in tasks], tasks, tok, generator=gen),
                            tok.pad_id)
        prepared = ppo_prepare(policy, critic, batch, lam)
        varied = batch['rewards'].std() > 0
        ev = arl.explained_variance(prepared[1], mc_returns(batch), batch['action_mask']) if varied else float('nan')
        per_epoch = ppo_epochs(policy, critic, popt, vopt, batch, prepared, epochs, minibatches, gen, target_kl)
        history.append(dict(reward=batch['rewards'].mean().item(), ev=ev, per_epoch=per_epoch))
    return policy, history
```

```python
runs = {lam: run_ppo(lam) for lam in [1.0, 0.95]}
for lam, (policy, history) in runs.items():
    evs = [h['ev'] for h in history if h['ev'] == h['ev']]                     # skip NaN (all-equal batches)
    curve = [sum(h['reward'] for h in history[i:i + 10]) / 10 for i in range(0, 30, 10)]
    train = arl.rollout_stats(sample(policy, train_tasks, 16, seed=7))['pass_rate']
    held = arl.rollout_stats(sample(policy, heldout_tasks, 16, seed=7))['pass_rate']
    print(f'lambda {lam}: batch reward by 10 iterations {[round(c, 2) for c in curve]} | critic EV mean'
          f' {sum(evs) / len(evs):+.2f} | pass train {train:.3f} held-out {held:.3f}')
```

Output:

```text
lambda 1.0: batch reward by 10 iterations [0.49, 0.88, 0.95] | critic EV mean +0.12 | pass train 0.900 held-out 0.312
lambda 0.95: batch reward by 10 iterations [0.42, 0.85, 0.91] | critic EV mean +0.26 | pass train 0.925 held-out 0.331
```

两次运行都在 30 次迭代内达到约 0.9 的训练通过率，按每个采样回合计算比第 6 章的 REINFORCE 更快，后者用了 80 个批次才达到 0.78：现在同一批数据用于 16 次更新，而不是 1 次。以每个新批次的蒙特卡洛回报衡量（在 critic 用该批次训练之前），critic 在 $\lambda = 1$ 时平均解释 12% 的方差，在 $\lambda = 0.95$ 时为 26%。这比第 7 章在新回合上的 2% 到 8% 要好，因为预热过的 critic 一直在它所评估的策略上持续训练，但仍远不足以在回合内部分配学习信号。留出任务上的通过率为 0.31 和 0.33，低于 REINFORCE 的 0.46。把每个批次复用 16 次会更充分地拟合训练任务，包括第 6 章发现策略在记忆的那些函数名。PPO 让优化更节省样本，却不能保证策略学到正确的任务规律。[第 9 章](09-grpo-and-variants.md)在环境中诊断这个问题，并在那里修复它。

> **证据。** 本书测得，每种设置一个随机种子，30 次迭代，每次 64 个回合。$\lambda = 1$ 与 $\lambda = 0.95$ 之间的比较处于单次运行的噪声范围内，不应被解读为对某一方的偏好。这次运行确实支持的是机制层面的结论：critic 这么弱时，GAE 带来的增益很小，而 critic 要付出第二个网络和第二个优化器的代价。

## 8.5 诊断与提前停止

有三个数字可以告诉工程师复用是否安全，每个 PPO 系列的训练器都应按轮次记录它们。

- **裁剪比例**（clip fraction）：裁剪分支起作用的策略 token 所占的比例。接近 0 时，裁剪处于闲置状态，更新就是普通的策略梯度。在各轮次间上升时，说明策略正在到达信赖域的边缘。
- $\pi_{\text{old}}$ 与 $\pi_\theta$ 之间的**近似 KL**，即在批次上的 $k_3$ 估计。它用信赖域界所使用的量来衡量策略移动了多远。
- **比率统计量**：策略 token 上比率的最小值和最大值，能捕捉被平均值掩盖的少数 token。

$\lambda = 1$ 那次运行的逐轮次统计量，在其 30 次迭代上取平均：

```python
history = runs[1.0][1]
for e in range(4):
    stats = [h['per_epoch'][e] for h in history]
    mean = {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}
    print(f'epoch {e + 1}: clip_frac {mean["clip_frac"]:.3f} | approx_kl {mean["approx_kl"]:.4f}'
          f' | ratio range {mean["ratio_min"]:.2f} to {mean["ratio_max"]:.2f}')
```

Output:

```text
epoch 1: clip_frac 0.013 | approx_kl 0.0049 | ratio range 0.72 to 1.49
epoch 2: clip_frac 0.060 | approx_kl 0.0371 | ratio range 0.26 to 2.88
epoch 3: clip_frac 0.075 | approx_kl 0.0685 | ratio range 0.16 to 4.70
epoch 4: clip_frac 0.089 | approx_kl 0.1047 | ratio range 0.12 to 7.02
```

第一轮几乎是 on-policy 的：它的第一个小批次的比率恰好为 1，后面的小批次最多移动了三步梯度，因此 1.3% 的 token 被裁剪，KL 为 0.005。之后每一轮都移动得更远。到第四轮，9% 的 token 被裁剪，KL 是第一轮的 21 倍，平均最小和最大比率 0.12 和 7.0 远在裁剪区间之外，正如 8.3 节所预测的。

![图 8.3：每个 PPO 轮次的裁剪比例（左）和近似 KL（右）。分别为 λ = 1 运行在 30 次迭代上的平均（4 个轮次），以及一个更新了 16 个轮次的批次，并标出了 0.02 的提前停止目标。](../../figures/ch08-epochs.svg)

图 8.3 画出了训练运行以及一个更新了 16 个轮次的大批次（8.6 节）的相同统计量。两者的裁剪比例都随每一轮上升。训练运行的 KL 在第二轮期间越过了虚线所示的提前停止目标，下面的提前停止正是在这里截断了大多数迭代。

**提前停止**（early stopping）以 KL 作为停止规则：每轮结束后，如果近似 KL 超过目标（通常为 0.01 到 0.05）就停止。它把轮次数从常数变成了预算。`ppo_epochs` 已经接受 `target_kl`；我们以 0.02 为目标重新运行 $\lambda = 1$ 的设置，并统计每次迭代用了多少轮。

```python
policy, history = run_ppo(1.0, target_kl=0.02)
used = [len(h['per_epoch']) for h in history]
train = arl.rollout_stats(sample(policy, train_tasks, 16, seed=7))['pass_rate']
held = arl.rollout_stats(sample(policy, heldout_tasks, 16, seed=7))['pass_rate']
print(f'epochs used per iteration: {used}')
print(f'mean {sum(used) / len(used):.2f} of 4 | pass train {train:.3f} held-out {held:.3f}')
```

Output:

```text
epochs used per iteration: [2, 3, 3, 4, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 4, 3, 2, 2, 4, 2, 2, 2, 2, 2, 3, 3, 2, 4]
mean 2.53 of 4 | pass train 0.912 held-out 0.206
```

大多数迭代在第二或第三轮之后越过目标，因此这次运行使用了固定轮次运行约 63% 的梯度步，并达到了相同的训练通过率 0.91。它在留出任务上的通过率为 0.21，低于固定轮次的运行。这只是一个随机种子的一次 160 回合测量，比它们低约三个标准误，并且主导这一数字的是本章未处理的记忆问题；它不能证明提前停止的泛化更差。

## 8.6 实践中的设计问题

> **问题：轮次太多。**
>
> **现象：** 裁剪比例和 KL 在各轮次间攀升，策略的熵坍缩，或在许多轮次之后发生跳变，训练奖励变得不稳定。
>
> **原因：** 每一轮都使策略离 $\pi_{\text{old}}$ 更远（8.3 节），代理目标的误差随这一距离增长（8.1 节），而裁剪并不能阻止漂移。
>
> **可选方案：** (a) 减少轮次，这对每个样本浪费的计算比看上去少，因为后面的轮次大部分被裁剪了。(b) 基于 KL 目标的提前停止，按批次自适应调整轮次数。(c) 更小的学习率，使每一轮的 KL 开销更小，但会减慢学习。
>
> **建议做法：** 对 LLM 智能体用 1 到 2 个轮次，以提前停止作为保护，因为钱主要花在 rollout 上，但一个被破坏的策略会浪费掉所有 rollout。下面的代码块在一个批次上更新 16 个轮次，每四轮打印一次。

```python
batch = old_batch
torch.manual_seed(0)                                                         # the critic head's initialization
policy, critic = copy.deepcopy(base), warm_critic(base)
popt, vopt = torch.optim.Adam(policy.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
prepared_16 = ppo_prepare(policy, critic, batch, 1.0)
many = ppo_epochs(policy, critic, popt, vopt, batch, prepared_16, epochs=16, minibatches=4,
                  gen=torch.Generator().manual_seed(0))
policy_16 = policy
with torch.no_grad():
    entropy = [arl.masked_mean(arl.entropy_from_logits(m(batch['ids'])[:, :-1]), mask).item() for m in (base, policy)]
for e in [0, 3, 7, 11, 15]:
    print(f'epoch {e + 1:2d}: clip_frac {many[e]["clip_frac"]:.3f} | approx_kl {many[e]["approx_kl"]:.3f}'
          f' | max ratio {many[e]["ratio_max"]:.1f}')
print(f'policy-token entropy before {entropy[0]:.3f} -> after 16 epochs {entropy[1]:.3f}')
```

Output:

```text
epoch  1: clip_frac 0.043 | approx_kl 0.007 | max ratio 1.7
epoch  4: clip_frac 0.097 | approx_kl 0.018 | max ratio 3.9
epoch  8: clip_frac 0.135 | approx_kl 0.023 | max ratio 4.2
epoch 12: clip_frac 0.155 | approx_kl 0.024 | max ratio 4.3
epoch 16: clip_frac 0.170 | approx_kl 0.027 | max ratio 4.3
policy-token entropy before 0.515 -> after 16 epochs 0.530
```

裁剪比例从第一轮的 4% 上升到第十六轮的 17%，因此后面的轮次把计算花在越来越少的 token 上：到最后，每六个 token 中就有一个不贡献梯度。KL 在前五轮迅速增长，然后在 0.027 附近趋于平稳，从第五轮起高于提前停止目标；最大比率稳定在 4.3 附近，是裁剪边界的三倍多。这两种效应合起来就是裁剪在复用下的作用：到达区间边界的 token 不再贡献，仍在区间内的 token 继续移动共享参数，策略以每轮越来越少的学习信号缓慢地持续漂移。这个批次上的策略 token 熵几乎没变（0.52 到 0.53），所以这种漂移不是坍缩；它是浪费的计算，把策略推离其数据所描述的分布。图 8.3 把这个批次与训练运行作对比，后者每轮的 KL 更大，因为它的 64 回合批次只给每个小批次 16 个回合，于是每一步更用力地拟合更少的样本。

> **问题：小批次大小与陈旧度。**
>
> **现象：** 把一个批次划分成更多的小批次，起初会加快按样本计的学习，随后会使其不稳定。
>
> **原因：** 一轮中的第 $j$ 个小批次是在一个已经在同一批次上走了 $j - 1$ 步的策略下评估的，因此即使在第一轮，后面的小批次也更加 off-policy。
>
> **可选方案：** (a) 更少、更大的小批次，更接近 on-policy，但每个样本对应的步数更少。(b) 更多的小批次配合更小的学习率。(c) 按小批次步数而非轮次计数的提前停止。
>
> **建议做法：** 每个批次 2 到 4 个小批次，并按小批次记录裁剪比例和 KL。下面的代码块在旧批次上用 1 到 16 个小批次各运行一轮，并报告最后一个小批次所见到的 KL。

```python
for minibatches in [1, 2, 4, 8, 16]:
    torch.manual_seed(0)
    policy, critic = copy.deepcopy(base), warm_critic(base)
    popt, vopt = torch.optim.Adam(policy.parameters(), lr=1e-3), torch.optim.Adam(critic.parameters(), lr=1e-3)
    prepared = ppo_prepare(policy, critic, old_batch, 1.0)
    old_lp, adv = prepared[0], prepared[2]
    ppo_epochs(policy, critic, popt, vopt, old_batch, prepared, 1, minibatches, torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, after = ppo_policy_loss(arl.sequence_log_probs(policy, old_batch['ids']), old_lp, adv, mask)
    print(f'{minibatches:2d} minibatches of {320 // minibatches:3d}: KL after one epoch {after["approx_kl"]:.4f}'
          f' | clip_frac {after["clip_frac"]:.3f}')
```

Output:

```text
 1 minibatches of 320: KL after one epoch 0.0062 | clip_frac 0.051
 2 minibatches of 160: KL after one epoch 0.0082 | clip_frac 0.078
 4 minibatches of  80: KL after one epoch 0.0112 | clip_frac 0.073
 8 minibatches of  40: KL after one epoch 0.0186 | clip_frac 0.088
16 minibatches of  20: KL after one epoch 0.0219 | clip_frac 0.091
```

即使在整个批次上只走一步，也会使策略移动 0.006 的 KL，因为 Adam 的前几步大小由学习率而非梯度决定。小批次越多意味着步数越多：使用 16 个小批次时，一轮使策略移动的距离是 3.5 倍，并且已经超过 0.02 的提前停止目标。因此，“一个轮次”不是陈旧度的单位；每个批次上的优化器步数才是。

> **问题：`old_logp` 来自训练器还是来自采样器。**
>
> **现象：** 一次运行从第一个小批次起裁剪比例就很高，或者一次运行无论走多少轮裁剪都从不激活。
>
> **原因：** $\pi_{\text{old}}$ 必须是实际采样这些 token 的分布。用训练器的前向计算重新计算 `old_logp`，只有在训练器与采样器计算出相同概率时才正确。在更新循环*内部*用当前参数重新计算它是一个 bug：此时每个比率都恰好为 1。
>
> **可选方案：** (a) 在第一次更新之前用训练器重新计算一次 `old_logp`，本章和大多数开源训练器都这样做。(b) 使用采样器的对数概率，这对行为策略而言是精确的，但会让比率吸收两个引擎之间的任何数值不一致。(c) 两者都用：训练器的用于 PPO 比率，采样器的用于显式的不一致修正（[第 13 章](13-rollout-systems.md)）。
>
> **建议做法：** 在像这里这样的单进程设置中用 (a)，在生产中用 (c)。下面的代码块比较第 5 章存储的 `sample_logp` 与训练器的重新计算结果，然后复现这个 bug。

```python
diff = ((old_logp - old_batch['sample_logp']).abs() * mask).max().item()
print(f'max |trainer log-prob - sampler log-prob| over {int(mask.sum())} policy tokens: {diff:.1e}')
policy = copy.deepcopy(base)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
for step in range(8):                                                       # the bug: old_logp recomputed every step
    logp = arl.sequence_log_probs(policy, old_batch['ids'][:64])
    loss, stats = ppo_policy_loss(logp, logp.detach(), prepared_16[2][:64], mask[:64])
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f'after 8 steps with old_logp = logp.detach(): clip_frac {stats["clip_frac"]:.3f}, approx_kl {stats["approx_kl"]:.4f}')
```

Output:

```text
max |trainer log-prob - sampler log-prob| over 3620 policy tokens: 2.1e-06
after 8 steps with old_logp = logp.detach(): clip_frac 0.000, approx_kl 0.0000
```

这里两个来源的差异在 $2 \times 10^{-6}$ 以内，这是带填充的同步前向计算与整批前向计算之间的 float32 舍入差异。它们一致，是因为 TinyAgent 在同一进程中采样和训练，使用相同的权重、相同的数据类型和相同的算子，温度为 1 且不使用 top-p。在生产中这些都不成立。采样器是一个推理引擎，有自己的算子，通常使用 bf16，归约依赖于批次组成，有时采样参数不同，权重可能落后训练器一次或多次更新。此时的不一致大到足以产生影响，[第 13 章](13-rollout-systems.md)在 Qwen3-0.6B 上测量了它。第二行就是那个 bug：`old_logp` 取自当前的前向计算时，每个比率都为 1，裁剪从不起作用，无论运行多少步，损失都只是普通的 REINFORCE。监控面板会显示恰好为 0 的裁剪比例和恰好为 0 的 KL，这本身就是该 bug 的特征。

> **问题：非对称裁剪。**
>
> **现象：** 训练过程中熵稳步下降，最终被证明是好的罕见动作，其概率增长得非常缓慢。
>
> **原因：** 上裁剪把一个好 token 概率在每个批次上的*相对*增长限制在 $1 + \epsilon_{\text{high}}$ 倍，因此在 $\epsilon_{\text{high}} = 0.2$ 时，$\pi_{\text{old}} = 0.01$ 的 token 最多只能达到 0.012，而 0.5 的 token 可以达到 0.6。下裁剪限制了从坏 token 移走概率的速度，而被移走的概率质量主要流向本来就很可能的 token。两种效应都使分布更集中。
>
> **可选方案：** (a) **clip-higher**：提高 $\epsilon_{\text{high}}$（DAPO 使用 0.28，$\epsilon_{\text{low}} = 0.2$；[第 9 章](09-grpo-and-variants.md)），让罕见的好 token 增长得更快，代价是信赖域更松。(b) 熵奖励项（[第 10 章](10-kl-entropy-regularization.md)）。(c) 降低 $\epsilon_{\text{low}}$ 以减缓概率质量的移走。
>
> **建议做法：** 以对称的 0.2 作为基线，测得熵坍缩时再使用 clip-higher。下面的代码块按 token 在 $\pi_{\text{old}}$ 下的概率分组，查看在 16 轮的运行中上裁剪实际拦住了哪些 token。

```python
with torch.no_grad():
    ratio = (arl.sequence_log_probs(policy_16, old_batch['ids']) - old_logp).exp()   # [B, T-1] after 16 epochs
old_p, on = old_logp.exp(), mask > 0                                         # pi_old of each sampled token
for lo, hi in [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]:
    good = on & (prepared_16[2] > 0) & (old_p >= lo) & (old_p < hi + 1e-6)   # positive-advantage tokens
    above = (ratio[good] > 1.2).float().mean().item()
    print(f'pi_old in [{lo:.1f}, {hi:.1f}]: {int(good.sum()):4d} tokens | ratio above 1.2: {above:.2f}'
          f' | median ratio {ratio[good].median():.2f}')
```

Output:

```text
pi_old in [0.0, 0.1]:   59 tokens | ratio above 1.2: 0.73 | median ratio 1.39
pi_old in [0.1, 0.5]:  235 tokens | ratio above 1.2: 0.60 | median ratio 1.23
pi_old in [0.5, 1.0]:  469 tokens | ratio above 1.2: 0.06 | median ratio 1.00
```

在 $\pi_{\text{old}}$ 给出概率低于 0.1 的正优势 token 中，73% 最终比率高于 1.2，中位数为 1.39：这些就是被上裁剪拦住的 token。在概率高于 0.5 的 token 中，只有 6% 如此，因为比率不可能超过 $1/\pi_{\text{old}} \le 2$，而且它们大多已经接近 1。因此，上裁剪主要恰恰作用在那些增长意味着探索的罕见 token 上，这正是 DAPO 主张提高 $\epsilon_{\text{high}}$ 的理由。裁剪所限制的概率增量是一个绝对量，即每个批次 $\epsilon_{\text{high}}\, \pi_{\text{old}}$，与该 token 当前的概率成正比。

## 小结

- 在一次更新之后复用批次需要重新加权。精确的轨迹权重是策略 token 上逐 token 比率的乘积（环境的项相互抵消），其方差随策略 token 数指数增长：三次 REINFORCE 更新之后，320 个基座智能体回合只相当于其数量的三分之一，新通过率的 IS 估计远低于真实值。
- PPO 的代理目标使用逐 token 比率。它修正每个动作的概率，却忽略了所访问前缀的变化，因此在一阶上与真实改进一致，并随策略移动而失去准确性：它对一次和三次更新的改进预测误差在半个百分点以内，对十次更新则少估了六分之一。
- 裁剪目标消除了把 token 比率朝其优势所要求的方向推过 $1 \pm \epsilon$ 的动力，并保留撤销错误方向移动的梯度。它是一个软信赖域：在一个小批次上走 40 步后，41% 的比率落在区间之外，尽管 KL 比不裁剪时小 25 倍。双重裁剪限制了负优势 token 上极大比率的影响。
- 带预热 critic 的 PPO 在 30 个批次内学会了训练任务。critic 解释了回报方差的 12% 到 26%，因此主要起基线的作用；大量复用使第 6 章的记忆问题在留出任务上更严重。
- 按轮次记录裁剪比例、近似 KL 和比率极值。前两者随每一个轮次和每一个额外的小批次增长；以 0.02 为 KL 目标的提前停止用约 63% 的梯度步达到了相同的训练通过率。在更新之前重新计算一次 `old_logp`，绝不要在循环内部计算。

## 练习

1. 证明只要两个策略面对同一个环境，即使环境是随机的，式 (8.1) 中环境的因子也会相互抵消。当工具输出依赖于实际耗时或共享缓存时，哪个假设不成立？比率会发生什么？
2. 沿 $\pi_\theta$ 的一条轨迹对 $\sum_i \big(r_i + V^{\text{old}}(c_{\mathrm{nx}(i)}) - V^{\text{old}}(c_i)\big)$ 做裂项求和，从 $A^{\text{old}}$ 的定义推导 8.1 节的性能差分恒等式。逐 token 代理目标究竟在哪一步做了近似？
3. 改变一个假设：用自归一化重要性采样 $\sum_n w_n R_n / \sum_n w_n$ 重做 8.1 节的 IS 表格。它是否更接近新样本上的通过率？它是有偏的；偏向哪个方向？为什么这种偏差常常更受青睐？
4. 实现 PPO 的 KL 惩罚变体（“PPO-penalty”）：不裁剪，在损失中加入 $\beta\, \widehat{\mathrm{KL}}$ 项，并在每个批次后将 $\beta$ 加倍或减半以跟踪 KL 目标。在 8.4 节的运行上，将它的逐轮次 KL 与裁剪版本比较。
5. 分别用 `value_loss(..., clip=None)` 和 `clip=0.05` 运行 8.4 节的 PPO。跟踪 critic 在新回合上的解释方差。价值裁剪对这个 critic 是有帮助还是有害？
6. 在 16 轮的运行中，分别测量正优势和负优势 token 的裁剪比例。哪一侧更常起作用？把 $\epsilon_{\text{high}}$ 提高到 0.28 会如何改变 16 轮后的熵？
7. 设计评审：一位队友提议每个批次做 8 个轮次、每轮 8 个小批次，“让 rollout 便宜八倍”。利用 8.6 节的测量，列出你预期在监控面板上看到什么、你会要求做什么实验，以及在同意之前你会要求加上什么保护措施。
