# 10. KL、熵与正则化

[第 9 章](09-grpo-and-variants.md#94-环境设计中的捷径特征)的默认 GRPO 运行成功了，但它的日志里有两个数字，后训练工程师在宣布它健康之前一定会追问。策略 token 的熵（entropy）在 80 步内从 0.57 nats 降到 0.08 nats；到最后，每五个组中有四个奖励完全相同，不带任何梯度。策略已经变得几乎是确定性的。在这个 gym 上，这正是成功的样子，因为每个任务只有一种正确流程。但在真实的代码任务分布上，一个已经停止探索（exploration）的策略以后无法再发现更好的流程，而坍缩了的采样器会让 pass@$k$ 不比 pass@1 好。相反方向的担忧同样真实：一个没有任何东西把策略约束在起点附近的 RL 目标，可以自由地漂移到奖励出错、语言退化，或者奖励不衡量的技能被遗忘的区域。

本章研究应对这些担忧的两种标准正则化手段。**KL 惩罚**把策略拉向一个冻结的**参考策略**（reference policy），通常是初始检查点。**熵奖励**（entropy bonus）把策略推离确定性。两者看起来都只是损失里多出的一行，但都暗藏细节，会改变训练器（trainer）实际计算的梯度：三种常用的 KL 估计量（estimator）用作损失时会产生三种不同的梯度，其中一种期望为零，另一种是另一个散度的梯度；对序列而言，把 KL 放进奖励和放进损失是不同的；而在整个词表（vocabulary）上的熵奖励会把概率推到策略永远不该输出的 token 上。我们逐一推导这些事实，在小到可以穷举的分布上做数值检验，然后在 TinyAgent 上比较各种正则化手段：不加正则、不同强度和放置位置的 KL 惩罚、熵奖励，以及更高的温度，以熵、留出（held-out）通过率和 pass@$k$ 衡量；clip-higher 另做一个单批次实验。本章保存两个函数 `kl_estimate` 和 `entropy_bonus`，以及使用它们的更新、训练循环和序列 KL 辅助函数。

```python
#@save
import copy
import math
import torch
import agentrl as arl
```

TinyAgent 实验使用第 9 章确定的本书默认设置：匿名化的一次编辑 gym、RL 基座智能体（base agent），以及训练集和留出集划分。

```python
torch.set_num_threads(4)              # fixed CPU threads: fast and repeatable on a shared check host
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
rl_base = arl.make_rl_base(tok)
```

## 10.1 参考策略

**参考策略** $\pi_{\text{ref}}$ 是一个固定的策略，实践中是 RL 开始时模型的一份冻结副本（SFT 或基座检查点）。KL 正则化的目标是

$$
J_\beta(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

其中序列 KL 沿策略自身的前缀分解为逐 token 的项（[附录 A.6](A-math.md#a6-kl-散度)）。文献中反复出现三个使用它的理由，而它们是不同的理由。第一，在 RLHF 中奖励是一个学出来的模型，只在其训练数据附近准确；KL 把策略留在奖励可信的区域。第二，参考策略携带着奖励不衡量的能力（流畅性、格式、知识），KL 限制了策略可以拿多少这类能力去交换。第三，KL 是一种*累积的*信赖域（trust region）：PPO 的裁剪（第 8 章）限制一次更新离 $\pi_{\text{old}}$ 多远，而 KL 限制整个运行离 $\pi_{\text{ref}}$ 多远。对于隐藏测试这类可验证奖励，第一个理由不复存在，这正是主张去掉 KL 的核心论据（10.7 节）。

正则化的目标想要什么？对于单次决策（老虎机（bandit）），答案有闭式解。记 $\pi$ 为 $K$ 个动作上的分布，奖励为 $r(a)$。则

$$
\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) = -\beta \sum_a \pi(a) \log \frac{\pi(a)}{\pi_{\text{ref}}(a)\, e^{r(a)/\beta}} = \beta \log Z - \beta\, \mathrm{KL}(\pi \,\|\, \pi^*),
$$

其中 $\pi^*(a) = \pi_{\text{ref}}(a)\, e^{r(a)/\beta} / Z$，$Z = \sum_a \pi_{\text{ref}}(a)\, e^{r(a)/\beta}$。由于 KL 非负，且仅在 $\pi = \pi^*$ 时为零，最优解是 $\pi^*$，最优值为 $\beta \log Z$。最优解用指数化的奖励对参考策略做*倾斜*：$\beta$ 是一个温度，在 $\pi_{\text{ref}}$（$\beta \to \infty$）与**参考策略支撑集内**的最优动作（$\beta \to 0$）之间插值。参考策略从不采取的动作，对任何 $\beta$ 概率都保持为零。在 $\pi_{\text{ref}}$ 为均匀分布时做同样的推导，就得到熵奖励，因为 $\mathrm{KL}(\pi \,\|\, \mathrm{Uniform}) = \log K - H(\pi)$：最大化 $\mathbb{E}[r] + \alpha H(\pi)$ 等价于对均匀分布施加 KL 惩罚，最优解为 $\mathrm{softmax}(r/\alpha)$。[第 15 章](15-sft-and-rejection-sampling.md)用同一个闭式解推导 DPO。

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

下面的老虎机有五个动作。动作 4 奖励最高，但参考策略从不采取它；动作 0 和 1 都不错，而参考策略偏好较弱的那个。

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

在 $\beta = 10$ 时，最优解就是参考策略，KL 为 0.001。随着 $\beta$ 减小，概率质量先移向参考策略本就偏好的动作 1，再移向动作 0，即参考策略很少采取的更好动作；在 $\beta = 0.01$ 时，策略确定性地选择动作 0，期望奖励 1.0，熵为 0。KL 饱和于 $\log(1/0.05) = 2.996$，这是把全部质量放在一个参考策略只有 5% 概率采取的动作上的代价。价值 1.2 的动作 4 始终得不到任何质量。$\alpha = 0.3$ 的熵奖励表现不同：它隐含的参考分布是均匀的，因此它在动作 4 上放了 0.59，期望奖励达到 1.065，高于任何 KL 设置。在语言模型中，“参考策略支撑集之外的动作”大多是垃圾 token，均匀参考分布对它们一视同仁，正是熵奖励的主要隐患（10.6 节）。我们用 $\pi^*$ 的随机扰动来检验这个闭式解：

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

$\pi^*$ 处的值等于 $\beta \log Z$，没有任何扰动能做得更好，正如上面的恒等式所要求的。

## 10.2 从样本估计 KL

训练器从来看不到序列上的完整分布。它看到的是采样得到的 token $y \sim \pi_\theta(\cdot \mid c)$，以及每个 token 的两个数：$\log \pi_\theta(y \mid c)$ 和 $\log \pi_{\text{ref}}(y \mid c)$。把它们的差记为 $\ell = \log \pi_{\text{ref}}(y \mid c) - \log \pi_\theta(y \mid c)$。John Schulman 的笔记 [*Approximating KL Divergence*](http://joschu.net/blog/kl-approx.html)（2020）比较了 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ 的三种单样本估计量：

$$
k_1 = -\ell, \qquad k_2 = \tfrac{1}{2}\ell^2, \qquad k_3 = e^{\ell} - 1 - \ell .
$$

$k_1$ 按定义无偏，在许多样本上取负值。$k_3$ 加上了 $e^\ell - 1$，由于 $\mathbb{E}_{\pi_\theta}[\pi_{\text{ref}}/\pi_\theta] = 1$，这一项在 $\pi_\theta$ 下均值为零；它就是加了控制变量（control variate）的 $k_1$（[附录 A.3](A-math.md)），无偏，并且由于 $e^\ell \ge 1 + \ell$ 而永不为负。$k_2$ 有偏：泰勒展开表明它在 $\ell$ 的二阶上与 KL 一致，更高阶则高估。第 9 章把这三者都保存为 `kl_penalty`。[附录 A.7](A-math.md#a7-从样本估计-kl) 在两个距离上对它们做了采样；这里我们穷举一个 50 类分布的每个结果，在策略与参考策略之间的四个距离上精确计算它们的均值和单样本标准差。

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

均值是精确计算的，所以 $k_1$ 和 $k_3$ 在每个距离上的比值都是 1.000，而 $k_2$ 的偏差随距离增大：KL 为 0.001 时为 1%，KL 为 0.32 时为 41%，KL 为 1.08 时达到 2 倍。离散程度说明了另一半。在 KL 为 0.001 时（相邻检查点之间逐 token KL 的典型量级），单个 $k_1$ 样本的标准差是被估计量的 45 倍，$k_3$ 则是 1.25 倍：把 $k_1$ 平均到同样的精度，大约需要 $(44.84/1.25)^2 \approx 1300$ 倍的 token。原因是 $k_1$ 是 $\ell$ 的一阶量，其离散程度与 $|\ell|$ 成比例，而 KL 本身是二阶量。随着两个分布分离，$k_3$ 的优势缩小，在 KL 为 1.08 时它的离散程度与 $k_1$ 相当。若要*记录* KL，$k_3$ 是正确的默认选择：无偏、非负，并且在训练所处的范围内方差小。把估计量用作*损失*则是另一个问题。

## 10.3 各估计量用作损失时产生的梯度

损失中的 KL 项是这样实现的：在批次中采样得到的 token 上计算估计量，把 $\beta$ 乘以其均值加到损失上。自动求导随后通过 $\log \pi_\theta(y \mid c)$ 对估计量求导，采样到的 token 保持固定。它*不会*对“这些 token 是从 $\pi_\theta$ 采样而来”这一事实求导。因此这种损失的期望梯度是 $\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta k(y)]$，而不是 $\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[k(y)]$：缺少了 [第 6 章](06-policy-gradient.md)的得分函数（score function）项 $\mathbb{E}[k(y)\, \nabla_\theta \log \pi_\theta(y)]$。

利用 $\nabla \ell = -\nabla \log \pi_\theta(y)$ 和 $\mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta] = 0$ 逐一计算。对 $k_1 = -\ell$，单样本梯度是 $\nabla \log \pi_\theta(y)$，其期望为零：$k_1$ 损失只贡献噪声。对 $k_2 = \ell^2/2$，单样本梯度是 $-\ell\, \nabla \log\pi_\theta(y) = \big(\log\pi_\theta(y) - \log\pi_{\text{ref}}(y)\big)\nabla\log\pi_\theta(y)$，其期望正是反向 KL（reverse KL）的真实梯度：

$$
\nabla_\theta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \sum_y \nabla \pi_\theta(y) \big(\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)\big) + \sum_y \pi_\theta(y)\, \nabla \log \pi_\theta(y) = \mathbb{E}_{\pi_\theta}\big[(\log \pi_\theta - \log \pi_{\text{ref}})\, \nabla \log \pi_\theta\big].
$$

有偏的估计量给出了无偏的梯度。对 $k_3 = e^\ell - 1 - \ell$，单样本梯度是 $(1 - e^{\ell})\nabla\log\pi_\theta(y)$，且

$$
\mathbb{E}_{\pi_\theta}\big[(1 - \tfrac{\pi_{\text{ref}}}{\pi_\theta})\nabla \log \pi_\theta\big] = 0 - \sum_y \pi_{\text{ref}}(y)\, \nabla \log \pi_\theta(y) = \nabla_\theta\, \mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta).
$$

无偏、低方差的估计量给出的却是**前向** KL（forward KL）的梯度，即另一个方向。在 $\pi_\theta = \pi_{\text{ref}}$ 附近，两个方向在二阶上一致，因此它们的梯度在一阶上一致，差异随距离增大。我们用每个结果的（detach 后的）采样概率对它加权来精确检验这一切，这就是大批量样本会趋近的期望。

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

保存的函数是第 9 章 `kl_penalty` 之上的薄封装，取值时直接调用它。它只增加了一点：以 `+` 结尾的类型使用**直通**（straight-through）构造 `g - g.detach() + v.detach()`，其前向值是估计量 $v$（因此记录的数字是比如 $k_3$），而梯度是 $g = k_2$ 的梯度（因此损失沿反向 KL 的梯度方向推动）。verl 用同样的 `+` 后缀实现了同一选项（[`core_algos.kl_penalty`](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py)，commit `00094bd`，2026-10-13 查阅），并附有注释说明 $k_1$ 和 $k_3$ 的期望值正确，但期望梯度不正确。

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

每一行都是用数字表达的推导。$k_1$ 的期望梯度为零，所以它相对任一目标的相对误差都恰好是 1。$k_2$ 的梯度在打印的每一位上都与反向 KL 梯度一致，$k_3$ 的梯度与*前向* KL 梯度一致，而与反向 KL 梯度相差 50%。直通的 `k3+` 恰好具有 $k_2$ 的梯度。在训练实际达到的距离上，$k_3$ 的误差有多大？

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

在逐 token KL 为 0.002 时，$k_3$ 损失与反向 KL 梯度相差在 3.4% 以内；在 0.06 时偏差为 17%。第 9 章的 GRPO 预设相对其参考策略的逐 token $k_3$ 平均为 0.17，介于第二行和第三行之间，该表给出的误差为 17% 到 34%。在这个范围内，用 $k_3$ 损失实现的“反向 KL 惩罚”（如 DeepSeekMath 的 GRPO 和 verl 的默认 `low_var_kl`）更接近前向 KL 惩罚。10.5 节说明方向为什么重要，10.7 节在 TinyAgent 上测量这一差别。

![图 10.1：策略远离参考策略时的各 KL 估计量，在一个 50 类分布上精确计算。左：各估计量取值的相对偏差。中：单样本标准差与真实 KL 之比（对数坐标）。右：各估计量期望损失梯度相对真实反向 KL 梯度的相对误差。](../../figures/ch10-kl-estimators.svg)

图 10.1 在真实 KL 跨越两个数量级的范围内并列展示了这三种性质。从左到右把三个面板当作三个不同的问题来读。记录的值对吗？$k_1$ 和 $k_3$ 对。单个样本有信息量吗？$k_2$ 和 $k_3$ 有，在 KL 较小时要好一个数量级。损失推动的方向对吗？只有 $k_2$（因而还有 `k3+`）在任何距离上都对。没有哪个估计量能同时赢下三项，这正是直通组合存在的原因。

> **陷阱。** 在 PPO 式的更新中，第一个小批次之后，批次中的行来自 $\pi_{\text{old}}$，而不是当前的 $\pi_\theta$。上面的推导假设样本是同策略（on-policy）的。在第 9 章默认设置的小比率下（裁剪比例约 0.1%），这只是一个较小的额外偏差；在多轮次（epoch）训练中，这是 KL 项的梯度只是近似等于其名称所指的又一个原因。

## 10.4 KL 放在奖励中与放在损失中

惩罚有两个可以放置的位置。**KL 放在奖励中**：从奖励中减去 $\beta$ 乘以逐 token 的 $k_1$（对数比率），交给策略梯度（policy gradient）处理，RLHF 风格的 PPO 就是这样做的。**KL 放在损失中**：把 $\beta$ 乘以逐 token 估计量加到损失上，GRPO 就是这样做的。对单个 token 而言，奖励版本就是以 $-k_1$ 为奖励的 REINFORCE，其梯度为 $-\mathbb{E}[k_1 \nabla\log\pi_\theta] $；这恰好是上面推导出的负反向 KL 梯度，因此对单个 token 而言，“把 $k_1$ 放在奖励中”与“把 $k_2$ 放在损失中”是一致的。

对序列则不然。由附录 A.6，序列 KL 为 $\mathbb{E}\big[\sum_t \mathrm{KL}_t(c_t)\big]$，其中 $\mathrm{KL}_t(c_t)$ 是前缀 $c_t$ 处的逐 token KL。它的梯度有两部分：在前缀固定时每个 $\mathrm{KL}_t$ 如何随参数变化，以及 token $y_t$ 的选择如何改变后续访问到哪些前缀，从而改变未来的哪些 KL。逐 token 的损失只捕捉了第一部分，因为每个 token 的损失项只是该 token 自身对数概率的函数。基于奖励的策略梯度两部分都捕捉到了，因为每个 token 的对数概率都被它之后所有内容的 KL 加权（即 [6.5 节](06-policy-gradient.md)的剩余回报（reward-to-go））。差别在于对*未来*偏离的信用分配（credit assignment）：一个把回合引向策略与参考策略分歧区域的 token，会被奖励版本惩罚，却不会被损失版本惩罚。

我们在一个词表大小为 4 的两 token 序列模型上检验这一点，它小到可以穷举全部 16 个序列：第一个 token 有自己的 logits，第二个 token 的 logits 取决于第一个。

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

`surrogate` 为每种放置方式构建自动求导会看到的期望损失。对损失放置，每个序列在两个 token 上都贡献估计量。对奖励放置，每个 token 的对数概率乘以从该 token 到结尾的 $k_1$ 之和（已 detach）。

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

奖励放置对两个 token 都复现了序列 KL 的精确梯度。$k_2$ 损失对第二个 token（最后一个，没有未来）是精确的，而对第一个 token 错了 68%，因为它忽略了第一个 token 对第二个 token 的 KL 的影响。$k_3$ 在两个 token 上都不对，$k_1$ 在两个 token 上都为零。逐 token 的 $k_2$ 损失是另一个目标的精确梯度：前缀分布保持固定时的 $\sum_t \mathrm{KL}_t$。

哪种放置更好是设计问题，而不是正确性问题。奖励放置惩罚的是正确的量，但它进入优势（advantage），因而继承了策略梯度的方差；在 GRPO 的组归一化下，它还会按每组奖励的离散程度重新缩放，所以它的强度因任务而异。损失放置是一个低方差的局部正则项，让每个 token 的分布都保持在参考策略附近，而不考虑回合走向何处。verl 两者都提供：`algorithm.use_kl_in_reward`，默认用 $k_1$ 惩罚，可选自适应控制器；以及 `actor.use_kl_loss`，配合 `kl_loss_type: low_var_kl`（$k_3$，截断到 $[-10, 10]$），默认系数 0.001（commit `00094bd`，2026-10-13 查阅）。对智能体而言，未来偏离项比单轮任务更重要，因为早期一次调用的 token 决定了回合会到达哪些观测，进而到达哪些后续上下文。

## 10.5 前向 KL 与反向 KL

反向 KL $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb{E}_{\pi_\theta}[\log \pi_\theta - \log \pi_{\text{ref}}]$ 是在策略下的期望。只要策略在参考策略没有质量的地方放了质量，它就很大；而对策略忽略的参考质量，它毫不在意。前向 KL $\mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$ 是在参考策略下的期望，只要参考策略有质量而策略缺少质量，它就很大。前者称为**寻模**（mode-seeking，或称零强制（zero-forcing）），后者称为**覆盖质量**（mass-covering）。最大似然和 SFT 最小化相对数据的前向 KL；[第 16 章](16-distillation.md)的同策略蒸馏（on-policy distillation）最小化相对教师的反向 KL。当一个简单的分布族必须逼近它无法表示的分布时，差别最为明显。我们用梯度下降分别在两个方向上，把一个离散化的高斯分布拟合到两个高斯的混合分布上。

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

前向拟合把中心放在两个峰之间，标准差为 2.84，质量在两侧平分，并把 30% 的质量放在谷中，而目标分布在那里只有 0.8%。反向拟合恰好落在右侧的峰上（均值 2.50，标准差 0.60），左侧什么也不放，KL 达到 0.693，即 $\log 2$：完美覆盖两个等大峰之一的代价。它选择哪个峰取决于初始化（我们从 1.0 开始，在中心右侧）。

![图 10.2：在 121 点网格上，用单个高斯分布拟合双峰目标。前向 KL 拟合铺展到两个峰和中间的谷；反向 KL 拟合锁定在一个峰上。](../../figures/ch10-forward-reverse.svg)

图 10.2 在目标分布上画出了两种拟合。对 RL 而言，后果是直接的。对 $\pi_{\text{ref}}$ 的反向 KL 惩罚允许策略无代价地放弃参考策略的峰（初始模型有时使用的策略），因此它本身并不保持多样性；同时它禁止在参考策略没有质量的地方放置质量，正如老虎机中的动作 4 所示。前向 KL 的拉力（也就是 $k_3$ 损失在中等距离上实际实现的东西，见 10.3 节）会惩罚放弃参考策略的峰，因而更像一个保持多样性的正则项。这是估计量梯度的副作用，而不是任何人写下的设计选择，值得弄清楚一个配置实际计算的是哪一个。

## 10.6 熵坍缩及其对策

**熵坍缩**（entropy collapse）是指 RL 过程中策略的逐 token 熵持续下降，趋向一个每次都采样同样 token 的策略。它不是某个组件的 bug：当 softmax 策略的高概率选择得到奖励时，策略梯度就会对它这样做。机制可以用一行写出。对 $K$ 个 token 上的 logits $z$，记 $\pi = \mathrm{softmax}(z)$，$H = -\sum_j \pi_j \log \pi_j$，求导得 $\partial H / \partial z_j = -\pi_j(\log \pi_j + H)$，因此 logits 的一个小变化 $\Delta z$ 使熵变化

$$
\Delta H \approx -\sum_j \pi_j\, (\log \pi_j + H)\, \Delta z_j = -\mathrm{Cov}_{j \sim \pi}\big(\log \pi_j,\ \Delta z_j\big),
$$

这里用到了 $\mathbb{E}_\pi[\log \pi] = -H$。当更新提高已经很可能的 token 的 logits 时，熵下降；当它提高不太可能的 token 的 logits 时，熵上升。Cui et al. 的 [*The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*](https://arxiv.org/abs/2505.22617)（2025-05-28）对策略梯度和自然梯度更新推导了这一协方差形式，并报告在不加干预时，熵与验证集性能沿一条拟合曲线此消彼长。在优势为 $A$ 的采样 token $a$ 上做一步策略梯度，logits 的变化为 $\Delta z = \eta A (e_a - \pi)$。四种情况涵盖了所有可能。

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

一阶预测在每种情况下都与实际变化相差不到 1%，符号也符合协方差的规律。奖励高概率 token（概率 0.647）会降低熵；奖励稀有 token（概率 0.0044）会提高熵；惩罚其中任何一个则相反。现在考虑训练后期的一个 GRPO 批次。在一个组内，成功的尝试大多由策略的高概率 token 组成，所以大多数正优势更新属于第一种情况。失败的尝试则由各种 token 混合而成，而惩罚稀有 token（第四行）同样会降低熵。只要策略对的时候多于错的时候，四种情况中就有三种在降低熵，这就是为什么坍缩是同策略 RL 的默认动力学，而不是某次运行的偶然。

各种对策以不同方式作用于这个协方差。

- **熵奖励**在损失中加入 $-\alpha H$，它对每个 token 的 logits 的梯度为 $\alpha\,\pi_j(\log\pi_j + H)$：它提高每个对数概率低于 $-H$ 的 token，降低高概率的 token。在 316 个词的词表上（Qwen3 则是 151k 个 token），“每个不太可能的 token”大多是策略永远不该输出的 token。10.1 节用闭式解说明了同一件事：熵奖励就是朝均匀分布的 KL 惩罚。
- **KL 惩罚**把每个 token 的分布拉向参考策略的分布，后者具有参考策略的熵。它只能在参考策略本身不确定的范围内抵抗坍缩；而且作为反向 KL，它不惩罚丢弃参考策略原有的备选（10.5 节）。
- **Clip-higher**（DAPO）提高 $\epsilon_{\text{high}}$。当 $A > 0$ 时，一旦 $\pi_\theta(a) > (1 + \epsilon_{\text{high}})\,\pi_{\text{old}}(a)$，PPO 的裁剪就会去掉该 token 的梯度，因此一个 token 在裁剪生效之前能获得的绝对增量是 $\epsilon_{\text{high}}\,\pi_{\text{old}}(a)$：概率为 0.9 的 token 是 0.18，概率为 0.01 的 token 是 0.002。所以上裁剪界作用于具有正优势的稀有 token，而这些恰好是上表中提高熵的更新。DAPO 给出的理由相同：“不太可能的‘探索’ token 的概率被限制得太紧，无法被提升”（“the probability of an unlikely 'exploration' token is too tightly bounded to be uplifted”），并把 $\epsilon_{\text{high}}$ 从 0.2 提高到 0.28。只有裁剪生效时这个手段才起作用，这要求策略在一个批次内发生移动（多个小批次或多个轮次）。
- **采样温度** $T$ 从 $\mathrm{softmax}(z/T)$ 中采样。由于训练器在同一个 $T$ 下计算对数概率（第 9 章的 `rl_update` 把 logits 除以 `cfg.temperature`），被优化的策略*就是*加温后的策略，网络可以把 $z$ 变得更尖锐，从而抵消任何固定的 $T$。更高的温度只能推迟坍缩，而不能阻止它；ProRL 报告，在提高 rollout 温度后，“熵仍随训练推进而稳定下降”（“entropy continues to decline steadily as training progresses”）。在比训练更低的温度下评估则是另一个需要专门做出的选择（[第 17 章](17-evaluation.md)）。

`train_rl` 没有添加额外损失项的钩子，而第 9 章的函数又不能修改。因此我们保存 `entropy_bonus`，以及第 9 章更新和循环的正则化版本，它们由同样的已保存组件（`sample_step`、`behavior_stats`、`compute_advantages`、`policy_loss`）构建。`entropy_bonus` 是策略位置上全词表熵的掩码均值；与 KL 估计量不同，它在每个访问到的前缀处是精确值，而不是单样本估计，因为整个词表的 logits 已经算好了。

```python
#@save
def entropy_bonus(logits, mask):
    """Mean full-vocabulary entropy over policy positions: logits [B, T, V], mask [B, T] -> differentiable scalar."""
    return arl.masked_mean(arl.entropy_from_logits(logits), mask)
```

`regularized_update` 就是第 9 章的 `rl_update`，有两处不同：损失减去 `entropy_coef` 乘以熵奖励，KL 项使用 `kl_estimate`，因此 `cfg.kl_kind` 可以是 `'k3+'`。记录的 `entropy` 是每个小批次中熵奖励在更新步之前的值。

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

`train_regularized` 是去掉周期性评估的 `train_rl`，外加 10.4 节的奖励放置。当 `kl_in_reward=True` 时，每个回合的奖励变为 $r_i - \beta \sum_t k_{1,t}$，在计算优势之前由采样器的对数概率和参考策略算出，此时损失中不含 KL 项。`sequence_kl` 计算这个求和。

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

两个正则项都关闭时，这个循环必须与第 9 章的完全相同。它消耗随机生成器的顺序与 `train_rl` 相同，所以从同一起点各跑三步，参数应当完全一致。

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

差异恰好为零，因此下面每一次正则化运行与第 9 章 `train_rl` 的区别，只在于它打开的那个正则项。

### 单批次上的 clip-higher

上裁剪界是否生效，取决于策略在一个批次内移动了多远，而第 9 章的默认设置（一个轮次、两个小批次）只裁剪了 0.13% 的 token，太少了，$\epsilon_{\text{high}}$ 无关紧要。为了看清裁剪生效时的作用，我们从 RL 基座智能体取一个 64 个回合的批次，在其上分别以 $\epsilon_{\text{high}}$ 为 0.2、DAPO 的 0.28 和实际上无上界的设置运行 8 个全批次更新步。8 步之后，我们统计比率超过 $1 + \epsilon_{\text{high}}$ 的正优势策略 token 数，以及其中有多少起始概率低于 0.5。

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

在两种有裁剪的设置中，最终超过上裁剪界的正优势 token 大多起始概率低于 0.5（$\epsilon_{\text{high}} = 0.2$ 时 20 个中有 15 个，0.28 时 15 个中有 14 个），尽管 121 个正优势 token 中只有 17 个起始于这个区间。DAPO 论证的这一部分成立：上裁剪界是对稀有 token 的约束。熵这一列却不符合该论证的结论。8 步之后，裁剪最紧时熵下降最少（0.547 到 0.538），完全没有上裁剪界时下降最多（到 0.508）。去掉上限会释放每一个正优势 token，包括那些增长会让分布变尖锐的中等概率 token，在这个批次上，这一效应超过了给稀有 token 留出的额外空间。DAPO 报告 clip-higher 在 32B 规模的长推理训练中提高了熵；在 TinyAgent 上我们没有看到这一点，无论是这里还是第 9 章的消融实验，所以下面的比较不包括 clip-higher，这一论断留给其出处。

## 10.7 在 TinyAgent 上比较各种正则化手段

现在我们用默认配置训练 40 步（第 9 章的运行在这个阶段还有提升空间），共九种设置：不加正则；梯度正确（`k3+`）的损失内 KL 惩罚，$\beta$ 取 0.05、0.3 和 1.0；GRPO 风格的 $k_3$ 损失，取 0.3；放在奖励中的 KL，取 0.3（10.4 节）；熵奖励 0.03 和 0.1；以及采样温度 1.3。每次运行都从同一个 RL 基座智能体出发，随机种子为 0。除了留出 pass@1，我们还测量多样性。对一个 $n$ 次尝试中有 $c$ 次通过的任务，不放回地抽取 $k$ 次尝试、其中至少有一次通过的概率为 $1 - \binom{n-c}{k}/\binom{n}{k}$；[第 17 章](17-evaluation.md)推导并保存这个估计量，所以这里它只是一个局部辅助函数。我们还统计每个任务 16 次尝试中不同动作序列的数量，并用 $k_3$ 估计相对初始策略的逐 token KL。

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

按列来读这张表，并记住这只是一个随机种子、10 个留出任务、每个任务 16 次尝试的结果，所以当某个任务从未被解出时，pass@8 以约 0.1 的步长变化。

- **熵。** 不加正则时，策略 token 的熵在 40 步内从 0.52 降到 0.12。除最弱的 KL 外，每种正则化手段都减缓了下降，减缓程度大致与其强度成比例：KL 系数 0.3 把熵保持在 0.28，1.0 把它保持在起始水平（0.54），熵奖励 0.03 保持在 0.31。熵奖励 0.1 则把策略推向了均匀分布：5.707 nats，而 $\log 316 = 5.756$，每个回合六次格式错误的调用，通过率为零。
- **pass@1 与 pass@8。** 不加正则的运行 pass@1 为 0.725，pass@8 只有 0.80：有两个留出任务在 16 次尝试中从未被解出，因为坍缩的策略在它们上面反复执行同一个错误流程。最弱的 KL（0.05）没有改变熵（0.124），却仍达到 1.000 的 pass@8，pass@1 也是表中最高的 0.80；在单个随机种子下，我们无法把这与噪声区分开。更强的 KL 以 pass@1 为代价换来了多样性：$\beta = 0.3$ 时每个任务的不同动作序列数翻倍（8.3 对 4.3），pass@8 达到 0.99，而 pass@1 降到 0.59。$\beta = 1.0$ 时，读代码没有占据主导（每回合 0.53 次读取），pass@1 为 0.23，接近 0.2 的盲目上限，而 pass@8 仍有 0.90，因为接近参考策略的策略会采样许多不同的猜测。
- **估计量。** $\beta = 0.3$ 时，$k_3$ 损失和 `k3+` 损失最终的熵（0.294 和 0.284）、pass@1（0.594 和 0.588）以及与起点的距离几乎相同。在这些逐 token KL 下（约 0.1），10.3 节给出的梯度差异约为 25%，单个随机种子分辨不出它的影响。
- **放置位置。** 同样 $\beta = 0.3$ 时，放在奖励中的 KL 比两种损失放置都强得多：pass@1 为 0.34，读取 0.48 次，与起点的距离是除 $\beta = 1.0$ 之外最小的（0.059）。这就是 10.4 节所说的尺度差异：奖励版本对回合中 12 到 14 个策略 token 的对数比率求和而不是取平均，然后进入一个按每组离散程度归一化的优势。
- **熵奖励与温度。** 两者都让熵保持得更高，同时却让策略离起点*更远*，超过不加正则的情况（KL 0.325 和 0.517，对比 0.237），而且都增加了格式错误的调用（每回合 0.28 和 0.25 次，对比 0.04）。熵奖励把质量分散到任何策略都不该输出的 token 上；温度运行中的网络把 logits 变尖锐以补偿 $T = 1.3$，因此在 $T = 1$ 下评估时，它离初始策略比除坍缩的熵奖励之外的任何设置都远。它在训练时的熵（在 $T = 1.3$ 下测量）仍从 0.78 降到了 0.47。

![图 10.3：从 RL 基座智能体出发、40 步 GRPO 中的策略 token 熵（上）与训练批次奖励（下），单个随机种子。左：使用反向 KL 梯度（k3+）的 KL 惩罚，三种系数。右：熵奖励、k3 损失和放在奖励中的 KL。每个面板中的橙色参考线是不加正则的运行。](../../figures/ch10-entropy-curves.svg)

图 10.3 展示了表格背后的动态过程。左列按系数排列各 KL 运行：惩罚越强，熵下降得越晚、越平缓，奖励曲线也越低。右列展示其他放置方式。熵奖励的奖励曲线紧跟不加正则的运行，同时把熵保持在 0.3 附近；$k_3$ 损失与相同系数的 `k3+` 损失走的路径几乎相同（两条绿色曲线，各在一列中）；放在奖励中的 KL 让熵保持得最高、奖励最低。在这个每个任务只有一种正确流程的 gym 上，没有任何正则化手段在 40 步内提高训练奖励：它们换来的是样本的多样性，代价是 pass@1 的下降，且随强度增大。

> **证据。** 本书测得：每种设置一个随机种子，40 步，每步 64 个回合，在 10 个留出任务上评估，每个任务 16 次尝试。小于约 0.08 的 pass@1 差异在这次评估的噪声之内（第 9 章测得默认运行在三个随机种子间的范围为 0.71 到 0.81）。按强度的定性排序（熵、不同序列数、与起点的距离）才是本节演示的机制。

### 主张去掉 KL 的理由

DAPO 去掉了 KL 项，理由是在训练长推理时，“模型分布可能与初始模型显著偏离，因此这一限制并无必要”（“the model distribution can diverge significantly from the initial model, thus this restriction is not necessary”）（[DAPO](https://arxiv.org/abs/2503.14476)，第 2.3 节，2025-03-18）。[第 20 章](20-frontier.md)中公开的代码智能体配方（DeepSWE、SkyRL-Agent）也报告没有使用 KL 损失和熵损失。当奖励是验证器（verifier）时，10.1 节的论证支持这一做法：没有可被利用的学习型奖励，所以需要参考策略的第一个理由不复存在，其余理由（保留未被衡量的能力、累积的信赖域）可以通过在其他任务上评估和 PPO 的逐步裁剪来处理。相反的立场是 ProRL（[2505.24864](https://arxiv.org/abs/2505.24864)，2025-05-30），它训练了 2,000 多步，报告“通过 KL 散度惩罚进行显式正则化提供了更强、更稳定的解决方案”（“explicit regularization via a KL divergence penalty provides a stronger and more stable solution”），有助于“保持熵”（“maintain entropy”），同时指出随着时间推移，“KL 项可能越来越主导损失，导致策略更新逐渐减弱”（“the KL term may increasingly dominate the loss, leading to diminishing policy updates”）。它的解决办法是重置参考策略，10.8 节复现了这一做法。两者都是大规模单轮推理上的报告，不是针对智能体的受控证据。

## 10.8 实践中的设计问题

> **问题：训练早期的熵坍缩。** *症状：*策略 token 的熵在运行的前十分之一内减半，而奖励仍然很低，零优势组的比例攀升，留出任务上的 pass@$k$ 趋近 pass@1。*原因：*10.6 节的协方差：在每个组中，成功的尝试大多由策略已经高概率的 token 组成，所以大多数正优势更新都让分布变尖锐，而步长（学习率、轮次数）决定了变尖锐的速度。*可选方案：*(a) 降低学习率或轮次数，这也会减慢学习；(b) 对初始策略施加 KL 惩罚，把熵保持在参考策略的水平附近，代价是 pass@1；(c) 熵奖励，使用小系数并监控格式错误的输出；(d) clip-higher，只有裁剪生效时才有帮助；(e) 更高的采样温度，能推迟坍缩。*默认做法：*本书的建议是从第一步就记录熵和 pass@$k$，只有当 pass@$k$ 下降或某个已知良好的行为消失时才把坍缩当作问题；先尝试使用 `k3+` 梯度的 (b)，再尝试 (c)。下面的历史记录显示了默认运行坍缩得有多快，以及 (b) 和 (c) 改变了什么。

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

不加正则时，熵在前 10 步内下降了三分之一（0.56 到 0.37），而此时批次奖励仍只有 0.41，到第 20 步降到 0.18；第 9 章的 80 步运行则继续降到 0.08。KL 惩罚和熵奖励都把第 20 步的熵的下降减缓到约 0.30。熵奖励做到这一点时第 39 步的批次奖励相同（0.86），KL 则付出了代价（0.73）。默认轨迹在留出任务上的代价见上面的表格：那两个它从未解出的任务。

> **问题：KL 系数过强或过弱。** *症状：*过强时，记录的 KL 一直接近零，奖励过早进入平台期，参考策略很少表现出的行为（这里是读代码）始终无法占据主导；过弱时，KL 无限增长，这一项没有任何可测量的作用。*原因：*在 10.1 节的最优解处，策略用 $e^{r/\beta}$ 对参考策略做倾斜，因此 $\beta$ 必须与奖励差异相比较，逐 token KL 必须与逐 token 优势相比较。尺度还取决于放置位置：损失中的逐 token 均值与奖励中的逐回合求和相差策略 token 的数量（这里约 13 个，智能体则是数千个）。*可选方案：*(a) 在对数网格上扫描 $\beta$，同时记录 KL；(b) 使用自适应控制器，把 $\beta$ 向目标 KL 调整（verl 的 `AdaptiveKLController`）；(c) 不用 KL，配合留出的回归测试集。*默认做法：*对可验证奖励用 (c)，遵循公开的代码智能体配方；当回路中有学习型奖励或评判模型时用 (b)。三次 `k3+` 运行的 KL 日志展示了这几种状态。

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

$\beta = 0.05$ 时，到参考策略的 KL 到第 20 步攀升到 0.20，接近不加正则的运行最终与起点的距离（表中留出回合上为 0.237），所以这一项几乎没有改变什么。$\beta = 1.0$ 时，它在整个运行中把策略保持在逐 token KL 0.03 以内，而参考策略大约每四五个回合才读一次代码（第 9 章测得每回合 0.21 到 0.24 次读取），读代码始终没有占据主导（每回合 0.53 次读取，pass@1 为 0.23）。$\beta = 0.3$ 是中间状态：KL 在 0.1 附近进入平台期，而奖励继续缓慢上升。

> **问题：参考模型的显存与计算成本。** *症状：*启用 KL 项后训练器显存不足，或者步时间增加了一大块。*原因：*参考模型是第二个完整模型，必须为每个训练 token 产生对数概率。它只需要权重（不需要梯度或优化器状态），但它给一个训练步增加了一次前向传播，而原本一个训练步只包括训练器为 $\pi_{\text{old}}$ 计算对数概率的一次传播（第 9 章会重新计算它）以及一次前向和一次反向传播。*可选方案：*(a) 在 rollout 期间由推理集群计算参考对数概率，随批次一起传输；(b) 单独卸载或切分参考模型；(c) 在 LoRA 训练中，把禁用适配器的基座模型用作参考模型，不需要额外显存；(d) 去掉 KL。*默认做法：*奖励可验证时用 (d)，否则用 (c) 或 (a)。下面是几种模型规模的计算，假设使用 bf16 权重和梯度，以及 fp32 主权重和 Adam 矩（每个训练参数 16 字节）：

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

在每种规模下这个比例都是 12.5%，因为每个参数是两个字节对十六个字节。绝对数值决定了可行性：对 32B 策略，参考模型增加的 60 GiB 必须与切分后的训练状态一起放进每个计算参考对数概率的节点；它还在每个训练 token 上多一次前向传播：把一次反向传播算作约两次前向传播，没有参考模型时一个训练步约为四个前向传播单位（$\pi_{\text{old}}$ 那次传播、前向、反向），参考模型增加第五个，即训练器计算量多 25%。TinyAgent 只有 157k 参数，参考模型在这里没有成本，所以本章每次运行都负担得起；在大规模下，这个决定不是免费的。

> **问题：重置参考策略。** *症状：*使用 KL 惩罚时，记录的 KL 稳定在一个平台上，而奖励不再提高；起约束作用的已经是惩罚项，而不是任务本身。*原因：*参考策略是初始策略，所以训练需要离起点越远，KL 梯度抵消掉的每次更新的部分就越多。*可选方案：*(a) 降低 $\beta$；(b) 把参考策略硬重置为当前策略（并像 ProRL 那样重置优化器状态），使 KL 成为局部信赖域，而不是锚定在起点；(c) 去掉 KL。*默认做法：*如果使用 KL，就按与验证集平台期挂钩的时间表重置，正如 ProRL 报告的做法。我们把 $\beta = 0.3$ 的运行继续训练 20 步两次，一次锚定在起点，一次把参考策略重置为当前策略；两次都重新初始化优化器。

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

锚定在起点时，策略到参考策略的 KL 保持在 0.18，pass@1 升到 0.731。把参考策略重置为当前策略后，到（新）参考策略的 KL 只有 0.029，熵降到 0.135，pass@1 进一步升到 0.787，不同序列更少（4.6 对 6.0）。重置做了 ProRL 所描述的事：它去掉了累积的、指向起点的拉力，让策略得以继续改进，同时也失去了 KL 保留下来的一部分多样性。一个随机种子和 160 个评估回合无法对 0.731 与 0.787 排序；每一列的变化方向才是机制所在。

## 小结

- KL 正则化目标有闭式最优解 $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$：用指数化的奖励对参考策略做倾斜，永远不会离开参考策略的支撑集。熵奖励是参考分布为均匀分布时的同一个目标。
- 在 Schulman 的估计量中，$k_1$ 和 $k_3$ 的取值无偏，并且在 KL 较小时 $k_3$ 的离散程度远低于其他两者。用作损失时，它们给出不同的梯度：$k_1$ 的期望梯度为零，$k_2$ 给出反向 KL 梯度，$k_3$ 给出前向 KL 梯度，其相对反向方向的误差随距离增大。`kl_estimate(..., 'k3+')` 记录 $k_3$ 的值，而按 $k_2$ 求导。
- 对序列而言，逐 token 的 KL 损失忽略了一个 token 如何改变其后各 token 的 KL；放在奖励中的 KL 通过策略梯度给出精确的序列梯度。在 TinyAgent 上，同样的 $\beta$ 下奖励放置的惩罚要强得多，因为它对 token 求和并进入优势。
- 反向 KL 是寻模的，前向 KL 是覆盖质量的；反向 KL 惩罚本身并不保留参考策略的备选。
- 当更新提高已经高概率的 token 时，熵下降，$\Delta H \approx -\mathrm{Cov}(\log\pi, \Delta z)$。在 TinyAgent 上，不加正则的运行在 40 步内从 0.52 nats 降到 0.12 nats，且十个留出任务中有两个在 16 次尝试内从未被解出；KL 惩罚和熵奖励减缓了坍缩并提高了 pass@8，代价是 pass@1；0.1 的熵奖励破坏了动作格式；clip-higher 对熵没有可测量的影响。这些是单随机种子的机制演示。
- 对于可验证奖励，去掉 KL（DAPO、DeepSWE）省掉了一个参考模型和一个需要调的旋钮；保留 KL（ProRL）则以 pass@1 换取熵，并需要重置参考策略，以免训练被锚定在起点。

## 练习

1. 证明给 $k_1$ 损失加上缺失的得分函数项 $\mathbb{E}[\nabla k_1 + \mathrm{sg}(k_1)\nabla\log\pi_\theta]$ 后得到反向 KL 梯度，并证明对 $k_3$ 做同样的修正也是如此。用 `expected_loss_grad` 式的穷举检验两者。为什么没有人实现修正后的 $k_3$ 来代替 `k3+`？
2. 把 $\mathbb{E}_{\pi_\theta}[k_2]$ 展开到 $\ell$ 的三阶，预测 10.2 节中小 KL 时的偏差列。你的近似在 KL 多大时失效？
3. 改变一个假设：只用不加正则和 `KL k3+ 0.05` 两种设置，把 10.7 节重新运行 80 步而不是 40 步。当不加正则的运行的熵达到第 9 章的 0.08 时，它的 pass@8 会低于正则化运行吗？对照三个随机种子的离散范围报告这一差异。
4. 把 10.4 节的两 token 模型扩展到三个 token，并实现一种精确的损失放置：逐 token 的 $k_2$ 损失，加上一个携带每个 token 未来 KL 的 REINFORCE 项。用精确梯度验证它，然后说明 GRPO 训练器必须在其损失中加入哪一部分才能实现它。
5. 在 `train_regularized` 的副本中实现自适应 KL 系数（每步 $\beta \leftarrow \beta(1 + \mathrm{clip}(\mathrm{KL}/\mathrm{KL}_{\text{target}} - 1, \pm 0.2)/10)$）。哪个目标 KL 能让 TinyAgent 学会读代码，同时让 pass@8 至少与不加正则的运行一样高？
6. 把熵奖励限制在每次工具观测之后的第一个策略 token 上（ARPO 认定的高熵位置；[第 20 章](20-frontier.md)）。在平均熵相同的条件下，与均匀熵奖励比较格式错误的调用和 pass@8。
7. 设计评审：某团队用 verl 的默认设置 `use_kl_loss=True`、`kl_loss_type=low_var_kl`、`kl_loss_coef=0.001`，在测试执行奖励上训练 SWE 智能体。记录的 `kl_loss` 为 0.02，策略熵在 200 步内下降了 70%。根据本章，论证这个 KL 项是否在起作用、它计算的是哪个梯度、你接下来会记录什么，以及如果奖励是 LLM 评判而不是测试，你会做哪些改变。
