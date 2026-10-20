# 9. GRPO 与不依赖 critic 的方法

第 8 章用 PPO 裁剪目标复用样本，但 advantage 依赖 critic；在本书的实验中，这个 critic 只能解释一小部分回报方差，却让训练器的显存占用翻了一倍。第 7 章提供了另一条路：对同一任务采样多次，让这些尝试互为基线。**组相对策略优化**（GRPO）把组基线与 PPO 的裁剪目标结合起来，省去 critic。它及其变体在公开的推理、编程模型训练中应用广泛，相关来源见第 20 章。

本章围绕 GRPO 构建全书的主训练循环 `train_rl`，并把每个已发表的变体（Dr. GRPO、DAPO、RLOO、GSPO）表达为这一个循环的若干配置字段，而不是各自独立的程序。第 10 章到第 19 章都用它训练。随后我们在 BugFix gym 上运行它，遇到一个任何目标函数都修不好的问题：智能体（agent）学会了不读代码就通过训练任务。用行为统计诊断这个问题、并通过修改*环境*堵住捷径，是 9.4 节的核心，它也决定了全书其余部分所用的任务设置。本章最后对各变体做小规模的受控消融实验（ablation），并如实标注它们的性质：这是在一个 158k 参数模型上的机制演示（mechanism demonstration），不是基准评测证据。

```python
#@save
import copy
import dataclasses
import random
from dataclasses import dataclass
import torch
import agentrl as arl
```

```python
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
base = arl.make_base_agent(tok)
```

## 9.1 DeepSeekMath 中的 GRPO

DeepSeekMath（Shao et al., 2024）针对一个问题 $q$，用从 $\pi_{\text{old}}$ 采样的 $G$ 个输出 $o_1, \dots, o_G$ 定义 GRPO：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_{i},\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_{i}\big) - \beta\, \hat{\mathbb{D}}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]_{i,t}\Big)\Big],
$$

其中结果优势 $\hat A_i = (r_i - \mathrm{mean}(\mathbf{r}))/\mathrm{std}(\mathbf{r})$ 由输出 $i$ 的每个 token 共享，逐 token 的 KL 估计为 $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$，即 $k_3$ 估计量（estimator）。对照前几章来读，它由四个选择组成：

1. **组采样。** 同一步内对每个任务做 $G$ 次尝试（DeepSeekMath 报告 $G = 64$）。
2. **不依赖 critic 的优势。** 以组均值为基线、组标准差为尺度：即 7.5 节的 `group_advantages`。
3. **PPO 的逐 token 裁剪比率**（8.2 节），聚合方式是先在每个序列内对 token 取平均，再对序列取平均：即 6.6 节的 `seq_mean_token_mean`。
4. **在损失中加入对冻结参考策略（reference policy）的 KL 惩罚**，系数为 $\beta$（报告值为 0.04），而不是像 RLHF 风格的 PPO 那样作为逐 token 的奖励。

DeepSeekMath 还报告“策略模型在每个探索阶段之后只做一次更新”（“the policy model only has a single update following each exploration stage”）。只做一次更新时，计算梯度那一刻 $\pi_\theta = \pi_{\text{old}}$，所有比率都是 1，裁剪不起作用；此时目标函数就是带 KL 项的组基线策略梯度（policy gradient）。许多实现则把每个批次切成若干小批次（minibatch），这使得除第一个之外的小批次都略微 off-policy，裁剪也就有了用武之地。因此，实践中的“GRPO”往往包含多种设置，本书把它们写成字段。

`RLConfig` 汇集了 `train_rl` 读取的全部设置。默认值是本书在 9.4 节最终得到的那个 gym 版本上的 GRPO 配置。属于某个变体的字段在 9.5 节介绍。

```python
#@save
@dataclass
class RLConfig:
    """Settings of train_rl; defaults are the book's GRPO on the anonymized one-edit gym (Section 9.4)."""
    # sampling
    group_size: int = 8              # G attempts per task
    prompts_per_step: int = 8        # tasks per step; a batch is G * prompts_per_step episodes
    temperature: float = 1.0
    max_turns: int = 6
    # environment (Section 9.4)
    rename: bool = True              # give every episode a random, uninformative function name
    max_edits: object = 1            # edits applied per episode; None = unlimited (the Chapter 4 gym)
    reward: object = 'hidden'        # 'hidden', 'public', or a callable(info) -> float
    # advantages
    adv_mode: str = 'grpo'           # 'grpo' (mean / std), 'mean' (Dr. GRPO), 'rloo'
    overlong_buffer: int = 0         # DAPO-style linear penalty over the last turns of the budget; 0 = off
    dynamic_sampling: bool = False   # DAPO: keep only groups whose rewards differ, resample to refill
    max_sample_rounds: int = 4
    # loss and update
    ratio_level: str = 'token'       # 'token' (PPO, GRPO) or 'sequence' (GSPO)
    clip_low: float = 0.2
    clip_high: float = 0.2
    loss_mode: str = 'token_mean'    # any arl.aggregate_loss mode
    norm_const: object = None        # for loss_mode='constant'; None = rows * max_turns * MAX_CALL_TOKENS
    kl_coef: float = 0.0             # beta of a KL penalty to a frozen reference policy
    kl_kind: str = 'k3'              # 'k1', 'k2', or 'k3' (Chapter 10)
    lr: float = 1e-3
    epochs: int = 1
    minibatches: int = 2
    max_grad_norm: float = 1.0
    seed: int = 0
```

有三个默认值偏离了 DeepSeekMath，需要说明理由。`token_mean` 取代了 `seq_mean_token_mean`，原因是 [6.6 节](06-policy-gradient.md#66-损失聚合)讨论的长度问题，9.7 节会再回到这一点。`kl_coef` 为 0，与 DAPO 以及第 20 章中公开的代码智能体配方一致；[第 10 章](10-kl-entropy-regularization.md)专门研究 KL 项。另外，每个批次切成 2 个小批次，这样裁剪会被触发，但更新仍然接近 on-policy（8.6 节）。

## 9.2 训练循环

`train_rl` 由五个部分组装而成，每部分都单独保存，以便后面的章节替换其中之一：环境工厂与组采样器、行为统计、优势、裁剪更新，以及循环本身。

### 按组采样

`make_env` 按配置的描述构建一个环境。它默认构建的环境 `EditBudgetEnv` 在 9.4 节定义；当 `max_edits=None` 时，它原样返回第 4 章的 `BugFixEnv`。`sample_groups` 把任务排成连续的、每组 $G$ 个的布局，这正是 `group_advantages` 所期望的布局，并用第 5 章的 `rollout_batch` 让所有回合（episode）同步推进。设置了 `cfg.rename` 时，每个回合的任务都会经过 `randomize_names`（9.4 节），它使用一个由 torch 生成器播种的 Python 随机流，因此整个步骤可以由一个随机种子完全复现。`env_factory` 参数是接入自定义环境和奖励的钩子（[第 11 章](11-rewards-and-verifiers.md)会用到）。

```python
#@save
def make_env(cfg):
    """One fresh environment as cfg describes it: budgets, reward, and edit limit."""
    if cfg.max_edits is None:
        return arl.BugFixEnv(cfg.max_turns, cfg.reward)
    return EditBudgetEnv(cfg.max_turns, cfg.reward, cfg.max_edits)


def sample_groups(model, env_factory, tasks, tok, cfg, generator):
    """cfg.group_size consecutive episodes per task -> list of len(tasks) * G Trajectory."""
    names = random.Random(int(torch.randint(2 ** 31, (1,), generator=generator)))   # renaming stream
    batch_tasks = [task for task in tasks for _ in range(cfg.group_size)]
    if cfg.rename:
        batch_tasks = [randomize_names(task, names) for task in batch_tasks]
    envs = [env_factory() for _ in batch_tasks]
    return arl.rollout_batch(model, envs, batch_tasks, tok, cfg.max_turns, cfg.temperature, generator)
```

### 智能体做了什么

仅凭奖励无法区分读代码的智能体和猜得准的智能体，而 9.4 节恰恰依赖于把两者区分开。`behavior_stats` 按回合报告：隐藏测试通过率、奖励、`read`、`test` 调用次数和实际生效的 edit 调用次数、第一次生效的编辑是否为该任务的正确操作、轮数，以及策略 token 数。每个任务的正确操作取自第 4 章的任务类型表，因此 9.4 节中改过名的任务仍按其原始身份计分。

```python
#@save
FIX_OP = {f"{family}/{variant['f']}": spec['fix_op']
          for family, spec in arl.FAMILIES.items() for variant in spec['variants']}


def behavior_stats(trajs):
    """Per-episode averages: pass_rate, reward, reads, tests, edits, first_edit_right, turns, length."""
    n = len(trajs)
    def per_episode(call):
        return sum(t.info['actions'].count(call) for t in trajs) / n
    return dict(pass_rate=sum(bool(t.info['submitted'] and t.info['hidden_pass']) for t in trajs) / n,
                reward=sum(t.reward for t in trajs) / n,
                reads=per_episode('<call> read </call>'), tests=per_episode('<call> test </call>'),
                edits=sum(len(t.info['ops']) for t in trajs) / n,
                first_edit_right=sum(t.info['ops'][:1] == [FIX_OP[t.info['task']]] for t in trajs) / n,
                turns=sum(t.info['n_turns'] for t in trajs) / n,
                length=sum(float(t.action_mask.sum()) for t in trajs) / n)
```

### 优势与 KL 项

`compute_advantages` 把一个整理好的批次转换为每个回合一个优势。奖励可以先经过 DAPO 的超长惩罚塑形（9.5 节）；三种优势模式都是第 7 章的估计量。`kl_penalty` 是用 $\pi_\theta$ 采样的 token 对 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ 做的逐 token 估计，把对数比率 $\log(\pi_{\text{ref}}/\pi_\theta)$ 记为 $\ell$：$k_1 = -\ell$，$k_2 = \ell^2/2$，$k_3 = e^{\ell} - 1 - \ell$。[第 10 章](10-kl-entropy-regularization.md)推导它们的偏差、方差，以及把它们用作损失时产生的梯度。

```python
#@save
def compute_advantages(batch, cfg):
    """One advantage per episode [B] from batch['rewards'] laid out in groups of cfg.group_size."""
    rewards = batch['rewards']
    if cfg.overlong_buffer:                        # linear penalty from 0 to -1 over the last turns of the budget
        over = (batch['n_turns'] - (cfg.max_turns - cfg.overlong_buffer)).clamp(min=0)
        rewards = rewards - (over / cfg.overlong_buffer).clamp(max=1.0)
    if cfg.adv_mode == 'rloo':
        return arl.rloo_advantages(rewards, cfg.group_size)
    if cfg.adv_mode in ('grpo', 'mean'):
        return arl.group_advantages(rewards, cfg.group_size, normalize_std=cfg.adv_mode == 'grpo')
    raise ValueError(f'unknown adv_mode {cfg.adv_mode!r}')


def kl_penalty(logp, ref_logp, kind='k3'):
    """Per-token estimate of KL(pi_theta || pi_ref) on tokens sampled from pi_theta: [B, T] -> [B, T]."""
    log_ratio = ref_logp.detach() - logp                       # log pi_ref / pi_theta
    if kind == 'k1':
        return -log_ratio
    if kind == 'k2':
        return 0.5 * log_ratio ** 2
    if kind == 'k3':
        return log_ratio.exp() - 1 - log_ratio
    raise ValueError(f'unknown KL estimator {kind!r}')
```

### 更新

`policy_loss` 按比率的层级分派：逐 token 比率交给第 8 章的 `ppo_policy_loss`，序列级比率交给 `gspo_policy_loss`（9.5 节）。`rl_update` 就是去掉 critic 的第 8 章 PPO 阶段。它通过对批次做一次无梯度前向传播来冻结 $\pi_{\text{old}}$，在给定参考模型时也冻结 $\pi_{\text{ref}}$（按 8.6 节的建议，训练器在采样温度下重新计算 `old_logp`），然后运行 `cfg.epochs` 遍、每遍 `cfg.minibatches` 个裁剪更新步。KL 项在策略 token 上取平均。熵（entropy）由更新本来就有的 logits 计算，所以记录它不需要额外代价。

```python
#@save
def policy_loss(logp, old_logp, advantages, mask, cfg):
    """The clipped surrogate at token or sequence level, as cfg says -> (loss, stats)."""
    if cfg.ratio_level == 'token':
        norm = cfg.norm_const or len(logp) * cfg.max_turns * arl.MAX_CALL_TOKENS
        return arl.ppo_policy_loss(logp, old_logp, advantages, mask, cfg.clip_low, cfg.clip_high,
                                   cfg.loss_mode, norm_const=norm)
    if cfg.ratio_level == 'sequence':
        return gspo_policy_loss(logp, old_logp, advantages, mask, cfg.clip_low, cfg.clip_high)
    raise ValueError(f'unknown ratio_level {cfg.ratio_level!r}')


def rl_update(model, optimizer, batch, advantages, cfg, ref_model=None, generator=None):
    """cfg.epochs passes of cfg.minibatches clipped steps on one batch -> dict of mean stats."""
    ids, mask = batch['ids'], batch['action_mask']                          # [B, T], [B, T-1]
    with torch.no_grad():                                                   # pi_old (and pi_ref), frozen
        old_logp = arl.selected_log_probs(model(ids)[:, :-1] / cfg.temperature, ids[:, 1:])
        ref_logp = None if ref_model is None else \
            arl.selected_log_probs(ref_model(ids)[:, :-1] / cfg.temperature, ids[:, 1:])
    logs = []
    for _ in range(cfg.epochs):
        for rows in torch.randperm(len(ids), generator=generator).chunk(cfg.minibatches):
            logits = model(ids[rows])[:, :-1] / cfg.temperature             # [b, T-1, V]
            logp = arl.selected_log_probs(logits, ids[rows, 1:])
            loss, stats = policy_loss(logp, old_logp[rows], advantages[rows], mask[rows], cfg)
            if ref_logp is not None and cfg.kl_coef:
                kl = arl.masked_mean(kl_penalty(logp, ref_logp[rows], cfg.kl_kind), mask[rows])
                loss, stats['kl_ref'] = loss + cfg.kl_coef * kl, kl.item()
            optimizer.zero_grad()
            loss.backward()
            stats['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            optimizer.step()
            stats['entropy'] = arl.masked_mean(arl.entropy_from_logits(logits.detach()), mask[rows]).item()
            logs.append(stats)
    return {k: sum(s[k] for s in logs) / len(logs) for k in logs[0]}
```

### 循环

`sample_step` 抽取 `prompts_per_step` 个互不相同的训练任务，并为它们采样组。启用动态采样时，它只保留奖励不全相同的组，并继续采样更多任务，直到批次填满或达到轮数上限；9.5 节解释这一选择。它同时返回保留的回合和所有采样过的回合，因为一个训练步的成本取决于采样了多少。

```python
#@save
def sample_step(model, env_factory, tasks, tok, cfg, generator):
    """Sample one step's groups -> (kept episodes for the update, all sampled episodes)."""
    want, kept, sampled = cfg.prompts_per_step * cfg.group_size, [], []
    for _ in range(cfg.max_sample_rounds if cfg.dynamic_sampling else 1):
        order = torch.randperm(len(tasks), generator=generator)[:cfg.prompts_per_step].tolist()
        trajs = sample_groups(model, env_factory, [tasks[i] for i in order], tok, cfg, generator)
        sampled += trajs
        for g in range(0, len(trajs), cfg.group_size):
            group = trajs[g:g + cfg.group_size]
            if not cfg.dynamic_sampling or len({t.reward for t in group}) > 1:
                kept += group
        if len(kept) >= want:
            break
    return kept[:want], sampled
```

`train_rl` 运行 `steps` 次“采样、计算优势、更新、记录”的迭代。它原地训练 `model`，并返回一个历史记录，每步一个字典：所有采样回合的行为统计；更新的统计量（`clip_frac`、`approx_kl`、比率范围、`entropy`、`grad_norm`，使用参考模型时还有 `kl_ref`）；采样的回合数；奖励全部相同的采样组所占比例（`zero_adv_frac`）；以及每隔 `eval_every` 步在 `eval_tasks` 上、每个任务 4 个回合的通过率。当 `kl_coef > 0` 且没有传入参考模型时，参考模型是初始模型的一份冻结副本。

```python
#@save
def train_rl(model, tasks, tok, cfg, steps, log_every=0, eval_tasks=None, eval_every=20, env_factory=None,
             ref_model=None):
    """Group-based RL on tasks; trains model in place -> history, one dict of statistics per step."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env_factory = env_factory or (lambda: make_env(cfg))
    if cfg.kl_coef and ref_model is None:
        ref_model = copy.deepcopy(model).eval()                     # frozen pi_ref = the starting policy
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for step in range(steps):
        kept, sampled = sample_step(model, env_factory, tasks, tok, cfg, gen)
        G = cfg.group_size
        flat = [len({t.reward for t in sampled[g:g + G]}) == 1 for g in range(0, len(sampled), G)]
        record = dict(step=step, n_episodes=len(sampled), zero_adv_frac=sum(flat) / len(flat),
                      **behavior_stats(sampled))
        if kept:
            batch = arl.collate(kept, tok.pad_id)
            batch['n_turns'] = torch.tensor([float(t.info['n_turns']) for t in kept])
            record.update(rl_update(model, optimizer, batch, compute_advantages(batch, cfg), cfg, ref_model, gen))
        if eval_tasks is not None and (step % eval_every == 0 or step == steps - 1):
            record['eval_pass'] = evaluate(model, eval_tasks, tok, 4, cfg.temperature,
                                           torch.Generator().manual_seed(step), cfg, env_factory)['pass_rate']
        history.append(record)
        if log_every and (step % log_every == 0 or step == steps - 1):
            held = f" | held-out {record['eval_pass']:.3f}" if 'eval_pass' in record else ''
            print(f"step {step:3d} | reward {record['reward']:.3f} | reads {record['reads']:.2f} | edits"
                  f" {record['edits']:.2f} | entropy {record.get('entropy', 0):.3f}{held}")
    return history
```

`evaluate` 在与训练相同的环境设置下，对每个任务采样 `n_samples` 个回合，返回行为统计。默认是温度 1 下采样的 pass@1；[第 17 章](17-evaluation.md)会认真讨论评估。

```python
#@save
def evaluate(model, tasks, tok, n_samples=1, temperature=1.0, generator=None, cfg=None, env_factory=None):
    """n_samples episodes per task in cfg's environment -> behavior_stats dict (pass_rate, reads, ...)."""
    cfg = dataclasses.replace(cfg or RLConfig(), group_size=n_samples, temperature=temperature)
    generator = generator or torch.Generator().manual_seed(0)
    trajs = sample_groups(model, env_factory or (lambda: make_env(cfg)), tasks, tok, cfg, generator)
    return behavior_stats(trajs)
```

## 9.3 在原始 gym 上运行 GRPO

我们先完全照搬 [第 6 章](06-policy-gradient.md#67-用-reinforce-训练-tinyagent)的设置：基座智能体（base agent），第 4 章的 gym，保留函数名、不限编辑次数。唯一的改变是更新方式，换成 GRPO，每组 8 个、每步 8 个任务。`rename=False` 和 `max_edits=None` 选择原始 gym。

```python
original = RLConfig(rename=False, max_edits=None)
policy_a = copy.deepcopy(base)
history_a = train_rl(policy_a, train_tasks, tok, original, steps=80, log_every=20)
```

Output:

```text
step   0 | reward 0.156 | reads 0.45 | edits 2.22 | entropy 0.533
step  20 | reward 0.516 | reads 0.08 | edits 1.94 | entropy 0.373
step  40 | reward 0.812 | reads 0.00 | edits 1.55 | entropy 0.208
step  60 | reward 0.969 | reads 0.00 | edits 1.80 | entropy 0.125
step  79 | reward 0.938 | reads 0.00 | edits 1.88 | entropy 0.099
```

```python
def compare(name, model, cfg, n=16):
    """Train vs held-out behavior of model in cfg's environment, 16 episodes per task."""
    for split, tasks in [('train', train_tasks), ('held-out', heldout_tasks)]:
        s = evaluate(model, tasks, tok, n, generator=torch.Generator().manual_seed(7), cfg=cfg)
        print(f"{name:<20}{split:<9} pass {s['pass_rate']:.3f} | reads {s['reads']:.2f} | edits {s['edits']:.2f}"
              f" | first edit right {s['first_edit_right']:.2f} | tests {s['tests']:.2f}")

compare('base agent', base, original)
compare('GRPO, original gym', policy_a, original)
```

Output:

```text
base agent          train     pass 0.144 | reads 0.27 | edits 1.96 | first edit right 0.19 | tests 0.29
base agent          held-out  pass 0.150 | reads 0.21 | edits 1.73 | first edit right 0.16 | tests 0.21
GRPO, original gym  train     pass 0.981 | reads 0.00 | edits 1.57 | first edit right 0.71 | tests 0.01
GRPO, original gym  held-out  pass 0.400 | reads 0.00 | edits 2.64 | first edit right 0.13 | tests 0.00
```

训练收敛得很快：批次奖励在第 60 步超过 0.9，策略 token 的熵从 0.53 降到 0.10。训练后的策略在训练任务上通过 98% 的回合，在留出任务（held-out tasks）上通过 40%，基座智能体则是 15%。训练任务上大幅提升而留出任务上提升小得多，这是记忆的特征，而行为统计的各列说明了记住的是什么。策略从不读代码，也几乎从不测试。它的第一次编辑在 71% 的训练回合中是正确操作，在留出回合中只有 13%，而在五个修复操作中均匀地猜一个也能达到 20%。在留出任务上，它退回到堆叠编辑，每个回合平均生效 2.6 次编辑。这是用另一种估计量重现了第 6 章的发现：学到什么，取决于环境和初始策略，而不取决于 REINFORCE。

## 9.4 环境设计中的捷径特征

**捷径特征**（shortcut feature）是这样一种输入：它在训练分布上能预测奖励，却不携带任务本应要求的信息。优化器之所以会找到捷径，是因为捷径比预期的技能更便宜。行为统计指向了 BugFix gym 的第一个捷径：issue 文本。

```python
for task in [train_tasks[0], train_tasks[4], heldout_tasks[0]]:
    print(f'{task.split:<8} {task.issue!r:<46} -> fix {task.fix_op}')
```

Output:

```text
train    'issue : clamp returns wrong results'          -> fix swap_minmax
train    'issue : sum_to returns wrong results'         -> fix fix_range_end
heldout  'issue : pin returns wrong results'            -> fix swap_minmax
```

issue 里写出了函数名，而在训练集中每个函数名恰好只出现在一个任务里，所以函数名就能确定任务，进而确定它的修复操作。一个把 `issue :` 之后第一个词映射到某个操作的策略，根本不需要从环境获取任何信息。在留出任务上，函数名都是它从未训练过的词，这个映射便失去了作用。真实基准也会以同样的方式泄漏（leakage）。SWE-Bench+（Aleithan et al., 2024）报告，在它检查的 SWE-bench 成功补丁中，有 32.67% 的解法“直接写在 issue 报告中”（“directly provided in the issue report”）；SWE-Bench Illusion（Liang et al., 2025）报告，模型仅凭 issue 文本就能以最高 76% 的准确率识别出有 bug 的文件路径，而在基准之外的仓库上最高降到 53%，作者将此解读为记忆（[第 20 章](20-frontier.md)）。两种情况下，issue 都携带了真实工程师必须通过读代码才能获得的信息。

修复应当放在环境里：让捷径不再携带信息。`randomize_names` 把任务 issue 和源码中的函数改名，新名字从 gym 全部 30 个函数名中随机抽取，并排除源码中已出现的词。所有名字都在第 5 章的词表（vocabulary）内，因此不需要修改分词器（tokenizer）。对每个回合独立地应用改名之后，函数名就成了噪声：每个名字与每个任务类型同样频繁地出现，策略再也无法凭它预测操作。

```python
#@save
FN_NAMES = sorted({variant['f'] for spec in arl.FAMILIES.values() for variant in spec['variants']})


def randomize_names(task, rng):
    """Copy of task whose function is renamed to a random gym function name not already in its source."""
    taken = set(arl.compact_source(task.source).split())
    new = rng.choice([name for name in FN_NAMES if name not in taken])
    return dataclasses.replace(task, fn_name=new, issue=task.issue.replace(task.fn_name, new),
                               source=task.source.replace(f'def {task.fn_name}(', f'def {new}('))
```

任务保留其 `name`，所以日志和 `FIX_OP` 仍能识别它。我们把 `clamp` 任务改名两次，并检查每个任务的每个改名后的 issue 和源码在编码时都没有未知 token。

```python
rng = random.Random(0)
for _ in range(2):
    renamed = randomize_names(train_tasks[0], rng)
    print(renamed.issue, '|', arl.compact_source(renamed.source).split('\n')[0])
unknown = sum(tok.unk_id in tok.encode(t.issue + ' ' + arl.compact_source(t.source))
              for t in arl.make_tasks('all') for t in [randomize_names(t, rng) for _ in range(20)])
print('renamed tasks with unknown tokens:', unknown)
```

Output:

```text
issue : tally returns wrong results | def tally ( value , lower , upper ) :
issue : pin returns wrong results | def pin ( value , lower , upper ) :
renamed tasks with unknown tokens: 0
```

我们从同一个基座智能体重新训练，打开改名，其他一切不变。

```python
renamed_cfg = RLConfig(max_edits=None)                       # rename=True is the default
policy_b = copy.deepcopy(base)
history_b = train_rl(policy_b, train_tasks, tok, renamed_cfg, steps=80)
compare('GRPO, renamed', policy_b, renamed_cfg)
```

Output:

```text
GRPO, renamed       train     pass 0.591 | reads 0.01 | edits 2.54 | first edit right 0.22 | tests 0.00
GRPO, renamed       held-out  pass 0.475 | reads 0.00 | edits 2.51 | first edit right 0.18 | tests 0.00
```

### 第二个捷径

改名把训练与留出通过率之间的差距从 0.58 缩小到 0.12（0.59 对 0.48），但并没有带来读代码的行为：每个回合 0.01 次读取。第一次编辑现在只在 22% 和 18% 的回合中正确，相当于均匀猜测的水平，而策略转而在每个回合中生效约 2.5 次编辑。它切换到了第 6 章发现的另一种策略：**堆叠**编辑。错误的操作通常不改变源码，或者只做无害的改动，因此按固定顺序应用几个操作，就能在不看代码的情况下修复好几个任务类型。这能走多远？`blind_pass_rate` 对每个任务应用一个固定的操作序列，然后运行隐藏测试。

```python
import itertools
def blind_pass_rate(ops, tasks):
    """Hidden pass rate of applying ops in order to every task without reading, then submitting."""
    passed = 0
    for task in tasks:
        source = task.source
        for op in ops:
            source = arl.EDIT_OPS[op](source, task.public_tests)
        passed += all(ok for ok, _ in arl.run_tests(source, task.hidden_tests, task.fn_name))
    return passed / len(tasks)

all_tasks = train_tasks + heldout_tasks
for n_ops in [1, 2, 3]:
    best = max((blind_pass_rate(ops, all_tasks), ops) for ops in itertools.product(arl.FIX_OPS, repeat=n_ops))
    print(f'best blind sequence of {n_ops} edit(s): pass rate {best[0]:.2f} with {" -> ".join(best[1])}')
```

Output:

```text
best blind sequence of 1 edit(s): pass rate 0.20 with swap_minmax
best blind sequence of 2 edit(s): pass rate 0.40 with swap_minmax -> fix_range_end
best blind sequence of 3 edit(s): pass rate 0.60 with swap_minmax -> fix_range_end -> fix_comparison
```

一个从不读代码的策略，用三次编辑加一次提交就能通过全部任务的 60%，远在六轮的预算之内。改名去掉了 issue 中的信息，而编辑菜单仍然允许智能体逐个枚举。这是第二个捷径特征，它是动作空间的性质，而不是观测的性质：错误的动作没有代价。真实环境很少这么宽容。改错代码的补丁通常会弄坏其他测试，而基准只对最终的一个补丁打分。gym 通过限制一个回合可以生效的编辑次数来模拟这一点。`EditBudgetEnv` 拒绝超过 `max_edits` 的编辑：这次调用消耗一轮，返回 `no change`，源码保持不变。

```python
#@save
class EditBudgetEnv(arl.BugFixEnv):
    """BugFixEnv that applies at most max_edits edits per episode; later edits use a turn and change nothing."""
    def __init__(self, max_turns=6, reward='hidden', max_edits=1):
        super().__init__(max_turns, reward)
        self.max_edits = max_edits

    def step(self, action_text):
        call = arl.parse_call(action_text)
        refused = call is not None and call[0] == 'edit' and call[1] in arl.EDIT_OPS \
            and len(self.ops) >= self.max_edits
        if not refused:
            return super().step(action_text)
        self.turn += 1
        self.done = self.turn >= self.max_turns
        if not self.done:
            return arl.MESSAGES['unchanged'], 0.0, False, {'turn': self.turn, 'event': 'refused'}
        info = self.final_info(submitted=False)
        return arl.MESSAGES['unchanged'], self.reward_fn(info), True, info
```

只允许一次编辑时，最好的盲目策略就是最好的单个操作，按上表为 0.20。现在任何高于 0.2 的通过率都需要来自环境的信息：`read` 的观测，或者编辑之前的 `test` 输出。我们再一次从基座智能体重新训练，这次使用本书的默认配置：函数改名、一次编辑。

```python
policy_c = copy.deepcopy(base)
history_c = train_rl(policy_c, train_tasks, tok, RLConfig(), steps=80, log_every=20)
compare('GRPO, one edit', policy_c, RLConfig())
```

Output:

```text
step   0 | reward 0.141 | reads 0.30 | edits 0.95 | entropy 0.510
step  20 | reward 0.156 | reads 0.02 | edits 0.97 | entropy 0.316
step  40 | reward 0.234 | reads 0.02 | edits 1.00 | entropy 0.288
step  60 | reward 0.344 | reads 0.02 | edits 1.00 | entropy 0.195
step  79 | reward 0.203 | reads 0.02 | edits 1.00 | entropy 0.228
GRPO, one edit      train     pass 0.194 | reads 0.01 | edits 0.97 | first edit right 0.20 | tests 0.01
GRPO, one edit      held-out  pass 0.150 | reads 0.03 | edits 0.97 | first edit right 0.15 | tests 0.01
```

### 更根本的探索问题

两个捷径都关闭之后，策略也没有学会读代码。它的通过率停在盲目上限，训练任务 0.19、留出任务 0.15，读取次数从第一步的每回合 0.30 次降到 0.02 次。随着策略把唯一的那次猜测从一个操作换到另一个，批次奖励在 0.14 到 0.34 之间游移。环境现在*要求*读代码，而策略梯度仍然找不到它。原因就在基座智能体自己的样本里。RL 是对策略已有的行为重新加权：一个动作序列只有在获得比同组其他尝试更高的奖励时才会提高概率。我们测量基座智能体在第一次编辑前读代码的频率，以及之后那次编辑正确的频率。

```python
def read_then_edit(model, n=16, seed=3):
    """Share of episodes that read before their first edit, and the first-edit accuracy of those episodes."""
    cfg = RLConfig(group_size=n)
    trajs = sample_groups(model, lambda: make_env(cfg), train_tasks, tok, cfg, torch.Generator().manual_seed(seed))
    readers = [t for t in trajs if t.info['ops'] and '<call> read </call>' in
               t.info['actions'][:t.info['actions'].index(f"<call> edit {t.info['ops'][0]} </call>")]]
    right = sum(t.info['ops'][0] == FIX_OP[t.info['task']] for t in readers) / max(len(readers), 1)
    return len(readers) / len(trajs), right

share, right = read_then_edit(base)
print(f'base agent: reads before its first edit in {share:.2f} of episodes; that edit is right {right:.2f} of the time')
```

Output:

```text
base agent: reads before its first edit in 0.07 of episodes; that edit is right 0.25 of the time
```

基座智能体只在 7% 的回合中先读后改，而此时那次编辑的正确率为 25%，接近猜测的 20%（约 20 个回合），因为它的示范来自 `random` 和 `blind` 两种玩法，读取之后从未跟随过有依据的选择。因此读与不读得到的奖励相同，读取的优势期望为零，噪声和改进更快的其他行为会把它淘汰掉。这是一个探索（exploration）失败，关闭捷径无法修复它：环境可以让一项技能变得*必要*，但 RL 只能放大策略偶尔已经表现出来的技能。

预训练语言模型的处境比这个基座智能体好：它见过代码如何被阅读和使用，有时会出于正确的理由做出正确的事。本书中与之对应的，是一个示范里包含少量根据观测选择正确操作的行为的基座智能体。`make_rl_base` 在每四组 `random` 和四组 `blind` 示范之外，加入一组来自第 4 章 `oracle` 的示范，这样每九条示范中有一条先读代码，再应用正确操作。

```python
#@save
def make_rl_base(tok, seed=0):
    """The RL starting point from Chapter 9 on: the base agent with 1 demonstration set in 9 from the oracle."""
    return arl.make_base_agent(tok, seed, kinds=('random', 'blind') * 4 + ('oracle',))
```

```python
rl_base = make_rl_base(tok)
share, right = read_then_edit(rl_base)
print(f'RL base agent: reads before its first edit in {share:.2f} of episodes; that edit is right {right:.2f} of the time')
compare('RL base agent', rl_base, RLConfig())
```

Output:

```text
RL base agent: reads before its first edit in 0.12 of episodes; that edit is right 0.41 of the time
RL base agent       train     pass 0.169 | reads 0.21 | edits 0.91 | first edit right 0.19 | tests 0.27
RL base agent       held-out  pass 0.156 | reads 0.24 | edits 0.93 | first edit right 0.19 | tests 0.32
```

新的起点与旧的起点通过率相同，训练任务 0.17、留出任务 0.16，所以它并不是更强的智能体。差别在于一个条件概率：它在 12% 的回合中先读，而之后的编辑有 41% 正确，是猜测的两倍。这足以让组基线发现信号。本章的主实验用默认配置训练它，并每 20 步在留出任务上评估一次。

```python
policy = copy.deepcopy(rl_base)
history = train_rl(policy, train_tasks, tok, RLConfig(), steps=80, log_every=20, eval_tasks=heldout_tasks)
compare('GRPO, default', policy, RLConfig())
```

Output:

```text
step   0 | reward 0.172 | reads 0.31 | edits 0.97 | entropy 0.565 | held-out 0.100
step  20 | reward 0.844 | reads 1.02 | edits 0.98 | entropy 0.184 | held-out 0.625
step  40 | reward 0.938 | reads 1.00 | edits 0.95 | entropy 0.107 | held-out 0.825
step  60 | reward 0.844 | reads 1.00 | edits 1.00 | entropy 0.092 | held-out 0.825
step  79 | reward 0.938 | reads 0.98 | edits 0.97 | entropy 0.079 | held-out 0.850
GRPO, default       train     pass 0.922 | reads 1.00 | edits 0.98 | first edit right 0.92 | tests 0.83
GRPO, default       held-out  pass 0.838 | reads 1.02 | edits 0.99 | first edit right 0.84 | tests 0.79
```

这一次，读代码在 20 步之内就扩散到整个批次，策略几乎在每个回合都会读。它的第一次编辑在 92% 的训练回合和 84% 的留出回合中正确，留出通过率随训练通过率一起上升：周期性评估中从第 0 步的 0.10 升到第 79 步的 0.85，160 个回合的评估中为 0.84。这 80 步在检查主机上耗时约 10 到 15 秒。最常见的留出回合展示了学到的流程。

```python
from collections import Counter
trajs = sample_groups(policy, lambda: make_env(RLConfig()), heldout_tasks, tok, RLConfig(group_size=16),
                      torch.Generator().manual_seed(11))
calls = Counter(' -> '.join(a.split()[1] if len(a.split()) == 3 else a.split()[2] for a in t.info['actions'])
                for t in trajs)
for sequence, count in calls.most_common(4):
    print(f'{count:3d} x {sequence}')
```

Output:

```text
 31 x read -> fix_init -> test -> submit
 30 x read -> fix_range_end -> test -> submit
 28 x read -> swap_minmax -> test -> submit
 28 x read -> fix_comparison -> test -> submit
```

最后做一个对照：既然基座智能体已经能够探索，环境修复还有必要吗？我们在原始 gym 上（保留函数名、不限编辑次数）训练 RL 基座智能体。

```python
policy_e = copy.deepcopy(rl_base)
train_rl(policy_e, train_tasks, tok, original, steps=80)
compare('RL base, original', policy_e, original)
```

Output:

```text
RL base, original   train     pass 0.978 | reads 1.00 | edits 1.05 | first edit right 0.99 | tests 0.98
RL base, original   held-out  pass 0.838 | reads 1.02 | edits 1.21 | first edit right 0.78 | tests 0.88
```

![图 9.1：五种设置下 80 步 GRPO 之后的留出行为：基座智能体分别在原始 gym、函数改名、改名加一次编辑的设置下；RL 基座智能体在改名加一次编辑（本书默认）和原始 gym 的设置下。柱形：留出通过率、每回合读取次数和第一次编辑的准确率。](../../figures/ch09-shortcuts.svg)

所有高频的留出回合都是 oracle 的流程：读代码，应用源码所需的操作，测试，提交。只有一次编辑时 `test` 调用并无必要，因为它的结果无法再被利用，策略保留它只是因为没有任何东西惩罚它。这再次印证了第 7 章的观点：结果优势也会为无用的步骤记功。

这个对照提供的信息出乎我们设计它时的预期。RL 基座智能体在原始 gym 上也学会了读代码，并达到相同的留出通过率 0.838，只是堆叠稍多一些（每个留出回合 1.21 次编辑，对比 0.99）。一旦初始策略会探索，读代码在两种 gym 中都是最优策略，捷径不再胜出。

图 9.1 汇总了这五次运行。两项改变承担的作用不同。初始策略决定了技能*是否*被学会：从第 6 章的基座智能体出发，任何环境都没有产生读代码的行为；从 RL 基座智能体出发，两种环境都产生了。环境决定的是*通过率意味着什么*。在原始 gym 中，0.6 的通过率可能来自读代码、来自函数名，也可能来自盲目堆叠，只有行为统计才能区分它们。在匿名化的一次编辑 gym 中，任何高于 0.2 的部分都必须来自回合中收集的信息，因此如果后续某章的改动损害了探索（过强的 KL 惩罚、熵坍缩、可被投机的奖励），它会表现为向 0.2 的下跌，而不是悄无声息地切换到某个捷径。

> **证据。** 本书测得，每种设置一个随机种子，80 步、每步 64 个回合。定性模式（先记忆，再堆叠，然后停滞，最后学会读代码）与我们准备本章时用其他学习率和随机种子做的探索性运行一致；那些运行未予报告。这些数字是小模型上的机制演示。一般性的结论针对的是机制：策略梯度会放大策略自身样本中任何能预测奖励的特征，因此必须从环境中移除捷径特征，而预期的技能必须在初始策略中出现，哪怕很少。

**全书其余部分的默认设置。** 从这里开始，各章使用 `RLConfig()` 的默认值训练，即匿名化的 gym（`rename=True`）、每回合一次编辑（`max_edits=1`），从 `make_rl_base(tok, seed)` 出发，并用 `evaluate(..., cfg=cfg)` 在 `heldout_tasks` 上评估。在这些设置下，稳定高于 0.2 的通过率，需要利用回合中收集的信息，而不能只靠盲猜，而每份历史记录中的行为统计使这一点可以核查。有意研究某个捷径的章节，比如 [第 11 章](11-rewards-and-verifiers.md)中的奖励投机（reward hacking），会通过 `RLConfig.reward` 或 `env_factory` 改变奖励或环境，其余一切保持不变。

## 9.5 各种变体改了什么

GRPO 的每个公开变体都只改变了 9.1 节所列选择中的几项。下表把每个变体表述为相对 DeepSeekMath GRPO 的差异，并给出实现它的 `RLConfig` 字段。“来源”一列概括了各论文的规定，记录见 [第 20 章](20-frontier.md#把-rlvr-目标写成相对-grpo-的差异)；日期为 arXiv 首版日期。

| 变体 | 相对 GRPO 的改变 | 字段 |
| --- | --- | --- |
| GRPO（DeepSeekMath，2024-02） | 基准：均值/标准差优势，对 token 项先在序列内取均值，KL 在损失中（$\beta = 0.04$） | `adv_mode='grpo'`、`loss_mode='seq_mean_token_mean'`、`kl_coef=0.04` |
| Dr. GRPO（2025-03） | 不除以标准差；用常数归一化项代替 $1/\lvert o_i \rvert$ | `adv_mode='mean'`、`loss_mode='constant'` |
| DAPO（2025-03） | clip-higher（$\epsilon_{\text{high}} = 0.28$）、动态采样、token 级损失、超长奖励塑形、无 KL | `clip_high=0.28`、`dynamic_sampling=True`、`loss_mode='token_mean'`、`overlong_buffer`、`kl_coef=0` |
| RLOO（2024-02） | 留一法基线；整个输出视为一个动作，没有比率和裁剪 | `adv_mode='rloo'`、`loss_mode='seq_mean_token_sum'`、`minibatches=1` |
| GSPO（2025-07） | 每个序列一个经长度归一化的比率，裁剪范围在 $10^{-4}$ 量级 | `ratio_level='sequence'`、`clip_low=3e-4`、`clip_high=4e-4` |

**Dr. GRPO** 去掉了 6.6 节和 7.5 节测量过其偏差的两种归一化：除以每个输出的长度，这会降低长失败输出中每个 token 受到的惩罚；以及除以组标准差，这会放大策略几乎总能解出或几乎总是失败的任务的权重。

**DAPO** 做了四项改变。*Clip-higher* 提高 $\epsilon_{\text{high}}$，让具有正优势的稀有 token 增长得更快（8.6 节测得上裁剪界主要作用在这类 token 上）。*动态采样*丢弃奖励全部相同的组（其优势恰为零），并采样更多任务直到批次填满，从而让每个保留的回合都带有梯度。*token 级损失*即 `token_mean`。*超长奖励塑形*加入一个惩罚，在达到最大长度之前的一段缓冲区内从 0 线性增长到 $-1$。在 gym 中起作用的长度是轮数预算，因此 `overlong_buffer=2` 会对用到第五轮或第六轮的回合分别施加 $-0.5$ 和 $-1$ 的惩罚。DAPO 还描述了**超长过滤**（overlong filtering），即把被截断的回合从损失中掩掉；DeepSWE 的“紧凑过滤”（compact filtering）把它推广到了超时和步数上限。

**RLOO** 使用 7.5 节的留一法基线，并把整个生成视为一个动作，因此它就是带有逐任务基线的 REINFORCE。它的优势是 Dr. GRPO 的 $\frac{G}{G-1}$ 倍，这个整体尺度大体会被 Adam 吸收。

**GSPO** 用每个序列一个比率取代逐 token 比率，即其 token 比率的几何平均 $s_i = \exp\big(\frac{1}{|o_i|}\sum_t \log \rho_{i,t}\big)$，并对 $s_i$ 做裁剪。这样一个序列的所有 token 共享同一个裁剪决定和同一个重要性权重。作者的动机是序列级奖励与 token 级修正之间的不匹配，并报告它稳定了混合专家模型的训练。由于 $s_i$ 是对数比率的平均，它的变动远小于单个 token 的比率，这就是它报告的裁剪范围比 PPO 小三个数量级的原因。

```python
#@save
def gspo_policy_loss(logp, old_logp, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    """Sequence-level clipped surrogate: logp, old_logp, mask [B, T]; advantages [B] -> (loss, stats)."""
    token_log_ratio = logp - old_logp.detach()
    seq_ratio = arl.masked_mean(token_log_ratio, mask, dim=-1).exp()      # s_i, [B]
    adv = advantages.detach()
    unclipped, clipped = seq_ratio * adv, seq_ratio.clamp(1 - clip_low, 1 + clip_high) * adv
    loss = -torch.minimum(unclipped, clipped).mean()                        # mean over sequences
    with torch.no_grad():
        stats = dict(clip_frac=(clipped < unclipped).float().mean().item(),
                     approx_kl=arl.masked_mean(token_log_ratio.exp() - 1 - token_log_ratio, mask).item(),
                     ratio_min=seq_ratio.min().item(), ratio_max=seq_ratio.max().item())
    return loss, stats


VARIANTS = {
    'grpo': dict(adv_mode='grpo', loss_mode='seq_mean_token_mean', kl_coef=0.04),
    'dr_grpo': dict(adv_mode='mean', loss_mode='constant'),
    'dapo': dict(adv_mode='grpo', loss_mode='token_mean', clip_high=0.28, dynamic_sampling=True, overlong_buffer=2),
    'rloo': dict(adv_mode='rloo', loss_mode='seq_mean_token_sum', minibatches=1),
    'gspo': dict(ratio_level='sequence', clip_low=3e-4, clip_high=4e-4),
}
```

`VARIANTS` 是实验的起点，而不是任何论文的复现：学习率、组大小、批次大小和模型都与论文不同，只有列出的字段发生了改变。每个预设从 RL 基座智能体出发运行 40 步。

```python
print(f'{"variant":<8} | reward, last 10 steps | held-out pass | clip_frac | kl_ref | episodes sampled')
for name, fields in VARIANTS.items():
    cfg, model = RLConfig(**fields), copy.deepcopy(rl_base)
    hist = train_rl(model, train_tasks, tok, cfg, steps=40)
    held = evaluate(model, heldout_tasks, tok, 8, generator=torch.Generator().manual_seed(0), cfg=cfg)
    mean = lambda key: sum(h.get(key, 0.0) for h in hist) / len(hist)
    print(f'{name:<8} | {sum(h["reward"] for h in hist[-10:]) / 10:21.3f} | {held["pass_rate"]:13.3f} |'
          f' {mean("clip_frac"):9.4f} | {mean("kl_ref"):6.4f} | {sum(h["n_episodes"] for h in hist):6d}')
```

Output:

```text
variant  | reward, last 10 steps | held-out pass | clip_frac | kl_ref | episodes sampled
grpo     |                 0.745 |         0.775 |    0.0010 | 0.1656 |   2560
dr_grpo  |                 0.795 |         0.825 |    0.0018 | 0.0000 |   2560
dapo     |                 0.847 |         0.850 |    0.0018 | 0.0000 |   5184
rloo     |                 0.728 |         0.738 |    0.0000 | 0.0000 |   2560
gspo     |                 0.673 |         0.713 |    0.3137 | 0.0000 |   2560
```

每个预设都在 40 步内学会了读代码；奖励和留出通过率的差别是单个随机种子在 80 个留出回合上的数字，标准误约为 0.04，9.6 节会测量随机种子带来的离散程度。有三列反映的是机制，而不是哪个变体最好。DeepSeekMath 的 GRPO 是唯一带参考模型的预设，它相对初始策略的 KL 估计平均达到 0.17，在 $\beta = 0.04$ 下，这一项与优势相比很小；[第 10 章](10-kl-entropy-regularization.md)研究它什么时候不小。DAPO 采样的回合数约为其他预设的两倍，这是动态采样的代价。GSPO 的裁剪比例为 0.31，而 token 级预设约为 0.002：论文的裁剪范围是为数千 token 长的输出选定的，在一个 12 token 的回合上，序列比率在一个小批次更新步之后就会离开这个区间，因此每第二个小批次的大部分都被裁掉了。RLOO 只有一个小批次，从不裁剪，正如 9.1 节所预测的。

## 9.6 小规模下的受控消融

预设一次改变好几个字段。受控消融则只改变默认配置的一个字段，其余一切保持固定，包括随机种子：下面每个实验组都从同一个 RL 基座智能体出发、用随机种子 0、1、2 各运行 40 步，并在留出任务上以每个任务 8 个回合进行评估。五个实验组共 15 次运行。

```python
ARMS = {'default': {}, 'no std': dict(adv_mode='mean'), 'seq_mean_token_mean': dict(loss_mode='seq_mean_token_mean'),
        'clip-higher 0.28': dict(clip_high=0.28), 'dynamic sampling': dict(dynamic_sampling=True)}
runs = {}
for arm, fields in ARMS.items():
    for seed in [0, 1, 2]:
        cfg, model = RLConfig(seed=seed, **fields), copy.deepcopy(rl_base)
        hist = train_rl(model, train_tasks, tok, cfg, steps=40)
        held = evaluate(model, heldout_tasks, tok, 8, generator=torch.Generator().manual_seed(seed), cfg=cfg)
        runs[arm, seed] = dict(history=hist, held=held['pass_rate'])
```

```python
print(f'{"arm":<20} | reward 30-39 (seeds)  | held-out (seeds)      | length | clip_frac | episodes/step')
for arm in ARMS:
    rs = [runs[arm, s] for s in range(3)]
    late = [sum(h['reward'] for h in r['history'][30:]) / 10 for r in rs]
    held = [r['held'] for r in rs]
    flat = [h for r in rs for h in r['history']]
    avg = lambda key: sum(h.get(key, 0.0) for h in flat) / len(flat)
    print(f'{arm:<20} | {" ".join(f"{x:.2f}" for x in late):<21} | {" ".join(f"{x:.2f}" for x in held):<21} |'
          f' {avg("length"):6.1f} | {avg("clip_frac"):9.4f} | {avg("n_episodes"):5.0f}')
```

Output:

```text
arm                  | reward 30-39 (seeds)  | held-out (seeds)      | length | clip_frac | episodes/step
default              | 0.83 0.85 0.71        | 0.80 0.76 0.81        |   12.9 |    0.0013 |    64
no std               | 0.78 0.77 0.73        | 0.79 0.71 0.80        |   12.5 |    0.0011 |    64
seq_mean_token_mean  | 0.80 0.81 0.80        | 0.79 0.71 0.78        |   14.9 |    0.0013 |    64
clip-higher 0.28     | 0.83 0.83 0.74        | 0.78 0.76 0.72        |   12.9 |    0.0015 |    64
dynamic sampling     | 0.83 0.87 0.85        | 0.81 0.85 0.84        |   12.7 |    0.0020 |   123
```

先用不同随机种子之间的波动来判断差异有多大。默认实验组的三个随机种子，后期训练奖励在 0.71 到 0.85 之间，留出通过率在 0.76 到 0.81 之间。去掉标准差归一化、换用 `seq_mean_token_mean`、把 $\epsilon_{\text{high}}$ 提高到 0.28，结果都落在这个范围之内。这个实验中，clip-higher 的影响本就很有限：在这个配置下裁剪只作用于 0.15% 的 token，所以它放宽的那条边界几乎从未被触及。更值得关注的是两个机制上的变化。`seq_mean_token_mean` 产生了最长的回合，策略 token 数为 14.9，对比 12.9（9.7 节会继续讨论）。动态采样在全部三个随机种子上都取得了最高的后期奖励和留出通过率（0.81 到 0.85），为此它每步采样 123 个回合，而不是 64 个。

![图 9.2：各消融设置与默认配置在 40 步 GRPO 中的训练批次奖励，色带覆盖三个随机种子的范围（最小到最大），曲线为其均值。RL 基座智能体，匿名化的一次编辑 gym，每步 64 个回合。](../../figures/ch09-ablations.svg)

图 9.2 展示了带随机种子色带的学习曲线。默认配置与每个单字段改动的色带在整个运行中都相互重叠；曲线之间的差别小于随机种子之间的差别。要真正消融这些选择，需要的随机种子比本章能负担的更多，[第 17 章](17-evaluation.md)会用配对比较把这一点量化。

![图 9.3：动态采样。左：默认实验组每步奖励全部相同的采样组所占比例（三个随机种子的均值）。右：默认实验组和动态采样实验组的训练批次奖励随累计采样回合数的变化。](../../figures/ch09-dynamic-sampling.svg)

图 9.3 把动态采样放到与其成本相关的坐标轴上。随着策略改进，越来越多的组在每次尝试中都通过，零优势组的比例上升，动态采样必须采样更多才能填满批次。以采样的回合数为横轴，它的曲线明显落后于默认配置：默认配置在约 2,000 个回合后达到 0.8 的批次奖励，动态采样则需要约 3,500 个。它每次更新学到的更多，每次 rollout 学到的更少。对智能体而言，一次 rollout 的成本远高于一次更新，因此应该以 rollout 为轴来比较。

> **证据。** 本书测得：三个随机种子，40 步，一个 158k 参数的模型，20 个训练任务。这些是机制演示。小于随机种子间离散程度的差异不算差异，这些结果也都无法预测同样的改动对大模型、长时程智能体或不同奖励的效果。来源论文中报告的这些选择的收益，来自规模大得多的数学推理任务（[第 20 章](20-frontier.md)）。

## 9.7 实践中的设计问题

> **问题：固定采样预算下的组大小与提示数。**
>
> **现象：** 在每步回合数相同的情况下，一种配置能学会难任务，另一种却在这些任务上停滞；或者每个批次的大部分都是零优势。
>
> **原因：** 对通过率为 $p$ 的任务做 $G$ 次尝试，这一组方差为零的概率是 $p^G + (1 - p)^G$。小组会在几乎总能解出或总是失败的任务上把尝试浪费在退化的组上，基线也更嘈杂。大组每步覆盖的不同任务更少，因此每步的梯度涉及的任务更少。
>
> **可选方案：** (a) 对困难、奖励稀疏的任务用大 $G$，因为成功很罕见，必须先采样到才能从中学习。(b) 对通过率中等、分布广泛的任务，用许多提示配小 $G$。(c) 根据每个任务的滑动通过率自适应地选择 $G$。
>
> **建议做法：** 在这个规模下 $G = 8$、8 个提示；改动之前先测量零方差比例和每步不同任务数。下面的代码块把预算固定为 64 个回合，对 RL 基座智能体和训练后的策略测量这两个量。

```python
for name, model in [('RL base agent', rl_base), ('trained (default run)', policy)]:
    cells = []
    for G in [4, 8, 16]:
        cfg = RLConfig(group_size=G)
        tasks = train_tasks[:64 // G]
        trajs = sample_groups(model, lambda: make_env(cfg), tasks, tok, cfg, torch.Generator().manual_seed(5))
        rewards = torch.tensor([t.reward for t in trajs]).view(-1, G)
        informative = (rewards.std(-1) > 0).float().mean().item()
        cells.append(f'G={G:2d}: {64 // G:2d} tasks, informative groups {informative:.2f}')
    print(f'{name:<22}', ' | '.join(cells))
```

Output:

```text
RL base agent          G= 4: 16 tasks, informative groups 0.62 | G= 8:  8 tasks, informative groups 0.88 | G=16:  4 tasks, informative groups 1.00
trained (default run)  G= 4: 16 tasks, informative groups 0.00 | G= 8:  8 tasks, informative groups 0.00 | G=16:  4 tasks, informative groups 0.25
```

对 RL 基座智能体而言，组越大，包含奖励差异的组越多，从 $G = 4$ 时的 62% 上升到 $G = 16$ 时的全部，代价是覆盖的任务只有四分之一。对训练后的策略，测量显示出另一种状态：它的通过率高于 0.9、熵低于 0.1，所以它对同一任务的各次尝试几乎总是一致，无论 $G$ 取多少，几乎没有组带有梯度。到了这一步，该调整的是任务分布（更难的任务，或去掉已解决的任务），而不是组大小。

> **问题：零优势组与过滤。**
>
> **现象：** 随着训练成功，全部通过的组的比例上升，有效批次缩小，进展变慢；在 `token_mean` 下，零优势回合仍计入归一化项，稀释了其他每个 token 的权重。
>
> **原因：** 零方差组在第 7 章的任何基线下优势都恰好为零，因此它的回合消耗了 rollout，却除了往分母里添 token 之外毫无贡献。
>
> **可选方案：** (a) 动态采样，保持有效批次满额，代价是额外的 rollout。(b) 过滤零优势组但不重新采样，成本不变，批次缩小。(c) 用课程学习去掉策略总能解出或从来解不出的任务（[第 14 章](14-scaling-and-efficiency.md)）。
>
> **建议做法：** 每步记录 `zero_adv_frac`；该比例适中时只过滤、不重新采样，整个任务饱和时再加入课程。只有当 rollout 相对更新很便宜时，动态采样才划算，而这恰好与智能体 RL 的情况相反。默认运行的历史记录显示了这一趋势。

```python
zero = [h['zero_adv_frac'] for h in history]
print('zero-advantage groups by 10-step window:', [round(sum(zero[i:i + 10]) / 10, 2) for i in range(0, 80, 10)])
ds = [runs['dynamic sampling', s]['history'] for s in range(3)]
per_step = [sum(h[i]['n_episodes'] for h in ds) / 3 for i in range(40)]
print('dynamic sampling, episodes sampled per step by 10-step window:',
      [round(sum(per_step[i:i + 10]) / 10) for i in range(0, 40, 10)])
```

Output:

```text
zero-advantage groups by 10-step window: [0.19, 0.12, 0.34, 0.45, 0.5, 0.62, 0.82, 0.75]
dynamic sampling, episodes sampled per step by 10-step window: [111, 98, 128, 154]
```

在默认运行中，零优势组的比例从前十步的 0.19 上升到最后二十步的约 0.8。到最后，每五个采样组中有四个是全部通过或全部失败，而 `token_mean` 仍把它们的 token 计入分母。动态采样会补满批次，而它的成本也随同一趋势上升，从早期每步约 110 个回合增加到后期约 150 个，超过名义值 64 的两倍。

> **问题：长度增长。**
>
> **现象：** 每个回合的平均策略 token 数或轮数在训练中上升，通过率却没有相应提高，增长往往出现在失败的回合中。
>
> **原因：** 聚合方式决定了每个 token 的权重（[6.6 节](06-policy-gradient.md#66-损失聚合)）。在 `seq_mean_token_mean` 下，一个失败回合的总惩罚是固定的，回合越长，每个 token 受到的惩罚越少，失败回合中多余的调用就很便宜。不对轮数计费的奖励也无法抵消这一点。
>
> **可选方案：** (a) 与长度成比例的聚合方式（`token_mean`、`constant` 或 `seq_mean_token_sum`）。(b) 超长塑形或步数成本，它们会改变任务本身，必须为有用的信息收集留出空间（[第 12 章](12-multi-turn-and-long-horizons.md)）。(c) 掩掉被截断的回合。
>
> **建议做法：** (a)，并分别记录成功回合和失败回合的长度。消融实验的历史记录直接比较了这两种聚合方式。

```python
for arm in ['default', 'seq_mean_token_mean']:
    hist = [runs[arm, s]['history'] for s in range(3)]
    length = [sum(h[i]['length'] for h in hist) / 3 for i in range(40)]
    turns = [sum(h[i]['turns'] for h in hist) / 3 for i in range(40)]
    reads = sum(h[i]['reads'] for h in hist for i in range(30, 40)) / 30
    print(f'{arm:<20} policy tokens per episode, steps 0-9 {sum(length[:10]) / 10:.1f} -> steps 30-39'
          f' {sum(length[30:]) / 10:.1f} | turns {sum(turns[30:]) / 10:.2f} | reads {reads:.2f}')
```

Output:

```text
default              policy tokens per episode, steps 0-9 11.6 -> steps 30-39 13.4 | turns 4.07 | reads 1.00
seq_mean_token_mean  policy tokens per episode, steps 0-9 14.0 -> steps 30-39 14.6 | turns 4.36 | reads 1.06
```

在 `seq_mean_token_mean` 下，回合从前十步起就有约 14 个策略 token，而 `token_mean` 下是从 11.6 上升到 13.4；轮数也更多（4.36 对 4.07），读取略多（每回合 1.06 对 1.00）。差异很早就出现，是因为基座智能体的失败回合很长，而按序列归一化对其中每个 token 的惩罚更少。在六轮预算、40 步的训练中，这个效应很小；在有数百轮的智能体回合中，它正是 Dr. GRPO 所指出的响应长度偏差的机制。

> **问题：选择损失聚合方式。**
>
> **现象：** 两次只差一个“归一化细节”的运行学习速度不同，或者长度发生漂移；批次组成的变化会改变有效学习率。
>
> **原因：** `token_mean` 让每个 token 的权重与批次的总长度耦合；`seq_mean_token_mean` 对各轨迹等权，对长轨迹中的 token 赋予较小权重；`constant` 和 `seq_mean_token_sum` 保持每个 token 的权重固定，需要针对其尺度调学习率。
>
> **可选方案：** `aggregate_loss` 的四种模式。
>
> **建议做法：** 本书使用 `token_mean`，因为它在批次内对长度中立、被广泛使用（DAPO），而且其依赖批次的尺度很容易通过记录的长度来监控；当训练中长度变化很大时使用 `constant`（Dr. GRPO），因为它连批次耦合也消除了。多轮智能体应避免 `seq_mean_token_mean`，因为它们的回合长度相差一个数量级，并且与成功相关；上一个代码块测量了它在这个 gym 上的效果。

## 小结

- GRPO 是 PPO 的裁剪代理目标，用组基线代替 critic；在 DeepSeekMath 的形式中，它还在序列内对 token 取均值，并在损失中加入 KL 惩罚。它已发表的变体只是少数几个字段上的差异；`RLConfig` 和 `train_rl` 把它们实现为同一个循环的选项。
- 在原始 gym 上，GRPO 重现了第 6 章的记忆现象：训练任务 0.98，留出任务 0.40，读取次数为零，第一次编辑根据函数名选择。
- 捷径特征存在于环境之中。逐回合给函数改名去掉了函数名这一捷径，也消除了训练与留出之间的大部分差距，策略随即转向堆叠编辑，而一个盲目序列就能把它推到 0.60。把每个回合限制为一次编辑去掉了这个捷径，却暴露出一个探索失败：基座智能体几乎从不在读代码之后做出有依据的编辑，因此没有可放大的东西。
- 一个九条示范中有一条有依据示范的基座智能体，在通过率相同的情况下，于 80 步内学会了“读代码、编辑、测试、提交”，留出任务达到 0.84，在匿名化 gym 上如此，作为对照在原始 gym 上也是如此。初始策略决定技能是否被学会；环境决定通过率能否证明这项技能。从这里开始，匿名化的一次编辑 gym 加上 `make_rl_base` 是本书的默认设置。
- 在小规模下，对标准差归一化、聚合方式和 clip-higher 的单字段消融都落在随机种子噪声之内，而在裁剪比例只有 0.15% 的情况下，clip-higher 根本没有作用对象。可以看到两个机制：`seq_mean_token_mean` 使回合变长；动态采样每次更新学到的更多，但采样的回合数是 1.9 倍，并且随着零优势组上升到批次的 80%，这一成本还在增长。使用论文裁剪范围的 GSPO 在 12 token 的回合上裁掉了 31% 的序列。这些是机制演示，不是基准评测证据。

## 练习

1. 证明在一个小批次、一轮次（epoch）的情况下，GRPO 的更新等于第 6 章使用组归一化优势的 REINFORCE 更新。然后说明使用两个小批次时有什么变化。第二个小批次中的哪些 token 可能被裁剪？
2. 改变一个假设：用 `make_base_agent(tok, kinds=('random', 'blind') * 8 + ('oracle',))`（十七条示范中有一条有依据示范）重新运行 9.4 节的默认配置。策略还能学会读代码吗？找出能在 80 步内学会读代码的最小有依据示范比例，并把它与初始策略实测的“先读后正确”概率联系起来。
3. 改名时从 gym 全部 30 个函数名中抽取名字，包括留出任务的函数名。把名字池限制为训练任务的函数名，重新测量默认运行的留出通过率。这一差别说明了留出函数名的嵌入起什么作用？
4. 把 CISPO（[第 20 章](20-frontier.md)）实现为 `ratio_level='cispo'`：逐 token 的损失为 $-\mathrm{sg}(\min(\rho, 1 + \epsilon_{\text{high}}))\, \hat A \log \pi_\theta$。哪些 token 保留了 PPO 的裁剪会去掉的梯度？在 9.6 节的运行上，把它与裁剪相关的统计量同默认配置做比较。
5. 让 KL 项发挥作用：用 `kl_coef` 取 {0.01, 0.1, 1.0} 训练默认配置，记录 `kl_ref` 和通过率。系数多大时策略不再学会读代码？[第 10 章](10-kl-entropy-regularization.md)会解释为什么答案取决于 `kl_kind`。
6. 测量 GSPO 预设的裁剪比例及其序列比率的分布范围。论文为长数学输出选定的裁剪范围适用于 12 token 的回合吗？假设 token 对数比率相互独立，推导 $\log s_i$ 的标准差如何随策略 token 数变化。
7. 设计评审：某团队的 SWE 智能体在训练仓库上达到 70%，在留出集上只有 35%，并且它的轨迹很少在编辑之前打开文件。根据 9.4 节，列出你会在任务数据和环境中寻找的捷径特征、你会要求查看的行为统计，以及你会考虑的两类修复。
