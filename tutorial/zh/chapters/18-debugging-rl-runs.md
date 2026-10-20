# 18. 调试强化学习训练

第 17 章说明了怎样判断评测差异是否可信。本章处理另一个更棘手的问题：训练在运行，奖励也在上升，但更新本身可能是错的。采样、环境、奖励、advantage 和参数更新互相影响，单看 loss 或 reward 很容易漏掉故障。下面会复现几个具体例子：掩码错位、旧概率使用错误、梯度累积归一化出错，都可能产生一条上升的奖励曲线。

第一，确定应记录哪些指标，并用 TinyAgent 的无故障运行给出一组参考范围。第二，构建一个加入诊断记录的精简训练器，检查它与 `train_rl` 完全一致，然后有意地破坏它，每次只引入一个故障：错位一格的掩码、被当前值覆盖的 `old_logp`、熵坍缩（entropy collapse）、长度偏差、符号错误、奖励投机（reward hacking）、过高的学习率、梯度累积 bug、温度不匹配，以及一个其实就是在线策略本身的参考模型。每个故障都给出它在监控面板上的特征，并与一次无故障训练并列对照。第三，把这些特征转化为代码库可以在第 1 步之前断言的不变量，以及一个从症状到测试的排查流程。这里每次训练都是一个 158k 参数模型在 20 个训练任务上的机制演示（mechanism demonstration）；这些特征可用于排查其他系统，具体数值阈值则需要重新测量。

```python
#@save
import copy
import dataclasses
import torch
import torch.nn.functional as F
import agentrl as arl
```

```python
torch.set_num_threads(4)                     # the check host is shared (see Chapter 10)
tok = arl.build_tokenizer(arl.make_tasks('all'))
train_tasks, heldout_tasks = arl.make_tasks('train'), arl.make_tasks('heldout')
base = arl.make_rl_base(tok)                 # Chapter 9's starting policy
```

## 18.1 记录什么，以及为什么

一个指标值得放上监控面板，当且仅当某种故障会在奖励变化之前移动它，或者在奖励不变时移动它。按这条规则，有用的数字可以分成四组。

1. **结果。** 训练奖励、训练批次的隐藏测试通过率，以及留出任务上的通过率。训练奖励是优化器看到的东西，因此它正是故障可以在无人察觉的情况下败坏的数字；在独立验证器下的留出通过率是对它的检验（[第 11 章](11-rewards-and-verifiers.md)，[第 17 章](17-evaluation.md)）。
2. **行为。** 智能体做了什么：每回合的读取、测试和编辑次数，格式错误的调用，提交，使用已知漏洞利用（`hardcode_tests`）的回合比例，以及每回合的策略 token 数，并按通过与失败的回合分开统计。它们说明奖励是*如何*获得的，而[第 9 章](09-grpo-and-variants.md#94-环境设计中的捷径特征)已经表明，这正是技能与捷径的区别。
3. **优化器状态。** 动作 token 上的策略熵、更新前后策略之间的近似 KL（`approx_kl`）、裁剪比例、比率 $\rho$ 的极值、裁剪前的梯度范数、使用参考策略时到参考策略的 KL，以及所有奖励都相等、因而不携带梯度的组所占比例（`zero_adv_frac`）。如果有 critic，还要加上其价值的解释方差（[7.4 节](07-advantages-and-credit.md#74-训练-critic)）；本章的训练都不使用 critic。
4. **一致性。** 在正确实现中应当是固定值的量，例如采样器记录的对数概率与训练器重新计算的对数概率之间的差距。无故障训练中它们处于浮点精度噪声水平；任何可见的数值都是 bug。这些是记录成本最低、诊断性最强的指标。

`log_metrics` 根据某一步采样的回合，构建监控面板一行中属于 rollout 的那一半。它复用第 9 章的 `behavior_stats`，并加上那一章不需要的指标：格式错误的调用、漏洞利用比例、提交，以及通过与失败回合的长度。属于更新的那一半来自下面的训练器，通过 `update` 合并进来。

```python
#@save
def log_metrics(sampled, cfg, update=None):
    """One dashboard row from a step's sampled Trajectory list (+ the update's stats) -> flat dict."""
    n, G = len(sampled), cfg.group_size
    rewards = torch.tensor([t.reward for t in sampled]).view(-1, G)                  # [P, G]
    passed = [bool(t.info['submitted'] and t.info['hidden_pass']) for t in sampled]
    lengths = [float(t.action_mask.sum()) for t in sampled]                           # policy tokens per episode
    def mean_length(want):
        picked = [length for length, p in zip(lengths, passed) if p == want]
        return sum(picked) / max(len(picked), 1)
    row = dict(arl.behavior_stats(sampled), n_episodes=n,
               zero_adv_frac=(rewards.max(-1).values == rewards.min(-1).values).float().mean().item(),
               malformed=sum(t.info['n_invalid'] for t in sampled) / n,
               hardcode=sum('hardcode_tests' in t.info['ops'] for t in sampled) / n,
               submitted=sum(bool(t.info['submitted']) for t in sampled) / n,
               length_pass=mean_length(True), length_fail=mean_length(False))
    return dict(row, **(update or {}))
```

## 18.2 可注入故障的带诊断记录的训练器

为了展示故障是什么样子，我们需要一个每种故障都对应一个开关的训练器。下面的训练器就是第 9 章的 `rl_update` 和 `train_rl`，只是在真实 bug 所在的位置被拆开，并带有一个打开某个故障的 `bug` 参数。`BUGS` 列出了代码故障；目录中的其他失败（过高的学习率、奖励投机、长度偏差）属于配置选择，不需要开关。

`prepare_batch` 产生损失所使用的三个张量：动作掩码、`old_logp` 和优势。这里有三个故障。错位一格的掩码由与 `ids` 而不是 `ids[1:]` 对齐的 token 归属构建，这会把每段策略 span 向右移动一个位置：每次调用都失去它的第一个 token `<call>`，并多出最后一个 token 之后的那个 token，即环境写入的 `<obs>` 分隔符。温度故障从一个以温度 0.7 运行的采样器获取 `old_logp`，就像一个直接信任推理引擎对数概率的系统所做的那样。符号错误把优势取反。`make_ref` 构建 KL 参考模型，在 `ref_is_policy` 下它直接返回在线模型本身，这在真实代码库中只是一个字符级的 bug（`ref = policy` 而不是深拷贝）。

```python
#@save
BUGS = {
    'mask_shift': '(1) action mask aligned to ids instead of ids[1:]: each policy span moves one token right',
    'stale_old_logp': '(2) old_logp recomputed from the current weights before every minibatch',
    'entropy_sign': '(3) entropy bonus added with the wrong sign: an entropy penalty of ENTROPY_COEF',
    'sign_flip': '(5) advantages negated',
    'accum_local': '(8) minibatch accumulated over 4 length-sorted microbatches, each token_mean / 4',
    'sampler_temp': '(9) sampler at SAMPLER_TEMP and its log-probs used as old_logp; trainer at cfg.temperature',
    'ref_is_policy': '(10) the KL reference is the live policy object',
}
SAMPLER_TEMP, ENTROPY_COEF = 0.7, 0.3


def prepare_batch(model, batch, cfg, bug=None):
    """What the trainer feeds its loss -> (mask, old_logp [B, T-1], advantages [B]), with an optional fault."""
    ids, mask = batch['ids'], batch['action_mask']                         # [B, T], [B, T-1]
    if bug == 'mask_shift':
        mask = F.pad(mask, (1, 0))[:, :-1]
    with torch.no_grad():
        old_logp = arl.selected_log_probs(model(ids)[:, :-1] / cfg.temperature, ids[:, 1:])
    if bug == 'sampler_temp':
        old_logp = batch['sample_logp']
    advantages = arl.compute_advantages(batch, cfg)
    return mask, old_logp, -advantages if bug == 'sign_flip' else advantages


def make_ref(model, cfg, bug=None):
    """The frozen KL reference (None when cfg.kl_coef is 0); the live model itself under 'ref_is_policy'."""
    if not cfg.kl_coef:
        return None
    return model if bug == 'ref_is_policy' else copy.deepcopy(model).eval()
```

`minibatch_backward` 计算一个小批次的损失并反向传播，但不执行优化器步。这里还有两个故障。在 `accum_local` 下，小批次被拆成按长度排序的四个微批次，就像按 token 预算调度时那样拆分，并且每个微批次的 `token_mean` 损失除以四：这正是 [14.4 节](14-scaling-and-efficiency.md#144-梯度累积与损失归一化)的归一化 bug。在 `entropy_sign` 下，熵奖励被加到损失上而不是减去，于是它变成了熵惩罚。到参考模型的 KL 在小批次内部计算，使用参考模型在同样行上的前向传播结果，所以当参考模型就是在线策略时，KL 恰好为零。

```python
#@save
def minibatch_backward(model, batch, rows, mask, old_logp, advantages, cfg, bug=None, ref_model=None):
    """Loss and backward for one minibatch, no optimizer step -> stats dict (clip_frac, ..., entropy, loss)."""
    ids = batch['ids']
    parts = rows[mask[rows].sum(-1).argsort()].chunk(4) if bug == 'accum_local' else [rows]
    logs = []
    for part in parts:
        logits = model(ids[part])[:, :-1] / cfg.temperature                    # [b, T-1, V]
        logp = arl.selected_log_probs(logits, ids[part, 1:])
        loss, stats = arl.policy_loss(logp, old_logp[part], advantages[part], mask[part], cfg)
        if bug == 'entropy_sign':
            loss = loss + ENTROPY_COEF * arl.entropy_bonus(logits, mask[part])
        if ref_model is not None:
            with torch.no_grad():
                ref_logp = arl.selected_log_probs(ref_model(ids[part])[:, :-1] / cfg.temperature, ids[part, 1:])
            kl = arl.masked_mean(arl.kl_penalty(logp, ref_logp, cfg.kl_kind), mask[part])
            loss, stats['kl_ref'] = loss + cfg.kl_coef * kl, kl.item()
        (loss / len(parts)).backward()
        stats['entropy'] = arl.masked_mean(arl.entropy_from_logits(logits.detach()), mask[part]).item()
        logs.append(dict(stats, loss=loss.item()))
    return {k: sum(s[k] for s in logs) / len(logs) for k in logs[0]}
```

`debug_update` 是由这两部分组装而成的 `rl_update`，并包含`old_logp` 被当前值覆盖的故障：在每个小批次之前，它都用当前权重重新计算“旧”对数概率，当不求梯度的前向传播被挪进小批次循环内部时，发生的就是这种情况。它还记录 18.1 节的一致性指标 `logp_gap`：在损失所训练的 token 上，采样器记录的对数概率与训练器自己重新计算的对数概率之间的平均绝对差。

```python
#@save
def debug_update(model, optimizer, batch, cfg, bug=None, ref_model=None, generator=None):
    """arl.rl_update with diagnostics and an optional fault from BUGS -> dict of mean stats."""
    mask, old_logp, advantages = prepare_batch(model, batch, cfg, bug)
    trainer_logp = prepare_batch(model, batch, cfg)[1] if bug else old_logp      # clean recomputation
    gap = (trainer_logp - batch['sample_logp']).abs()[mask > 0]                   # [N trained tokens]
    logs = []
    for _ in range(cfg.epochs):
        for rows in torch.randperm(len(advantages), generator=generator).chunk(cfg.minibatches):
            if bug == 'stale_old_logp':
                old_logp = prepare_batch(model, batch, cfg)[1]
            optimizer.zero_grad()
            stats = minibatch_backward(model, batch, rows, mask, old_logp, advantages, cfg, bug, ref_model)
            stats['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            optimizer.step()
            logs.append(stats)
    out = {k: sum(s[k] for s in logs) / len(logs) for k in logs[0]}
    return dict(out, logp_gap=gap.mean().item(), loss_tokens=mask.sum().item() / len(mask))
```

`train_logged` 就是 `train_rl`，加上 `log_metrics` 行以及温度故障所需的采样器配置。它每隔 `eval_every` 步在留出任务上评测一次，规程与 `train_rl` 相同：每个任务 4 个回合，隐藏测试通过率，在训练环境中进行。

```python
#@save
def train_logged(model, tasks, tok, cfg, steps, bug=None, eval_tasks=None, eval_every=10, env_factory=None):
    """arl.train_rl with log_metrics rows and an optional fault from BUGS; trains in place -> history."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env_factory = env_factory or (lambda: arl.make_env(cfg))
    sampler_cfg = dataclasses.replace(cfg, temperature=SAMPLER_TEMP) if bug == 'sampler_temp' else cfg
    ref_model = make_ref(model, cfg, bug)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for step in range(steps):
        kept, sampled = arl.sample_step(model, env_factory, tasks, tok, sampler_cfg, gen)
        batch = arl.collate(kept, tok.pad_id)
        batch['n_turns'] = torch.tensor([float(t.info['n_turns']) for t in kept])
        row = log_metrics(sampled, cfg, debug_update(model, optimizer, batch, cfg, bug, ref_model, gen))
        if eval_tasks is not None and (step % eval_every == 0 or step == steps - 1):
            row['eval_pass'] = arl.evaluate(model, eval_tasks, tok, 4, cfg.temperature,
                                            torch.Generator().manual_seed(step), cfg, env_factory)['pass_rate']
        history.append(dict(row, step=step))
    return history
```

一个调试框架只有在不注入故障时就是它所声称的那个训练器，才有用处。我们把 RL 基座智能体复制两份，各训练 5 步，一份用 `train_rl`，一份用 `train_logged`，然后比较每个参数以及两份历史记录都记录的每一项统计量。

```python
reference, logged = copy.deepcopy(base), copy.deepcopy(base)
hist_ref = arl.train_rl(reference, train_tasks, tok, arl.RLConfig(), steps=5)
hist_log = train_logged(logged, train_tasks, tok, arl.RLConfig(), steps=5)
same_params = all(torch.equal(p, q) for p, q in zip(reference.parameters(), logged.parameters()))
same_stats = all(a[k] == b[k] for a, b in zip(hist_ref, hist_log) for k in a)
print(f'parameters bit-identical: {same_params} | all {len(hist_ref[0])} train_rl statistics identical: {same_stats}')
```

Output:

```text
parameters bit-identical: True | all 17 train_rl statistics identical: True
```

无故障训练使用本书的默认配置，训练 24 步，每 8 步做一次留出评测。`run` 按名称保存每份历史记录，`show` 打印每次训练的若干指标：前四步的均值、一个箭头、后四步的均值，以及最终的留出通过率。

```python
RUNS = {}
def run(name, cfg=arl.RLConfig(), bug=None, steps=24):
    """Train a fresh copy of the RL base agent with train_logged; keep its history in RUNS[name]."""
    RUNS[name] = train_logged(copy.deepcopy(base), train_tasks, tok, cfg, steps, bug, heldout_tasks, eval_every=8)

def show(names, keys):
    """Per run: each metric as mean(first 4 steps) -> mean(last 4 steps), and the last held-out pass rate."""
    for name in names:
        h = RUNS[name]
        mean = lambda k, part: sum(r[k] for r in part) / len(part)
        cells = ' | '.join(f'{k} {mean(k, h[:4]):.3g} -> {mean(k, h[-4:]):.3g}' for k in keys)
        print(f"{name:<17}| {cells} | held-out {h[-1]['eval_pass']:.2f}")

run('healthy')
```

```python
TAXONOMY = ['reward', 'eval_pass', 'entropy', 'approx_kl', 'clip_frac', 'ratio_max', 'grad_norm', 'length',
            'zero_adv_frac', 'reads', 'malformed', 'hardcode', 'logp_gap']
healthy = RUNS['healthy']
for key in TAXONOMY:
    values = sorted(r[key] for r in healthy if key in r)
    print(f'{key:<14} step 0 {healthy[0][key]:9.3g} | last {healthy[-1][key]:9.3g} |'
          f' min {values[0]:9.3g} | median {values[len(values) // 2]:9.3g} | max {values[-1]:9.3g}')
```

Output:

```text
reward         step 0     0.172 | last     0.656 | min     0.172 | median     0.438 | max     0.859
eval_pass      step 0       0.1 | last       0.7 | min       0.1 | median     0.525 | max       0.7
entropy        step 0     0.565 | last      0.21 | min     0.169 | median     0.355 | max     0.565
approx_kl      step 0   0.00322 | last  0.000681 | min  9.61e-05 | median  0.000303 | max   0.00322
clip_frac      step 0    0.0139 | last         0 | min         0 | median         0 | max    0.0139
ratio_max      step 0      1.21 | last      1.27 | min      1.03 | median      1.11 | max      1.27
grad_norm      step 0     0.204 | last     0.243 | min     0.155 | median     0.226 | max     0.288
length         step 0      13.2 | last      14.4 | min        11 | median      13.7 | max      14.4
zero_adv_frac  step 0     0.375 | last     0.375 | min         0 | median      0.25 | max     0.625
reads          step 0     0.312 | last      1.08 | min     0.312 | median      1.11 | max      1.61
malformed      step 0    0.0625 | last    0.0469 | min    0.0312 | median    0.0781 | max     0.188
hardcode       step 0    0.0781 | last         0 | min         0 | median    0.0156 | max     0.125
logp_gap       step 0  3.05e-08 | last  6.98e-09 | min  6.98e-09 | median  3.07e-08 | max  5.02e-08
```

在没有故障时，这次 24 步训练把训练奖励从 0.17 提高到 0.66，把留出通过率从 0.10 提高到 0.70。各列给出了在这个设置下每个指标的正常范围，下面的每一种失败都对照它们来解读。有三组值得细看。一致性指标 `logp_gap` 始终在 $7 \times 10^{-9}$ 到 $5 \times 10^{-8}$ 之间：采样器和训练器在相同 token 上运行同一个 fp32 模型，二者一致到舍入误差的程度。更新幅度很小。`approx_kl` 在第 0 步是 0.003，之后约为 $3 \times 10^{-4}$，最大比率不超过 1.27，至少一半的步中裁剪比例为 0，因为两个小批次中的第一个恰好是 on-policy 的，第二个也几乎没有移动。此外，从第一步起，每步就有 0 到 63% 的组（中位数 25%）不携带梯度，因为基座智能体在某些任务上八次尝试全部失败。

下表把 18.1 节浓缩成一张参考卡。无故障一列是上面测得的；失败一列指向接下来的目录，编号与本章开头一致。

| 指标 | TinyAgent 上的正常值（24 步） | 偏离时提示什么 |
| --- | --- | --- |
| 奖励、批次通过率 | 0.17 → 0.66 | 一切问题，但在故障 (1)、(2)、(6)、(8) 下它都上升了 |
| 留出隐藏测试通过率 | 0.10 → 0.70 | (6) 投机、捷径；这是对奖励本身的唯一检验 |
| 动作 token 的熵 | 0.57 → 0.21，平滑下降 | 先于奖励下降：(3) 坍缩；上升：(1)、(7) |
| 每步 `approx_kl` | 第 0 步 0.003，之后约 3e-4 | (7) 学习率：第 0 步为 0.31；大量复用样本 |
| `clip_frac` | 0 到 0.014 | 整次训练恰好为 0：(2) 被当前值覆盖的 `old_logp` |
| 比率极值 | 最大值 1.03 到 1.27 | 远离 1：(1)、(9) |
| 裁剪前的 `grad_norm` | 0.16 到 0.29 | (1)：最高 1.5；恰好为 0：所有组内的奖励都相同 |
| 每回合策略 token 数，按结果分 | 11 到 14 | 失败回合变长：(4) 长度偏差 |
| `zero_adv_frac` | 0 到 0.63 | 低奖励时趋向 1：(3)、(5)、(7)、(9)；高奖励时：饱和 |
| 读取、格式错误调用、漏洞利用比例 | 读取 0.31 → 1.08；格式错误低于 0.19；`hardcode` → 0 | 读取消失：(3)、(9)；格式错误：(7)；漏洞利用：(6) |
| `logp_gap`（采样器对训练器） | 7e-9 到 5e-8 | (1)、(9)；重分词漂移（第 5 章）；引擎不一致（第 13 章） |
| `kl_ref`（有 KL 项时） | $\beta = 0.1$ 时 0.014 → 0.057 | 恰好为 0：(10) 参考模型 = 策略 |
| critic 解释方差 | 这里没有 critic；第 7–8 章中为 0.1 到 0.3 | critic 什么也没跟踪到 |

## 18.3 损失输入中的故障

前三个故障破坏的是损失的输入：它在哪些 token 上训练，以及它把哪些概率当作 $\pi_{\text{old}}$。如果 18.1 节的分组是对的，它们的特征应该出现在一致性指标中，早于任何曲线有机会变化。

```python
run('mask_shift', bug='mask_shift')
run('stale_old_logp', bug='stale_old_logp')
run('sampler_temp', bug='sampler_temp')
show(['healthy', 'mask_shift', 'stale_old_logp', 'sampler_temp'],
     ['reward', 'logp_gap', 'clip_frac', 'ratio_max', 'grad_norm'])
```

Output:

```text
healthy          | reward 0.191 -> 0.656 | logp_gap 3.79e-08 -> 8.86e-09 | clip_frac 0.00385 -> 0 | ratio_max 1.12 -> 1.14 | grad_norm 0.191 -> 0.229 | held-out 0.70
mask_shift       | reward 0.188 -> 0.52 | logp_gap 3.75 -> 3.1 | clip_frac 0.0129 -> 0.0437 | ratio_max 1.23 -> 2.96 | grad_norm 0.478 -> 1.5 | held-out 0.47
stale_old_logp   | reward 0.207 -> 0.715 | logp_gap 4.32e-08 -> 1.51e-08 | clip_frac 0 -> 0 | ratio_max 1 -> 1 | grad_norm 0.225 -> 0.238 | held-out 0.68
sampler_temp     | reward 0.227 -> 0.156 | logp_gap 0.0752 -> 0.0285 | clip_frac 0.0218 -> 0 | ratio_max 2.65 -> 2.96 | grad_norm 0.202 -> 0 | held-out 0.15
```

错位一格的掩码并没有让学习停止。它的训练奖励从 0.19 升到 0.52，只看奖励曲线的话，它会被当作无故障训练的一个较慢的随机种子而放行。一致性指标从第 0 步起就给出了不同的结论：前四步中 `logp_gap` 平均为每个被训练 token 3.75 nat，而正常值是 $4 \times 10^{-8}$。采样器只为它采样的 token 记录对数概率。在错位掩码现在覆盖的 `<obs>` 分隔符上，它记录的是占位值 0，而训练器的模型给这个分隔符的对数概率远低于零，因为从来没有任何示范训练策略去预测它。同样是这些 token 驱动了优化器指标。$\log \pi$ 对某个 token 的 logit 的梯度是 $1 - \pi$，对于模型认为几乎不可能的 token 接近 1，因此这少数几个 token 主导了更新：梯度范数在开始时为 0.48，结束时为 1.5，而正常值约为 0.2。它们的比率也偏离得很远（最大比率 3.0，裁剪比例 0.044）。这个故障让留出通过率损失了 23 个百分点，0.47 对 0.70。

错误处理 `old_logp` 的训练则是另一种情形。它的结果与无故障训练无法区分（训练奖励 0.72 对 0.66，留出 0.68 对 0.70），唯一的特征是两个优化器指标在整次训练中*恰好*停在它们的平凡值上：裁剪比例为 0，最大比率为 1。当 `old_logp` 用当前权重重新计算时，$\rho \equiv 1$，裁剪永远不会起作用，PPO 悄悄退化为一串普通的策略梯度步。在这次短程实验中，它没有造成明显差异，因为每个批次只用于两个小批次步，第二步几乎不动。若有更多轮次（epoch）或小批次，正是裁剪阻止了策略在一个批次上移动过远（[8.5 节](08-ppo.md#85-诊断与提前停止)测得裁剪比例和 KL 在四个轮次中增长），而这个 bug 去掉了裁剪，却在奖励上不留任何痕迹。一个在整次训练中恰好为 0 或恰好为 1 的指标是一个警告，而不是一种安慰。

温度故障是三者中破坏性最强的。采样器以温度 0.7 运行，而以温度 1 运行的训练器把采样器的对数概率当作 $\pi_{\text{old}}$。训练奖励从 0.23 降到 0.16，留出通过率最终为 0.15，低于盲猜一次的 0.20。`logp_gap` 在开始时为 0.075 nat，比正常运行的数值噪声高六个数量级；第 0 步的最大比率已经是 2.65，而正确的流水线在第一次更新之前应当恰好有 $\rho = 1$。行为指标显示了接下来发生了什么。

```python
show(['healthy', 'mask_shift', 'sampler_temp'], ['entropy', 'malformed', 'reads', 'zero_adv_frac'])
```

Output:

```text
healthy          | entropy 0.531 -> 0.195 | malformed 0.0625 -> 0.0664 | reads 0.473 -> 1.09 | zero_adv_frac 0.312 -> 0.375 | held-out 0.70
mask_shift       | entropy 0.579 -> 0.795 | malformed 0.043 -> 0.121 | reads 0.418 -> 1.2 | zero_adv_frac 0.312 -> 0.0938 | held-out 0.47
sampler_temp     | entropy 0.47 -> 0.156 | malformed 0 -> 0 | reads 0.0859 -> 0.00391 | zero_adv_frac 0.281 -> 1 | held-out 0.15
```

读取次数从每回合 0.09 降到 0.004，到最后每个组的奖励都相等（`zero_adv_frac` 为 1.0），梯度范数为 0：策略停止了探索，训练停止改进。相比之下，掩码故障提高了熵（0.58 到 0.80）和格式错误的调用（0.04 到 0.12），这是训练一个策略从不输出的 token、却不再训练开启每次调用的 `<call>` token 的副作用。

为什么温度不匹配会专门扼杀探索？降低温度会把概率从罕见 token 移向常见 token，因此对罕见 token，比率 $\pi_{T=1}/\mu_{T=0.7}$ 一开始就大于 1，而对常见 token 小于 1。PPO 的裁剪会去掉比率超过 $1 + \epsilon$ 的正优势 token 的梯度。我们在基座智能体以温度 0.7 采样的一个批次上、在任何更新之前测量这一点。`probe` 按给定（可能有故障的）流水线的采样方式采样 64 个回合；18.6 节会复用它。

```python
check_cfg = arl.RLConfig(kl_coef=0.1)          # with a KL term, so that Section 18.6's reference checks apply
def probe(bug=None, seed=1):
    """64 episodes of the base agent on 8 training tasks, sampled as the (possibly faulty) pipeline samples."""
    cfg = dataclasses.replace(check_cfg, temperature=SAMPLER_TEMP) if bug == 'sampler_temp' else check_cfg
    return arl.sample_groups(base, lambda: arl.make_env(cfg), train_tasks[:8], tok, cfg,
                             torch.Generator().manual_seed(seed))
```

```python
batch = arl.collate(probe('sampler_temp'), tok.pad_id)                       # sampled at T = 0.7
mask, old_logp, adv = prepare_batch(base, batch, check_cfg, 'sampler_temp')
ratio = (prepare_batch(base, batch, check_cfg)[1] - old_logp).exp()          # pi_T=1 / mu_T=0.7, [B, T-1]
rare = old_logp.exp() < 0.5                                                  # the sampler gave them < 0.5
positive = (adv > 0)[:, None] & (mask > 0)                                   # tokens the update should reinforce
for name, sel in [('rare (mu < 0.5)', rare & positive), ('common (mu >= 0.5)', ~rare & positive)]:
    print(f'{name:<19} tokens with A > 0: {int(sel.sum()):4d} | mean ratio {ratio[sel].mean():.2f} |'
          f' clipped (ratio > 1.2) {(ratio[sel] > 1.2).float().mean():.2f}')
```

Output:

```text
rare (mu < 0.5)     tokens with A > 0:   17 | mean ratio 1.17 | clipped (ratio > 1.2) 0.12
common (mu >= 0.5)  tokens with A > 0:   91 | mean ratio 0.95 | clipped (ratio > 1.2) 0.00
```

在更新本应强化的 token 中，罕见 token（采样器概率低于 0.5）的平均比率一开始就是 1.17，其中 12% 在策略还没有发生任何变化时就已被裁剪；常见 token 一开始为 0.95，没有一个被裁剪。本应防止策略移动过远的裁剪，在这里截掉了罕见高优势动作的一部分梯度，而罕见的好动作正是探索的组成部分。于是两种效应共同作用：更冷的采样器更少地抽到读取（第 0 步每回合 0.09 次，而温度 1 时为 0.47 次），而当一次读取获得奖励时，它的更新又有一部分被裁剪掉。罕见 token 的样本很少（17 个 token），本章也没有运行两端温度都一致为 0.7 的对照实验；在一次此处未报告的探索性训练中，以较低温度一致地采样和训练能够正常学习，这指向不匹配而非温度本身。练习 1 请你运行这个对照。修复方法不依赖于这种分解：训练器必须在采样器实际使用的分布下计算概率（[第 3 章](03-language-models-as-policies.md)），或者用重要性权重校正差异（[第 13 章](13-rollout-systems.md)）。

## 18.4 梯度中的故障

接下来的三个故障保持损失的输入不变，而破坏梯度：它的符号、大小和归一化。符号错误和学习率的训练在第 8 步停止，就像工程师在失败已经明确时会停掉它们一样。

```python
run('sign_flip', bug='sign_flip', steps=8)
run('lr 1e-2', arl.RLConfig(lr=1e-2), steps=8)
run('accum_local', bug='accum_local')
show(['healthy', 'sign_flip', 'lr 1e-2', 'accum_local'], ['reward', 'approx_kl', 'entropy', 'malformed', 'turns'])
for name in ['healthy', 'lr 1e-2']:
    print(f'{name:<17}', ' '.join(f"{r['approx_kl']:.4f}" for r in RUNS[name][:8]), '| approx_kl, steps 0-7')
```

Output:

```text
healthy          | reward 0.191 -> 0.656 | approx_kl 0.00112 -> 0.000442 | entropy 0.531 -> 0.195 | malformed 0.0625 -> 0.0664 | turns 3.52 -> 4.26 | held-out 0.70
sign_flip        | reward 0.125 -> 0.0273 | approx_kl 0.00145 -> 0.000167 | entropy 0.551 -> 0.501 | malformed 0.0664 -> 0.0312 | turns 4.21 -> 5.48 | held-out 0.03
lr 1e-2          | reward 0.043 -> 0 | approx_kl 0.0953 -> 0.00214 | entropy 0.417 -> 1.08 | malformed 0.762 -> 2.89 | turns 4.75 -> 6 | held-out 0.00
accum_local      | reward 0.219 -> 0.668 | approx_kl 0.00129 -> 0.000197 | entropy 0.499 -> 0.204 | malformed 0.0391 -> 0.16 | turns 3.64 -> 4.38 | held-out 0.65
healthy           0.0032 0.0007 0.0003 0.0003 0.0006 0.0005 0.0006 0.0002 | approx_kl, steps 0-7
lr 1e-2           0.3098 0.0642 0.0049 0.0024 0.0038 0.0021 0.0018 0.0009 | approx_kl, steps 0-7
```

优势被取反后，策略对奖励做的是梯度*下降*。训练奖励在八步内从 0.13 降到 0.03，留出通过率降到 0.03，回合从 4.2 轮拉长到 5.5 轮，因为用完预算而不提交是稳定拿到 0 的方法。优化器指标毫无异常：`approx_kl`、熵和格式错误的调用都在正常范围内。只有结果表明训练出了问题，这就是为什么 18.6 节要直接检验更新的方向。

学习率 $10^{-2}$ 是默认值的十倍，它首先表现在优化器和格式指标上。格式错误的调用从每回合 0.76 次升到 2.89 次，熵从 0.42 nat 升到 1.08 nat，后期每个回合都用满了全部 6 轮，奖励从 0.04 降到 0。输出的最后两行展示了每步 KL 中事件发生的顺序。

学习率 $10^{-2}$ 下的第一次更新使策略移动了每 token 0.31 nat 的近似 KL，约为正常训练第一步 0.0032 的 100 倍，第二次更新为 0.064。此后每步 KL 回落到接近正常水平。训练并没有恢复：它的奖励已经崩溃，各组奖励都相同，几乎没有剩余的梯度可用。这就是不稳定的一般形态：一两步出现很大的 `approx_kl` 或 `grad_norm`，随后指标恢复平静，策略却已难以继续学习。只显示平滑曲线的监控面板可能隐藏原因、只显示后果，所以要在记录均值的同时记录每步最大值。

累积故障完全没有特征。它的训练奖励（0.67）和留出通过率（0.65）在 40 回合评测的噪声范围内（通过率 0.7 时二项标准误约为 0.07）与无故障训练一致，每个优化器指标也是如此；格式错误的调用最后略高（0.16 对 0.07）。然而梯度是错的：18.6 节的不变量检查测得它相对单遍梯度的相对误差为 0.54（图 18.2），因为短微批次中的 token 获得的权重是长微批次中 token 的数倍（[14.4 节](14-scaling-and-efficiency.md#144-梯度累积与损失归一化)）。在这个 gym 中，回合很短且长度相近，这个误差就像对 token 做了一次带噪声的重新加权，训练得以存活。对于长尾的智能体回合，一个按长度排序的微批次可能只装下一条很长的轨迹，它的 token 只获得预期权重的一小部分。这是否会在监控面板上显现取决于具体训练；但它是否存在与此无关，只有对梯度的检验才能判断。

## 18.5 目标中的故障

最后四种失败来自目标本身，而不是更新中的缺陷：一个正则项、一种损失聚合、一个奖励和一个参考模型。我们预期它们的特征出现在行为统计中。

```python
run('entropy_sign', bug='entropy_sign')
show(['healthy', 'entropy_sign'], ['reward', 'entropy', 'reads', 'zero_adv_frac'])
```

Output:

```text
healthy          | reward 0.191 -> 0.656 | entropy 0.531 -> 0.195 | reads 0.473 -> 1.09 | zero_adv_frac 0.312 -> 0.375 | held-out 0.70
entropy_sign     | reward 0.176 -> 0.176 | entropy 0.445 -> 0.0268 | reads 0.191 -> 0.00781 | zero_adv_frac 0.344 -> 0.844 | held-out 0.20
```

我们用一个符号翻转的熵奖励来有意制造熵坍缩，它相当于对动作 token 平均熵施加 0.3 的惩罚。在真实训练中，坍缩通常有更缓慢的原因，例如较低的采样温度、偏爱某一模式的奖励，或每个批次过多的轮次（[第 10 章](10-kl-entropy-regularization.md)），但其特征是相同的。熵从 0.45 nat 降到 0.027 nat，读取也随之消失，从每回合 0.19 次降到 0.008 次。奖励停在 0.18，留出通过率最终为 0.20，即 [9.4 节](09-grpo-and-variants.md#94-环境设计中的捷径特征)的盲猜一次上限：策略在发现读取有回报之前，就坍缩到了“编辑、提交”上。`zero_adv_frac` 能把这种情况与成功区分开。收敛的策略和坍缩的策略都有低熵和大量同奖励组，但这里 84% 的组在奖励为 0.18 时奖励完全相同，所以几乎都是全失败组，而收敛策略的同奖励组是全通过组。要把全通过和全失败的比例分开记录，并把“熵的下降快于奖励的上升”当作早期预警。

```python
run('healthy, 12 turns', arl.RLConfig(max_turns=12), steps=16)
run('length bias', arl.RLConfig(max_turns=12, loss_mode='seq_mean_token_mean'), steps=16)
show(['healthy, 12 turns', 'length bias'], ['reward', 'length_pass', 'length_fail', 'turns', 'reads'])
```

Output:

```text
healthy, 12 turns| reward 0.23 -> 0.488 | length_pass 13.9 -> 13.2 | length_fail 16.1 -> 12.8 | turns 4.34 -> 3.98 | reads 0.57 -> 1.08 | held-out 0.55
length bias      | reward 0.195 -> 0.512 | length_pass 14.3 -> 12.7 | length_fail 20.8 -> 33.9 | turns 5.45 -> 6.77 | reads 0.75 -> 2.71 | held-out 0.55
```

长度偏差需要空间才能显现，所以这里两次训练都使用 12 轮预算、训练 16 步；它们只在损失聚合（loss aggregation）方式上不同。在 `token_mean` 下，失败回合变短了，从 16 个策略 token 降到 13 个。在 `seq_mean_token_mean` 下，失败回合从 21 个 token 增长到 34 个，回合平均使用 6.8 轮而不是 4.0 轮，每回合读取 2.7 次，而一次就够了。两次训练中通过的回合都保持在约 13 个 token，训练奖励（0.51 对 0.49）和留出通过率（都是 0.55）都无法区分二者。这正是 [6.6 节](06-policy-gradient.md#66-损失聚合)的机制：在按序列归一化下，每个失败回合承担固定的总惩罚，因此回合越长，其中每个 token 受到的惩罚越小，失败的最便宜方式就是慢慢地失败。特征是按结果拆分的长度；整体平均长度会把这一效应与失败比例的变化混在一起。对编程智能体而言，这种失败早在体现到奖励上之前，就已经在消耗 rollout 时间和上下文了。

```python
run('public reward', arl.RLConfig(reward='public'))
show(['healthy', 'public reward'], ['reward', 'pass_rate', 'hardcode', 'reads'])
```

Output:

```text
healthy          | reward 0.191 -> 0.656 | pass_rate 0.191 -> 0.656 | hardcode 0.0703 -> 0 | reads 0.473 -> 1.09 | held-out 0.70
public reward    | reward 0.273 -> 0.969 | pass_rate 0.176 -> 0.00781 | hardcode 0.125 -> 0.969 | reads 0.418 -> 0.0352 | held-out 0.00
```

奖励投机在 24 步内复现了[第 11 章](11-rewards-and-verifiers.md#113-复现奖励投机)的那次训练。训练奖励是公开测试验证器，它从 0.27 升到 0.97，是本章最好看的曲线。在同样的回合上，隐藏测试通过率从 0.18 降到 0.008，使用 `hardcode_tests` 的比例从 0.13 升到 0.97，读取降到每回合 0.04 次，留出通过率最终为 0。漏洞利用比例是最直接的指标，因为它点出了具体行为，但它之所以存在，只是因为有人早已知道这个漏洞。隐藏测试通过率需要一个独立的验证器，它也能捕捉到没有人预料到的漏洞利用。

```python
run('KL 0.1', arl.RLConfig(kl_coef=0.1), steps=8)
run('ref_is_policy', arl.RLConfig(kl_coef=0.1), bug='ref_is_policy', steps=8)
show(['KL 0.1', 'ref_is_policy'], ['reward', 'kl_ref'])
print('ref_is_policy identical to the run without KL:',
      all(a['reward'] == b['reward'] and a['grad_norm'] == b['grad_norm'] for a, b in zip(RUNS['ref_is_policy'], healthy)))
```

Output:

```text
KL 0.1           | reward 0.203 -> 0.27 | kl_ref 0.0142 -> 0.0571 | held-out 0.23
ref_is_policy    | reward 0.191 -> 0.277 | kl_ref 0 -> 0 | held-out 0.28
ref_is_policy identical to the run without KL: True
```

最后一个故障是参考模型就是在线策略。当 $\beta = 0.1$、参考模型是一份冻结副本时，随着策略离开其起点，到参考模型的 KL 在八步内从每 token 0.014 nat 增长到 0.057 nat。当参考模型是在线策略时，它在每一步都恰好为 0，而且这次训练与没有任何 KL 项的无故障训练逐位相同：当 $\pi_{\text{ref}} = \pi_\theta$ 时，$k_3$ 估计及其梯度都消失了。其他所有指标都是正常的，因为这次训练*就是*那次无故障训练。失败之处在于配置所要求的正则化并不存在，只有当有人需要它时，例如一次长训练发生漂移时，它才会暴露出来。

初始策略与参考策略相同时，KL 为零是正常的；但策略已更新多步、系数非零，KL 仍持续精确为零，就应检查参考模型是否误用了在线策略。

![图 18.1：十种失败的监控面板。每种失败对应两个面板：训练批次奖励（上）和暴露它的指标（下），红色为故障训练，蓝色为与之匹配的无故障训练。无故障训练采用默认配置，但长度偏差的对照使用 12 轮预算，参考模型故障的对照使用 0.1 的 KL 系数。每次训练 24 步、每步 64 个回合；符号错误和学习率的训练在第 8 步停止，长度偏差这一对训练了 16 步。](../../figures/ch18-dashboard.svg)

图 18.1 按监控面板的呈现方式汇集了这十种失败。先看每一对中的上方面板。十种中有四种（陈旧比率、长度偏差、累积 bug、在线参考模型）的奖励曲线落在无故障训练的逐步噪声之内；一种（掩码）上升得更慢但仍在上升；还有一种（奖励投机）是整页最好看的曲线。再看下方面板，它们说明了 18.1 节为何要把指标分组。一致性指标从第一步就标出了代码故障 (1)、(2)、(9) 和 (10)：对数概率差距是 $10^{0.5}$ 而不是 $10^{-8}$，裁剪比例钉死在 0，比率在个别步上超过 20，KL 钉死在 0。格式指标在两步之内就标出了学习率问题 (7)。行为指标标出了目标层面的失败 (3)、(4) 和 (6)，它们的训练奖励要么从未与无故障训练分开，要么朝错误的方向移动。还有一个故障 (8)，什么也没有标出。

## 18.6 每个强化学习代码库都应断言的不变量

18.3–18.5 节中每一个存在于代码中的故障，都违反了一个在正确训练器中于第 0 步、任何训练之前就成立的恒等式。检查这些恒等式只需在一个批次上做几次前向和反向传播，而一次失败的检查对故障的定位远比一条曲线精确。本书的清单如下：

1. **掩码 = 策略 token。** 损失掩码恰好覆盖那些目标 token 由策略采样得到的位置。
2. **采样器 = 训练器对数概率。** 在这些位置上，训练器重新计算的对数概率与采样器的记录在数值噪声范围内相等：这里两者都是 fp32，阈值为 $10^{-4}$；对 bf16 引擎，阈值应根据实测的不一致分布来设定（[第 13 章](13-rollout-systems.md)，13.6 节）。
3. **任何一步之前比率 = 1。** 在一步的第一个小批次中，$\rho = \pi_\theta / \pi_{\text{old}}$ 恰好为 1。
4. **组内优势均值 = 0**，针对组基线（[7.5 节](07-advantages-and-credit.md#75-用组内样本构造基线)），这也检查了组的排布是否符合优势函数的预期。
5. **损失对填充不变。** 向批次追加填充列不会改变任何东西。
6. **累积梯度 = 单遍梯度。** 梯度累积复现了对整个批次做一次反向传播所得的梯度。
7. **更新中 `old_logp` 被冻结。** 在第一个小批次步之后，后续小批次看到 $\rho \neq 1$，所以整个更新上的 `approx_kl` 为正。
8. **更新遵循优势的符号。** 一次更新之后，$A > 0$ 的回合的 $\log \pi(\tau)$ 上升，$A < 0$ 的回合下降：变化量与优势正相关。
9. **到参考模型的 KL 在初始化时为 0、一步之后为正**（使用参考模型时）。前半句在这个训练器中由构造保证；后半句能抓住在线参考模型。

`pad_batch` 和 `flat_grad` 是两个小辅助函数：一个添加右侧填充，另一个在整个批次上运行训练器的反向传播，并把梯度作为一个向量返回。`step_checks` 在模型的一份副本上执行一次真实更新，检验不变量 7 到 9。

```python
#@save
def pad_batch(batch, extra, pad_id=0):
    """The same batch with `extra` columns of right padding on every [B, T] and [B, T-1] tensor."""
    return dict(batch, **{k: F.pad(batch[k], (0, extra), value=pad_id if k == 'ids' else 0)
                          for k in ['ids', 'attention_mask', 'action_mask', 'sample_logp']})


def flat_grad(model, batch, mask, old_logp, advantages, cfg, bug=None):
    """The trainer's backward on the whole batch, no optimizer step -> (gradient vector, stats)."""
    model.zero_grad()
    stats = minibatch_backward(model, batch, torch.arange(len(advantages)), mask, old_logp, advantages, cfg, bug)
    grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    model.zero_grad()
    return grad, stats


def step_checks(model, batch, owned, trainer_logp, cfg, bug=None):
    """Invariants of one real update on a copy of model -> dict of (measured value, passed)."""
    trained = copy.deepcopy(model)
    ref = make_ref(trained, cfg, bug)
    stats = debug_update(trained, torch.optim.Adam(trained.parameters(), lr=cfg.lr), batch, cfg, bug, ref,
                         torch.Generator().manual_seed(0))
    new_logp = prepare_batch(trained, batch, cfg)[1]
    moved = ((new_logp - trainer_logp) * owned).sum(-1)                         # change of log pi(episode), [B]
    corr = torch.corrcoef(torch.stack([moved, arl.compute_advantages(batch, cfg)]))[0, 1].item()
    kl = 0.0 if ref is None else arl.masked_mean(arl.kl_penalty(new_logp, prepare_batch(ref, batch, cfg)[1]),
                                                 owned).item()
    return {'old_logp frozen in the update': (stats['approx_kl'], stats['approx_kl'] > 0),
            'update raises log pi where A > 0': (corr, corr > 0),
            'KL to reference > 0 after a step': (kl, ref is None or kl > 0)}
```

```python
#@save
def sanity_checks(model, trajs, tok, cfg, bug=None):
    """Invariants of the instrumented trainer on one batch at step 0 -> dict check -> (measured value, passed)."""
    batch = arl.collate(trajs, tok.pad_id)
    batch['n_turns'] = torch.tensor([float(t.info['n_turns']) for t in trajs])
    mask, old_logp, adv = prepare_batch(model, batch, cfg, bug)
    owned = (arl.pad_rows([t.owner[1:] for t in trajs], mask.shape[1], -1) == arl.OWNER['policy']).float()
    trainer_logp = prepare_batch(model, batch, cfg)[1]
    gap = (trainer_logp - batch['sample_logp']).abs()[mask > 0].max().item()
    full, _ = flat_grad(model, batch, mask, old_logp, adv, cfg, None if bug == 'accum_local' else bug)
    grad, stats = flat_grad(model, batch, mask, old_logp, adv, cfg, bug)          # the trainer's own path
    padded = pad_batch(batch, 8, tok.pad_id)
    _, pad_stats = flat_grad(model, padded, *prepare_batch(model, padded, cfg, bug), cfg, bug)
    ratio_err = max(stats['ratio_max'] - 1, 1 - stats['ratio_min'])
    group_mean = adv.view(-1, cfg.group_size).mean(-1).abs().max().item()
    pad_err = abs(pad_stats['loss'] - stats['loss'])
    accum_err = ((grad - full).norm() / full.norm()).item()
    checks = {'mask = policy tokens': (int((mask != owned).sum()), bool((mask == owned).all())),
              'sampler = trainer log-prob': (gap, gap < 1e-4),
              'ratio = 1 before any step': (ratio_err, ratio_err < 1e-4),
              'group advantage mean = 0': (group_mean, group_mean < 1e-5),
              'loss invariant to padding': (pad_err, pad_err < 1e-5),
              'accumulated = one-pass gradient': (accum_err, accum_err < 1e-4)}
    return dict(checks, **step_checks(model, batch, owned, trainer_logp, cfg, bug))
```

```python
for name, (value, ok) in sanity_checks(base, probe(), tok, check_cfg).items():
    print(f'{name:<34} {value:10.3g}  {"ok" if ok else "FAIL"}')
```

Output:

```text
mask = policy tokens                        0  ok
sampler = trainer log-prob           1.31e-06  ok
ratio = 1 before any step                   0  ok
group advantage mean = 0             2.24e-08  ok
loss invariant to padding            7.45e-09  ok
accumulated = one-pass gradient             0  ok
old_logp frozen in the update         0.00223  ok
update raises log pi where A > 0        0.514  ok
KL to reference > 0 after a step       0.0124  ok
```

`sanity_checks` 完全按照训练器的方式构建批次，运行六项静态检查和三项单步检查，返回每项检查的测量值和判定结果。轨迹带有各 token 的归属（[5.6 节](05-trajectories-and-templates.md#56-轨迹与掩码)），正是这一点让掩码检查独立于被检验的掩码。没有故障时，九项检查全部通过，数值显示了余量。掩码与归属完全一致。在 64 个回合上，采样器与训练器之间的最大差距为 $1.3 \times 10^{-6}$：同步采样以不同于训练器的批次形状运行这些回合，因此 fp32 舍入略有不同，这就是阈值不为零的原因。填充让损失变化 $7 \times 10^{-9}$。一步之后，回合对数概率的变化与优势的相关系数为 0.51；在两个小批次上做一次小的、带裁剪的更新，本来就不应期望相关系数接近 1，重要的是它的符号。现在打开每一个代码故障，再运行同样的检查，每个故障一行，每项检查一列，编号与上面的清单一致。

```python
MATRIX = {bug: sanity_checks(base, probe(bug), tok, check_cfg, bug) for bug in [None, *BUGS]}
names = list(MATRIX[None])
print(f'{"":<15}', ' '.join(f'{i + 1:>4}' for i in range(len(names))))
for bug, checks in MATRIX.items():
    print(f'{str(bug):<15}', ' '.join(f'{"ok" if ok else "FAIL":>4}' for _, ok in checks.values()))
```

Output:

```text
                   1    2    3    4    5    6    7    8    9
None              ok   ok   ok   ok   ok   ok   ok   ok   ok
mask_shift      FAIL FAIL   ok   ok FAIL   ok   ok   ok   ok
stale_old_logp    ok   ok   ok   ok   ok   ok FAIL   ok   ok
entropy_sign      ok   ok   ok   ok   ok   ok   ok   ok   ok
sign_flip         ok   ok   ok   ok   ok   ok   ok FAIL   ok
accum_local       ok   ok   ok   ok   ok FAIL   ok   ok   ok
sampler_temp      ok FAIL FAIL   ok   ok   ok   ok   ok   ok
ref_is_policy     ok   ok   ok   ok   ok   ok   ok   ok FAIL
```

本章注入的每个代码故障都至少让一项检查失败，而且失败的检查能识别出它：覆盖旧概率的故障只让检查 7 失败，符号错误只让检查 8 失败，累积 bug 只让检查 6 失败，在线参考模型只让检查 9 失败。图 18.2 展示了这个矩阵及其测量值。有三个结果值得说明。错位一格的掩码让三项检查失败，其中之一是填充不变性，而这并不是我们为它设计的：把掩码右移，会在未填充的批次中丢掉最后一列，但在填充过的批次中会把它移进一个填充列，于是损失依赖于填充。温度故障让两项对数概率检查都失败，第 0 步的比率高达 3，而它们本应全为 1。人为配置的熵惩罚则通过了全部九项：它是一个看起来合理、只是符号错了的项，更新仍然遵循优势的符号（图 18.2 中相关系数为 0.37），只有 18.5 节的监控面板显示出坍缩。不变量捕捉实现故障；目标故障需要监控面板。二者都抓不住奖励投机、学习率或长度偏差，它们是运行在正确代码上的配置选择，需要留出评测和无故障参照。

![图 18.2：哪个不变量捕捉哪个故障。行：无故障的带诊断记录的训练器，以及打开 `BUGS` 中每个代码故障的版本；列：`sanity_checks` 的九项检查，在任何训练之前于 RL 基座智能体的 64 个回合上运行一次。红色单元格表示失败。](../../figures/ch18-checks.svg)

## 18.7 排查：从症状到测试

监控面板告诉你*有*问题。排查是一套流程，把一个症状转化为能在其各种原因之间做出判断的最便宜的测试。本书的流程有四步，按成本排序。

1. **运行不变量检查。** `sanity_checks` 只需在一个批次上做几次前向和反向传播，就能识别或排除本章的每一个代码故障。在每次训练开始时运行它，并在每次改动训练器、模板、采样器或分批代码之后再次运行。
2. **查看一致性指标。** 高于噪声的 `logp_gap`、卡在平凡值上的比率或裁剪比例、恰好为 0 的 KL，这些都是 bug 而不是动态现象，不需要对照训练。
3. **与无故障参照比较。** 优化器指标和行为指标没有固定的正确值。它们的含义来自同一设置下的正常范围（18.2 节的表格）以及种子离散（[第 17 章](17-evaluation.md)）。相对于参照，每次只改变一样东西。
4. **用独立信号检验结果。** 在策略看不到的验证器下评测留出任务，是防范奖励测错东西的唯一手段。

图 18.3 把本章的每个症状映射到它的第一项测试。

![图 18.3：排查图。左：监控面板上的症状。中：首先要运行的测试。右：它所指向的故障，编号与本章目录一致。按行从左往右读；多个症状共享同一项测试，18.6 节的一致性检查排在最前，因为它们便宜且有决定性。](../../figures/ch18-triage.svg)

关于使用这张图有两点说明。症状会共享原因：低奖励下 `zero_adv_frac` 趋向 1，在熵坍缩、温度不匹配、符号错误和高学习率下都出现过，所以做判断的是测试，而不是症状。顺序也很重要：一致性故障可以模仿其下游的动态失败。掩码故障提高了梯度范数、裁剪比例、熵和格式错误的调用，这些合在一起看起来就像学习率过高；而降低学习率这个自然的反应对病因毫无作用。在调任何参数之前，先修好这些便宜的不变量。

> **生产实践。** 项目固定提交 `00094bd` 下的 verl（2026-10-13 在检查主机上阅读源码）记录了其中大部分组别：分数、奖励、优势和回报的统计量；`response_length/mean`、`max`、`min` 和 `clip_ratio`（在长度上限处被截断的回复所占比例）；`actor/entropy`；`actor/pg_clipfrac` 和 `actor/pg_clipfrac_lower`（双重裁剪）；`actor/ppo_kl`，即对数比率的掩码均值（$k_1$ 估计）；梯度范数；以及使用 critic 时的 `critic/vf_explained_var`。当 rollout 的对数概率被记录在批次中时，它还会计算 `training/rollout_probs_diff_mean`、`_max` 和 `_std`，即在回复掩码上 rollout 引擎与训练器之间 token *概率*的绝对差，以及它们的 Pearson 相关系数：这就是本章的一致性指标，只是作用在概率而不是对数概率上。任务相关的行为统计，例如每回合工具调用次数或漏洞利用比例，需要用户自行添加。

## 18.8 实践中的设计问题

> **问题：奖励上升却存在实现错误的训练。**
>
> **现象：** 训练奖励在上升，也许比以前的训练稍慢一点，于是没人再往下看。
>
> **原因：** 策略梯度对许多实现故障是鲁棒的，因为正确 token 上的梯度仍然存在。一个添加了错误项、丢掉了部分正确项或移除了某种安全机制的故障，会减缓或扭曲学习，但不会让它停止；18.3 节的掩码故障就是典型案例。
>
> **可选方案：** (a) 让每次训练在第 0 步都必须通过 `sanity_checks`；(b) 记录一致性指标（`logp_gap`，以及本应恰好处于平凡值的比率和 KL），对任何高于噪声的值发出警报；(c) 将奖励曲线与一次固定的无故障参照训练比较。
>
> **建议做法：** (a) 和 (b)，它们便宜且有决定性；仅靠 (c) 无法区分较慢的 bug 与较慢的随机种子。两条奖励曲线，每四步取一次：

```python
for name in ['healthy', 'mask_shift']:
    h = RUNS[name]
    print(f'{name:<11}| reward', ' '.join(f"{h[i]['reward']:.2f}" for i in range(0, 24, 4)),
          f"| held-out {h[-1]['eval_pass']:.2f} | logp_gap {h[0]['logp_gap']:.2g} | grad_norm {h[0]['grad_norm']:.2f}")
```

Output:

```text
healthy    | reward 0.17 0.34 0.36 0.44 0.53 0.72 | held-out 0.70 | logp_gap 3e-08 | grad_norm 0.20
mask_shift | reward 0.17 0.27 0.39 0.41 0.39 0.50 | held-out 0.47 | logp_gap 3.5 | grad_norm 0.48
```

直到第 12 步，故障训练的奖励（0.17、0.27、0.39、0.41）都在逐步噪声范围内跟随无故障训练（0.17、0.34、0.36、0.44）。故障训练只是在后来才落后，到第 20 步为 0.50 对 0.72，只看奖励的人可能把这个差距归因于随机种子。而同一行上第 0 步的数字是毫不含糊的：对数概率差距 3.5 对 $3 \times 10^{-8}$，梯度范数是正常值的 2.4 倍。

> **问题：只在留出评测中可见的静默失败。**
>
> **现象：** 训练侧的每个指标看起来都很好，甚至比平时更好：奖励上升、熵下降、KL 步长很小、梯度范数正常。
>
> **原因：** 训练信号与目标已经分离。奖励投机是最干净的例子（[第 11 章](11-rewards-and-verifiers.md)）；记住的捷径（[第 9 章](09-grpo-and-variants.md#94-环境设计中的捷径特征)）和衡量代理指标的奖励是其他例子。没有任何训练侧指标能看到它，因为它们衡量的都是策略与其所得奖励之间的关系。
>
> **可选方案：** (a) 按固定频率在独立验证器下做留出评测；(b) 为迄今发现的每个漏洞利用建立行为统计；(c) 定期由人或模型审阅采样的轨迹。
>
> **建议做法：** 始终采用 (a)，对已知漏洞利用加上 (b)，当 (a) 和 (b) 与轨迹所显示的情况不一致时采用 (c)。18.5 节的投机训练：

```python
h = RUNS['public reward']
for step in [0, 8, 16, 23]:
    r = h[step]
    print(f"step {step:2d} | reward {r['reward']:.3f} | entropy {r['entropy']:.3f} | grad_norm {r['grad_norm']:.3f}"
          f" | approx_kl {r['approx_kl']:.1e} | held-out hidden pass {r['eval_pass']:.3f}")
```

Output:

```text
step  0 | reward 0.234 | entropy 0.565 | grad_norm 0.252 | approx_kl 3.3e-03 | held-out hidden pass 0.175
step  8 | reward 0.750 | entropy 0.310 | grad_norm 0.404 | approx_kl 4.7e-04 | held-out hidden pass 0.125
step 16 | reward 0.969 | entropy 0.115 | grad_norm 0.155 | approx_kl 1.4e-04 | held-out hidden pass 0.025
step 23 | reward 0.969 | entropy 0.092 | grad_norm 0.200 | approx_kl 2.2e-05 | held-out hidden pass 0.000
```

不看最后一列来读第 16 步那一行：奖励 0.97，熵 0.12，梯度范数 0.16，`approx_kl` 为 $1.4 \times 10^{-4}$，这是一次干净收敛的训练的样貌。它的留出隐藏测试通过率是 0.025，开始时为 0.175，到第 23 步降到 0。同样的逻辑也适用于检查点选择：按训练奖励选出的检查点会是投机最严重的那个。应在留出的验证集上选择检查点（[17.8 节](17-evaluation.md#178-不碰测试集地选择检查点)），并把测试集留给最终报告。

> **问题：无法复现的尖峰。**
>
> **现象：** 一次训练在 `grad_norm`、`approx_kl` 或奖励上出现尖峰，而用同样配置重跑却没有出现，因此无法检查原因。
>
> **原因：** 要么重跑没有抽到相同的样本，要么数值上的不确定性改变了它们。
>
> **可选方案：** (a) 让每一步都可重放：记录随机种子和采样器状态，并在每一步（或每个指标越过阈值的步）保存批次（token id、采样器对数概率、奖励）以及模型检查点，然后确定性地重放更新并检查该批次；(b) 在判定尖峰异常之前，先估计该指标的种子离散（[第 17 章](17-evaluation.md)）；(c) 使用批次不变的算子，以吞吐量为代价消除数值来源（[第 20 章](20-frontier.md)中 Thinking Machines 的报告）。
>
> **建议做法：** (a) 和 (b)：可重放的尖峰是一份带证据的 bug 报告，而落在种子离散范围内的尖峰只是噪声。下面的代码块测量三种重跑对前六步的影响有多大：完全相同的重跑、从乘以 $1 + 10^{-5}$ 噪声的权重开始的重跑（代替非确定性归约），以及换一个随机种子的重跑。

```python
def signature(h):
    """What a rerun must match: per-step reward, length, and gradient norm."""
    return [(r['reward'], r['length'], r['grad_norm']) for r in h]

again = train_logged(copy.deepcopy(base), train_tasks, tok, arl.RLConfig(), 6)
nudged = copy.deepcopy(base)
with torch.no_grad():                                  # relative 1e-5 noise on every weight
    for p in nudged.parameters():
        p.mul_(1 + 1e-5 * torch.randn(p.shape, generator=torch.Generator().manual_seed(0)))
perturbed = train_logged(nudged, train_tasks, tok, arl.RLConfig(), 6)
other_seed = train_logged(copy.deepcopy(base), train_tasks, tok, arl.RLConfig(seed=1), 6)
print('same seed, 6 steps, identical:', signature(again) == signature(healthy[:6]))
for name, h in [('weights x (1 + 1e-5 noise)', perturbed), ('seed 1', other_seed)]:
    print(f'{name:<26} rewards equal at {sum(a["reward"] == b["reward"] for a, b in zip(h, again))}/6 steps |'
          f' max |grad_norm diff| {max(abs(a["grad_norm"] - b["grad_norm"]) for a, b in zip(h, again)):.1e}')
```

Output:

```text
same seed, 6 steps, identical: True
weights x (1 + 1e-5 noise) rewards equal at 6/6 steps | max |grad_norm diff| 8.3e-06
seed 1                     rewards equal at 0/6 steps | max |grad_norm diff| 1.7e-01
```

相同的随机种子精确复现了六步，尽管参照训练在训练中穿插了留出评测：评测使用自己的生成器，因此不会让训练的随机数流发生偏移。被扰动 $10^{-5}$ 的权重让梯度范数最多变化 $8 \times 10^{-6}$，没有改变任何奖励。六步中没有一个采样的回合发生变化，因为只有当均匀随机数恰好落在某个概率边界的扰动范围之内时，一次类别采样才会翻转。换一个随机种子则每一步的奖励都变了，梯度范数最多相差 0.17，超过了无故障训练梯度范数的整个范围（0.16 到 0.29）。因此，这次实验中尖峰的主要来源是采样差异，而非所测试量级的数值扰动。在大规模下二者交织在一起：非批次不变的 bf16 算子会随批次组成让对数概率改变约每 token 0.03 nat（13.6 节），足以在早期翻转采样的 token，此后两次训练除了配置之外再无共同之处。

> **问题：在两个代码版本之间二分定位回归。**
>
> **现象：** 当前代码训练得比上周的差，二者之间隔着二十个提交。
>
> **原因：** 其中任何一个。强化学习的回归很少出在 diff 看上去危险的地方；重构一个辅助函数或改变一个默认值，就可能翻转一个符号或错位一个掩码。
>
> **可选方案：** (a) 阅读 diff；(b) 二分查找：测试中点提交，保留包含转变的那一半，重复进行，对 $n$ 个提交需要 $\lceil \log_2 n \rceil$ 次测试；(c) 用成本最低、足以判定故障的测试做二分：有不变量失败时就用不变量，否则用一次随机种子匹配的短训练，这样每次探测都要一次训练，而且只有当回归超过种子离散时才有效（[第 17 章](17-evaluation.md)）。
>
> **建议做法：** (c)。下面的代码块模拟了一段八个提交的历史，其中提交 3 引入了符号错误；搜索在每个提交上只能看到不变量是否通过。

```python
COMMITS = ['cache eval tasks', 'log entropy per turn', 'vectorize collate', 'inline advantage helper',
           'bf16 autocast in sampler', 'rename config fields', 'new task split', 'faster sampler']
BAD = 3                                                # simulated history: commit 3 introduced a sign error
def checks_pass(commit):
    """Does the code at this commit pass every invariant? (The search below never reads BAD.)"""
    bug = 'sign_flip' if commit >= BAD else None
    return all(ok for _, ok in sanity_checks(base, probe(), tok, check_cfg, bug).values())

good, bad = 0, len(COMMITS) - 1                         # commit 0 known good, the last one known bad
while bad - good > 1:
    mid = (good + bad) // 2
    passed = checks_pass(mid)
    print(f'test commit {mid} ({COMMITS[mid]}): {"pass" if passed else "fail"}')
    good, bad = (mid, bad) if passed else (good, mid)
print(f'first bad commit: {bad} ({COMMITS[bad]})')
```

Output:

```text
test commit 3 (inline advantage helper): fail
test commit 1 (log entropy per turn): pass
test commit 2 (vectorize collate): pass
first bad commit: 3 (inline advantage helper)
```

三次测试就在八个提交中定位到了提交 3。以不变量作为测试，二分查找很快，而且答案毫不含糊。以训练作为测试，每次探测都要一次训练，小于种子离散的回归无法被可靠地二分。这就是一旦发现 bug 就为它添加一条不变量的理由：这样下一次同类回归只需一个批次就能找到。`git bisect run` 能在真实历史上自动完成同样的搜索。

## 小结

- 有用的监控面板包含四组指标：结果、行为、优化器状态和一致性。一致性指标（采样器与训练器的对数概率差距、必须恰好为 1 或 0 的比率和 KL）成本最低、最具决定性；在 TinyAgent 上，正常的差距低于 $10^{-7}$。
- 加入诊断记录的训练器副本逐位复现了 `train_rl`，并在其上复现了十种失败。在掩码、陈旧比率、累积和投机故障下，训练奖励都在上升；掩码故障在奖励上升的同时让留出通过率损失了 23 个百分点，奖励投机产生了最好看的奖励曲线和 0 的留出通过率。
- 每种失败都在某一组指标中有其特征：掩码、陈旧比率、温度和在线参考模型故障体现在一致性指标中；学习率体现在优化器和格式指标中；熵坍缩、长度偏差和投机体现在行为指标中。累积 bug 没有任何特征。
- 九项第 0 步不变量（保存为 `sanity_checks`）捕捉到了每一个代码故障，而且各自通过能点名该故障的检查。它们无法捕捉目标或配置层面的失败，那些需要监控面板、无故障参照和留出评测。
- 排查先运行不变量，再看一致性指标，然后与无故障参照和种子离散比较，并始终用独立验证器检验结果。让每一步可重放，有不变量时用不变量二分定位回归。这里所有训练都是小规模的机制演示：这些特征可用于排查其他系统，具体数值阈值则需要重新测量。

## 练习

1. 改变一个假设：运行 18.3 节留下的对照实验。用 `RLConfig(temperature=0.7)` 且不引入故障进行训练，使采样器与训练器都在 0.7 上一致。策略会学会读取吗？然后分别用 `SAMPLER_TEMP = 0.9` 和 `0.5` 重跑该故障。随着温度差距变化，罕见正优势 token 被裁剪的比例和最终留出通过率如何变化？
2. 覆盖 `old_logp` 的故障，在一个 epoch、两个小批次的实验中没有造成明显影响。用 `epochs=4, minibatches=4` 重跑它和无故障训练，并按小批次记录 `approx_kl`。缺失的裁剪从什么时候开始产生影响？
3. 编写一个能在一步之内捕捉熵惩罚故障的不变量。一个候选方案：把所有优势设为 0，除非有意使用正则项，否则正确的更新应保持策略不变。实现它，并列出它会误报的配置。
4. 注入一个九项检查都抓不到的故障，例如在批次被打乱、脱离组排布之后再计算优势。哪项检查最接近抓住它？什么新的不变量能抓住它？
5. `logp_gap` 的阈值 $10^{-4}$ 假设两侧都是 fp32。利用 13.6 节的 bf16 测量结果，提出一个阈值和一个统计量（均值、p99 或最大值），使其能在 bf16 引擎与训练器的组合中把数值不一致与错位一格的掩码区分开。
6. 让尖峰可重放：扩展 `train_logged`，使其在 `grad_norm` 超过其滑动中位数三倍时保存批次和模型状态，并编写 `replay(step)`，精确复现该步记录的统计量。
7. 设计评审：你的 SWE 智能体的训练奖励已经稳定上升了 300 步，熵平滑下降，而留出通过率在最近 100 步中持平。一位同事提议把学习率加倍。写出你会首先执行的排查，按顺序列出每一步读取的指标或测试，以及它支持的决策。
