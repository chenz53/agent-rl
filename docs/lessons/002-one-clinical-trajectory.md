# 第二讲：从一条交互轨迹到策略梯度

[第一课](001-clinical-reasoning-as-an-environment.md) 定义了环境。本课要解释：模型选择查询、观察结果、再给答案，这整段过程为何能用一个最终奖励来训练？又为什么这个奖励不足以告诉我们每一步的贡献？

本课多轮轨迹是教学构造。已运行的 H200 实验只优化最后的 A/B 判断；不要把二者合并描述成已完成的多轮训练。

## 1. 把 rollout 展开，而不是只写一句“Agent 与环境交互”

一个 episode 是任务从 reset 到结束的运行区间；rollout 是让策略在环境中实际运行并收集结果的过程；trajectory τ 是该过程保存下来的数据序列。这些词有时在代码里被混用，我们这里区分它们。

用统一工具 `get_clinical_record` 写一条轨迹：

| t | 动作前模型可见历史 h_t | 模型动作 a_t | 新观测 o_{t+1} | 步骤奖励 r_t |
| --- | --- | --- | --- | --- |
| 0 | 主诉 | 查询 exposure | positive | 0 |
| 1 | 主诉、第一次查询及结果 | 查询 test_x | negative | 0 |
| 2 | 上述全部历史 | 回答 B | 结束标志 | 1 |

如果给查询加成本，前两个 r_t 可以变成 -0.05；本表先只用终局正确性奖励。

用标准库构造一个可检查的记录例子：

```python
trajectory = [
    {"action": {"field": "exposure"}, "observation": "positive", "reward": 0.0},
    {"action": {"field": "test_x"}, "observation": "negative", "reward": 0.0},
    {"action": {"answer": "B"}, "observation": "terminated", "reward": 1.0},
]
print(sum(step["reward"] for step in trajectory))  # 1.0
```

真实训练还需要保存策略版本、原始 generated token IDs、采样 log-probabilities、mask 和终止原因。只存一段自然语言对话，不足以可靠重建训练 batch。

## 2. 轨迹概率：策略和环境各负责哪一部分

条件于任务 x，令工具环境为 E：

\[
p_\theta(\tau\mid x,E)
=p_E(o_0\mid x)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)P_E(o_{t+1}\mid h_t,a_t).
\]

πθ 是模型选择动作的概率；P_E 是环境产生新观测的概率。病例读取通常是确定性的：给定正确字段，返回固定值，对这条实际返回的条件概率是 1。其他环境可以有随机性，例如检索返回波动或模拟器随机响应。

**人为数值例子：**三次模型动作的条件概率依次为 0.4、0.5、0.8；环境返回确定，则这条轨迹的策略部分概率为：

```python
from math import log
probs = [0.4, 0.5, 0.8]
print(probs[0] * probs[1] * probs[2])  # 0.16
print(sum(log(p) for p in probs))     # 约 -1.832581
```

0.8 是“已经查询并看到结果后回答 B”的概率，不是“只看主诉时回答 B”的概率。忽略条件上下文会把不同策略问题混在一起。

## 3. 为什么环境不需要可微

本课的策略梯度推导取有限 episode、γ=1。第四节会给出一般 return 的定义；若改成折扣目标，需要一致处理时间权重，不能只替换一个 return 公式就认为整套推导不变。

目标是提升期望回报：

\[
J(\theta)=E_{\tau\sim p_\theta}[R(\tau)].
\]

在 reward 和环境转移没有直接 θ 依赖、并满足通常的可微与积分交换条件下，使用 likelihood-ratio identity：

\[
\nabla J=\sum_\tau R(\tau)\nabla p_\theta(\tau)
=E_\tau[R(\tau)\nabla\log p_\theta(\tau)].
\]

如果环境分布固定，log 轨迹概率的显式参数梯度来自模型动作项：

\[
\nabla\log p_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t\mid h_t).
\]

我们不对数据库结果“positive”求导，而对“选择了这个查询”的概率求导。环境反馈仍然非常重要：它改变后续 history 和最终 reward，进而影响采样到什么梯度。

这并不是环境分布对学习没有影响。随着策略改变，访问到的 history 分布也改变；策略梯度的轨迹期望正包含这种影响。

## 4. reward 与 return-to-go：未来结果为什么影响早期动作

给定步骤奖励 `[-0.05,-0.05,1]`，γ=1，从每个动作开始累计的 return-to-go 分别为 `[0.90,0.95,1.00]`。

独立小例子：

```python
rewards = [-0.05, -0.05, 1.0]
gamma = 1.0
returns = [
    sum((gamma ** (k-t)) * rewards[k] for k in range(t, len(rewards)))
    for t in range(len(rewards))
]
print(returns)  # [0.9, 0.95, 1.0]
```

第 0 步动作影响后面整个过程，所以其 return 包含后续成功奖励；第 2 步不应该为发生在它之前的查询成本承担因果影响。用 return-to-go 去掉过去奖励，可以降低无用噪声，而不改变标准 on-policy 策略梯度的期望。

只在最终给 1 分且 γ=1 时，三个动作的 return 都是 1。这不意味着它们贡献相等，只意味着这份反馈没有提供更细的区分。

## 5. baseline：为什么“比通常表现好多少”比原始 reward 更有用

最简单的 REINFORCE 梯度估计为 `R × Σ∇logπ`。每个回报都乘上动作 log-probability 的梯度；即使所有轨迹回报都差不多，也会有很大采样噪声。

加入不依赖当前采样动作的 baseline b(h_t)：

\[
\hat g=\sum_t(G_t-b(h_t))\nabla\log\pi_\theta(a_t\mid h_t).
\]

为什么可以减掉它？对固定 h，动作概率加起来是 1：

\[
E_{a\sim\pi}[b(h)\nabla\log\pi(a\mid h)]
=b(h)\sum_a\nabla\pi(a\mid h)
=b(h)\nabla1=0.
\]

因此，在上述条件下 baseline 项期望为 0，可以改善方差而不改变梯度期望。若 baseline 是学习到的 value，policy loss 中通常将它视为常量，value 自己通过独立目标训练。

**具体数值：**假设某类起始上下文通常成功率 0.4，我们先用 b=0.4：

| 结果 | R | R−b | 对采样动作项的权重 |
| --- | --- | --- | --- |
| 成功 | 1 | +0.6 | 正权重 |
| 失败 | 0 | −0.4 | 负权重 |

没有 baseline 时，R=0 的失败轨迹在这个最简单估计里贡献为 0。加入 baseline 后失败也能提供负权重；这不代表它凭空增加了真实信息，而是改变了估计器的方差结构。

GRPO 的同组均值含有当前样本自己的 reward，再除以同组标准差。它不是本节“严格动作无关 baseline”证明的直接套用，不能因此宣称所有 GRPO 估计都严格无偏。后续用组内相对信号解释它，不把两种估计混为一谈。

## 6. 写成 loss：为什么是负号

我们通常用优化器最小化 loss，而 J 是要最大化的期望回报。在当前策略采样、仅作一次局部梯度解释时，可以写：

\[
L=-\sum_t\operatorname{stopgrad}(G_t-b_t)\log\pi_\theta(a_t\mid h_t).
\]

下面使用彼此独立的 log-probability 变量，演示 loss 对这些中间变量的导数；它们不是一个完整归一化策略，也不是模型参数：

```python
import torch
logp = torch.tensor([-1.0, -0.7, -0.3], requires_grad=True)
weight = torch.tensor([0.6, 0.6, 0.6])
loss = -(weight.detach() * logp).sum()
loss.backward()
print(loss.item())   # 约 1.2
print(logp.grad)     # tensor([-0.6, -0.6, -0.6])
```

梯度下降会在这个局部视角中推高这些 log-probability。但真实模型参数共享，一个 batch 更新不保证每个动作概率都单调增加。

若对旧策略采到的 batch 重复更新，不能一直当作每次都是当前策略新样本。PPO/GRPO 的 ratio 等机制用于构造受控 surrogate；第四讲会实际计算。

这里的 **on-policy** 指数据由要优化的当前策略采样；**off-policy** 指数据来自不同的行为策略。比如采样时 P(B)=0.2，经过几次更新后 P(B)=0.8，同一个旧 batch 已经不是新策略下的原始样本。PPO 常先采一批数据，再固定采样时的概率做若干小步更新；称它是 on-policy 家族，不意味着这些小步之间的分布完全没变。

## 7. 信用分配：成功轨迹里的无用动作也会拿到正权重

假设第一步查询 exposure 实际与任务标签独立，第二步 test_x 才有帮助。该轨迹最终成功，使用上面的终局权重时，第一步仍拿到 +0.6。

策略梯度不是逐步因果归因算法。大量轨迹中无用查询可能不再与额外收益关联；加入成本也会改变它的净价值，但单条成功轨迹本身没有告诉训练器哪一步必要。

要判断查询是否有帮助，可比较在相同起始信息下：允许/禁止该查询、替换查询结果、固定后续策略等条件。但反事实对照必须明确定义；直接删除一段文本后让模型重新回答，也改变了后续行为，不能自动解释为该查询的纯因果贡献。

## 8. 一个工具动作如何变成 token-level log-probability

一次 JSON 工具调用是 token 序列 y_1...y_m：

\[
\log\pi_\theta(a\mid h)=\sum_{j=1}^{m}\log p_\theta(y_j\mid h,y_{<j}).
\]

如果用终局权重给整次动作里的 token 同一个 advantage，是在提高/降低这段 sampled sequence 的概率。它并不意味着每个 token 在语义上同样关键。

这里有个常见工程错误：把生成文本 decode 后再 encode，认为必定还原相同 token IDs。特殊 token、模板、空格和 tokenizer 行为可能打破这一点。训练记录应优先保留推理时的原始 sampled IDs 与对应 log-probability。

更细的难点是预测位置对齐：causal LM 第 k 个位置的 logits 预测下一个 token，不能不作 shift 就拿同一位置的 token 当 label。

## 9. attention mask 与 response/loss mask 是不同东西

考虑下面六个**人工 token 位置**：

```text
位置              0       1        2       3       4       5
来源            查询名   参数     工具值   工具值   答案前缀  B
attention mask    1       1        1       1       1       1
response mask     1       1        0       0       1       1
```

attention mask 说这些位置是不是有效上下文；response mask 说哪些 sampled action 位置进入策略目标。工具值需要被模型读到，所以不是从 attention 中删除，只是不作为策略采样动作计算 policy loss。

用独立 log-probability 变量演示：

```python
import torch
logp = torch.tensor([-1., -1., -1., -1., -1., -1.], requires_grad=True)
mask = torch.tensor([1., 1., 0., 0., 1., 1.])
advantage = 0.6
loss = -(logp * advantage * mask).sum() / mask.sum()
loss.backward()
print(logp.grad)
# tensor([-0.15, -0.15, 0., 0., -0.15, -0.15])
```

这些 0 只说明工具位置不作为独立动作 likelihood 项。工具 token 的 embedding、表示和注意力仍可能通过后续答案的 loss 收到参数梯度；不要把 mask=0 理解成模型完全不学习如何读取工具信息。

本例除以有效动作 token 数 4。若错误地除以总长度 6，每段工具文本的长度都会改变 loss 的缩放。不同的 token-mean/sequence-mean 聚合策略也会改变长短轨迹的相对权重，必须读具体实现。

## 10. 把理论变成需要保存的数据

| 字段 | 例子 | 少了它会出现什么问题 |
| --- | --- | --- |
| task/prompt 标识 | 同一病例和同一可见上下文的版本 | GRPO 分组错误 |
| 策略版本 | rollout 时 checkpoint/step | 不清楚数据是否过旧 |
| 生成 token IDs | 模型原始采样结果 | decode/re-encode 后概率错位 |
| 采样 log-probabilities | 每个动作 token 的旧策略概率 | ratio 的分母不可信 |
| response mask | assistant=1，工具/填充=0 | 在环境输出上错误训练策略 |
| 最终分数及诊断指标 | score、format_valid、tool_error | 把格式失败或工具故障混为任务失败 |
| stop reason | 回答完成/预算/超时 | 无法判断失败的来源 |

已有 probe 用的是人工 token 位置，真实多轮模型轨迹尚需后续验证。下一讲沿 verl 的工具、数据与 reward 调用链，说明这些概念落在哪些代码接口上。

## 11. 自检与答案线索

- **为什么只给最终 reward 也能训练前面的查询？** 轨迹 likelihood 含有查询概率，score-function 梯度把最终 return 乘回这些 log-probability 梯度。
- **为什么工具不需要可微？** 优化的是选择动作的概率，不是直接对数据库求导。
- **为什么把工具位置 mask 掉不会让模型“看不见工具”？** attention/context 与 policy-loss 位置承担不同职责。
- **为什么一个成功轨迹不足以证明所有步骤有效？** reward 缺少逐步反事实信息，终局权重会同时作用于有用和无用的 sampled actions。

理论背景：[Sutton 等人的策略梯度论文](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)。本文推导与玩具代码是教学解释，不是该论文的实验结果。
