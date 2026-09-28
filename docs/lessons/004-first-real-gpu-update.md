# 第四讲：逐行解释 H200 上的一次真实 GRPO 参数更新

这节课回答：从模型输出两个 logits 开始，到一次 `optimizer.step()` 结束，究竟发生了什么？每个中间量是什么，为什么需要它，实际数值是多少？

先读 [第一课](001-clinical-reasoning-as-an-environment.md) 的环境与信息边界、[第二讲](002-one-clinical-trajectory.md) 的策略梯度，再结合 [第三讲](003-verl-code-walkthrough.md) 的工具接口。

本节中有三种证据，不能混淆：

- **H200 实测**：真实 Qwen3-0.6B 的采样与一个参数更新，见 [原始结果](../../experiments/results/002-gpu-update.json)。
- **数值复核**：用真实 verl/PyTorch 验证小张量计算，见 [检查脚本](../../scripts/verify_lesson_math.py) 与 [输出](../../experiments/results/003-lesson-math.json)。
- **教学设定**：A/B 是人工标签；没有任何医学依据说明这些病史和检查应对应 B。

完整实际代码：[demo_gpu_update.py](../../scripts/demo_gpu_update.py)。下面标为“实际脚本片段”的代码需要脚本中已有的变量；标为“独立小例子”的代码可在有 PyTorch 的 Python 环境运行。

## 1. 先说清楚这次优化的决策

病例的初始主诉是发热、咳嗽。脚本先调用真实 verl 工具，得到：

```json
[
  {"status": "ok", "field": "exposure", "value": "positive"},
  {"status": "ok", "field": "test_x", "value": "negative"}
]
```

模型看到主诉与上述结果，随后只能选一个动作：A 或 B。它不能选择是否查询、不能改变查询顺序，也不生成长篇推理。

因此，这次实验可视为一个 **contextual bandit**：给定上下文，选择一次动作，立即获得奖励。它是完整序贯 Agent 的一个最小切片。

| 部分 | 谁决定 | 本次是否被 RL 优化 |
| --- | --- | --- |
| 查询 exposure | 脚本 | 否 |
| 查询 test_x | 脚本 | 否 |
| 选择 A 或 B | 模型采样 | 是 |
| 包装成 `<final>...</final>` | 脚本 | 否 |
| 判定答案是否匹配 B | scorer | 固定评分规则，不是策略 |

“真实参数更新”与“完整 Agent 训练”是两个层次。我们已经做了前者，尚未验证模型自主查询。

先给整个数据流的形状建立直觉：

| 数据 | 本次形状 | 每个维度的意义 |
| --- | --- | --- |
| 输入 IDs | `[1,L]` | 1 个相同上下文，L 个输入 token |
| 完整模型 logits | `[1,L,V]` | 每个位置对词表 V 的预测 |
| A/B logits 与 log-probabilities | `[2]` | 两个允许动作 |
| sampled actions | `[16]` | 16 次抽样的动作索引 |
| old/current log-probabilities | `[16,1]` | 16 条 response，每条 1 个动作 token |
| rewards、advantages、mask | `[16,1]` | 与训练动作位置对齐 |
| 最终 loss | 标量 | 对有效动作位置聚合后的目标 |

## 2. 输入如何进入模型：messages、chat template 与 token

实际脚本片段：

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=False
).to("cuda")
```

`messages` 是角色化的消息列表；chat template 把它转成模型训练时使用的序列格式。`add_generation_prompt=True` 让序列停在 assistant 应继续生成的位置。`enable_thinking=False` 是这个模型模板接受的参数，不是所有模型通用的开关。

本次真实 prompt 尾部含有：

```text
<|im_start|>assistant
<think>

</think>

```

这段模板文字已经在输入里，不是本次策略采样的动作。它虽然不是训练目标，仍会影响下一 token 的分布。

`inputs["input_ids"]` 的形状是 `[1, L]`，这里 1 是输入上下文数量，L 是 token 数。我们对同一个上下文采样 16 次，不需要先复制 16 份相同的 forward 输入。

## 3. 从 logits 到策略：π 究竟是什么

### 3.1 logits 不是概率

完整语言模型给出 `[batch, length, vocabulary]` 形状的 logits。我们取最后一个位置，再只取 A/B 对应的两个坐标：

```python
logits = model(**inputs).logits[0, -1, allowed_ids]
log_probs = torch.log_softmax(logits / temperature, dim=-1)
```

本次 `allowed_ids=[32,33]`，分别对应单 token A 和 B。`logits` 与 `log_probs` 的形状都为 `[2]`。

限制动作空间后的策略定义为：

\[
\pi_\theta(a\mid h)=\frac{\exp(z_a/T)}{\exp(z_A/T)+\exp(z_B/T)},\quad a\in\{A,B\}.
\]

这里 θ 是模型参数，h 是完整可见上下文，z 是原始 logits，T 是温度。分母只有 A/B 两项，所以这不是全词表生成概率。

### 3.2 用实际概率重建一个等价的 logits 例子

**独立小例子；重建出的 logits 不是从 Qwen 直接记录的原始 logits。** softmax 对所有 logits 加同一个常量不敏感，所以同一概率有无穷多组等价 logits。

```python
import torch
p = torch.tensor([0.7391070127, 0.2608930171], dtype=torch.float64)
z = 2.0 * p.log()             # 构造一个等价 logits 向量
for temperature in [1.0, 2.0, 4.0]:
    print(temperature, torch.softmax(z / temperature, dim=-1).tolist())
```

数值复核结果约为：

| 温度 | P(A) | P(B) |
| --- | --- | --- |
| 1 | 0.889207 | 0.110793 |
| 2 | 0.739107 | 0.260893 |
| 4 | 0.627304 | 0.372696 |

提高温度使这个两动作分布更平坦，B 更容易被探索到。它不会凭空创造可用工具、正确知识或有效奖励。

**工程要求：采样和训练必须使用同一个概率定义。** 如果按这个二选一分布采样，却用全词表 softmax 的 log-probability 训练，分母变了；如果采样 T=2 而计算训练概率 T=1，也变成了另一策略。重要性比率需要显式处理这种差异，不能假装相同。

## 4. 采样不是 argmax，样本比例也不等于策略概率

实际脚本片段：

```python
with torch.no_grad():
    old_distribution = policy_log_probs()  # 两个 log-probabilities
    actions = torch.multinomial(
        old_distribution.exp(), 16, replacement=True
    )
    old_log_probs = old_distribution[actions].unsqueeze(1).clone()
```

逐项看：

- `exp()`：把 log-probabilities 还原成概率。
- `multinomial(..., replacement=True)`：按相同策略进行 16 次分类抽样；A/B 可以重复出现。
- `actions`：形状 `[16]`，内容是 0 或 1，不是 token ID 32 或 33。
- `old_distribution[actions]`：取出每个实际抽到动作的 log-probability。
- `unsqueeze(1)`：得到 `[16,1]`，适配上游的 `[response 数量, response token 数]` 接口。

实测抽到 14 个 A、2 个 B。因此经验成功率为 2/16=12.5%，但采样前策略 P(B)=26.09%。两者不同不是 bug。

按固定上下文、独立抽样假设，B 的次数 K 服从二项分布：

\[
K\sim\operatorname{Binomial}(16,0.260893),\quad E[K]=4.1743.
\]

正好抽到 2 个 B 的概率约为 11.86%。16 个样本不足以要求经验频率等于模型概率。

**已核验的独立标准库例子：**

```python
from math import comb
p = 0.2608930170536041
print(16 * p)                          # 4.174288...
print(comb(16, 2) * p**2 * (1-p)**14)  # 0.118579...
```

## 5. 旧策略与当前策略：数值相同，计算图角色不同

采样发生时的策略记为 π_old。计算 loss 时正在优化的策略记为 πθ。首次更新前参数还没改变，所以它们的概率值相同；但 π_old 必须作为常量保留。

实际脚本片段：

```python
# 上一个代码块在 no_grad 中产生了 old_log_probs。
current_log_probs = policy_log_probs()[actions].unsqueeze(1)
ratio_error = (current_log_probs.detach() - old_log_probs).abs().max()
```

实测 `ratio_error=0`。这只验证当前受限动作实验的采样/训练概率一致，不验证以后 vLLM 与训练后端的一致性。

为什么重新 forward？因为采样 forward 在 `no_grad()` 下运行，没有留下参数梯度的计算图。训练需要一个保留计算图的 forward。

本例另有：

```python
model.eval()
model.config.use_cache = False
```

- `eval()` 改变 dropout 等层的行为，不关闭 autograd，也不冻结参数。
- `no_grad()` 才是在代码块中关闭梯度记录。
- `use_cache=False` 禁用本例训练 forward 不需要的 KV cache。

把 `model.eval()` 理解成“只能推理、不能训练”是错误的。

## 6. 从模型动作到 reward：0/1 是人为定义的目标

实际脚本片段：

```python
chosen = [labels[i] for i in actions.tolist()]
scores = [
    compute_score(
        "nanoastra_synthetic",
        '<final>' + json.dumps({"diagnosis": label}) + '</final>',
        DEMO_CASE.label,
    )["score"]
    for label in chosen
]
rewards = torch.tensor(scores, device="cuda").unsqueeze(1)
mask = torch.ones_like(rewards)
```

`DEMO_CASE.label` 是 B，所以 14 个 A 得 0、2 个 B 得 1。这里 JSON 格式由脚本保证，模型不需要学会正确包装。

reward tensor 形状为 `[16,1]`，不是 `[1,16]`：16 条独立 sampled responses，每条只有一个被优化的动作 token。

如果一条 response 有几十个生成 token，而只有最终结果奖励，框架常把这个 scalar 放在最后有效位置，其他位置放 0，再沿序列求和获得整条轨迹得分。奖励存放位置和每个 token 最终拿到的 advantage 不是同一件事。

## 7. GRPO advantage：用组内相对结果构造学习信号

GRPO 是 Group Relative Policy Optimization。本节只分析固定 verl 中的 outcome advantage 估计；具体训练还可以组合不同 policy loss、KL 与采样设置，不能把名字当成一套永远相同的完整实现。

### 7.1 reward、value 与 advantage 的区别

- Reward：这次实际发生的评分，如 0 或 1。
- Value：在某个状态/历史下继续执行，预期能得到多少 return。
- Advantage：某个动作相对于参考水平有多好。

经典定义是 `Aπ(h,a)=Qπ(h,a)-Vπ(h)`。本例没有训练 critic 去估计 V；使用同一 prompt 的一组结果构造相对分数。这是 outcome-only 的 GRPO 估计，不能等同于知道每一步的真实 Q−V。

### 7.2 逐项代入本次 16 个样本

组内均值：

\[
\bar R=\frac{14\times0+2\times1}{16}=0.125.
\]

固定 verl 实现调用 `torch.std`，默认用样本标准差，即除以 N−1：

\[
s=\sqrt{\frac{14(0-0.125)^2+2(1-0.125)^2}{15}}
=\sqrt{\frac{1.75}{15}}\approx0.341565.
\]

以 ε=10⁻⁶ 防止零除：

\[
\hat A_A=\frac{0-0.125}{s+\epsilon}\approx-0.365961,
\qquad
\hat A_B=\frac{1-0.125}{s+\epsilon}\approx2.561730.
\]

可直接运行的 PyTorch 小例子：

```python
import torch
r = torch.tensor([0.0] * 14 + [1.0] * 2, dtype=torch.float64)
advantage = (r - r.mean()) / (r.std(correction=1) + 1e-6)
print(r.mean().item(), r.std(correction=1).item())
print(advantage[0].item(), advantage[-1].item())
# 0.125, 0.341565...
# -0.365961..., 2.561730...
```

实际训练脚本没有自己实现上述归一化，而是调用：

```python
advantages, _ = compute_grpo_outcome_advantage(
    rewards, mask, np.array([DEMO_CASE.case_id] * 16)
)
```

`index` 必须标识相同 prompt 的多次采样。本例只有同一个病例、同一个 prompt，所以重复同一个 ID 正确。若同一病例变换了可见上下文，它们也不能仅因为 case ID 一样就随意混为一组。

**group 不等于 minibatch。** 一个 optimizer minibatch 可以包含多个 prompt 的多个 group。归一化要尊重 group 边界；之后的 loss 再按指定方式聚合。把整个 minibatch 的奖励一起标准化会改变相对比较的对象，第三讲提供了具体反例。

### 7.3 组全对或全错时发生什么

对于大小大于 1 的组，若 reward 是 `[0,0,0,0]` 或 `[1,1,1,1]`，每个样本都等于组均值，advantage 全为 0。ε 防止 NaN，不会创造学习信号。

这是 **这个 outcome-relative 项** 没有信号，不代表 KL、熵或其他辅助项也一定没有梯度。本次实验没有添加这些项。

若成功策略出现概率只有 1%，16 次采样全部失败的概率是 `0.99**16≈85.15%`。即使存在好策略，采不到它，组内相对奖励也可能无法指引模型。

本上游实现对 singleton group 有特殊分支，不能把上述结论无条件推广到 N=1；示范要求 N≥2。

### 7.4 mask 如何作用

上游把每条 response 的 scalar advantage 广播到 token 维，再乘 response mask。比如：

```text
scalar advantage = 0.866024
response_mask    = [1, 1, 0, 0, 1, 1]
result           = [0.866024, 0.866024, 0, 0, 0.866024, 0.866024]
```

中间两个位置是假想工具返回的 token。这个示意在真实 verl CPU probe 中验证过；H200 模型实验每条只有一个动作 token，mask 为 `[1]`，没有实际多轮 tokenization。

## 8. 概率比率 ratio：当前策略偏离采样策略多少

对样本 i 的动作 a_i：

\[
\rho_i(\theta)=\frac{\pi_\theta(a_i\mid h_i)}{\pi_{old}(a_i\mid h_i)}
=\exp(\log\pi_\theta(a_i\mid h_i)-\log\pi_{old}(a_i\mid h_i)).
\]

例如一个 B 样本，采样时概率 0.25、当前概率 0.30，ratio=1.2，表示该动作概率相对增加 20%，不是增加 20 个百分点。

```python
import math
print(math.exp(math.log(0.30) - math.log(0.25)))  # 1.2
```

保存 old log-probability 后，可对采样 batch 做多个优化步骤；随着参数更新，ratio 会偏离 1。它在 surrogate 中反映这种变化。PPO 仍然是近似更新方法，不能把 token-level ratio 理解成任意旧轨迹分布都已被完全校正。

本次只有一次更新，计算 loss 时还没有改参数，所以所有 ratio=1。

## 9. Clipped policy loss：正负 advantage 要分别看

这里使用 PPO（Proximal Policy Optimization）风格的 clipped surrogate。surrogate 是用于参数更新的代理目标；它不是临床 reward 本身，也不是“当前模型准确率”的另一种表示。

### 9.1 从最大化收益到最小化 loss

忽略额外修正时，逐 token 的标准 clipped loss 为：

\[
\ell_i=\max\left(-\rho_i A_i,
-\operatorname{clip}(\rho_i,1-\epsilon_c,1+\epsilon_c)A_i\right).
\]

这里 ε_c=0.2 是策略裁剪范围；不要和 GRPO 分母里的 ε=10⁻⁶ 混淆。

这是最大化 clipped surrogate 后取负号的结果，所以是 `max`。最终对 response mask 允许的位置聚合；本例 token-mean 等于 16 个单 token loss 的均值。

### 9.2 正 advantage：别继续奖励已经过大的概率增长

令 A=2，ratio=1.5：

```text
未裁剪 loss = -1.5 × 2 = -3.0
裁剪后 loss = -1.2 × 2 = -2.4
最终取 max = -2.4
```

在这一段，继续增大 ratio 不再让该项 loss 下降，其对 log-probability 的导数为 0。

但若 A=2、ratio=0.6，最终是 max(-1.2,-1.6)=-1.2，梯度仍存在：一个好动作的概率反而下降，应当受到纠正。

### 9.3 负 advantage：别继续奖励已经过大的概率下降

令 A=-1、ratio=0.5：未裁剪为 0.5，裁剪为 0.8，取 max 得 0.8，继续降低概率没有额外收益。

反过来，A=-1、ratio=1.5，最终 loss=1.5，梯度仍存在：坏动作的概率正在升高，需要纠正。

用真实上游函数验证的结果：

| A | ratio | 最终 loss | 对 current log-prob 的导数 |
| --- | --- | --- | --- |
| +2 | 1.5 | -2.4 | 0 |
| +2 | 0.6 | -1.2 | -1.2 |
| -1 | 0.5 | 0.8 | 0 |
| -1 | 1.5 | 1.5 | +1.5 |

因此“ratio 超出 [0.8,1.2] 就一定没有梯度”是错的。也不能把 PPO clipping 当成硬性保证所有策略概率都留在区间内；共享参数和其他样本仍会改变它们。

### 9.4 我们用到的上游还有 dual-clip

固定版本 `compute_policy_loss_vanilla` 对负 advantage 额外使用 c=3：`min(standard_loss, -c*A)`。例如 A=-1、ratio=5，标准 loss 为 5，dual-clip 后为 3，防止这类极端项继续增大。

本次实际 ratio=1，这个额外分支没有改变结果。后续比较不同实现时，不能只看到函数叫 vanilla 就默认与最简 PPO 公式完全一致。

## 10. loss≈0 为何不是梯度≈0：算出具体导数

首次更新时 ratio=1，组内 advantage 的和为 0，所以：

\[
L(\theta_{old})=-\frac1{16}\sum_i A_i\approx0.
\]

但 **ratio 在这一点的值为 1，不代表它是常数函数**：

\[
\nabla_\theta L\big|_{\theta_{old}}
=-\frac1{16}\sum_i A_i\nabla_\theta\log\pi_\theta(a_i\mid h).
\]

### 10.1 对两个 logits 手算

令 z_B 是 softmax 前的 B logit，T=2：

\[
\frac{\partial\log\pi(a)}{\partial z_B}
=\frac{\mathbf1[a=B]-\pi(B)}{T}.
\]

代入 14 个 A、2 个 B 与 P(B)=0.260893，得到：

\[
\frac{\partial L}{\partial z_B}
=-\frac1{16}\left[
14(-0.365961)\frac{-0.260893}{2}
+2(2.561730)\frac{1-0.260893}{2}
\right]\approx-0.160108.
\]

同理 ∂L/∂z_A≈+0.160108。做梯度下降时，局部方向是降低 A logit、提高 B logit。

### 10.2 独立 autograd 例子

这是等价两 logit 小张量，不是重新加载 Qwen：

```python
import torch
p = torch.tensor([0.7391070127, 0.2608930171], dtype=torch.float64)
z = (2 * p.log()).requires_grad_()
actions = torch.tensor([0] * 14 + [1] * 2)
r = (actions == 1).double()
adv = (r - r.mean()) / (r.std(correction=1) + 1e-6)
logp = torch.log_softmax(z / 2, dim=-1)[actions]
old_logp = logp.detach()
ratio = (logp - old_logp).exp()
loss = -(ratio * adv).mean()  # 此点 ratio=1，处于未裁剪区
loss.backward()
print(loss.item(), z.grad.tolist())
# loss ≈ 0
# gradient ≈ [+0.160108, -0.160108]
```

这个数值例子也已用上游真实 clipped loss 复核。

### 10.3 一个会让训练静默失效的错误

若把上一段的 `old_logp = logp.detach()` 改成：

```python
old_logp = logp  # 错：新旧 log-probability 使用同一计算图
```

ratio 变成 `exp(f(θ)-f(θ))`，整个函数恒等于 1，导数抵消。已核验两 logit 的梯度变为 `[0,0]`。

注意 `.clone()` 本身不会 detach。实际脚本中的 clone 安全，是因为 old log-probability 在 `torch.no_grad()` 内计算。

## 11. 从 logits 梯度到 6 亿参数的更新

Qwen 实际优化的是 θ，不是把两个 logits 当独立参数。链式法则为：

\[
\nabla_\theta L=\sum_{a\in\{A,B\}}\frac{\partial L}{\partial z_a}\nabla_\theta z_a.
\]

所以两 logit 梯度约 ±0.1601，与全模型梯度范数 142.12 没有矛盾；二者对应不同坐标空间。

实际脚本片段：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

本例 optimizer 是 `SGD(model.parameters(), lr=1e-4)`，没有 momentum 或 weight decay。`zero_grad` 清理上一次的梯度；PyTorch 默认 backward 会累加梯度，不能遗漏这一步。

`clip_grad_norm_` 返回的是 **裁剪前** 的全局 L2 范数。实测为 142.1208，大于 1，所以梯度大约乘以：

\[
1/142.1208\approx0.00703627.
\]

然后做 θ_new=θ_old−10⁻⁴g_clipped。在精确算术下、仅这个 SGD 更新且裁剪后范数为 1 时，参数增量的 L2 范数约为 10⁻⁴；浮点舍入会影响实测增量。小的参数位移仍可能通过网络 Jacobian 导致可见的输出概率变化。

**梯度裁剪不是策略 ratio clipping。** 前者限制参数梯度向量范数；后者改变 surrogate loss 的局部形状。本次 ratio clipping 没触发，梯度范数裁剪触发了。

## 12. 如何读实测结果，避免得出过强结论

| 量 | 实测值 | 能说明什么 |
| --- | --- | --- |
| 16 次采样 | 14 A、2 B | 这次 batch 的动作与奖励组成 |
| P(B) 更新前 | 0.260893 | 更新前受限策略的概率 |
| P(B) 更新后 | 0.269492 | 同一上下文下，概率增加约 0.8599 个百分点 |
| loss 更新前 | 1.49×10⁻⁸ | 居中的 advantage 使初始 surrogate 值接近 0 |
| 梯度范数 | 142.1208 | 模型获得非零训练信号 |
| clip fraction | 0 | 更新前未落入产生 clipping 的区域 |
| approx KL | 0 | 更新前 old/current log-probability 一致 |
| 峰值 allocated | 5.676 GiB | PyTorch 记录的峰值张量显存；不是所有设备开销 |

没有计算完整临床评估，也没有建立收敛曲线。26.95% 不是测试集准确率。approx KL 是上游记录的采样 log-probability 差值估计，也不是对 reference model 的显式 KL 正则项；本实验没有加 reference KL。

### 12.1 old policy、reference policy 与 KL 不要混为一谈

old policy 是生成当前 batch 的策略，用于 ratio 分母。reference policy 通常是另一个固定或按方案更新的参考模型，用于约束行为偏移；它不必等于本 batch 的 old policy。

在同一上下文下，KL divergence 可写为 `KL(p||q)=Σ_a p(a)log[p(a)/q(a)]`。它比较两个分布，方向不同一般数值不同。例如 p=[0.8,0.2]、q=[0.5,0.5] 时，KL(p||q)≈0.1927 nats。它不是两个模型答案是否相同的 0/1 指标。

上游这里记录 `mean(logπ_old−logπ_current)`，在动作确实来自 old policy 时对应 KL(old||current) 的一个采样估计，有限样本下可有噪声。首次更新前二者相等，所以记录为 0；这与是否另加 `β×KL(πθ||π_ref)` 正则是两个问题。本实验没有 reference model。

### 12.2 entropy 是什么，为什么也可能提供梯度

Entropy 为 `H(π)=−Σ_a π(a)logπ(a)`，度量一个策略分布有多分散。二选一均匀分布 [0.5,0.5] 的 entropy 是 log2≈0.6931 nats；确定选一个动作时趋于 0。

若目标额外奖励 entropy，即使 outcome advantage 全为 0，也可能因 entropy 项而更新策略。但更高 entropy 只是更多不确定性/探索，不等于更有用的临床行为。本次 loss 不含 entropy 项，不能用它解释本次观察到的更新。

独立标准库例子，同时计算两个量：

```python
import math
p, q = [0.8, 0.2], [0.5, 0.5]
kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
entropy_p = -sum(pi * math.log(pi) for pi in p)
entropy_q = -sum(qi * math.log(qi) for qi in q)
print(round(kl, 6), round(entropy_p, 6), round(entropy_q, 6))
# 0.192745 0.500402 0.693147
```

本实验只检验一次奖励驱动更新的方向与接口。模型可能只是朝一个病例的人工标签移动；泛化需要独立病例集、多 seed 与对照实验。

## 13. 运行与复查

在已配置的 vp 上运行数值复核，不加载模型：

```sh
cd /home/user/nanoAstra
.venv/bin/python scripts/verify_lesson_math.py
```

重做真实单步模型实验：

```sh
.venv/bin/python scripts/demo_gpu_update.py \
  --model /home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --seed 42 --samples 16 --temperature 2 --learning-rate 0.0001
```

模型参数仅在进程内更新，结束后没有保存 checkpoint。seed 与 snapshot 固定能减少变化，但不承诺跨硬件/软件逐 bit 一致。

读源码时重点对照：

- [GRPO 的 group mean/std 与 mask](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268)。
- [ratio、标准 clip 与 dual-clip](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286)。

## 14. 自检：现在应能解释的三个现象

1. **loss 为零，但 P(B) 上升。** 解释应涉及函数值与导数、old log-probability 的常量角色。
2. **全成功的 batch 反而没有 outcome advantage。** 解释应涉及组内相对奖励，而不是声称模型已经最优。
3. **更高温度可能让训练起步，却也改变 log-probability。** 解释应同时涉及探索和采样/训练策略一致性。

下一阶段才把查询动作纳入策略，观察这套机制在多轮轨迹上如何变化。
