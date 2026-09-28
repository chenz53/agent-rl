# 第四讲：H200 上第一次真实参数更新

本讲基于 2026-09-28 的实测。代码：`scripts/demo_gpu_update.py`。原始数值：`experiments/results/002-gpu-update.json`。

## 1. 这次究竟训练了什么

使用缓存的 Qwen3-0.6B，调用真实 verl 工具 registry 读取合成病例的两个字段。查询由脚本固定；模型只在看到这些信息后，采样一个 A/B 标签。奖励由合成标签 B 决定。

模型的所有参数都参与反向传播，使用 PyTorch SGD 做一个更新步。优势计算和 clipped policy loss 直接调用固定版本 verl。没有启动分布式 trainer、FSDP、vLLM 或完整 ToolAgentLoop。

这是把真实模型接到 RL 目标上的最小实验，不是多轮查询策略训练，也不代表临床能力改善。

## 2. 从 logits 定义策略

代码中的策略不是全词表采样，而是一个明确受限的二选一分布：

```python
logits = model(**inputs).logits[0, -1, allowed_ids]
return torch.log_softmax(logits / args.temperature, dim=-1)
```

`allowed_ids = [32, 33]` 是该 tokenizer 中 A、B 的单 token ID。temperature=2。

这定义了 π(A|h)、π(B|h)。采样和训练必须使用完全相同的限制与归一化。如果采样只允许 A/B，却用全词表 softmax 的 log-probability 构造重要性比率，就混用了不同策略。

格式 `<final>{...}</final>` 在本实验由脚本包装，不是模型生成动作。这次没有训练 JSON 输出能力。

## 3. 保留旧策略概率，再重新计算带梯度的概率

```python
with torch.no_grad():
    old_distribution = policy_log_probs()
    actions = torch.multinomial(old_distribution.exp(), args.samples, replacement=True)
    old_log_probs = old_distribution[actions].unsqueeze(1).clone()

current_log_probs = policy_log_probs()[actions].unsqueeze(1)
```

`old_log_probs` 是采样时保存的常量；`current_log_probs` 是当前模型的可微计算结果。首次更新前数值相同，计算图角色不同。

本次测得两者最大误差为 0。模型用 eval 模式关闭 dropout，但 eval 不会关闭梯度。

## 4. 实际采样与 advantage

同一病例采样 16 次，没有人为筛选成功样本：14 次 A，2 次 B。A 得 0 分，B 得 1 分。组内平均奖励为 0.125。

调用真实框架：

```python
advantages, _ = compute_grpo_outcome_advantage(
    rewards, mask, np.array([DEMO_CASE.case_id] * args.samples)
)
```

结果：A 样本的 advantage 约 -0.36596；B 样本约 +2.56173。这里使用上游的样本标准差和 epsilon。本次每个 response 只有一个动作 token，所以 mask 全为 1；工具 token 的 mask=0 在单独的集成 probe 中验证，不能说本实验做了真实多轮 token 对齐。

## 5. 为什么 loss 接近 0，梯度仍然很大

在首次更新点，新旧策略相同，ratio=1，且处于 clipping 区间内部。此时 surrogate loss 的值是：

\[
L(\theta_{old})=-\frac{1}{N}\sum_i A_i\approx0.
\]

但其梯度是：

\[
\nabla_\theta L\big|_{\theta_{old}}
=-\frac{1}{N}\sum_i A_i\nabla_\theta\log\pi_\theta(a_i\mid h)\big|_{\theta_{old}}.
\]

advantage 的和为 0，不代表 advantage 加权的 log-probability 梯度之和为 0。

特别注意：ratio 在当前点数值为 1，不代表它是常数函数。若让 old_log_probs 也参与同一计算图，可能错误地抵消梯度。

本次数据：

| 指标 | 实测 |
| --- | --- |
| 更新前 surrogate loss | 1.49×10⁻⁸ |
| 裁剪前全模型梯度 L2 范数 | 142.1208 |
| 梯度裁剪上限 | 1.0 |
| 学习率 | 0.0001 |
| 更新前 P(B) | 0.260893 |
| 更新后 P(B) | 0.269492 |
| PyTorch 峰值 allocated 显存 | 5.676 GiB |

因此，这个实验里不能用“loss 是否显著下降”来判断是否学习。更直接的观察是正确动作的概率、梯度、reward 分布与策略变化。clip fraction 和 approx KL 在更新前均为 0，也不意味着更新后的策略没有变化。

## 6. 两个重要边界

1. 这是一条合成病例上的一个 SGD 步。P(B) 上升验证的是更新方向与数据通路，不能说明泛化。
2. 这里的 π 是温度 2 下 A/B 限制分布，26.09%→26.95% 不是临床正确率，也不是全词表生成 B 的无条件概率。

下一步让“是否查询、查询哪个字段、何时回答”也成为模型采样的动作，再追踪真实多轮 token、工具观测 mask 和每条轨迹的 reward。
