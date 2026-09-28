# nanoAstra 教材与概念索引

教材面向有基础模型训练经验的学习者。每课围绕具体任务与代码讲解：定义 → 机制 → 数值实例 → snippet → 输出 → 失败模式。

## 四课主线

| 课程 | 主问题 | 代码或证据 |
| --- | --- | --- |
| [1. 临床推理环境](001-clinical-reasoning-as-an-environment.md) | 到底让模型做什么决策？ | `clinical.py` 与合成状态转移 |
| [2. 轨迹与策略梯度](002-one-clinical-trajectory.md) | 最终奖励怎样影响前面的动作？ | REINFORCE、return、mask 小张量 |
| [3. verl 调用链](003-verl-code-walkthrough.md) | 样本如何经过工具与评分器？ | 真实 registry/BaseTool/reward 接口 |
| [4. H200 真实更新](004-first-real-gpu-update.md) | 每个 tensor 怎样形成一次参数更新？ | Qwen 实测与数值复核 |

## 按概念查找

| 概念 | 位置 | 具体例子 |
| --- | --- | --- |
| state / observation / history | 第 1 课 §2 | 完整病例、一次查询返回、交互历史 |
| POMDP / belief | 第 1 课 §3 | 相同主诉、不同隐藏状态与 Bayes 更新 |
| action 粒度 | 第 1 课 §4 | token、工具调用、整条轨迹 |
| transition / reset / termination | 第 1 课 §5–6 | 查询扣预算与新 episode 清理 |
| reward / return / 科学效用 | 第 1 课 §7–8 | 0.95−0.05 与 0.80 的比较 |
| rollout / trajectory | 第 2 课 §1–2 | 三步交互及轨迹概率 0.16 |
| score-function gradient | 第 2 课 §3 | 为什么不对数据库求导 |
| return-to-go | 第 2 课 §4 | [0.90,0.95,1.00] |
| baseline / advantage / credit | 第 2 课 §5–7 | 成功 +0.6、失败 −0.4，无用动作也获正权重 |
| on/off-policy | 第 2 课 §6 | 旧 batch 与更新后的策略 |
| token alignment / masks | 第 2 课 §8–9 | 环境位置梯度项为 0，但仍是上下文 |
| schema / runtime 校验 | 第 3 课 §3–5 | 模型参数与环境注入参数的区别 |
| 工具生命周期 | 第 3 课 §5 | create→execute→release 后实例数为 0 |
| reward 数据流 / 分组 | 第 3 课 §7–9 | 未接入的查询成本、跨 prompt 混组错误 |
| logits / temperature / policy | 第 4 课 §3 | T=1、2、4 下 P(B) 的变化 |
| sampling / old log-prob | 第 4 课 §4–5 | 16 次采样与固定旧计算图 |
| GRPO / std / group / mask | 第 4 课 §7 | 14 负例、2 正例与 ±不同 advantage |
| ratio / PPO clipping / dual-clip | 第 4 课 §8–9 | 正负 advantage 下的 5 个裁剪例子 |
| loss 与 gradient | 第 4 课 §10 | loss≈0、logit 梯度±0.160108 |
| gradient clipping / SGD | 第 4 课 §11 | 范数 142.12→1，P(B) 上升 |
| old/reference policy / KL / entropy | 第 4 课 §12 | 采样偏移、参考约束与探索的区别 |

## 代码验证范围

42 个 Python 片段通过 AST 解析；其中 29 个作为可运行片段在 vp 现有环境中执行，输出保存到 [snippet 记录](../../experiments/results/003-lesson-snippets.json)。有上下文依赖的方法片段与故意错误的片段没有被当成独立程序运行。

[数学检查](../../scripts/verify_lesson_math.py) 另使用真实 verl 验证组内标准化、分组错误、mask、裁剪和梯度；[结果](../../experiments/results/003-lesson-math.json) 与 [验证说明](../../experiments/003-teaching-verification.md) 可核对。

这里的数值检查不重新训练模型。H200 实测仍引用原始实验 002，未将推导数值伪装成新的模型结果。
