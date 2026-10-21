# 智能体强化学习：从策略梯度到编程智能体

> 本书中文版译自英文版。英文版是唯一的规范版本，代码、运行输出与图均与英文版完全相同。

本书讲解如何用强化学习训练语言模型智能体：让模型调用工具、读取结果，并根据多轮交互的最终结果改进策略。验证器（verifier）负责给结果打分。全书从策略梯度（policy gradient）的定义出发，最终搭建出一套完整、经过测试的小型编程智能体训练栈。沿途会讨论后训练团队每天都要做的设计决策：如何在变长的工具轨迹上聚合损失，何时值得为 critic（价值模型）付出额外开销，如何保持推理引擎与训练器一致，如何在奖励投机（reward hacking）进入评测之前发现它，以及如何读懂一个正在出问题的训练监控面板。

## 本书的组织方式

本书逐章搭建**同一个系统**：

- **BugFix gym**（第 4 章）是一组小型 Python 修复任务。智能体使用 `read`、`test`、`edit` 和 `submit` 这几个工具来修复它们。测试会执行真实代码，隐藏测试用于检查公开用例之外的修复效果，而一个漏洞利用（exploit）操作让奖励投机真实地发生。
- **TinyAgent**（第 3 章）是一个小型 transformer 策略，在 CPU 上用强化学习训练约一分钟即可完成。它能复现大规模训练中出现的现象：格式学习、熵坍缩、长度效应、跨轮次的回报归因以及奖励投机。
- **真实模型实验**在模型特定行为起作用的地方使用真实的 Qwen3-0.6B 分词器和模型，例如对话模板、工具调用 token、重分词漂移，以及推理引擎与训练器之间的对数概率不一致。

每个机制都先推导，再在一个短代码块中从零实现，然后运行。其打印输出直接出现在代码下方。仿照 [D2L](https://github.com/d2l-ai/d2l-zh) 的做法，某一章构建的函数会标记为 `#@save` 并收集进 `agentrl` 包，因此后续章节直接调用 `arl.ppo_policy_loss(...)`，而不必重复实现。到了综合实战（capstone）一章，读者已经亲手写过训练栈的每一个部分。生成包按主题组织为 `policy.py`、`environments.py`、`trajectories.py`、`grpo.py` 等模块，`__init__.py` 只导出接口。[模块索引](../../agentrl/README.md)列出了章节与模块的对应关系。

每章都以**实践中的设计问题**收尾。这一节选取工程师真实遇到的问题，对每个问题给出症状、原因、可选方案和推荐的默认做法。之后每章还有小结和练习。[附录 B](chapters/B-design-review.md) 收录了开放式的设计评审问题，并附参考答案。

## 目录

| 部分 | 章节 | 回答的问题 |
| --- | --- | --- |
| | [0. 符号说明与本书用法](chapters/00-notation.md) | 如何阅读和运行本书？ |
| **I. 基础** | [1. 什么是智能体强化学习？](chapters/01-introduction.md) | 当模型在环境中行动时，什么发生了变化？ |
| | [2. 强化学习基础](chapters/02-rl-fundamentals.md) | 什么是策略、回报和价值？LLM 智能体为何是一个 MDP？ |
| | [3. 作为策略的语言模型](chapters/03-language-models-as-policies.md) | 一个动作的概率究竟是什么？ |
| | [4. 环境与工具：BugFix gym](chapters/04-environments-and-tools.md) | 智能体作用于什么？什么算作成功？ |
| | [5. 轨迹、模板与掩码](chapters/05-trajectories-and-templates.md) | 一个多轮回合如何变成训练张量？ |
| **II. 策略优化** | [6. 策略梯度](chapters/06-policy-gradient.md) | 奖励如何变成梯度？ |
| | [7. 优势与回报归因](chapters/07-advantages-and-credit.md) | 哪些动作应当获得正向反馈？ |
| | [8. PPO：安全地复用样本](chapters/08-ppo.md) | 昂贵的 rollout 如何在多次更新中复用？ |
| | [9. GRPO 与不依赖 critic 的方法](chapters/09-grpo-and-variants.md) | GRPO、Dr. GRPO、DAPO、RLOO 和 GSPO 各自改变了什么？ |
| | [10. KL、熵与正则化](chapters/10-kl-entropy-regularization.md) | 如何防止策略漂移或坍缩？ |
| **III. 奖励与智能体** | [11. 奖励、验证器与奖励投机](chapters/11-rewards-and-verifiers.md) | 应当奖励什么？智能体会如何作弊？ |
| | [12. 多轮智能体与长时程](chapters/12-multi-turn-and-long-horizons.md) | 回合变长时什么会出问题？ |
| **IV. 系统** | [13. Rollout 系统与 off-policy 校正](chapters/13-rollout-systems.md) | 推理引擎、训练器和环境如何协同工作？ |
| | [14. 规模化与效率](chapters/14-scaling-and-efficiency.md) | token、显存和 GPU 时数都花在了哪里？ |
| **V. 超越纯强化学习** | [15. SFT、拒绝采样与偏好](chapters/15-sft-and-rejection-sampling.md) | 何时示范或比较是更好的信号？ |
| | [16. 面向智能体的蒸馏](chapters/16-distillation.md) | 教师如何为学生自己的轨迹打分？ |
| **VI. 实践** | [17. 评测](chapters/17-evaluation.md) | 智能体真的变好了吗？ |
| | [18. 调试强化学习训练](chapters/18-debugging-rl-runs.md) | 常见故障在监控面板上是什么样子？ |
| | [19. 综合实战：端到端训练一个编程智能体](chapters/19-capstone.md) | 我们能否组装并评测整个训练栈？ |
| | [20. 研究前沿与开放问题](chapters/20-frontier.md) | 哪些问题尚无定论？我们会如何检验？ |
| **附录** | [A. 数学工具](chapters/A-math.md) | 各章依赖哪些恒等式和估计量？ |
| | [B. 设计评审问题](chapters/B-design-review.md) | 我能否推理一个真实的后训练设计问题？ |

## 运行代码

每个 Python 代码块都可以运行。检查器按顺序执行各章，从前面章节的 `#@save` 块重新构建 `agentrl`，并将每个代码块的标准输出与记录的输出进行比较：

```sh
scripts/remote_check.sh <workdir> [chapter-prefix ...]   # runs on the GPU host; CPU is enough
python3 scripts/build_library.py                         # regenerate agentrl/ from the chapters
scripts/remote_figures.sh <workdir> [chNN ...]           # regenerate data figures
python3 scripts/build_tutorial_site.py                   # regenerate the Mintlify edition in website/
```

[真实模型实验](labs/README.md)在一块 H200 上用 BugFix gym 训练 Qwen3-0.6B，第 19 章报告了其实测结果。

[编辑指南](../editorial-guide.md)规定了本书的写作约定，[来源台账](../sources.md)记录了每一个外部来源、其查验日期以及查验范围。

## 实验能说明什么，不能说明什么

TinyAgent 实验均实际运行过，但模型和任务集都很小。它们用来解释机制、复现故障，不能作为大模型性能的基准证据。论文报告的结果会标注为文献报告并注明日期。建议会标注为本书的建议。环境只执行为本书编写的可信代码；第 4 章解释了真实的模型生成代码所需要的隔离。
