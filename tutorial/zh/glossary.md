# 术语表与翻译约定

本书中文版由英文版翻译而来。英文版（`tutorial/chapters/`）是唯一的规范来源：代码、运行输出、数学公式与图都在英文版中执行和校验，中文版逐字保留它们，只翻译正文。本页规定译名，保证全书用词一致。

## 翻译约定

1. **只翻译正文。** 标题、段落、列表、表格中的文字、图注（`![...]` 方括号内的文字）、提示框（Pitfall、In production、Evidence 等）、小结与练习都要翻译。代码块、`Output:` 之后的输出块、`$$` 数学块、Mermaid 图、图片路径一律原样保留，不改一个字符。`scripts/check_translation.py` 会逐字比对。
2. **`Output:` 标签保持英文。** 它是校验器识别输出块的标记。
3. **术语首次出现时附英文。** 每章第一次出现某个术语时写成“策略梯度（policy gradient）”，之后只用中文。缩写（PPO、GRPO、KL、SFT、RLVR、GAE、DPO 等）直接使用，不翻译。
4. **行内代码保持原样。** `` `train_rl` ``、`` `action_mask` ``、`` `<call> read </call>` `` 等不翻译。函数名、参数名、配置字段、文件名、工具名（`read`、`test`、`edit`、`submit`）都保持英文。
5. **专有名称不翻译。** BugFix gym、TinyAgent、Qwen3-0.6B、DeepSWE、SWE-bench、verl、vLLM 等保持原文。
6. **章节编号与结构不变。** 标题保留编号（如 `## 6.1 ...`），标题数量与层级必须和英文版一致。
7. **链接。** 章节之间的链接指向中文版的同名文件（如 `06-policy-gradient.md`）。带锚点的链接要改成中文标题生成的锚点：锚点规则是小写，去掉标点，空格换成连字符，保留汉字。例如 `## 6.1 得分函数估计量` 的锚点是 `#61-得分函数估计量`。指向来源台账、编辑指南等只有英文的页面时，路径改为 `../../sources.md` 这样的相对路径。图片路径从 `../figures/` 改为 `../../figures/`，图片本身不变。
8. **文风。** 面向有经验的大模型工程师，使用准确、简洁的技术中文。可参考《动手学深度学习》（d2l-zh）的行文，但不要照搬其文字。数字、单位、百分比与英文版完全一致。中英文之间不强制加空格，但全书保持一致：中文与英文单词、数字之间加一个空格。使用中文全角标点，行内代码与公式两侧不加全角引号。
9. **证据标签不变。** 英文版中“本书测得”“论文报告”“假设或建议”的区分，必须在中文中同样清楚，不得把报告的结果写成测得的结果。

## 译名表

| English | 中文 |
| --- | --- |
| agent | 智能体 |
| agentic RL | 智能体强化学习（agentic RL） |
| policy | 策略 |
| behavior policy | 行为策略 |
| reference policy | 参考策略 |
| reward | 奖励 |
| return | 回报 |
| value function | 价值函数 |
| action-value function | 动作价值函数 |
| advantage | 优势 |
| baseline | 基线 |
| critic / value model | 评论家 / 价值模型 |
| actor | 行动者 |
| trajectory | 轨迹 |
| episode | 回合 |
| rollout | rollout（采样轨迹的过程；可译“轨迹采样”，正文中多数保留 rollout） |
| step / turn | 步 / 轮 |
| multi-turn | 多轮 |
| horizon | 时间跨度（long horizon：长时程） |
| observation | 观测 |
| action | 动作 |
| state | 状态 |
| environment | 环境 |
| tool / tool call | 工具 / 工具调用 |
| verifier | 验证器 |
| reward hacking | 奖励投机（reward hacking） |
| exploit | 漏洞利用 |
| reward model | 奖励模型 |
| process reward | 过程奖励 |
| outcome reward | 结果奖励 |
| reward shaping | 奖励塑形 |
| potential-based shaping | 基于势函数的塑形 |
| credit assignment | 信用分配 |
| discount factor | 折扣因子 |
| policy gradient | 策略梯度 |
| score function | 得分函数 |
| surrogate loss / objective | 替代损失 / 替代目标 |
| importance sampling | 重要性采样 |
| importance ratio / probability ratio | 重要性比率 / 概率比 |
| clipping | 裁剪 |
| clip fraction | 裁剪比例 |
| trust region | 信赖域 |
| group (in GRPO) | 组 |
| group size | 组大小 |
| on-policy / off-policy | 同策略 / 异策略 |
| staleness | 陈旧度 |
| asynchronous | 异步 |
| entropy | 熵 |
| entropy collapse | 熵坍缩 |
| KL divergence | KL 散度 |
| forward / reverse KL | 前向 KL / 反向 KL |
| estimator | 估计量 |
| unbiased / bias / variance | 无偏 / 偏差 / 方差 |
| control variate | 控制变量 |
| exploration | 探索 |
| bandit | 老虎机（bandit） |
| Markov decision process (MDP) | 马尔可夫决策过程（MDP） |
| partially observable | 部分可观测 |
| Bellman equation | 贝尔曼方程 |
| policy evaluation | 策略评估 |
| value iteration | 价值迭代 |
| temporal-difference (TD) error | 时序差分（TD）误差 |
| generalized advantage estimation | 广义优势估计（GAE） |
| token | token（不译为“词元”，保持英文） |
| logits / log-probability | logits / 对数概率 |
| vocabulary | 词表 |
| tokenizer | 分词器 |
| chat template | 对话模板 |
| retokenization drift | 重分词漂移 |
| action mask / loss mask | 动作掩码 / 损失掩码 |
| padding / packing | 填充 / 打包 |
| loss aggregation | 损失聚合 |
| gradient accumulation | 梯度累积 |
| microbatch / minibatch | 微批次 / 小批次 |
| epoch | 轮次（epoch） |
| learning rate | 学习率 |
| warm start / cold start | 热启动 / 冷启动 |
| base model / base agent | 基座模型 / 基座智能体 |
| supervised fine-tuning (SFT) | 监督微调（SFT） |
| rejection sampling | 拒绝采样 |
| expert iteration | 专家迭代 |
| behavior cloning | 行为克隆 |
| preference | 偏好 |
| distillation (on-policy / off-policy) | 蒸馏（同策略 / 异策略） |
| teacher / student | 教师 / 学生 |
| exposure bias | 暴露偏差 |
| inference engine | 推理引擎 |
| trainer | 训练器 |
| weight sync | 权重同步 |
| training–inference mismatch | 训推不一致 |
| throughput | 吞吐量 |
| straggler / long tail | 拖尾任务 / 长尾 |
| sandbox / isolation | 沙箱 / 隔离 |
| public tests / hidden tests | 公开测试 / 隐藏测试 |
| held-out | 留出（held-out tasks：留出任务） |
| train split | 训练集 |
| pass rate / pass@k | 通过率 / pass@k |
| confidence interval | 置信区间 |
| bootstrap | 自助法（bootstrap） |
| ablation | 消融实验 |
| seed | 随机种子 |
| harness | 评测框架（harness） |
| contamination / leakage | 污染 / 泄漏 |
| shortcut | 捷径 |
| dashboard | 监控面板 |
| sanity check | 健全性检查 |
| design review | 设计评审 |
| Design problems in practice | 实践中的设计问题 |
| Problem / Symptom / Cause / Options / Default | 问题 / 症状 / 原因 / 可选方案 / 默认做法 |
| Pitfall / In production / Evidence | 陷阱 / 生产实践 / 证据 |
| Summary / Exercises | 小结 / 练习 |
| measured here / reported / hypothesis | 本书测得 / 文献报告 / 假设 |
| mechanism demonstration | 机制演示 |

## 补充译名（翻译过程中确定）

| English | 中文 |
| --- | --- |
| issue (task text) | 问题描述（issue） |
| bug family / variant | 缺陷族（简称“族”）/ 变体 |
| edit op / repair op | 编辑操作 / 修复操作 |
| scripted policy / baseline | 脚本化策略 / 脚本化基线 |
| oracle, blind, random, hacker (policy names) | 保留英文 |
| demonstration | 示范 |
| owner / ownership (of tokens) | 所有者 / 所有权 |
| predicted positions | 被预测位置 |
| prefix / prefix stability | 前缀 / 前缀稳定性 |
| thinking / non-thinking mode | 思考模式 / 非思考模式 |
| token-in/token-out | 保留英文 |
| turn budget / token budget | 轮次预算 / token 预算 |
| step penalty / step cost | 步惩罚 / 步代价 |
| truncation / truncated episode | 截断 / 截断的回合 |
| malformed call | 格式错误的调用 |
| off-by-one | 差一错误（off-by-one） |
| temperature / greedy decoding / nucleus sampling | 温度 / 贪心解码 / 核采样 |
| KV cache, kernel, nats, rank, all-reduce, roofline | 保留英文 |
| reward-to-go / return-to-go | 剩余回报 |
| n-step return | n 步回报 |
| bootstrap (from a value) | 自举（区别于统计学的自助法） |
| explained variance | 解释方差 |
| effective sample size (ESS) | 有效样本量 |
| whitening | 白化 |
| group mean / leave-one-out baseline | 组均值 / 留一法基线 |
| zero-advantage group / informative group | 零优势组 / 有信息量的组 |
| difficulty bias / response-length bias | 难度偏差 / 响应长度偏差 |
| turn-level / token-level | 轮级 / token 级 |
| critic-free | 无评论家 |
| dynamic sampling / clip-higher | 动态采样 / clip-higher（保留英文） |
| dual clipping / approximate KL / early stopping | 双重裁剪 / 近似 KL / 提前停止 |
| mode-seeking / mode-covering | 寻模 / 覆盖模式 |
| entropy bonus | 熵奖励 |
| curriculum | 课程学习 |
| shortcut feature | 捷径特征 |
| stacking edits | 叠加编辑 |
| verifiable reward / partial credit / proxy reward | 可验证奖励 / 部分得分 / 代理奖励 |
| dense / sparse reward | 稠密 / 稀疏奖励 |
| LLM judge / rubric | LLM 评判模型 / 评分细则 |
| flaky test | 不稳定测试 |
| false positive / false negative | 假阳性 / 假阴性 |
| context overflow / observation masking | 上下文溢出 / 观测掩蔽 |
| partial rollout / staleness bound | 部分 rollout / 陈旧度上界 |
| colocated / disaggregated | 共置 / 分离式部署 |
| continuous batching / paged attention | 连续批处理 / 分页注意力 |
| TIS / MIS | 截断重要性采样（TIS）/ 掩码重要性采样（MIS） |
| batch-invariant | 批不变（batch-invariant） |
| attention leak / block-diagonal causal mask | 注意力泄漏 / 块对角因果掩码 |
| recompute (activation checkpointing) | 重新计算 |
| rejection-sampling fine-tuning (RFT) | 拒绝采样微调（RFT） |
| chosen / rejected (DPO) | 被选 / 被拒 |
| design effect / cluster bootstrap | 设计效应 / 聚类自助法 |
| Wald / Wilson interval | Wald 区间 / Wilson 区间 |
| plug-in estimator / standard error | 代入估计量 / 标准误 |
| power calculation / winner's curse | 功效分析 / 赢家诅咒 |
| validation set / test set / checkpoint | 验证集 / 测试集 / 检查点 |
| harness drift / decontamination | 评测框架漂移 / 去污染 |
| invariant / triage / bisect | 不变量 / 排查 / 二分定位 |
| capstone | 综合实战（capstone） |
| test-time scaling / self-play | 测试时扩展 / 自博弈 |
| wall-clock time | 挂钟时间 |
