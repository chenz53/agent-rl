# 第三讲：沿代码把临床任务接到 verl

后续状态更新：本节保留当时未安装依赖的教学记录；用户随后授权在 vp 运行，工具与 GRPO probe 已通过。最新实测见 [第四讲](004-first-real-gpu-update.md) 与实验 002。

从这节开始采用：真实框架代码 → 可运行的领域示例 → 数学解释 → 失败分析。

## 1. 现在可以运行什么

在 nanoAstra 仓库根目录运行，无需新增依赖：

```sh
python3 scripts/demo_case.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

这是脚本预设动作调用领域函数，不是 LLM rollout、不是 verl runtime，也没有参数更新。

已经验证的输出：

```text
get_clinical_record('exposure') -> {"status": "ok", "field": "exposure", "value": "positive"}
get_clinical_record('test_x') -> {"status": "ok", "field": "test_x", "value": "negative"}
get_clinical_record('label') -> {"status": "invalid_field"}
<final>{"diagnosis":"B"}</final> -> score=1.0, format_valid=1.0
<final>{"diagnosis":"A"}</final> -> score=0.0, format_valid=1.0
B -> score=0.0, format_valid=0.0
```

注意：A/B 与检查的关系是人为构造的。一个病例只能检查数据通路，不能用来证明模型学会临床推理。

## 2. 我们自己的代码只负责任务

### 病例与信息边界

阅读 `src/nanoastra/clinical.py` 的 `Case.tool_record` 与 `to_verl_row`：

- `prompt`：模型起初可见的信息。
- `extra_info.tools_kwargs.get_clinical_record.create_kwargs.record`：工具允许访问的字段，不直接加入 prompt。
- `reward_model.ground_truth`：评分器使用的标签；不进入工具记录。

这些是不同用途的数据容器。仅仅“不显示 label 字段”还不够：必须检查序列化后的 prompt 和工具输出有没有泄漏。本轮测试验证这个最小边界；真实系统还要检查检索、日志、case ID、提示模板和数据拆分。

### 查询工具

阅读 `src/nanoastra/verl_tool.py`。它直接继承上游 `BaseTool`，没有伪造框架类，也没有缺依赖时的 mock fallback。

```python
result = read_record(self._records[instance_id], parameters["field"])
return ToolResponse(text=json.dumps(result)), 0.0, {"record_queries": 1}
```

三个返回值分别是观测、工具步骤 reward 和 metrics。这里步骤 reward 固定为 0，终局结果另外评分。这个 metrics 返回不等于已经配置了整条轨迹的查询次数汇总。

### 最终奖励

`src/nanoastra/reward.py` 使用上游自定义 reward 回调签名：

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
```

格式必须满足约定，标签正确才得分。`accuracy` 与 `format_valid` 分开记录，便于区分“答错”与“输出格式不合法”。严格格式本身也会改变学习问题；后续应验证奖励是否主要驱动格式学习。

当前上游 reward manager 可能传入解码后的多轮 response 文本，所以 scorer 不能假设 `solution_str` 只有最后一条 assistant 内容。我们要求末尾恰好一个 `<final>...</final>` 块。真实外部检索内容可能包含类似标记，届时需要基于消息角色的隔离，不能把该玩具解析器直接当成生产评分器。

## 3. 按这个顺序读上游

以下链接全部固定在同一个 commit，而不是随时变化的 main。

| 顺序 | 代码 | 要追踪的问题 |
| --- | --- | --- |
| 1 | [BaseTool](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/base_tool.py#L24) | 哪些方法是框架约定？ |
| 2 | [ToolAgentLoop 工具执行](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L471) | 谁执行 create、execute、release？ |
| 3 | [AgentData](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L57) | 跨轮状态、mask 和工具分数保存在哪里？ |
| 4 | [自定义终局 reward 的调用](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/reward_loop/reward_manager/naive.py#L34) | scorer 实际收到哪段文本和哪些 metadata？ |
| 5 | [GRPO advantage](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268) | scalar score 怎样分组并扩展到 token？ |
| 6 | [Vanilla policy loss](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286) | ratio、clip、mask 怎样组合？ |

源码证据比方法名更可靠：BaseTool 文档描述 trajectory instance，但当前 ToolAgentLoop 的这条执行路径在每次调用中 create，并在 finally 中 release。因此我们的 adapter 只做无跨调用副作用的信息查询。需要跨轮状态时，须明确放入 trajectory 级状态并验证并行调用行为。

另一个值得追的问题：上游将 `tool_rewards` 放入 extra_fields，不能只看到 `execute()` 返回 -0.05 就认定训练已经扣掉查询成本。必须追到最终 token-level reward 的构建并测试。本轮不添加这个尚未验证的成本。

## 4. 用真实 GRPO 源码解释上节的问题

固定源码里的 `compute_grpo_outcome_advantage` 做四件事：

1. 对一条 response 的 token reward 求和，得到一个 scalar score。
2. 用 index 将同一 prompt 的多条 rollout 分组。
3. 在组内减均值，可选再除以标准差。
4. 将每条轨迹的 scalar advantage 广播到其 token，并乘 `response_mask`。

同一 prompt 的四条轨迹分数为 `[1, 1, 0, 0]` 时，均值为 0.5。该实现使用 PyTorch 默认样本标准差，所以标准化结果约为 `[+0.866, +0.866, -0.866, -0.866]`，而不是 `[+1, +1, -1, -1]`。

假设某条轨迹的示意 mask 是 `[1, 1, 0, 0, 1, 1]`，中间两个位置表示环境 token，那么对应 advantage 约为 `[0.866, 0.866, 0, 0, 0.866, 0.866]`。这些只是人工 token 位置，不代表真实 tokenizer 的输出。

这说明：该 outcome-only 估计没有区分一条成功轨迹里哪次查询更有价值。

如果同一 prompt 的一组 rollout 全得 0，组内相对信号就全为 0（这里组大小 >1）。即使存在一个高收益查询策略，只要当前采样没发现它，这个组就没有 outcome advantage 来推动它。KL 或其他辅助项可能仍有梯度，不能据此声称整个训练的所有梯度都为零。

## 5. 集成 probe 与远端训练

`scripts/probe_verl.py` 已写好，用于在具备依赖的环境中实际调用：

- verl 原生 tool registry 加载我们的 YAML 和 BaseTool 子类；
- 工具的 create → execute → release；
- 原生 `compute_grpo_outcome_advantage`，检查工具位置为 0 与全失败组为 0。

本轮未运行该 probe：本机的依赖安装请求没有获得批准。没有通过复制算法、替换 torch 或伪造 verl 模块来宣称集成成功。

CPU 候选依赖写在 `pyproject.toml`，需要后续批准或在远端合适环境中安装、解析和锁定；它不是完整 GPU 安装方案。准备好兼容依赖后，从仓库根目录运行：

```sh
python scripts/fetch_upstream.py
python scripts/probe_verl.py
```

通过 probe 仍不代表完整 AgentLoop、真实 token mask、推理后端或优化器已经验证。远程 GPU 型号和环境明确后，才补充经过实测的训练配置。
