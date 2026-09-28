# 第三讲：沿真实代码追踪病例、工具、轨迹与奖励

这一讲不再列框架功能清单，而是回答：一条病例数据进入 verl 后，谁看见哪些字段，谁创建工具，工具返回后数据流向哪里，最终 scorer 又收到了什么？

上游固定为 `00094bd9cd3fef9cf8903daf60ea4a2bcf832efc`。下文针对这个版本，不能直接套到任何版本的 verl。

## 1. 当前可运行的三层示范

| 层次 | 命令 | 验证内容 | 未验证内容 |
| --- | --- | --- | --- |
| 领域逻辑 | `python3 scripts/demo_case.py` | 查询与评分规则 | 没有模型或 verl runtime |
| 框架 probe | `.venv/bin/python scripts/probe_verl.py` | 真实 registry、BaseTool、GRPO | 没有完整 AgentLoop/模型 |
| 模型更新 | `.venv/bin/python scripts/demo_gpu_update.py` | Qwen 的受限 A/B 采样和一次梯度更新 | 查询动作由脚本固定 |

后两项已在 `vp:/home/user/nanoAstra` 运行。当前尚未执行完整的模型自主多轮查询。这个边界重要：手动调用 `tool.execute()` 成功，不等于 LLM 能产生合法工具调用，更不等于 reward 已正确进入 trainer。

## 2. 先看一个样本：数据字段各去哪里

实际代码是 [clinical.py](../../src/nanoastra/clinical.py) 中的 `to_verl_row`。

在仓库根目录用 `PYTHONPATH=src python3` 进入 Python 后：

```python
from nanoastra.clinical import DEMO_CASE, to_verl_row
row = to_verl_row(DEMO_CASE)
print(row["agent_name"])
print(row["prompt"][-1]["content"])
print(row["extra_info"]["tools_kwargs"])
print(row["reward_model"])
```

对应结果：

```text
tool_agent
Synthetic patient: fever and cough. Candidate labels are A and B.
{'get_clinical_record': {'create_kwargs': {'record': {'exposure': 'positive', 'test_x': 'negative'}}}}
{'style': 'rule', 'ground_truth': 'B'}
```

逐项解释：

- `agent_name="tool_agent"`：告诉相应 rollout 路径选择工具型 Agent loop；它本身不会训练模型使用工具。
- `prompt`：将被渲染进模型输入的消息。
- `tools_kwargs`：工具初始化时的旁路参数，不等于模型生成的工具参数。
- `reward_model.ground_truth`：用于评分的隐藏标签。
- `data_source`：让 scorer 区分任务来源。本例不允许用这个人工 scorer 给真实临床数据打分。

**最容易混淆的两种参数：**

```text
模型生成 parameters:       {"field": "test_x"}
环境注入 create_kwargs:    {"record": {"exposure": "positive", "test_x": "negative"}}
```

前者是 Agent 的选择，后者是环境拥有的允许访问记录。如果让模型自己传入检查结果，再相信这个结果，环境就失去了验证意义。

## 3. 工具 schema 是模型可见的操作说明

我们的 [clinical_tools.yaml](../../configs/clinical_tools.yaml) 描述工具名、参数及允许字段。核心部分为：

```yaml
function:
  name: get_clinical_record
  description: Read one available field from the current synthetic case.
  parameters:
    type: object
    properties:
      field:
        type: string
        enum: [exposure, test_x]
    required: [field]
```

schema 告诉模型怎样调用工具，但不是安全或正确性的充分保证。模型可能仍生成 `{"field":"label"}`，也可能少字段或输出非法 JSON。运行时必须再次校验。

本工具只有两个可查询字段。如果 prompt 中暗示存在更多工具而 schema 没有提供，模型无法凭 RL 学会调用不存在的能力；环境定义和提示必须一致。

## 4. 从 YAML 到一个真实 Python 工具对象

下面在 vp 环境、仓库根目录中可运行；没有伪造 BaseTool：

```python
import sys
sys.path.insert(0, "src")
from nanoastra.upstream import activate_verl
activate_verl()
from verl.tools.tool_registry import initialize_tools_from_config

tool = initialize_tools_from_config("configs/clinical_tools.yaml")[0]
print(type(tool).__name__)
print(tool.name)
# ClinicalRecordTool
# get_clinical_record
```

初始化还会输出 schema。registry 读取 `class_name`，导入我们自己的类，再传入 config 与解析过的 tool schema。这说明适配器遵循了真实框架接口，不意味着模型已经调用过它。

`activate_verl()` 检查本地上游 checkout 的 SHA 和是否有修改，然后加入 import path。它不是一个训练器，也不会自动安装依赖。

## 5. create → execute → release，逐行看状态的生命周期

[实际适配器](../../src/nanoastra/verl_tool.py) 继承 `BaseTool`。

### 5.1 create：为一次调用准备允许访问的数据

实际方法片段：

```python
source = kwargs["create_kwargs"]["record"]
key = instance_id or str(uuid4())
self._records[key] = dict(source)
return key, ToolResponse()
```

省略的部分还检查字段、值类型与重复 instance ID。`dict(source)` 复制记录，避免共享可变对象。随机 key 用于隔离调用，不是病例 label。

初始 `ToolResponse()` 为空：创建实例不应一次性披露所有字段。虽然当前所读 AgentLoop 路径丢弃了 create 返回的 response，适配器仍保持清晰的信息边界。

### 5.2 execute：只响应本次请求的字段

实际片段：

```python
if not isinstance(parameters, dict) or set(parameters) != {"field"}:
    result = {"status": "invalid_arguments"}
else:
    result = read_record(self._records[instance_id], parameters["field"])
return ToolResponse(text=json.dumps(result)), 0.0, {"record_queries": 1}
```

三个返回值的语义：

1. `ToolResponse.text`：下次模型决策能够看到的新观测。
2. `0.0`：这个调用返回的工具 reward；不等于整条轨迹最终评分。
3. `record_queries`：诊断指标；没有配置汇总就不能当作总查询数。

即使 caller 意外提供了带 `label` 的字典，适配器 create 会拒绝额外字段，查询函数也有 whitelist。这里有两道明确的边界，而不是单靠模型遵守提示。

### 5.3 release：无论成功还是失败，都清理实例

实际方法：

```python
async def release(self, instance_id, **kwargs):
    self._records.pop(instance_id, None)
```

完整调用小例子，接上节创建的 `tool`：

```python
import asyncio
from nanoastra.clinical import DEMO_CASE

async def one_call():
    key, initial = await tool.create(
        create_kwargs={"record": DEMO_CASE.tool_record()}
    )
    try:
        response, reward, metrics = await tool.execute(key, {"field": "test_x"})
        print(response.text)
        print(reward, metrics)
    finally:
        await tool.release(key)
    print(len(tool._records))

asyncio.run(one_call())
```

预期输出，已由真实框架 probe 验证同一调用路径：

```text
{"status": "ok", "field": "test_x", "value": "negative"}
0.0 {'record_queries': 1}
0
```

### 5.4 关键源码发现：实例不是天然跨轮持久化的

固定版本 [ToolAgentLoop](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L471) 的这个执行路径，对每次工具调用都 create，在 finally 中 release。

因此，在 `_records[key]` 中保存 `query_count += 1`，下一次调用可能已经是新实例，计数又从 0 开始。不要因为 BaseTool 文档提到 trajectory，就假定每一条调用路径都保证这种生命周期。

跨轮状态需要明确归属，例如对应轨迹的 `AgentData.extra_fields` 或有生命周期管理的外部 session。若允许并行工具调用，还必须考虑并发更新预算的原子性。当前适配器只做只读查询，尚未实现跨轮预算。

## 6. 从模型 token 到工具观测：完整框架还有哪些步骤

预期完整路径是：

```text
模型生成 token
  → 根据模型格式解析工具名与 JSON 参数
  → 查 registry 找到工具
  → create / execute / release
  → 将 ToolResponse 转成下一段消息
  → 更新 token/context 与 response mask
  → 再调用模型或终止
```

每个箭头都可能失败。比如模型产生了接近合法但缺逗号的 JSON：工具可能根本没有执行；不能把“工具没有返回证据”归因为病例缺信息。

上游 [AgentData](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L57) 保存 messages、response_mask、response_logprobs、tool_rewards、turn counts 等数据。这些是本次手工工具 probe 没有完整覆盖的部分。

当前 H200 脚本把查询结果写入同一个 user context，然后训练最终一个 token；不能把它的 mask 验证外推成所有真实多轮模板都正确。

## 7. reward 回调收到什么，怎么验证

[实际 scorer](../../src/nanoastra/reward.py) 的接口是：

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
```

其中 solution_str 在固定上游 [naive reward manager](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/reward_loop/reward_manager/naive.py#L34) 中来自 response token 的解码，不应默认它只有最后一句答案。

可在本地 `PYTHONPATH=src python3` 执行：

```python
from nanoastra.reward import compute_score
for text in [
    '<final>{"diagnosis":"B"}</final>',
    '<final>{"diagnosis":"A"}</final>',
    'B',
]:
    print(compute_score("nanoastra_synthetic", text, "B"))
```

已验证输出：

```text
{'score': 1.0, 'accuracy': 1.0, 'format_valid': 1.0}
{'score': 0.0, 'accuracy': 0.0, 'format_valid': 1.0}
{'score': 0.0, 'accuracy': 0.0, 'format_valid': 0.0}
```

为什么分开 accuracy 和 format_valid？如果初期所有样本 score=0，你需要区分是判断错误，还是答案根本未被解析。只看 reward 曲线无法区分两者。

本例要求恰好一个位于结尾的 `<final>...</final>`。如果将来外部文献能带入伪造的 final 标记，基于整个字符串的解析会变得不可靠；应保留角色边界并仅对可信的 assistant 最终输出评分。当前只用固定合成记录，不声称这个 parser 能处理任意外部内容。

## 8. reward 数据流：返回了分数，不等于 trainer 使用了分数

框架可能同时有工具步骤 reward、最终 scorer reward、KL penalty 等。必须追问最后进入 `token_level_rewards` 的究竟是什么。

固定源码中工具分数被保存为 `tool_rewards`，并进入额外字段。看见某个工具返回 -0.05，并不足以证明最终 surrogate 已经扣了这笔成本。

要验证，可构造两条最终答案完全相同、只有查询次数不同的轨迹，逐层打印：

```text
tool.execute 返回值
→ AgentData.tool_rewards
→ reward manager 收到的 extra_info
→ 最终 scalar score
→ token_level_rewards 的序列求和
```

若最终 sum 仍相同，说明你想要的成本没有进入目标。本项目目前工具 reward 固定 0，仅优化最终正确性；不要把第一课的成本示例误当成已上线的训练规则。

## 9. group ID 是算法的一部分，不是普通 metadata

假设有两个 prompt：easy 的两条结果 `[1,1]`，hard 的两条结果 `[0,0]`。

正确分组后，每个组内的结果一样，所以 outcome advantage 全为 0。若误把四条都当成同一组，会得到 `[+0.866,+0.866,-0.866,-0.866]`。这变成跨任务比较难度，而不是比较同一任务的策略选择。

在 vp 完成 `activate_verl()` 后运行：

```python
import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
r = torch.tensor([[1.], [1.], [0.], [0.]])
m = torch.ones_like(r)
correct, _ = compute_grpo_outcome_advantage(r, m, np.array(["easy", "easy", "hard", "hard"]))
wrong, _ = compute_grpo_outcome_advantage(r, m, np.array(["all"] * 4))
print(correct[:, 0].tolist())
print(wrong[:, 0].tolist())
# [0.0, 0.0, 0.0, 0.0]
# 约 [0.866024, 0.866024, -0.866024, -0.866024]
```

本例已经用真实函数复核。输出 tensor 形状正确不代表分组语义正确，这是一个典型的静默训练错误。

## 10. 源码阅读路线与检查点

| 顺序 | 源码 | 读完应能回答 |
| --- | --- | --- |
| 1 | [BaseTool](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/base_tool.py#L24) | 三种返回数据分别是什么？ |
| 2 | [Tool registry](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/tool_registry.py) | YAML 中的类如何被实例化？ |
| 3 | [工具调用路径](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L471) | 参数由谁注入？何时清理？ |
| 4 | [Reward manager](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/reward_loop/reward_manager/naive.py#L34) | 评分看到哪些 token 和 metadata？ |
| 5 | [GRPO advantage](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268) | 怎样分组、归一化、广播到 token？ |
| 6 | [Policy loss](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286) | 哪些比率被 clip，怎样聚合？ |

第四讲会沿同一版源码把每个数学量代入 H200 的真实采样结果。源码是理解框架语义的依据，跑通的小例子是验证这些理解的证据，二者缺一不可。
