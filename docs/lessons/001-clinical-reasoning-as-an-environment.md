# 第一课：把临床推理定义成一个可训练的决策问题

本课的目标是：给你一份病例、一组工具和一个基础模型，你能明确指出环境状态、模型可见信息、动作、奖励以及终止规则各是什么，并知道它们分别应该放进哪些代码结构。

我们用匿名诊断 A/B 和人为设定的病例讲训练机制，不用它们表达医学结论。

## 1. 从一个具体任务开始

假设完整病例包含四项信息：

```python
case = {
    "presentation": "Synthetic patient: fever and cough.",
    "exposure": "positive",
    "test_x": "negative",
    "label": "B",
}
```

我们要训练的未来 Agent 起初只看到 `presentation`，然后可以：

1. 查询 `exposure`；
2. 查询 `test_x`；
3. 提交最终 A/B 标签。

这个设定与“一次把全部字段给模型，再让它回答”不同：查询动作决定下一次决策时模型能看到哪些信息。

当前已实现的 H200 单步示范先固定两次查询，只训练最终 A/B 判断。完整自主查询是后续里程碑；本课定义的是它的目标问题。

## 2. state、observation 与 history：三个不同的对象

### 2.1 state：环境用于推进任务的完整信息

完整状态可以写为：

\[
s_t=(c,\,d_t,\,q_t,\,u_t).
\]

- c：完整病例，包括尚未披露的字段。
- d_t：已经披露的字段集合。
- q_t：剩余查询预算。
- u_t：是否已经终止。

病例内容不变，并不代表环境状态不变。查询一次后，已披露集合和预算会改变。

**教学伪实现，不是仓库已经接入的完整环境：**

```python
state = {
    "case": case,
    "revealed": set(),
    "remaining_queries": 2,
    "terminated": False,
}
```

### 2.2 observation：这一步给模型看的信息

初始观测可以只有：

```python
observation_0 = {"presentation": case["presentation"]}
```

查询 exposure 后的新观测为：

```python
observation_1 = {"field": "exposure", "value": "positive"}
```

observation 是有意限制的视图，不应把 `state` 整个序列化给模型，否则隐藏信息和评分标签也会被暴露。

### 2.3 history：截至当前，模型实际收到的交互序列

```python
history = [
    {"role": "user", "content": observation_0},
    {"role": "assistant", "action": {"field": "exposure"}},
    {"role": "tool", "content": observation_1},
]
```

符号上可以写成 `h_t=(o_0,a_0,o_1,...,a_{t-1},o_t)`。LLM 策略通常以这个序列的 token 表示作为输入。

如果框架做了摘要或截断，模型的有效输入不再是完整历史，而是 `C(h_t)`。这会改变策略问题：早期关键证据可能不再可见。因此上下文管理也属于环境/Agent 设计，不能当成与训练无关的显示细节。

## 3. 为什么这通常是部分可观测问题

想象两个病例的初始主诉相同，但隐藏检查结果不同、人工标签也不同。初始 observation 一样，完整 state 不一样。模型不能从 observation 唯一确定 state，这就是部分可观测性。

MDP 的建模要求是：给定当前完整状态和动作，下一状态的分布不再依赖更早历史。POMDP 则允许 Agent 只观察到完整状态的一部分。

在静态记录查询任务中，环境可以近乎确定；部分可观测仍然存在。“确定性环境”与“完全可观测环境”不是同义词。

有时引入 belief `b_t(s)=P(s|h_t)` 表示对隐藏状态的概率判断。它是一个状态分布，不是某个单独诊断的文本标签。我们实际实现暂不维护显式 belief tensor，而让 LLM 根据历史决策。

**只用于解释 Bayes 更新的人工例子：**先验 P(A)=P(B)=0.5；设某结果出现概率 P(+|A)=0.2、P(+|B)=0.8，则观测到 + 后：

\[
P(B\mid +)=\frac{0.8\times0.5}{0.2\times0.5+0.8\times0.5}=0.8.
\]

```python
prior_b = 0.5
posterior_b = (0.8 * prior_b) / (0.8 * prior_b + 0.2 * (1 - prior_b))
print(posterior_b)  # 0.8
```

这是信息如何改变判断的示意，不是声称我们的模型输出已经校准成了这个 posterior。

## 4. action：改变后续交互的模型输出

在工具级视角，一个动作可以是：

```json
{"name": "get_clinical_record", "arguments": {"field": "test_x"}}
```

在 token 级视角，它是许多自回归生成 token 组成的序列。同一个任务可以用不同时间粒度分析：

| 粒度 | 一步是什么 | 适合讨论什么 |
| --- | --- | --- |
| token | 生成下一个 token | log-probability、loss mask、解码 |
| 工具动作 | 一个完整工具名与参数 | 查询策略、信息获取、调用成本 |
| trajectory | 从开始到终止的整个过程 | 任务成功、总成本、评估 |

不能因为一次工具调用有 30 个 token，就把它误当成 30 次独立检查。也不能因为一个动作是 JSON，就认为它不可用于策略梯度；其概率仍由模型生成 token 的概率定义。

## 5. transition：动作之后环境怎么变化

对静态记录读取，一个理想化转移是：查询字段 → 返回已有值 → 扣除预算。下面是能运行的**教学转移例子**，没有接入 trainer：

```python
case = {"presentation": "fever and cough", "exposure": "positive", "test_x": "negative", "label": "B"}
state = {"revealed": set(), "remaining_queries": 2, "terminated": False}
field = "test_x"
assert not state["terminated"]
assert state["remaining_queries"] > 0
assert field in ("exposure", "test_x")
state["revealed"].add(field)
state["remaining_queries"] -= 1
observation = {"field": field, "value": case[field]}
print(observation)
print(state["remaining_queries"])
# {'field': 'test_x', 'value': 'negative'}
# 1
```

真实查询失败时，需要区别：

- 字段非法：模型请求了不允许访问的字段。
- 信息缺失：病例允许该字段，但没有记录。
- 基础设施错误：数据库超时、连接失败等。

把第三种直接当成“临床推理错误”会污染奖励。简单地让失败返回空字符串也会混淆这些情况。

现有 `read_record` 的行为已验证：

```python
from nanoastra.clinical import read_record
print(read_record({}, "test_x"))
print(read_record({"label": "B"}, "label"))
# {'status': 'unavailable', 'field': 'test_x'}
# {'status': 'invalid_field'}
```

上述仓库导入例子在根目录执行 `PYTHONPATH=src python3` 后可运行。

## 6. termination、truncation 与 reset

**Termination** 是任务按定义完成，例如提交最终答案。**Truncation** 是被外部限制截断，例如 token 上限或超时。在不同框架中，这些信息可能以 stop reason、done 或多个字段表示，不能仅凭名字推断语义。

假设轨迹没有最终答案便耗尽预算：你可以在任务规范中把它定义为失败并给 0 分；但这是一项设计决定，不是 RL 数学自动推出的唯一结论。若以后用 value bootstrap，是否真正终止还会影响 return 估计。

reset 必须清除该 episode 的已披露信息、预算与结束标志，而不只是重新加载病例。

```python
def reset_runtime():
    return {"revealed": set(), "remaining_queries": 2, "terminated": False}

first = reset_runtime()
first["revealed"].add("test_x")
second = reset_runtime()
print(second["revealed"])  # set()
```

如果第二条 rollout 偷看到了第一条 rollout 留下的证据，两条采样就不再是你以为的同一初始条件下独立运行。

## 7. reward、return 与真实效用

Reward `r_t` 是某一步拿到的数值反馈；return 是从某时刻起累计的奖励。例如两次查询各花 0.05 分，最终正确得 1 分：

```python
rewards = [-0.05, -0.05, 1.0]
print(sum(rewards))  # 0.9
```

带折扣的 return 定义为：

\[
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
\]

γ=1 时不额外折扣延迟；γ<1 时，更晚的奖励权重更小。对有限病例任务，可以先用 γ=1 加显式查询成本。不能无意中同时用强折扣和高成本，随后把偏好早停全归因于模型。

真实效用 U 可能包含正确性、证据充分性、风险和资源成本；可计算 reward R 往往只覆盖一部分。例如，只查最终标签会把“幸运猜对”和“充分获取证据后答对”给成同分。

因此需要独立评估。如果以输出长度奖励“解释充分”，模型可能增加措辞；若以引用数奖励“证据扎实”，模型可能堆引用。指标优化与科学目标一致，需要证据而不是默认成立。

## 8. 算一次信息获取的价值

人为设定：直接回答正确率 80%；查询一次后正确率 95%；查询成本 c=0.05。只关心最终 0/1 正确性与该成本，则：

\[
E[R\mid stop]=0.8,\quad E[R\mid query]=0.95-0.05=0.9.
\]

```python
p_before, p_after, cost = 0.80, 0.95, 0.05
print(p_after - cost - p_before)  # 约 0.10
```

在这些假设下，查询提高期望奖励 0.10。临界条件是 `p_after-p_before > cost`。若 cost=0.20，查询的期望奖励只有 0.75，目标反而偏好直接回答。

这解释了“是否该查询”由任务目标共同决定，不能只凭查询后的准确率更高下结论。这里的概率和成本都是教学设定，不是临床成本模型。

## 9. 代码里怎样避免把答案暴露给 Agent

[实际数据行构造](../../src/nanoastra/clinical.py) 将信息分为三个容器：

```python
row = {
    "prompt": [...],                       # 初始可见信息
    "extra_info": {
        "tools_kwargs": {
            "get_clinical_record": {
                "create_kwargs": {"record": {"exposure": "positive", "test_x": "negative"}}
            }
        }
    },
    "reward_model": {"ground_truth": "B"}  # 仅用于评分
}
```

这是结构示意，不是完整训练数据。字段分开放不等于自动安全：如果后续代码把整行 dump 到 prompt，仍然泄漏。应检查实际渲染后的 prompt、工具输入/输出以及 label 是否通过 ID 或缺失模式间接透露。

最低限度的实际检查：

```python
from nanoastra.clinical import DEMO_CASE, to_verl_row
row = to_verl_row(DEMO_CASE)
record = row["extra_info"]["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
print(sorted(record))
print(row["reward_model"])
# ['exposure', 'test_x']
# {'style': 'rule', 'ground_truth': 'B'}
```

## 10. 怎么判断后续 RL 改善了什么

至少比较四个条件：初始信息直接回答、完整允许信息直接回答、未经过 RL 的可查询 Agent、相同预算下经过 RL 的 Agent。

举一个**假想结果，不是本项目实测**：

| 条件 | 正确率 | 平均查询数 |
| --- | --- | --- |
| 初始信息直接回答 | 60% | 0 |
| 全部信息直接回答 | 90% | 固定提供全部信息 |
| 未训练查询 Agent | 65% | 1.8 |
| RL 查询 Agent | 82% | 1.2 |

这个结果提示信息获取策略可能有提升空间，但不能仅据表格证明机制。还要匹配 token/采样预算、固定评估任务并排除泄漏。若“全部信息”条件也只有 61%，应先调查模型能力或标签/任务定义，而不是直接增大 RL 训练量。

至此，你应该能把“训练临床 Agent”拆成可审查的任务契约。下一讲解释一条这样的交互轨迹如何形成梯度。
