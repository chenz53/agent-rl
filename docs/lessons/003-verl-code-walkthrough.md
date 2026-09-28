# Lesson 3: Trace records, tools, trajectories, and rewards through real code

The question is concrete: who sees each field, creates the tool, receives the observation, and scores the response? All upstream statements refer to verl commit `00094bd9cd3fef9cf8903daf60ea4a2bcf832efc`, not arbitrary releases.

## 1. Three distinct executable layers

| Layer | Script | Verified | Not verified |
| --- | --- | --- | --- |
| Domain logic | `scripts/demo_case.py` | Lookup and artificial scoring | Model or verl runtime |
| Framework probe | `scripts/probe_verl.py` | Real registry, BaseTool, CPU GRPO | Complete AgentLoop |
| GPU update | `scripts/demo_gpu_update.py` | Qwen A/B sampling and one update | Autonomous query selection |

The last two ran in `/home/user/nanoAstra` on vp. Calling `tool.execute()` manually does not show that a model can emit legal calls or that reward reaches a full trainer.

## 2. Follow one data row

Run repository snippets with `PYTHONPATH=src python3` from the root:

```python
from nanoastra.clinical import DEMO_CASE, to_verl_row
row = to_verl_row(DEMO_CASE)
print(row["agent_name"])
print(row["prompt"][-1]["content"])
print(row["extra_info"]["tools_kwargs"])
print(row["reward_model"])
```

Output identifies `tool_agent`, the artificial presentation, a tool record containing `exposure` and `test_x`, and hidden `ground_truth=B`.

- `agent_name` selects the applicable rollout loop; it does not train tool use.
- `prompt` contains messages rendered into model input.
- `tools_kwargs` carries environment-supplied initialization data.
- `reward_model.ground_truth` carries the scoring label.
- `data_source` restricts the scorer to supported tasks; the A/B scorer must not score oncology prognosis.

Model-generated parameters are `{"field":"test_x"}`. Environment-injected initialization is `{"record":{"exposure":"positive","test_x":"negative"}}`. Letting the model supply the result itself would destroy the lookup's evidentiary role.

## 3. Schema describes actions; runtime validation enforces them

The [original YAML](../../configs/clinical_tools.yaml) supplies the name, argument type, required key, and enum:

```yaml
function:
  name: get_clinical_record
  parameters:
    type: object
    properties:
      field:
        type: string
        enum: [exposure, test_x]
    required: [field]
```

A schema does not guarantee legal generation. The model can still request a label, omit keys, or produce malformed JSON. Runtime checks remain necessary. Conversely, a prompt cannot make an unavailable tool callable through RL; capability descriptions must match the tool contract.

## 4. From YAML to a genuine tool instance

In the configured vp environment:

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

This prints `ClinicalRecordTool` and `get_clinical_record`; initialization also prints the schema. The registry imports `class_name` and passes config plus parsed schema to the class. No substitute BaseTool is used.

`activate_verl()` checks the upstream SHA and modifications before adding the checkout to the import path. It does not install dependencies or start training.

## 5. The create, execute, release lifecycle

### Create only the permitted context

Contextual adapter excerpt:

```python
source = kwargs["create_kwargs"]["record"]
key = instance_id or str(uuid4())
self._records[key] = dict(source)
return key, ToolResponse()
```

The omitted validation checks keys, text values, and duplicate instance IDs. Copying the source avoids shared mutations. The random instance key isolates calls; it is not a patient label. An empty initial `ToolResponse` avoids disclosing all evidence at creation, even though the inspected AgentLoop path discards that initial response.

### Execute the requested lookup

The original two-field logic is:

```python
if not isinstance(parameters, dict) or set(parameters) != {"field"}:
    result = {"status": "invalid_arguments"}
else:
    result = read_record(self._records[instance_id], parameters["field"])
return ToolResponse(text=json.dumps(result)), 0.0, {"record_queries": 1}
```

The current adapter additionally passes its configured field allowlist to `read_record`, enabling the oncology table schema without changing the lifecycle. Return values have separate roles: response text is the observation, 0.0 is a tool-level reward, and `record_queries` is a diagnostic. It is not a trajectory total unless explicitly aggregated.

Creation rejects unapproved record keys, and lookup validates requested keys. These are code boundaries rather than reliance on a model obeying prose instructions.

### Release state even after failures

```python
async def release(self, instance_id, **kwargs):
    self._records.pop(instance_id, None)
```

Continuing from the instantiated tool:

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

The lookup returns negative `test_x`, reward 0, one query metric, and zero remaining instances. `finally` ensures cleanup even if execution fails.

### Instances are not necessarily persistent across turns

The pinned [ToolAgentLoop call path](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L471) creates and releases an instance for each call. A query counter stored only in `_records[key]` may therefore reset on the next call.

Persistent budgets need a trajectory-owned state location, such as appropriate AgentData extra fields or a managed external session. Concurrent calls also need atomic budget updates. Our adapter performs read-only lookup and implements no cross-turn budget.

## 6. The complete loop includes more than tool execution

```text
Generated tokens -> parse tool name/arguments -> resolve registry entry
-> create/execute/release -> append tool response -> update context/masks
-> generate again or stop
```

A missing JSON comma may prevent any tool execution. Do not classify that as absent clinical evidence. The pinned [AgentData](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L57) includes messages, response masks/log-probabilities, tool rewards, and turn counts. The manual probe does not cover the full path.

The GPU script inserts fixed query results into a user context and optimizes one final token. Its mask check cannot establish correctness for every multi-turn chat template.

## 7. Inspect the scoring callback

The [scorer](../../src/nanoastra/reward.py) exposes this contextual signature:

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
```

In the pinned [naive reward manager](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/reward_loop/reward_manager/naive.py#L34), response tokens are decoded into `solution_str`. Do not assume it contains only the last sentence.

```python
from nanoastra.reward import compute_score
for text in [
    '<final>{"diagnosis":"B"}</final>',
    '<final>{"diagnosis":"A"}</final>',
    'B',
]:
    print(compute_score("nanoastra_synthetic", text, "B"))
```

The three outputs distinguish correct/valid `(1,1)`, wrong/valid `(0,1)`, and invalid-format `(0,0)` for accuracy and format validity. A flat zero reward curve can otherwise conceal a parser problem.

The toy parser requires exactly one terminal `<final>...</final>` block. With arbitrary external text, forged markers could make whole-string parsing unreliable; preserve role boundaries and score only the intended assistant output. The current fixed synthetic task does not validate this parser against adversarial external content.

## 8. Returned reward is not automatically optimized reward

Tool rewards, final scores, and KL penalties may enter different paths. Trace the actual value used in `token_level_rewards`. Seeing a tool return -0.05 does not prove the policy loss incorporates that cost.

A useful audit holds the final answer fixed and varies query count, inspecting tool returns, `AgentData.tool_rewards`, reward-manager metadata, scalar score, and the sequence sum of token rewards. If the final sums remain equal, the intended cost has not reached the objective.

The current tool reward is zero. Lesson 1's query-cost calculation is conceptual, not a deployed training rule.

## 9. Group identity is part of the algorithm

Suppose the easy prompt gets `[1,1]` and the hard prompt `[0,0]`. Correct within-prompt groups have zero relative advantage. Mixing all four samples produces approximately `[+0.866,+0.866,-0.866,-0.866]`: a comparison of task difficulty rather than alternative behavior on the same task.

After activating verl:

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
```

This is checked against the real upstream function. Valid tensor shapes do not imply correct grouping semantics. Even one patient's records at two different index times can constitute different prompts; patient identity alone is not a sufficient group key.

## 10. Source-reading checkpoints

| Source | Question to answer |
| --- | --- |
| [BaseTool](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/base_tool.py#L24) | What do response, reward, and metrics mean? |
| [Registry](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/tool_registry.py) | How is the configured class instantiated? |
| [Tool loop](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py#L471) | Who injects context and cleans up? |
| [Reward manager](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/reward_loop/reward_manager/naive.py#L34) | Which tokens and metadata reach scoring? |
| [GRPO](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268) | How are groups normalized and masked? |
| [Policy loss](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286) | Which ratios are clipped and how are losses reduced? |

[Lesson 4](004-first-real-gpu-update.md) substitutes real sampled outcomes into these interfaces. Source reading establishes semantics; executable examples test that interpretation.
