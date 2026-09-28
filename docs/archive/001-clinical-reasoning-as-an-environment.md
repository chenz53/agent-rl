> Historical material. Follow the [current oncology curriculum](../lessons/README.md) instead.

# Lesson 1: Define clinical reasoning as a trainable decision problem

The goal is to take a case, tools, and a base model and identify the environment state, visible information, actions, rewards, and stopping rules. Every object should have a clear home in the implementation.

This lesson retains the original artificial A/B task to explain RL mechanics and reproduce the GPU experiment. A and B have no medical meaning. The active application is now [cancer prognosis from CSV exports](006-oncology-prognosis-from-csv.md).

## 1. Begin with an explicit task

```python
case = {
    "presentation": "Synthetic patient: fever and cough.",
    "exposure": "positive",
    "test_x": "negative",
    "label": "B",
}
```

The future agent initially sees only `presentation`. It can query `exposure`, query `test_x`, or submit A/B. Querying changes the information available for the next decision. Supplying every field at the beginning would define a different policy problem.

The completed H200 experiment fixes both queries and trains only the final choice. Autonomous querying is a later milestone; the sequential environment described here is the intended extension.

## 2. State, observation, and history

### State: everything the environment needs

Write the full state as \(s_t=(c,d_t,q_t,u_t)\): the complete case, disclosed fields, remaining query budget, and termination flag. A static case can still have changing runtime state. After a query, the disclosed set grows and the budget falls.

The following is a teaching sketch, not a complete implemented environment:

```python
state = {
    "case": case,
    "revealed": set(),
    "remaining_queries": 2,
    "terminated": False,
}
```

### Observation: what the agent receives now

The initial observation deliberately excludes hidden fields and the label:

```python
observation_0 = {"presentation": case["presentation"]}
```

After a query, a new observation might be:

```python
observation_1 = {"field": "exposure", "value": "positive"}
```

An observation is a restricted view. Serializing the full state into the prompt would expose hidden evidence and the answer key. Keeping a label in a differently named dictionary does not help if the prompt renderer later dumps that dictionary.

### History: the interaction the model actually received

```python
history = [
    {"role": "user", "content": observation_0},
    {"role": "assistant", "action": {"field": "exposure"}},
    {"role": "tool", "content": observation_1},
]
```

Formally, \(h_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t)\). An LLM normally conditions on a tokenized representation of this sequence. If the framework summarizes or truncates it, the effective input is \(C(h_t)\), not the complete history. Losing an early result changes the policy problem; context management belongs in the environment specification.

## 3. Why this is partially observable

Two cases may have the same presentation but different hidden findings and labels. Their initial observations are identical although their states differ. The agent therefore cannot uniquely infer state from observation.

An MDP assumes that the full current state and action suffice to determine the next-state distribution. A POMDP allows the agent to observe only part of that state. A deterministic record lookup can still be partially observable: determinism and observability are distinct properties.

A belief \(b_t(s)=P(s\mid h_t)\) is a distribution over hidden states, not a diagnosis string. Our code does not maintain an explicit belief tensor; the LLM would act from its history.

For an invented Bayesian illustration, let P(A)=P(B)=0.5 and P(+|A)=0.2, P(+|B)=0.8. Then:

\[
P(B\mid+)=\frac{0.8\cdot0.5}{0.2\cdot0.5+0.8\cdot0.5}=0.8.
\]

```python
prior_b = 0.5
posterior_b = (0.8 * prior_b) / (0.8 * prior_b + 0.2 * (1 - prior_b))
print(posterior_b)  # 0.8
```

The result is 0.8 under these stipulated probabilities. It does not establish that a model's verbal confidence is a calibrated posterior.

## 4. An action has several useful granularities

A tool-level action can be `{"name":"get_clinical_record","arguments":{"field":"test_x"}}`. At the model level, that JSON is an autoregressive token sequence.

| Granularity | One step | Useful for |
| --- | --- | --- |
| Token | Generate the next token | Log-probabilities, decoding, loss masks |
| Tool action | Generate a complete call | Information acquisition and call cost |
| Trajectory | Complete interaction | Success, total cost, evaluation |

Thirty tokens describing one lookup do not mean thirty clinical tests. Conversely, a JSON action remains trainable by policy gradient because its probability is defined by the probabilities of its generated tokens.

## 5. Transition: what changes after an action?

A simplified record transition reads an existing value and consumes a budget unit. This runnable example is not wired into the trainer:

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

The output is the negative `test_x` observation and a remaining budget of 1. The case itself has not changed, but the runtime state has.

A failed lookup must distinguish invalid fields, missing information, and infrastructure failures. A database timeout is not evidence that the patient lacks a finding; assigning it a clinical failure reward contaminates training. An empty string is also insufficient to distinguish these cases.

The implemented lookup demonstrates two explicit outcomes:

```python
from nanoastra.clinical import read_record
print(read_record({}, "test_x"))
print(read_record({"label": "B"}, "label"))
# {'status': 'unavailable', 'field': 'test_x'}
# {'status': 'invalid_field'}
```

Run repository imports from the root with `PYTHONPATH=src python3`. Missing `test_x` returns `unavailable`; requesting the hidden label returns `invalid_field` even if a caller accidentally supplied it.

## 6. Termination, truncation, and reset

Termination means the task ended according to its definition, such as submitting an answer. Truncation means an external limit stopped execution, such as a timeout or token cap. Framework names alone do not establish these semantics.

A task designer may give zero reward to a trajectory that exhausts its budget without answering. That is a design choice, not a theorem of RL. If a value function later bootstraps from a final state, true termination versus truncation can also affect return estimation.

Reset must clear disclosed evidence, budget, and ending state, not merely reload the case:

```python
def reset_runtime():
    return {"revealed": set(), "remaining_queries": 2, "terminated": False}

first = reset_runtime()
first["revealed"].add("test_x")
second = reset_runtime()
print(second["revealed"])  # set()
```

The second runtime prints `set()`. If it inherits evidence from the first rollout, the samples no longer represent independent runs from the specified initial condition.

## 7. Reward, return, and utility

A reward \(r_t\) is a numerical feedback at a step. A return aggregates future rewards. Two queries costing 0.05 each and a correct answer worth 1 give:

```python
rewards = [-0.05, -0.05, 1.0]
print(sum(rewards))  # 0.9
```

The undiscounted total is 0.9. More generally:

\[
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
\]

With gamma=1, delay has no extra discount. With gamma<1, later reward has lower weight. A finite record task can begin with gamma=1 and explicit query costs. Combining large query costs and heavy discounting may strongly favor premature stopping; do not attribute that behavior solely to the model.

Scientific utility U may include correctness, sufficient evidence, risk, and resource use. Computable reward R typically covers only part of U. Final-label scoring gives a lucky guess and a well-supported correct answer the same reward. Length bonuses can encourage verbosity, and citation-count bonuses can encourage irrelevant citations. Their relationship to utility requires independent evaluation.

## 8. Calculate the value of information

Assume direct answering succeeds with probability 0.80, querying raises it to 0.95, and the query costs 0.05. Under this invented objective:

\[
E[R\mid stop]=0.80,\qquad E[R\mid query]=0.95-0.05=0.90.
\]

```python
p_before, p_after, cost = 0.80, 0.95, 0.05
print(p_after - cost - p_before)
```

Expected improvement is approximately 0.10. Querying is preferred exactly when `p_after - p_before > cost`. At cost 0.20, querying yields 0.75 and the same objective prefers stopping. Higher post-query accuracy alone does not determine the optimal action. These probabilities and costs are pedagogical, not a clinical resource model.

## 9. Separate agent evidence from scoring labels

The [actual row builder](../../src/nanoastra/clinical.py) separates initial messages, tool initialization data, and the hidden reward label:

```python
row = {
    "prompt": [...],
    "extra_info": {
        "tools_kwargs": {
            "get_clinical_record": {
                "create_kwargs": {"record": {"exposure": "positive", "test_x": "negative"}}
            }
        }
    },
    "reward_model": {"ground_truth": "B"}
}
```

This block illustrates the structure, not a complete training row. The renderer, tool inputs, and tool outputs must be inspected: labels can leak through dumped metadata, informative identifiers, or missingness patterns despite apparently separate containers.

A concrete inspection is:

```python
from nanoastra.clinical import DEMO_CASE, to_verl_row
row = to_verl_row(DEMO_CASE)
record = row["extra_info"]["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
print(sorted(record))
print(row["reward_model"])
# ['exposure', 'test_x']
# {'style': 'rule', 'ground_truth': 'B'}
```

The tool record contains only `exposure` and `test_x`; the reward container contains the artificial label B. In the oncology task, a stronger temporal boundary additionally excludes unavailable record versions and mortality outcomes.

## 10. What would an RL improvement mean?

Compare initial-information answering, full-allowed-information answering, an untrained querying agent, and an RL querying agent at matched budgets.

| Hypothetical condition | Accuracy | Mean queries |
| --- | --- | --- |
| Initial information only | 60% | 0 |
| All allowed information | 90% | Supplied in advance |
| Untrained querying policy | 65% | 1.8 |
| RL querying policy | 82% | 1.2 |

These are invented numbers, not nanoAstra results. They suggest an information-acquisition bottleneck, but matched token/sampling budgets, fixed evaluation tasks, and leakage checks are still necessary. If full-information accuracy were only 61%, investigate model competence and task/label quality before scaling RL.

The deliverable is an auditable task contract. [Lesson 2](002-one-clinical-trajectory.md) connects that contract to gradients.
