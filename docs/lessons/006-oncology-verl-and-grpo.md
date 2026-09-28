# Lesson 6: Connect oncology tools and learning signals to real verl

[Lesson 5](005-oncology-trajectories-and-gradients.md) established why final feedback can train earlier queries. Now we use the actual fixed verl code for tool creation, group advantage, and clipped policy loss. We do not write a second RL framework.

Run the blocks in order in the existing vp environment, from `/home/user/nanoAstra` with `PYTHONPATH=src .venv/bin/python`. The exact upstream revision is `00094bd9cd3fef9cf8903daf60ea4a2bcf832efc`. All outputs are captured from this environment. No Qwen model is loaded and no clinical agent is trained in this lesson.

## 1. Instantiate the oncology tool through the real registry

The YAML declares allowed clinical table names. Initialization data come from the environment, while the model would generate only the requested field. Registry initialization prints its schema; the snippet captures that routine print so the output focuses on the class and clinical result.

```python
import asyncio
import contextlib
import io
import json
from nanoastra.upstream import activate_verl
from nanoastra.oncology import tool_view
activate_verl()
from verl.tools.tool_registry import initialize_tools_from_config
with contextlib.redirect_stdout(io.StringIO()):
    tool = initialize_tools_from_config("configs/oncology_tools.yaml")[0]

async def read_function():
    key, initial = await tool.create(create_kwargs={"record": tool_view()})
    try:
        response, reward, metrics = await tool.execute(key, {"field": "performance_status"})
        payload = json.loads(response.text)
        rows = json.loads(payload["value"])["rows"]
        print("class:", type(tool).__name__)
        print("initial response empty:", initial.is_empty())
        print("records:", [(r["record_id"], r["score"]) for r in rows])
        print("tool reward:", reward, "query count:", metrics["record_queries"])
    finally:
        await tool.release(key)
    print("remaining instances:", len(tool._records))

asyncio.run(read_function())
```

Output:

```text
class: ClinicalRecordTool
initial response empty: True
records: [('PS-001', '1'), ('PS-002', '2')]
tool reward: 0.0 query count: 1
remaining instances: 0
```

Creation receives permitted records but returns no evidence automatically. Execute reveals the selected table; release removes the per-call instance. The pinned ToolAgentLoop performs this lifecycle per call. A trajectory budget stored only inside one such instance would disappear on the next call. The episode class in lesson 3 therefore illustrated a separate trajectory-owned responsibility.

This verifies the adapter and registry, not a model's ability to emit legal JSON or a complete AgentLoop's tokenization. Those are additional layers requiring their own tests.

## 2. Group comparable trajectories, not arbitrary patients

Suppose four scripted candidate trajectories for the same patient, index, and initial prompt receive teaching scores `[0.4,0.4,0.9,0.9]`. The higher scores represent a hypothetical evaluator preferring better evidence acquisition; these are not validated clinical rewards.

GRPO constructs relative outcome weights within that group. The pinned implementation uses sample standard deviation:

\[
\hat A_i=\frac{R_i-\bar R}{s_R+10^{-6}}.
\]

```python
import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
scores = torch.tensor([[0.4], [0.4], [0.9], [0.9]], dtype=torch.float64)
mask = torch.ones_like(scores)
groups = np.array(["SYN-ONC-001@index-v1@prompt-v1"] * 4)
advantages, _ = compute_grpo_outcome_advantage(scores, mask, groups)
print("shape:", list(advantages.shape))
print("mean:", round(scores.mean().item(), 6))
print("sample std:", round(scores.std(correction=1).item(), 6))
print("advantages:", [round(x, 6) for x in advantages[:, 0].tolist()])
```

Output:

```text
shape: [4, 1]
mean: 0.65
sample std: 0.288675
advantages: [-0.866022, -0.866022, 0.866022, 0.866022]
```

The four rows represent candidate responses, not four tokens of one response. Relative scores do not identify the true causal contribution of an intermediate query. The group must preserve the same initial task conditions; the same patient at a different cutoff is a different prompt.

## 3. See what incorrect grouping and uniform outcomes do

```python
correct, _ = compute_grpo_outcome_advantage(scores, mask, np.array(["prompt-A", "prompt-A", "prompt-B", "prompt-B"]))
print("two uniform groups:", correct[:, 0].tolist())
print("incorrectly merged groups:", [round(x, 6) for x in advantages[:, 0].tolist()])
constant, _ = compute_grpo_outcome_advantage(torch.ones_like(scores), mask, groups)
print("all equal in one group:", constant[:, 0].tolist())
```

Output:

```text
two uniform groups: [0.0, 0.0, 0.0, 0.0]
incorrectly merged groups: [-0.866022, -0.866022, 0.866022, 0.866022]
all equal in one group: [0.0, 0.0, 0.0, 0.0]
```

If the first two scores belonged to one prompt and the last two to another, every within-prompt outcome is equal and the relative term is zero. Merging them compares task difficulty rather than alternative behavior. Tensor shapes still look correct, so this can be a silent training error.

A homogeneous group provides no outcome-relative signal; epsilon prevents division by zero but does not create useful variation. Other loss terms could still have gradients. The singleton branch has separate upstream behavior; this example uses groups larger than one.

## 4. Old probabilities must be constants

We now use a two-action **toy acquisition policy**, `stop_early` versus `read_more`, to make one gradient visible. The four chosen actions are scripted, not sampled model outputs. Its probabilities are policy action probabilities, not patient mortality estimates.

For sampled action a, the ratio is:

\[
\rho=\exp(\log\pi_{current}(a)-\log\pi_{old}(a)).
\]

The old probabilities record the behavior policy at collection time and must not retain the current graph. The current probabilities require a gradient graph. Equal values before the first update do not make them interchangeable objects.

```python
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.config import ActorConfig
config = ActorConfig(strategy="fsdp", rollout_n=4, ppo_micro_batch_size_per_gpu=1)
logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
actions = torch.tensor([0, 0, 1, 1])
current = torch.log_softmax(logits, dim=-1)[actions].unsqueeze(1)
old = current.detach().clone()
loss, _ = compute_policy_loss_vanilla(old, current, advantages, mask, config=config)
loss.backward()
print("ratios:", (current-old).exp().detach()[:, 0].tolist())
print("loss near zero:", abs(loss.item()) < 1e-10)
print("logit gradients:", [round(x, 6) for x in logits.grad.tolist()])
```

Output:

```text
ratios: [1.0, 1.0, 1.0, 1.0]
loss near zero: True
logit gradients: [0.433011, -0.433011]
```

The centered advantages make the loss value near zero at ratio one, but the derivative is not zero. `read_more` receives the positive relative scores, so the local descent direction increases its probability. This follows from stipulated rewards, not proof that reading more is always desirable in oncology.

The intentionally wrong graph construction can be checked without damaging any model:

```python
broken_logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
broken = torch.log_softmax(broken_logits, dim=-1)[actions].unsqueeze(1)
broken_loss, _ = compute_policy_loss_vanilla(broken, broken, advantages, mask, config=config)
broken_loss.backward()
print("shared-graph gradients:", broken_logits.grad.tolist())
```

Output:

```text
shared-graph gradients: [0.0, 0.0]
```

Now the ratio is `exp(f(theta)-f(theta))`, identically one as a function, so derivatives cancel. `.clone()` alone would not fix this; detaching or computing old probabilities under `no_grad()` is essential.

## 5. Clipping depends on the advantage sign

The standard minimized surrogate is:

\[
\ell=\max(-\rho A,-\mathrm{clip}(\rho,0.8,1.2)A).
\]

The actual pinned function also uses a dual-clip cap for sufficiently extreme negative-advantage cases. We verify representative values rather than assume a function named vanilla equals only the textbook expression:

```python
import math
for a, ratio in [(2.,1.5), (2.,0.6), (-1.,0.5), (-1.,1.5), (-1.,5.)]:
    lp = torch.tensor([[math.log(ratio)]], dtype=torch.float64, requires_grad=True)
    value, _ = compute_policy_loss_vanilla(torch.zeros_like(lp), lp,
        torch.tensor([[a]], dtype=torch.float64), torch.ones_like(lp), config=config)
    value.backward()
    print(f"A={a:+.1f} ratio={ratio:.1f} loss={value.item():.1f} gradient={lp.grad.item():.1f}")
```

Output:

```text
A=+2.0 ratio=1.5 loss=-2.4 gradient=0.0
A=+2.0 ratio=0.6 loss=-1.2 gradient=-1.2
A=-1.0 ratio=0.5 loss=0.8 gradient=0.0
A=-1.0 ratio=1.5 loss=1.5 gradient=1.5
A=-1.0 ratio=5.0 loss=3.0 gradient=0.0
```

For a positive advantage, an excessively increased probability stops receiving extra benefit. For a negative advantage, an excessively decreased probability stops receiving extra benefit. Moving in the wrong direction can still have an active gradient outside the clip interval. Thus “outside [0.8,1.2] means no gradient” is false.

Clipping shapes the surrogate; it does not guarantee every action probability remains in the interval. Other samples and shared model parameters can move a probability further. The final example uses the upstream negative-advantage dual-clip cap of 3.

## 6. Follow one small optimizer step

The following updates only the two toy logits above, not Qwen or an oncology model. It illustrates why a loss near zero can nevertheless change behavior:

```python
optimizer = torch.optim.SGD([logits], lr=0.1)
before = torch.softmax(logits.detach(), dim=-1)[1].item()
# Gradients already came from the earlier backward on these logits.
optimizer.step()
after = torch.softmax(logits.detach(), dim=-1)[1].item()
print("P(read_more) before:", round(before, 6))
print("P(read_more) after:", round(after, 6))
optimizer.zero_grad(set_to_none=True)
print("gradient cleared:", logits.grad is None)
```

Output:

```text
P(read_more) before: 0.5
P(read_more) after: 0.521637
gradient cleared: True
```

In a real loop, clear stale gradients before backward, compute current probabilities, backpropagate, optionally clip the parameter-gradient norm, and step. Gradient-norm clipping differs from the probability-ratio clipping above: one limits the parameter gradient vector, the other modifies the loss surface.

Sampling temperature and restricted action spaces must match the probabilities used in the loss, or be handled explicitly by the estimator. The historical Qwen experiment used a constrained action distribution; its result must not be relabeled as oncology training.

## 7. Old policy, reference policy, and entropy serve different purposes

The old policy generated the batch. A reference policy may separately constrain drift from a chosen model. KL compares distributions; entropy describes how dispersed one distribution is:

```python
p, q = [0.8, 0.2], [0.5, 0.5]
kl = sum(pi*math.log(pi/qi) for pi,qi in zip(p,q))
entropy = -sum(pi*math.log(pi) for pi in p)
print("KL(p||q):", round(kl, 6))
print("H(p):", round(entropy, 6))
```

Output:

```text
KL(p||q): 0.192745
H(p): 0.500402
```

Neither quantity measures clinical correctness. An entropy bonus can encourage dispersion, and a reference penalty can restrict behavior change, but neither substitutes for a defensible task reward. The toy loss above includes no added reference-model or entropy term.

## 8. The boundary between this lesson and actual agent training

We have verified clinical data projection, a genuine framework tool, group-relative weights, and policy-loss mechanics. We have not generated autonomous oncology trajectories, established a calibrated mortality model, or validated a clinical reward.

The next implementation milestone is one bounded non-RL model rollout over these tables, preserving original sampled IDs, tool observations, response masks, policy version, and stop reason. Next compare it with full-information and fixed-query baselines. Only then connect a scientifically defensible evaluation signal to a complete trainer.

Inspect the pinned [ToolAgentLoop](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py), [GRPO estimator](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268), and [policy loss](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286). Historical experiments are in the [archive](../archive/README.md); they are optional supporting evidence, not steps to read before lesson 1.
