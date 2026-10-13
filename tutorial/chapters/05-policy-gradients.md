# From an expected reward to a PyTorch update

A numerical reward usually has no differentiable path through a tool, evaluator, or discrete sampler to policy parameters. The score-function identity provides the connection. We first derive it and then perform an actual optimizer step on four logits.

## The derivation

Let J(theta)=E[R(tau)] under policy-generated trajectories. Assuming the reward/environment has no direct theta dependence, differentiating the trajectory distribution gives

$$\nabla J=\mathbb{E}\left[R(\tau)\sum_{i\in\text{actions}}\nabla\log p_\theta(y_i\mid c_i)\right].$$

Substitute an appropriate baseline or estimated advantage to obtain a surrogate sampled loss -A times the sum of action log-probabilities. Detach the weight during the policy update. This surrogate's numerical value is not a calibrated measure of task quality; its derivative is what matters.

## One real parameter update

The next example updates categorical logits, not a language model or clinical agent. The sampled token is stipulated to isolate the derivative, so this is an optimizer demonstration rather than an unbiased empirical rollout estimate. A positive weight should increase its probability after a sufficiently small step.

```python
import torch
from nanoastra.posttraining import selected_log_probs
z = torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
optimizer = torch.optim.SGD([z], lr=0.1)
y = torch.tensor([2])
before = z.softmax(-1)[0, 2].item()
optimizer.zero_grad()
loss = -0.3 * selected_log_probs(z, y).sum()
loss.backward()
optimizer.step()
after = z.softmax(-1)[0, 2].item()
print("selected probability before:", round(before, 6))
print("selected probability after:", round(after, 6))
print("increased:", after > before)
```

Output:

```text
selected probability before: 0.25
selected probability after: 0.255667
increased: True
```

## Sequence sum versus token mean

A sum of log-probabilities follows the trajectory score-function derivation. Dividing each trajectory by its own action length changes relative trajectory weighting. Pooling all tokens changes weighting again. These can be useful design choices, but they optimize differently. Do not silently replace one reduction while claiming identical objectives.

An advantage shared by the entire rollout also reinforces unnecessary tokens in a successful trajectory. A baseline improves a statistical estimator; it does not automatically perform causal attribution. Compare redundant-query and efficient trajectories, process supervision, or value-based estimates with clear evaluator assumptions.

**Exercise.** Repeat the optimizer step with negative advantage and verify the selected probability decreases. Explain why shared model parameters make per-token probability changes less independent than this tiny experiment.

**Next.** Understand why reusing samples introduces [policy ratios](06-ppo-and-grpo.md).
