# From an expected reward to a PyTorch update

For the clamp issue, we have a generated action sequence, its selected-token log-probabilities, and an evaluator-derived weight. Now connect those quantities to an update. In the vocabulary `["return", "min", "max", "value"]`, ID 2 stands for `max` after `return `. This miniature update isolates the direction of learning rather than training a whole coding model.

A numerical reward usually has no differentiable path through a tool, evaluator, or discrete sampler to policy parameters. The score-function identity provides the connection. We first derive it and then perform an actual optimizer step on four logits.

## The derivation

Let J(theta)=E[R(tau)] under policy-generated trajectories. Assuming the reward/environment has no direct theta dependence, differentiating the trajectory distribution gives

$$
\nabla J=\mathbb{E}\left[R(\tau)\sum_{i\in\text{actions}}\nabla\log p_\theta(y_i\mid c_i)\right].
$$

Substitute an appropriate baseline or estimated advantage to obtain a surrogate sampled loss -A times the sum of action log-probabilities. Detach the weight during the policy update. This surrogate's numerical value is not a calibrated measure of task quality; its derivative is what matters.

## One real parameter update

The next example updates categorical logits, not a language model or repository agent. The sampled token is stipulated to isolate the derivative, so this is an optimizer demonstration rather than an unbiased empirical rollout estimate. A positive weight should increase its probability after a sufficiently small step.

## Construct the token-score function again

This chapter stands alone, so the selected-log-probability function is defined again. First normalize over vocabulary with `log_softmax`. Next add an index axis, select the observed code token with `gather`, and remove the index axis. The remainder constructs a trainable parameter, clears old gradients, differentiates the weighted negative log-probability, and takes one SGD step. `zero_grad` is necessary because PyTorch otherwise accumulates gradients between updates.

```python
import torch
def selected_log_probs(logits, token_ids):
    """Inputs [..., V], [...] -> [...]; vocabulary normalization remains full."""
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("Token IDs must match every non-vocabulary axis")
    log_distribution = logits.log_softmax(dim=-1)
    indices = token_ids.unsqueeze(-1)
    selected = log_distribution.gather(dim=-1, index=indices)
    return selected.squeeze(-1)
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

## Construct a masked sequence loss

Now include a tool-result position explicitly. The four supplied log-probabilities below represent generated reasoning, a generated read-file call, an environment test result, and a generated submission. Only the three policy positions contribute. These are stipulated scalar scores so the mask's derivative can be inspected directly.

`policy_gradient_loss` first requires log-probabilities and the mask to have identical `[N,T]` shapes. One detached advantage per trajectory is expanded to `[N,1]`. Multiplication broadcasts it over time, applies ownership, and sums within each trajectory. The final mean gives each trajectory equal weight on its sequence sum. The function does not silently switch to length normalization.

```python
def policy_gradient_loss(log_probs, advantages, action_mask):
    if log_probs.shape != action_mask.shape:
        raise ValueError("Action mask must align with scored token positions")
    if advantages.shape != log_probs.shape[:-1]:
        raise ValueError("Need one advantage per trajectory")
    weights = advantages.detach().unsqueeze(-1)
    weighted_tokens = weights * log_probs * action_mask
    trajectory_objectives = weighted_tokens.sum(dim=-1)
    return -trajectory_objectives.mean()

scores = torch.tensor([[-1.0, -2.0, -8.0, -4.0]], dtype=torch.float64, requires_grad=True)
action_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
advantages = torch.tensor([0.3], dtype=torch.float64)
masked_loss = policy_gradient_loss(scores, advantages, action_mask)
masked_loss.backward()
print("masked loss:", round(masked_loss.item(), 4))
print("position gradients:", [round(v, 4) for v in scores.grad[0].tolist()])
print("tool-result gradient excluded:", scores.grad[0, 2].item() == 0.0)
```

Output:

```text
masked loss: 2.1
position gradients: [-0.3, -0.3, -0.0, -0.3]
tool-result gradient excluded: True
```

The zero is the direct gradient with respect to the excluded tool-result score. In a real transformer, processing the tool-result text can still affect later generated tokens and receive gradient through that conditioning computation.

## Sequence sum versus token mean

A sum of log-probabilities follows the trajectory score-function derivation. Dividing each trajectory by its own action length changes relative trajectory weighting. Pooling all tokens changes weighting again. These can be useful design choices, but they optimize differently. Do not silently replace one reduction while claiming identical objectives.

An advantage shared by the entire rollout also reinforces unnecessary tokens in a successful trajectory. A baseline improves a statistical estimator; it does not automatically perform causal attribution. Compare redundant-query and efficient trajectories, process supervision, or value-based estimates with clear evaluator assumptions.

**Exercise.** Repeat the optimizer step with negative advantage and verify the selected probability decreases. Explain why shared model parameters make per-token probability changes less independent than this tiny experiment.

**Next question.** Can different positions use different expected-return references? Derive [values and returns](07-values-and-returns.md) before introducing policy ratios.
