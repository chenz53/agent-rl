# Returns, value functions, and GAE

The previous policy-gradient chapter used one final scalar for every action. A critic instead estimates expected future return at each state or visible history. This provides a state-dependent reference; it does not magically label each reasoning sentence's causal contribution.

## Reward is local; return includes the future

For per-step rewards, define return-to-go as the discounted sum of rewards from that step onward. Gamma chooses the discount convention. A token-indexed discount penalizes later rewards differently from a tool-step discount, so the timestep definition belongs in the algorithm contract. With only a final reward and gamma=1, every preceding step has the same Monte Carlo return.

A value function V(c) estimates expected return conditional on the agent's available history. In a partially observed environment, access to privileged state needs an explicit training design; quietly feeding outcomes to the policy is leakage. The value prediction is trained against return targets, while the policy uses detached advantage estimates.

## Temporal-difference residuals

At step t, delta is r_t + gamma V(c_next) - V(c_t), with the bootstrap term removed at a true terminal boundary. Generalized advantage estimation recursively combines these residuals using gamma times lambda. Lambda=1 approaches a bootstrapped Monte Carlo estimate; lambda=0 uses one-step TD. This is a bias/variance choice relative to critic quality and rollout boundaries.

The example has three abstract decisions and a true terminal endpoint. The value predictions and final reward are stipulated. It uses gamma=1 and lambda=0.95 to show that estimated advantages can vary by step.

```python
import torch
rewards = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
values = torch.tensor([0.2, 0.4, 0.6, 0.0], dtype=torch.float64)
gamma, lam = 1.0, 0.95
deltas = rewards + gamma * values[1:] - values[:-1]
advantages = torch.zeros_like(rewards)
accumulator = 0.0
for t in reversed(range(len(rewards))):
    accumulator = deltas[t] + gamma * lam * accumulator
    advantages[t] = accumulator
returns = advantages + values[:-1]
print("TD residuals:", [round(v, 4) for v in deltas.tolist()])
print("GAE advantages:", [round(v, 4) for v in advantages.tolist()])
print("lambda-return targets:", [round(v, 4) for v in returns.tolist()])
```

Output:

```text
TD residuals: [0.2, 0.2, 0.4]
GAE advantages: [0.751, 0.58, 0.4]
lambda-return targets: [0.951, 0.98, 1.0]
```

## Why training a critic is not free

A critic adds memory, prediction error, and its own optimization. Poor values can introduce biased or noisy weights; monitor value error, explained variance, and advantages by trajectory stage. Group-relative estimators avoid a learned critic in some setups, trading it for repeated rollout compute and group-dependent statistics.

External truncation differs from a true terminal event when the intended task continues beyond the collection boundary. Bootstrapping then depends on whether a next-state value is available and whether the task defines that boundary as terminal. Never apply one rule without recording the stop reason.

Read the original [GAE paper](https://arxiv.org/abs/1506.02438) for the estimator. This code is an abstract three-step example, not a transformer critic implementation.

**Exercises.** Set lambda to zero and one and explain the targets. Place a timeout between the second and third step and specify when you would retain a bootstrap value. Then define whether a timestep in your agent is a token, an assistant segment, or a completed tool interaction.
