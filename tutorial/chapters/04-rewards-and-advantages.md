# Rewards, returns, and grouped advantages

A reward evaluates behavior according to a chosen signal. An advantage compares a decision's expected return with a reference at the same state. Neither is a token probability. In a terminal-only example we often attach one estimated advantage to every generated position; that convenience does not identify each action's causal contribution.

## Derive the baseline identity

For an action-independent baseline b(s), the expectation of b(s) times the score function is zero because the probabilities sum to one. Subtracting such a baseline can reduce variance without changing the expected on-policy gradient. A learned value model estimates expected return; it is trained with its own target and loss.

For B tasks and G rollouts, terminal rewards have shape [B,G]. Group-centered weights subtract the per-task mean. If that mean includes the sample's own reward, it is not independent of that sampled action: in the simple iid setting it scales the expected gradient by (G-1)/G relative to a leave-one-out baseline. Group standard-deviation normalization adds a further data-dependent weighting. Calling all of these estimators identical or unbiased would be misleading.

## Make the estimator an explicit choice

The following rewards are teaching numbers. Population standard deviation is used when normalization is requested; real frameworks may choose another convention. The experiment contrasts ordinary centering and leave-one-out.

```python
import torch
from nanoastra.posttraining import group_advantages
r = torch.tensor([[0.9, 0.7, 0.3, 0.5], [0.2, 0.4, 0.6, 0.8]], dtype=torch.float64)
centered = group_advantages(r)
loo = group_advantages(r, leave_one_out=True)
print("rewards shape:", tuple(r.shape))
print("centered:", centered)
print("leave-one-out:", loo)
print("G/(G-1) identity:", torch.allclose(loo, centered * 4/3))
print("all-equal group:", group_advantages(torch.ones(1, 4), normalize=True))
```

Output:

```text
rewards shape: (2, 4)
centered: tensor([[ 0.3000,  0.1000, -0.3000, -0.1000],
        [-0.3000, -0.1000,  0.1000,  0.3000]], dtype=torch.float64)
leave-one-out: tensor([[ 0.4000,  0.1333, -0.4000, -0.1333],
        [-0.4000, -0.1333,  0.1333,  0.4000]], dtype=torch.float64)
G/(G-1) identity: True
all-equal group: tensor([[0., 0., 0., 0.]])
```

## Design problems

An all-equal group provides no group-relative learning signal, even when all answers are correct or all are wrong. Filtering these groups changes the effective task distribution and consumes extra rollout compute. Increasing G improves some estimates but reduces task diversity at fixed generation budget; report both unique tasks and sampled trajectories.

Outcome-only feedback spreads credit across the trajectory. Process rewards offer finer labels but can reward plausible-sounding text instead of useful decisions. In oncology, a future mortality label is not itself a per-step reward for a table query. Separate proper predictive scoring, record fidelity, and query costs.

**Exercise.** Compare G=2 and G=8 at equal total sampled tokens. What must remain fixed to attribute an improvement to the estimator rather than more compute?

**Next.** Derive [the policy gradient](05-policy-gradients.md).
