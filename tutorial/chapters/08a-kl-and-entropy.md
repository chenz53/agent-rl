# Reference KL, entropy, and what a sampled estimate means

A reference policy, a teacher, and an old behavior policy serve different purposes. A reference can constrain drift, a teacher supplies learning targets, and behavior probabilities describe how the data were collected. Identical tensor shapes do not make them interchangeable.

## Exact categorical quantities

At a fixed prefix, entropy is minus the sum of p log p over vocabulary. Reverse KL to a reference is the sum of p times log p minus log q. Both depend on unchosen tokens. A selected-token log-ratio sampled under p estimates that KL value in expectation; a single value can be negative even though the full KL is nonnegative.

Whether KL is added directly to a differentiable loss or used to modify a reward changes gradient handling. A detached reward needs a score-function route; an exact differentiable KL already has a direct distribution gradient. Adding both without deriving the target can double-count regularization.

## Inspect a negative pointwise log-ratio

The following example prints the exact divergence and every possible sampled value. A negative sampled value is normal, rather than evidence that the KL implementation is broken.

```python
import torch
p = torch.tensor([0.8, 0.2], dtype=torch.float64)
q = torch.tensor([0.5, 0.5], dtype=torch.float64)
pointwise = p.log() - q.log()
kl = (p * pointwise).sum()
entropy = -(p * p.log()).sum()
print("pointwise log-ratios:", [round(v, 6) for v in pointwise.tolist()])
print("exact KL:", round(kl.item(), 6))
print("entropy:", round(entropy.item(), 6))
print("negative sampled value exists:", bool((pointwise < 0).any()))
```

Output:

```text
pointwise log-ratios: [0.470004, -0.916291]
exact KL: 0.192745
entropy: 0.500402
negative sampled value exists: True
```

## The measure under an estimator matters

The commonly used expression exp(-d)-1+d, with d=log p-log q, has expectation KL(p||q) when sampling from p, since the expectation of q/p equals one on the appropriate common support. Sampling from an old behavior distribution changes this identity unless the necessary correction is included. Its derivative also requires a separate analysis: value-estimator unbiasedness does not establish gradient-estimator unbiasedness.

Large KL, low entropy, and high reward can coexist. An evaluator shortcut may collapse the model onto one behavior. Monitor exact or consistently estimated KL, entropy, length, and held-out quality jointly. A reference trained only on general text may also discourage legitimate domain adaptation, so its coefficient and choice need ablations.

**Exercises.** Enumerate both vocabulary tokens to verify the estimator expectation. Repeat with sampling weights from a third distribution and show the mismatch. Compare loss-level KL and reward-level KL with explicitly written stop-gradients.

**Next.** Read the systems chapter before translating these tiny tensors into a distributed trainer.
