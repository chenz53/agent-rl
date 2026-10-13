# Why gather the selected token probability?

Chapter 3 identified policy-generated segments in the clamp repair. To change the probability of those segments during training, we need to measure how likely the model was to generate their tokens. This is why token probability appears here. We start with the prefix `return ` and a tiny illustrative vocabulary `["return", "min", "max", "value"]`; sampled ID 2 means `max`. A real tokenizer would split the entire reasoning and call text into many subword tokens.

The question is not simply how to index a tensor. It is which mathematical objective the tensor represents. Sampled-action policy gradients, exact distribution matching, and entropy regularization need different quantities. This chapter derives the sampled-action quantity and then tests what its gradient actually does.

## From trajectory likelihood to a token

At a student-generated position, let the prefix be c and vocabulary logits be z. The model defines

$$
p(v\mid c)=\frac{\exp z_v}{\sum_u\exp z_u}.
$$

If the sampled token is y, its log-probability is

$$
\log p(y\mid c)=z_y-\operatorname{logsumexp}_u z_u.
$$

Gathering only z_y gives an unnormalized logit. It is not the action log-probability. Adding the same constant to every logit leaves the distribution unchanged but changes z_y, which already proves that z_y alone cannot represent it.

For a policy-generated trajectory, its conditional action likelihood is the product of the token probabilities at generated positions. Environment observations condition subsequent decisions, but the environment's likelihood factors have no policy-parameter derivative when the environment is independent of theta. The score-function gradient consequently contains the sum of selected-token log-probabilities. We do not have reward observations for every unchosen vocabulary token merely because the model can score them.

## Shapes and causal alignment

Let B denote tasks, G rollouts per task, T aligned positions, and V vocabulary size. Logical logits have shape [B,G,T,V], targets and selected log-probabilities [B,G,T]. Model batches often flatten B and G to N=B*G. A causal output at position t predicts the token at t+1, so a full-sequence forward aligns `logits[..., :-1, :]` with `input_ids[..., 1:]`. The action mask must be shifted to those same target positions.

## Gathering still updates all vocabulary logits

For this local gradient inspection, A is simply a fixed positive weight. Chapter 5 will derive learning weights from patch rewards; no baseline knowledge is required here. For loss -A log p(y), the derivative with respect to z_v is

$$
\frac{\partial L}{\partial z_v}=A\big(p(v)-\mathbf{1}[v=y]\big).
$$

The selected token supplies the observed action, but the softmax normalizer couples it to every vocabulary logit. The experiment uses a four-token vocabulary and one sampled action; these are stipulated tensors, not a model rollout.

## Construct selected log-probability step by step

The function first checks that target IDs have the same non-vocabulary axes as logits. `log_softmax(-1)` computes the vocabulary-normalized log distribution stably. `unsqueeze(-1)` adds the singleton index axis needed by `gather`; `gather` selects the observed token at every position; `squeeze(-1)` removes only that final singleton. This preserves task, rollout, and time axes. The validation branch catches misaligned targets rather than silently broadcasting them.

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
torch.set_printoptions(precision=4, sci_mode=False)
z = torch.tensor([[0.0, 1.0, 2.0, -1.0]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([2])
logp = selected_log_probs(z, y)
loss = -0.3 * logp.sum()
loss.backward()
expected = 0.3 * (z.detach().softmax(-1) - torch.nn.functional.one_hot(y, 4))
print("full distribution:", z.detach().softmax(-1))
print("selected shape:", tuple(logp.shape))
print("selected log-probability:", logp.detach())
print("all-logit gradients:", z.grad)
print("gradient identity:", torch.allclose(z.grad, expected))
print("shift invariant:", torch.allclose(logp.detach(), selected_log_probs(z.detach()+100, y)))
```

Output:

```text
full distribution: tensor([[0.0871, 0.2369, 0.6439, 0.0321]], dtype=torch.float64)
selected shape: (1,)
selected log-probability: tensor([-0.4402], dtype=torch.float64)
all-logit gradients: tensor([[ 0.0261,  0.0711, -0.1068,  0.0096]], dtype=torch.float64)
gradient identity: True
shift invariant: True
```

## Execute the grouped shape and alignment contract

Now use two tasks, three rollouts each, five total sequence positions, and a seven-token vocabulary. The logits at the first four positions predict the last four input tokens. These deterministic numbers isolate indexing; they are not sampled language-model predictions.

```python
B, G, L, V = 2, 3, 5, 7
ids = torch.arange(B*G*L).reshape(B, G, L) % V
full_logits = torch.arange(B*G*L*V, dtype=torch.float64).reshape(B, G, L, V) / 100
aligned_logits = full_logits[..., :-1, :]
targets = ids[..., 1:]
selected = selected_log_probs(aligned_logits, targets)
reference = -torch.nn.functional.cross_entropy(
    aligned_logits.reshape(-1, V), targets.reshape(-1), reduction="none"
).reshape(B, G, L-1)
print("full logits:", tuple(full_logits.shape))
print("aligned targets:", tuple(targets.shape))
print("selected log-probs:", tuple(selected.shape))
print("flattened model batch:", tuple(ids.reshape(B*G, L).shape))
print("cross-entropy agreement:", torch.allclose(selected, reference))
```

Output:

```text
full logits: (2, 3, 5, 7)
aligned targets: (2, 3, 4)
selected log-probs: (2, 3, 4)
flattened model batch: (6, 5)
cross-entropy agreement: True
```

The gather operation removes only the vocabulary axis. Group membership remains available for computing per-task baselines. The reduced time axis has L-1 positions because the final input position has no next target in this complete-sequence example.

## What is mainstream, and what is an alternative?

Sampled-token log-probabilities appear in standard categorical policy-gradient and PPO-style surrogates. Our pinned verl `logprobs_from_logits` has optimized implementations of this quantity. PyTorch's [gather](https://pytorch.org/docs/stable/generated/torch.gather.html) describes the indexing operation; [PPO](https://arxiv.org/abs/1707.06347) supplies the sampled-action ratio objective. This is an inspectable public pattern, not evidence of identical private vendor pipelines.

Exact entropy uses a sum over all vocabulary probabilities. Full-vocabulary distillation uses a teacher distribution, supplying targets for unchosen alternatives. Top-k distillation sparsifies that distribution; it is not just a faster implementation of the same sampled-reward estimator. See [distillation](10-on-policy-distillation.md) and [research alternatives](14-research-frontier.md).

A fused selected-log-probability kernel may avoid materializing [N,T,V] probabilities, but normalization still involves the vocabulary or a distributed equivalent. Local shard normalization is wrong for vocabulary-parallel logits unless the global partition function is included. A top-k teacher response may also have omitted probability mass; silently renormalizing it changes the target.

## Behavior distribution matters

Temperature and top-p modify the rollout behavior distribution. A stored log-probability must say whether it belongs to raw model logits or the actual sampling distribution. Ratios comparing different conventions do not describe the intended importance correction. Greedy output is not an unbiased sample from the raw softmax policy. Truncated support creates additional support issues; importance ratios cannot recover actions the behavior policy never sampled.

## Exercises

1. Replace y=2 with y=0 and inspect every vocabulary gradient. Explain the signs.
2. Remove the log-softmax and use a gathered logit as the loss. Which shift-invariance assertion fails?
3. For [2,4,7,1000] logits, identify the gather output shape and the final loss shape.
4. Write a three-token alignment test where an unshifted implementation scores the wrong targets.

**Next.** Construct the action mask in [trajectory ownership](03-trajectories-and-masks.md).
