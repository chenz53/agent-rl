# A dated research map, not a universal recipe

Research snapshot: 2026-10-13. This chapter separates established mathematical identities, inspectable public implementations, and recent empirical proposals. A new preprint is a candidate experiment; it is not evidence that every model vendor uses the method or that it transfers to oncology.

## Two axes of alternatives

Vocabulary support determines which next-token alternatives are supervised: one sampled token, top-k, or the full vocabulary. Position selection determines which generated timesteps receive supervision. Changing either may alter cost, variance, bias, or the objective. A sparse implementation can be faster yet estimate a different gradient.

| Question | Public research | What to test before adopting it |
| --- | --- | --- |
| Can a small support retain a fixed-prefix reverse-KL gradient? | [Unbiased Top-k Estimation, 2026-09-28](https://arxiv.org/abs/2609.34447) proposes sampled tail correction | Exact enumeration on a small vocabulary; top-k construction; teacher normalization; communication cost |
| Should sparse support follow only probability rank? | [Look Before You Select, 2026-09-28](https://arxiv.org/abs/2609.34386) investigates vocabulary sparsification | Equal support/compute budget; omitted mass; teacher/student mismatch |
| Must every sampled position be supervised? | [DIAL-OPD, 2026-10-08](https://arxiv.org/abs/2610.11659) studies supervision allocation | Keep vocabulary sparsity distinct; match tokens/updates; test selection bias and domain transfer |
| Can sampled-token variance be reduced? | [KL for a KL, 2026-05-08](https://arxiv.org/abs/2605.07865) proposes a control-variate baseline | Check detachment and unbiasedness assumptions; account for value-model cost |
| Should the divergence depend on uncertainty? | [Entropy-Aware OPD, ICML 2026](https://proceedings.mlr.press/v306/jin26e.html) combines divergence directions using teacher entropy | Inspect the gating rule, temperature dependence, and multi-answer behavior |

These are author proposals and author-reported results. Only the following small numerical example is executed here; none of the listed methods has been reproduced at model-training scale in this repository.

## Why top-k renormalization changes the object

Take two distributions that differ in their tails. Selecting two entries and renormalizing them produces new distributions. The resulting KL cannot be presented as the full-vocabulary KL. A correct sparse estimator requires its own derivation and assumptions.

```python
import torch
p = torch.tensor([0.6, 0.2, 0.15, 0.05], dtype=torch.float64)
q = torch.tensor([0.5, 0.2, 0.1, 0.2], dtype=torch.float64)
full = (p * (p.log()-q.log())).sum()
idx = p.topk(2).indices
pk, qk = p[idx], q[idx]
pk, qk = pk/pk.sum(), qk/qk.sum()
restricted = (pk * (pk.log()-qk.log())).sum()
print("full reverse KL:", round(full.item(), 6))
print("renormalized top-2 KL:", round(restricted.item(), 6))
print("discarded student mass:", round((1-p[idx].sum()).item(), 6))
```

Output:

```text
full reverse KL: 0.100898
renormalized top-2 KL: 0.00321
discarded student mass: 0.2
```

## Evidence discipline

For every frontier entry, record the paper version/date, released code commit when inspected, objective, estimator, stop-gradient placements, sampling distribution, benchmark, and compute accounting. Mark whether a result is a reported claim or our measurement. A repository name alone is not an implementation audit.

An unbiased gradient conditional on a fixed prefix does not establish an unbiased gradient through the entire student-induced context distribution. Better math benchmark scores do not establish calibrated medical prognosis. Retaining these distinctions is necessary for useful research transfer.

**Exercise.** Pick one row and write a falsifiable ablation with a failure criterion. Start with exact small-vocabulary enumeration before running a large-model benchmark.

**Next.** Execute the [bounded capstone](12-capstone.md), then use the oncology track for domain integration.
