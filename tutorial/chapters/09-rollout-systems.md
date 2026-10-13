# Rollout systems: throughput is part of the experiment

A correct loss can still fail when the generation engine and training engine disagree about tokenization, policy versions, or probability conventions. The system must preserve the statistical contract while distributing expensive generation and updates.

## Collect, score, update, synchronize

A synchronous cycle freezes a behavior policy, samples task groups, executes tools, evaluates results, recomputes current log-probabilities, updates, then synchronizes rollout weights. Asynchronous collection overlaps work but creates policy lag. A version number should travel with each sampled token stream rather than being inferred from batch arrival time.

Use original sampled IDs to score the observed actions. Re-tokenizing an assistant string after tools are inserted may produce a different sequence; rebuilding a full chat template may insert or remove delimiters. Our earlier tokenizer chapter is an inspection exercise, not proof of exact runtime replay.

## Reductions are system-sensitive

Gradient accumulation and distributed workers need the intended global denominator. Averaging per-microbatch token means weights short and long microbatches differently from a global token mean. The same problem appears when trajectories have different action lengths.

The following experiment demonstrates three different reductions on the same artificial token values.

```python
import torch
from nanoastra.posttraining import reduce_tokens
values = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
for reduction in ["sequence_sum", "sequence_mean", "token_mean"]:
    print(reduction, reduce_tokens(values, mask, reduction).item())
```

Output:

```text
sequence_sum 5.0
sequence_mean 2.0
token_mean 2.5
```

## Design review

Plan KV-cache memory, teacher serving, tool concurrency, rollout length tails, and cancellation. Increasing parallel tool calls changes the interaction semantics if actions were intended to depend on preceding results. A database read and an irreversible external action need different retry/idempotency handling.

An invalid tool schema can cause collection failures long before the optimizer runs. Instrument parser success, tool exceptions, tool latency, stop reasons, and fraction of usable trajectories. Keep throughput metrics alongside quality; reward per update can conceal unequal generation budgets.

Use the pinned verl case study to inspect an existing implementation. These teaching kernels do not replace its scheduling, sequence packing, or distributed normalization. No full asynchronous trainer has been implemented here.

**Exercise.** Two workers produce 10 and 100 action tokens. Derive the global token-mean denominator and compare it with averaging their local means. Describe what metadata is needed to reject stale rollouts.

**Next.** Build [an evaluation contract](10-evaluation-and-design.md).
