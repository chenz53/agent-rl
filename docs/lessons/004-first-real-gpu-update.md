# Lesson 4: Explain one real GRPO parameter update on an H200

We follow the computation from two model logits to `optimizer.step()`. Separate three kinds of evidence: a real Qwen3-0.6B update on H200, small-tensor numerical verification with actual verl/PyTorch, and an artificial A/B task with no medical interpretation.

The [raw GPU result](../../experiments/results/002-gpu-update.json), [numerical checker](../../scripts/verify_lesson_math.py), and [training script](../../scripts/demo_gpu_update.py) are the evidence. Contextual excerpts require variables from that script; standalone examples are marked below.

## 1. What decision was optimized?

The script reads `exposure=positive` and `test_x=negative` using real verl tools. The model then chooses one action, A or B. It cannot choose whether or in what order to query. The script wraps the sampled label in final-answer JSON.

This is a contextual bandit: one action conditioned on a fixed context, followed by reward. A real parameter update has been demonstrated; autonomous multi-turn agent training has not.

| Quantity | Shape | Interpretation |
| --- | --- | --- |
| Input IDs | `[1,L]` | One shared context of L tokens |
| Full logits | `[1,L,V]` | Vocabulary predictions at each position |
| Restricted logits/log-probabilities | `[2]` | Allowed actions A and B |
| Sampled action indices | `[16]` | Independent categorical draws |
| Old/current action log-probabilities | `[16,1]` | Sixteen responses, one action token each |
| Rewards, advantages, masks | `[16,1]` | Aligned with those actions |
| Loss | Scalar | Reduction over effective action positions |

## 2. Messages, templates, and input tokens

Contextual script excerpt:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=False
).to("cuda")
```

The chat template turns role-tagged messages into the model's expected format. `add_generation_prompt=True` positions the continuation at the assistant turn. `enable_thinking=False` is supported by this model's template, not a universal tokenizer option.

The actual input ends with an assistant marker and an empty think block. Those template tokens are context, not actions sampled in this experiment. They nevertheless influence the next-token distribution.

The forward input need not be copied sixteen times: all draws use the same context and initial distribution. That shortcut is specific to this one-action setup.

## 3. Logits are not probabilities

The script selects the final prediction position and the two permitted token coordinates:

```python
logits = model(**inputs).logits[0, -1, allowed_ids]
log_probs = torch.log_softmax(logits / temperature, dim=-1)
```

`allowed_ids=[32,33]` represent single-token A and B. The restricted policy is:

\[
\pi_\theta(a\mid h)=\frac{\exp(z_a/T)}{\exp(z_A/T)+\exp(z_B/T)},\qquad a\in\{A,B\}.
\]

The denominator includes only A/B, not the full vocabulary. Temperature T is part of the probability definition.

### Reconstruct equivalent logits from the recorded probabilities

Softmax is unchanged by adding a common constant to all logits. The following standalone example reconstructs one equivalent vector; these are not Qwen's recorded raw logits:

```python
import torch
p = torch.tensor([0.7391070127, 0.2608930171], dtype=torch.float64)
z = 2.0 * p.log()
for temperature in [1.0, 2.0, 4.0]:
    print(temperature, torch.softmax(z / temperature, dim=-1).tolist())
```

| Temperature | P(A) | P(B) |
| --- | --- | --- |
| 1 | 0.889207 | 0.110793 |
| 2 | 0.739107 | 0.260893 |
| 4 | 0.627304 | 0.372696 |

Higher temperature increases exploration of B in this example. It does not create useful tools, reliable knowledge, or a valid reward.

Sampling and training must use the same probability definition, or explicitly account for their difference. Sampling from a restricted T=2 distribution and evaluating log-probabilities with full-vocabulary T=1 softmax silently changes the policy and invalidates a naive ratio calculation.

## 4. Sampling frequencies are not policy probabilities

Contextual excerpt:

```python
with torch.no_grad():
    old_distribution = policy_log_probs()
    actions = torch.multinomial(
        old_distribution.exp(), 16, replacement=True
    )
    old_log_probs = old_distribution[actions].unsqueeze(1).clone()
```

Exponentiation recovers probabilities, `multinomial` samples with replacement, and each result is action index 0 or 1 rather than token ID 32 or 33. Indexing retrieves the probability of each sampled action; `unsqueeze(1)` produces `[16,1]` for the upstream response/token interface.

The actual batch contained fourteen A and two B. Empirical success was 12.5%, although the policy assigned B 26.09%. Under independent sampling, \(K\sim\mathrm{Binomial}(16,0.260893)\), so expected B count is 4.1743, while exactly two B draws occur with probability about 11.86%.

Standalone calculation:

```python
from math import comb
p = 0.2608930170536041
print(16 * p)                          # 4.174288...
print(comb(16, 2) * p**2 * (1-p)**14)  # 0.118579...
```

A small batch's empirical frequency need not match the distribution exactly. This difference is not by itself a sampling bug.

## 5. Old and current policies: same values, different graph roles

The sampling policy is \(\pi_{old}\); the optimized policy is \(\pi_\theta\). Before the first update their parameters and probability values match, but the old log-probabilities must be constants in the loss.

```python
current_log_probs = policy_log_probs()[actions].unsqueeze(1)
ratio_error = (current_log_probs.detach() - old_log_probs).abs().max()
```

The recorded maximum difference was zero. This checks the probability definition in this restricted experiment, not future parity between inference and training backends.

Sampling ran inside `no_grad()`, so a new forward builds the training graph. The script also uses:

```python
model.eval()
model.config.use_cache = False
```

`eval()` changes layer behavior such as dropout; it does not disable autograd or freeze parameters. `no_grad()` disables graph recording within its scope. Disabling the KV cache avoids an inference cache unnecessary for this forward.

## 6. Actions become artificial rewards

Contextual excerpt:

```python
chosen = [labels[i] for i in actions.tolist()]
scores = [
    compute_score(
        "nanoastra_synthetic",
        '<final>' + json.dumps({"diagnosis": label}) + '</final>',
        DEMO_CASE.label,
    )["score"]
    for label in chosen
]
rewards = torch.tensor(scores, device="cuda").unsqueeze(1)
mask = torch.ones_like(rewards)
```

B is the stipulated label. Fourteen A samples receive zero and two B samples receive one. JSON formatting is guaranteed by the script, not learned by the model.

The tensor is `[16,1]`, not `[1,16]`. In a longer response, a scalar terminal reward is often placed at the last valid token and summed across positions into a sequence score. Reward placement and the positions receiving a broadcast advantage are different concepts.

## 7. Derive the GRPO outcome advantage

Reward is observed feedback. A value is expected future return under a continuation policy. The classical advantage is \(Q^\pi(h,a)-V^\pi(h)\). This experiment trains no critic: it builds relative scores from several outcomes for the same prompt. Those scores are not known stepwise causal advantages.

### Substitute the actual sixteen outcomes

\[
\bar R=(14\cdot0+2\cdot1)/16=0.125.
\]

The pinned implementation uses sample standard deviation (N-1 denominator):

\[
s=\sqrt{\frac{14(0-0.125)^2+2(1-0.125)^2}{15}}
=\sqrt{1.75/15}\approx0.341565.
\]

With numerical epsilon \(10^{-6}\):

\[
\hat A_A\approx-0.365961,\qquad \hat A_B\approx2.561730.
\]

Standalone PyTorch example:

```python
import torch
r = torch.tensor([0.0] * 14 + [1.0] * 2, dtype=torch.float64)
advantage = (r - r.mean()) / (r.std(correction=1) + 1e-6)
print(r.mean().item(), r.std(correction=1).item())
print(advantage[0].item(), advantage[-1].item())
# 0.125, 0.341565...
# -0.365961..., 2.561730...
```

The training script calls the actual upstream estimator:

```python
advantages, _ = compute_grpo_outcome_advantage(
    rewards, mask, np.array([DEMO_CASE.case_id] * 16)
)
```

The group key must represent the same prompt/visible context sampled repeatedly. A patient at two different dates is not automatically one group. A minibatch can contain several groups: normalizing the entire minibatch together changes the comparison from within-task behavior to a mixture of task difficulty and behavior.

### Uniform outcome groups produce no relative signal

For groups larger than one, all-zero or all-one rewards yield zero outcome advantage. Epsilon prevents division by zero but creates no information. Other loss terms, such as KL or entropy, might still have gradients; this experiment includes neither.

If success probability is 1%, sixteen draws all fail with probability `0.99**16`, approximately 85.15%. Exploration can therefore determine whether a usable relative outcome signal exists. The pinned function has a special singleton branch, so do not generalize this statement to group size one without reading it.

### Broadcast to valid action tokens

A scalar 0.866024 and mask `[1,1,0,0,1,1]` produce `[0.866024,0.866024,0,0,0.866024,0.866024]`. The CPU probe checks this with real verl; the H200 experiment itself has only a one-token mask `[1]`.

## 8. Ratio: current probability relative to sampling probability

\[
\rho_i(\theta)=\frac{\pi_\theta(a_i\mid h_i)}{\pi_{old}(a_i\mid h_i)}
=\exp(\log\pi_\theta-\log\pi_{old}).
\]

Standalone example:

```python
import math
print(math.exp(math.log(0.30) - math.log(0.25)))  # 1.2
```

A change from 0.25 to 0.30 gives ratio 1.2: a twenty-percent relative increase, not twenty percentage points. Holding old probabilities fixed allows several surrogate updates on a sampled batch, but token-level ratios do not magically correct arbitrary stale trajectory distributions.

For the one-step experiment, all ratios are one when the loss is computed because parameters have not yet changed.

## 9. Clipping depends on the sign of advantage

For minimization, the standard per-token PPO loss is:

\[
\ell_i=\max\left(-\rho_i A_i,
-\operatorname{clip}(\rho_i,1-\epsilon_c,1+\epsilon_c)A_i\right).
\]

Here \(\epsilon_c=0.2\), distinct from the tiny GRPO denominator epsilon. The maximum appears because we negate a maximization surrogate. Reduction uses valid response positions; with one token per response it is the sixteen-sample mean.

For A=+2 and ratio=1.5, raw loss is -3.0 and clipped loss -2.4; the maximum is -2.4. Increasing the already large probability has no further benefit from that term. With A=+2 and ratio=0.6, the maximum is -1.2 rather than -1.6, retaining a gradient that corrects the decrease of a good action.

For A=-1 and ratio=0.5, the maximum is 0.8 rather than 0.5: further probability reduction has no additional benefit. For A=-1 and ratio=1.5, loss is 1.5 and the gradient remains active because the bad action became more probable.

| Advantage | Ratio | Loss | Derivative with respect to current log-probability |
| --- | --- | --- | --- |
| +2 | 1.5 | -2.4 | 0 |
| +2 | 0.6 | -1.2 | -1.2 |
| -1 | 0.5 | 0.8 | 0 |
| -1 | 1.5 | 1.5 | +1.5 |

These values were checked against upstream. A ratio outside [0.8,1.2] does not always imply zero gradient. Clipping is also not a hard guarantee that all probabilities stay inside that interval: shared parameters and other samples can still move them.

### The pinned implementation also applies dual-clip

For negative advantages, the function uses `min(standard_loss, -c*A)` with c=3. At A=-1, ratio=5, loss is capped from 5 to 3. That branch did not change the actual update at ratio=1. A function named `vanilla` need not match only the simplest textbook formula.

## 10. Near-zero loss can have a nonzero gradient

At the old parameters, centered advantages sum to zero:

\[
L(\theta_{old})=-\frac1{16}\sum_i A_i\approx0.
\]

But ratio equals one at that point; it is not a constant function:

\[
\nabla L\big|_{\theta_{old}}
=-\frac1{16}\sum_iA_i\nabla\log\pi_\theta(a_i\mid h).
\]

For the B logit at T=2:

\[
\frac{\partial\log\pi(a)}{\partial z_B}
=\frac{\mathbf1[a=B]-\pi(B)}{2}.
\]

Substitute fourteen A samples, two B samples, and P(B)=0.260893:

\[
\frac{\partial L}{\partial z_B}
=-\frac1{16}\left[14(-0.365961)\frac{-0.260893}{2}
+2(2.561730)\frac{1-0.260893}{2}\right]\approx-0.160108.
\]

The A derivative is approximately +0.160108. Gradient descent locally lowers A's logit and raises B's.

Standalone equivalent-logit autograd example, not another Qwen run:

```python
import torch
p = torch.tensor([0.7391070127, 0.2608930171], dtype=torch.float64)
z = (2 * p.log()).requires_grad_()
actions = torch.tensor([0] * 14 + [1] * 2)
r = (actions == 1).double()
adv = (r - r.mean()) / (r.std(correction=1) + 1e-6)
logp = torch.log_softmax(z / 2, dim=-1)[actions]
old_logp = logp.detach()
ratio = (logp - old_logp).exp()
loss = -(ratio * adv).mean()
loss.backward()
print(loss.item(), z.grad.tolist())
# loss ≈ 0
# gradient ≈ [+0.160108, -0.160108]
```

This yields near-zero loss and the nonzero derivatives above; the actual clipped upstream loss was also checked.

### A silent graph error

The following intentionally incorrect contextual line replaces the detached old probability:

```python
old_logp = logp
```

Then the ratio is `exp(f(theta)-f(theta))`, identically one, and the gradients cancel. The verified toy gradients become `[0,0]`. A `.clone()` does not itself detach; cloning in the actual script is safe because the source was computed under `no_grad()`.

## 11. From two logits to the full model

The actual optimized coordinates are model parameters, not two independent logits:

\[
\nabla_\theta L=\sum_{a\in\{A,B\}}
\frac{\partial L}{\partial z_a}\nabla_\theta z_a.
\]

Logit derivatives around 0.1601 and a model gradient norm around 142.12 describe different coordinate spaces and are consistent.

Contextual update excerpt:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The optimizer is SGD with learning rate `1e-4`, no momentum, and no weight decay. Clearing gradients matters because backward accumulates by default.

`clip_grad_norm_` returns the pre-clipping norm, measured as 142.1208. Clipping to one scales the gradient by approximately 0.00703627. Exact-arithmetic SGD would then have an update norm near `1e-4`; floating-point rounding can change the measured parameter displacement. Network Jacobians can turn a small parameter move into a visible output change.

Gradient-norm clipping limits the parameter gradient vector; PPO clipping changes the surrogate's local shape. The former activated in this experiment; the latter did not at the initial ratio.

## 12. Interpret the measurements without overclaiming

| Measurement | Value | Meaning |
| --- | --- | --- |
| Samples | 14 A, 2 B | Composition of this batch |
| P(B), before | 0.260893 | Restricted policy under the fixed context |
| P(B), after | 0.269492 | Increase of about 0.8599 percentage points |
| Initial loss | 1.49e-8 | Centered surrogate value near zero |
| Gradient norm | 142.1208 | Nonzero full-model training signal |
| Clip fraction | 0 | No effective policy clipping before update |
| Approximate KL | 0 | Old/current probabilities equal before update |
| Peak allocated memory | 5.676 GiB | PyTorch tensor memory, not all device overhead |

26.95% is not test-set accuracy. There is no convergence curve, independent clinical assessment, or demonstrated generalization.

### Old policy versus reference policy

The old policy generated this batch and supplies the ratio denominator. A reference policy constrains behavioral drift and can be a separate fixed model. They need not be identical.

For distributions p and q, `KL(p||q)=sum p*log(p/q)`. The pinned loss logs a sampled mean of `log_pi_old-log_pi_current`, estimating KL(old||current) under appropriate old-policy sampling. Finite estimates can be noisy. This diagnostic is distinct from adding a reference-model KL penalty; no reference model was used here.

### Entropy measures dispersion, not clinical quality

Entropy is `-sum p*log(p)`. A uniform two-action distribution has log(2) nats; a deterministic one approaches zero. An entropy bonus can create a gradient even when the relative outcome term is zero, but the present loss has no entropy term.

Standalone calculation:

```python
import math
p, q = [0.8, 0.2], [0.5, 0.5]
kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
entropy_p = -sum(pi * math.log(pi) for pi in p)
entropy_q = -sum(qi * math.log(qi) for qi in q)
print(round(kl, 6), round(entropy_p, 6), round(entropy_q, 6))
# 0.192745 0.500402 0.693147
```

For p=[0.8,0.2] and q=[0.5,0.5], KL is 0.192745 and the entropies are 0.500402 and 0.693147. More entropy means more dispersion, not necessarily better exploration of useful clinical actions.

## 13. Reproduce the checks

In the configured vp checkout:

```sh
.venv/bin/python scripts/verify_lesson_math.py
.venv/bin/python scripts/demo_gpu_update.py \
  --model /home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --seed 42 --samples 16 --temperature 2 --learning-rate 0.0001
```

The first command checks tensors without loading Qwen. The second performs the real update; it does not save a checkpoint. Fixed seed and snapshot reduce variation but do not guarantee bitwise reproducibility across hardware/software.

Inspect the pinned [advantage estimator](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L268) and [policy loss](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/trainer/ppo/core_algos.py#L1286).

## 14. Explain these three outcomes

1. Loss is near zero but P(B) increases: distinguish function value from derivative and keep old probabilities constant.
2. An all-success group has no relative outcome advantage: within-group comparison provides no variation; this does not prove the model is optimal.
3. Higher temperature may help discover rewarded behavior but changes action probabilities: exploration and sampling/training consistency must be analyzed together.

The next training milestone includes query actions in the policy. The oncology data workflow first establishes the evidence and outcome boundaries that such a policy must respect.
