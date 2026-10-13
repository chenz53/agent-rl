# On-policy distillation: prefixes, distributions, and gradients

On-policy distillation is defined first by who generates the prefixes: the student. A teacher then supplies supervision at those prefixes. It is broader than one KL formula. We separate the visited-context distribution from the next-token objective so that an implementation's gradient can be checked rather than inferred from its name.

## Why student prefixes?

Training only on teacher-written completions exposes the student to teacher prefixes. At deployment the student visits its own prefixes, including its own mistakes. [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) studies training on student-generated sequences with teacher feedback. This motivates the data source; it does not guarantee that every teacher is suitable for every domain.

At a fixed visited context c, denote student probabilities by p and teacher probabilities by q. Here both use the same vocabulary and tokenization. The reverse KL is

$$D_{KL}(p\Vert q)=\sum_v p_v(\log p_v-\log q_v).$$

Forward KL exchanges p and q. Their gradients and support behavior differ. State the direction in the loss name rather than calling both simply KL. A teacher must assign nonzero probability where reverse KL is evaluated; numerical clamping changes the objective.

## Exact distribution loss

Compute both normalized distributions at each student prefix, detach teacher logits, and differentiate the sum across vocabulary. This supplies information about unchosen alternatives. The experiment compares its gradient with an exactly enumerated sampled-token surrogate at one fixed prefix.

## Why a naive sampled log-ratio is not enough

A sampled value `log p(y)-log q(y)` is an unbiased estimate of the KL value when y is drawn from p. But if y is treated as fixed by autograd, differentiating that value alone omits the derivative of the sampling distribution. An unbiased value estimator is not automatically an unbiased gradient estimator.

For a fixed prefix, the reverse-KL gradient can be represented as

$$\mathbb{E}_{y\sim p}[(\log p_y-\log q_y)\nabla\log p_y].$$

The extra constant from differentiating p log p cancels because the expected score function is zero. A corresponding sampled surrogate is `stopgrad(logp-logq) * logp`. Equivalently, a policy-gradient reward can use the negative log-ratio. The sign and stop-gradient must agree with the chosen minimization/maximization convention.

We enumerate all possible sampled tokens and weight their surrogate gradients by detached sampling probabilities. This removes Monte Carlo noise so we can test equality. These tiny teacher/student distributions are invented.

```python
import torch
from nanoastra.posttraining import reverse_kl
student = torch.tensor([0.0, 1.0, -0.5, 0.5], dtype=torch.float64, requires_grad=True)
teacher = torch.tensor([0.5, -0.2, 0.1, 1.0], dtype=torch.float64)
exact = reverse_kl(student, teacher)
exact_grad, = torch.autograd.grad(exact, student)
logp = student.log_softmax(-1)
logq = teacher.log_softmax(-1)
surrogate = (logp.exp().detach() * (logp-logq).detach() * logp).sum()
estimated_grad, = torch.autograd.grad(surrogate, student)
logp = student.log_softmax(-1)  # Fresh graph for the separate naive-gradient test.
naive_expected = (logp.exp().detach() * (logp-logq)).sum()
naive_grad, = torch.autograd.grad(naive_expected, student)
print("fixed-prefix reverse KL:", round(exact.item(), 6))
print("exact gradient:", exact_grad)
print("enumerated score-function gradient:", estimated_grad)
print("gradient match:", torch.allclose(exact_grad, estimated_grad, atol=1e-12))
print("naive expected gradient nearly zero:", naive_grad.abs().max().item() < 1e-12)
```

Output:

```text
fixed-prefix reverse KL: 0.315205
exact gradient: tensor([-0.1278,  0.4262, -0.0877, -0.2107], dtype=torch.float64)
enumerated score-function gradient: tensor([-0.1278,  0.4262, -0.0877, -0.2107], dtype=torch.float64)
gradient match: True
naive expected gradient nearly zero: True
```

## What this test does and does not establish

The identity holds conditional on a fixed context. Differentiating the complete sequence objective under a parameter-dependent distribution of earlier prefixes adds state-distribution terms. Many practical OPD recipes treat sampled prefixes as fixed during updates. That local objective is not automatically the exact gradient of full-sequence KL. Future teacher-guidance rewards can also be handled with return-to-go estimates; write down the intended target before claiming equivalence.

A single sampled token has noisy gradient information. Full vocabulary uses more distributional information at greater teacher communication and memory cost. Top-k reduces communication but may drop tail contributions or renormalize a different distribution. Token-position filtering chooses which timesteps to train; vocabulary sparsification chooses which alternatives at a timestep to compare. These are distinct axes.

## A deployment contract for the teacher

Store student policy version, sampled IDs, teacher version, tokenizer hash, temperature, prompt rendering, target positions, and whether scores are full-normalized or top-k-normalized. If teacher and student tokenizers differ, direct vocabulary-index KL is undefined; text-based distillation or an explicit alignment method is needed. A hosted teacher may provide only text or limited log-probabilities, making exact full-vocabulary KL unavailable.

For the oncology agent, a teacher with privileged outcome labels might accidentally teach leakage or hindsight. Define permitted teacher information and test whether the student answer cites records available at the prediction index. Teacher agreement is not clinical validation.

## Alternatives to test

[The frontier chapter](11-research-frontier.md) tracks tail-corrected sparse estimators, context-dependent divergence choices, control variates, and token-selection studies. Their published mathematics and reported benchmarks are separate from our tiny verified gradient identity.

**Exercises.** Reverse the KL direction and derive its gradient. Then change the teacher temperature, measure entropy, and explain why this changes the target rather than merely runtime speed. Finally, propose a full-vocabulary versus sampled-token comparison at matched teacher-query cost.

**Next.** Move from loss formulas to [rollout systems](09-rollout-systems.md).
