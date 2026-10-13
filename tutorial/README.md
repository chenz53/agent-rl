# Hands-on Post-training for Coding Agents

An English, PyTorch-only tutorial that starts from a coding issue and builds every teaching function visibly. DeepSWE is the public system reference; our authored `clamp` repair supplies a small, executable thread across all examples. D2L informs the teaching progression, not the text or framework implementation.

## Why this order?

A probability calculation needs an action whose likelihood matters. An action needs a task and an environment. A loss needs both action likelihood and a learning signal. We therefore introduce these dependencies in that order instead of jumping from a broad training map to token tensors.

| Order | Question left by the preceding chapter | Chapter |
| --- | --- | --- |
| 1 | What bug should the agent fix, and what counts as success? | [Coding task and DeepSWE context](chapters/01-coding-task.md) |
| 2 | How can it read, edit, test, and submit code? | [Coding environment from scratch](chapters/02-coding-environment.md) |
| 3 | Which parts of an interaction belong to the policy? | [Trajectories and ownership](chapters/03-trajectories-and-masks.md) |
| 4 | How likely was the generated reasoning or tool request? | [Token probabilities and shapes](chapters/04-token-probabilities.md) |
| 5 | Was the resulting repair better than the reference level? | [Rewards and advantages](chapters/05-rewards-and-advantages.md) |
| 6 | How does that score update the model? | [Policy gradient from scratch](chapters/06-policy-gradients.md) |
| 7 | Can different decisions use different expected-return references? | [Values, returns, and GAE](chapters/07-values-and-returns.md) |
| 8 | What if we reuse a trace after changing the policy? | [PPO and GRPO](chapters/08-ppo-and-grpo.md) |
| 9 | What changes when supervision is a demonstration or preference? | [SFT and DPO](chapters/09-sft-and-preferences.md) |
| 10 | What changes when a teacher scores the student's own prefixes? | [On-policy distillation](chapters/10-on-policy-distillation.md) |
| 11 | How do teacher guidance, reference KL, and entropy differ? | [KL and entropy](chapters/11-kl-and-entropy.md) |
| 12 | How do unequal trajectories and serving systems affect the update? | [Rollout systems](chapters/12-rollout-systems.md) |
| 13 | Can the patch exploit the verifier instead of solving the issue? | [Evaluation design](chapters/13-evaluation-and-design.md) |
| 14 | Which frontier alternatives deserve a controlled experiment? | [Research map](chapters/14-research-frontier.md) |
| 15 | Can we combine execution, sampling, scoring, and updating? | [Coding patch capstone](chapters/15-coding-capstone.md) |

Read in order. Every chapter starts by connecting to the coding interaction already established. New functions are defined in the chapter, including their repeated definitions when required for an independent session. PyTorch primitives and standard-library functions are used directly; there are no project helper imports.

## Execute and inspect

The fifteen chapters contain twenty-two executable Python blocks, each immediately followed by captured stdout. Blocks share a namespace within a chapter; each chapter receives a fresh namespace. The verified environment is the existing `vp` Python/PyTorch installation, and compact examples run on CPU without loading model weights.

```sh
# From /home/user/nanoAstra on vp:
.venv/bin/python scripts/check_tutorial_snippets.py
```

This checker also rejects `nanoastra` helper imports. The teaching functions are intentionally visible even when an optimized production kernel would be preferable. Tiny source execution accepts only trusted authored candidate strings; arbitrary generated code requires real process/container isolation.

Canonical Markdown lives in `tutorial/chapters/`. Generate the Mintlify reading edition with:

```sh
python3 scripts/build_tutorial_site.py
cd website
mint broken-links
mint dev
```

[Experiment 009](../experiments/009-coding-tutorial.md) records actual results. [Sources](sources.md) pin the inspected DeepSWE infrastructure reference and label research evidence. [The editorial guide](editorial-guide.md) specifies code, math, and visualization conventions.

## What the capstone establishes

The small policy learns to choose among four authored patches using real test execution. It does not train a transformer to write new code or reproduce DeepSWE. The next model-backed milestone adds tokenization, generated tool calls, isolated repository execution, policy-version tracking, and independent held-out evaluation using existing production infrastructure.
