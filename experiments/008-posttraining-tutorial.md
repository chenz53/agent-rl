# Experiment 008: PyTorch post-training tutorial and Mintlify edition

Date: 2026-10-13. This expands nanoAstra into a broader post-training teaching project while keeping its synthetic oncology case study.

## Content and boundaries

Fourteen new chapters explain the training map, selected token log-probabilities and gradients, trajectory masks, grouped advantages, score-function updates, values/GAE, PPO/GRPO design, SFT/DPO, OPD, KL/entropy, rollout systems, evaluation, current research, and a small complete training loop. Fifteen Python blocks have adjacent captured stdout.

D2L supplies a teaching-structure reference; the chapters and compact numerical examples are original. All new tensor demonstrations use PyTorch. The Mintlify MDX edition is generated from canonical Markdown, including the previous seven current oncology lesson files and supporting reference pages.

The categorical capstone samples actions on-policy and applies 80 small SGD updates. It does not train a language model. Exact fixed-prefix KL gradients are checked against enumerated sampled-token surrogate gradients. Top-k renormalization and loss-reduction examples expose objectives that are not interchangeable.

## Execution environment

Existing `vp` environment: `/home/user/nanoAstra/.venv`, PyTorch `2.9.1+cu126`. New laboratories use CPU tensors; no model weights or dependencies were downloaded. No local Python dependencies were installed.

```sh
# On vp, from /home/user/nanoAstra:
.venv/bin/python scripts/check_tutorial_snippets.py
```

[Captured results](results/008-posttraining-tutorial.json) record stdout for every new block. Refresh is used once when authoring outputs; a subsequent strict run must compare exactly without refreshing.

## Site reproduction

```sh
python3 scripts/build_tutorial_site.py
cd website
mint broken-links
mint dev
```

The available local `mint` CLI is version 4.1.91; this older CLI lacks the `validate` command described in newer official documentation. Site checks use its supported commands and development compiler. Actual check and preview outcomes are appended after execution below.

## Evidence boundaries

Current research entries are checked as of 2026-10-13 and mark author proposals/results. Listed frontier methods are not benchmark reproductions. Private vendor pipeline details remain unknown. The exact OPD gradient check holds at a fixed prefix, not for the entire parameter-dependent context distribution. Full autonomous oncology RL and real-data clinical validation remain pending.

## Observed checks

- Fourteen new chapters and all fifteen executable Python blocks passed a strict fresh-run stdout comparison.
- Grouped shape example agrees with cross-entropy after causal alignment; selected-token gradient agrees with the full-logit derivative.
- Exact fixed-prefix reverse-KL gradient agrees with enumerated score-function gradients; the naive sampled-value derivative fails to supply that signal.
- The sampled categorical capstone improves best-action probability from 0.25 to approximately 0.9169 in its stipulated environment.
- `mint broken-links` passed after escaping prose-only MDX syntax while preserving code/math.
- The existing Mintlify CLI compiled and served the local reading site at `http://localhost:3333`. Browser inspection confirmed navigation, rendered equations, code, and captured outputs in the selected-token chapter. No hosted deployment occurred.
