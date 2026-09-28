# Lessons and concept index

The learner is experienced in foundation-model training. Lessons use definitions, mechanisms, worked numbers, executable snippets, output interpretation, and failure analysis.

## Current application

Start with [Lesson 6: cancer prognosis from CSV exports](006-oncology-prognosis-from-csv.md). It defines the data and task before introducing a learned policy. Lessons 1-4 retain their artificial A/B example so the existing mathematical and GPU results remain reproducible.

| Lesson | Main question | Evidence/code |
| --- | --- | --- |
| [1. Environment](001-clinical-reasoning-as-an-environment.md) | What decisions are being optimized? | State, observations, transitions, costs |
| [2. Policy gradients](002-one-clinical-trajectory.md) | How does final reward train earlier actions? | Likelihood, returns, baselines, masks |
| [3. verl interfaces](003-verl-code-walkthrough.md) | Where do data, tool calls, and scores flow? | Registry/BaseTool/reward/grouping |
| [4. Real H200 update](004-first-real-gpu-update.md) | How do logits become a parameter update? | Qwen measurements and numerical checks |
| [5. Historical emergency case](005-realistic-clinical-case.md) | How does evidence change diagnostic certainty? | Superseded staged record replay |
| [6. Oncology CSV workflow](006-oncology-prognosis-from-csv.md) | What was knowable at a prognostic index? | CSV parsing, versions, temporal filtering, censoring |

## Find a concept

| Concept | Location | Worked example |
| --- | --- | --- |
| State / observation / history | Lesson 1 sections 2-3 | Hidden case and restricted views |
| POMDP / belief | Lesson 1 section 3 | Bayesian update to 0.8 |
| Transition / reset / ending | Lesson 1 sections 5-6 | Query budget and clean episodes |
| Reward / return / utility | Lesson 1 sections 7-8 | 0.95 minus 0.05 query value |
| Rollout likelihood | Lesson 2 sections 1-3 | Probability 0.16 |
| Return-to-go / baseline | Lesson 2 sections 4-5 | [0.90,0.95,1.00], +0.6/-0.4 |
| On-policy / stale data | Lesson 2 section 6 | Old versus updated action probabilities |
| Credit assignment | Lesson 2 section 7 | Unnecessary query in a successful trajectory |
| Token alignment / masks | Lesson 2 sections 8-9 | Tool context versus sampled action terms |
| Schema / lifecycle | Lesson 3 sections 3-5 | Per-call create/execute/release |
| Reward aggregation / groups | Lesson 3 sections 8-9 | Costs not reaching loss; mixed prompts |
| Temperature / sampling | Lesson 4 sections 3-4 | Sixteen samples are not exact probabilities |
| GRPO normalization | Lesson 4 section 7 | Fourteen zeros and two ones |
| Ratio / clipping / dual-clip | Lesson 4 sections 8-9 | Five positive/negative advantage cases |
| Loss versus gradient | Lesson 4 section 10 | Near-zero loss, gradients +/-0.160108 |
| Gradient clipping / SGD | Lesson 4 section 11 | Norm 142.12 clipped to 1 |
| Old/reference policy / KL / entropy | Lesson 4 section 12 | Sampling drift versus regularization |
| Event versus availability time | Lesson 6 sections 4-5 | Late lab result and corrected albumin |
| Longitudinal features / units | Lesson 6 section 6 | 11.111% weight loss, g/L versus g/dL |
| Index / horizon / censoring | Lesson 6 sections 1 and 7 | Event, known non-event, unknown outcome |

## Verification

The original first four lessons contain 42 Python blocks; 29 are explicitly selected for execution. Contextual model excerpts and intentionally incorrect examples are syntax-checked but not run as standalone programs. The selection remains explicit in `scripts/check_lesson_snippets.py`.

Historical verification is recorded in [experiment 003](../../experiments/003-teaching-verification.md). English-edition and oncology verification is recorded in [experiment 005](../../experiments/005-oncology-csv.md). These checks do not repeat the GPU update or demonstrate clinical validity.
