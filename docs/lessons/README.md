# The oncology-first curriculum

This is historical teaching material. The active course is now the [coding-first post-training tutorial](../../tutorial/README.md). The sequence below is retained for reproducing earlier project experiments and is not exported in the current site.

Read these lessons **in numerical order**. One synthetic patient, one baseline index, and one prognosis task connect the entire course. You do not need to read the historical A/B or emergency cases first.

Each lesson introduces a problem that the preceding lesson leaves unresolved. Code blocks run in order within a fresh session for that lesson; an actual captured Output block immediately follows every Python block. The main lessons 1-4 use the standard library. The detailed chapter 3A and lessons 5-6 use the existing vp environment; chapter 3A also uses the cached Qwen tokenizer.

| Order | Lesson | What you build or explain |
| --- | --- | --- |
| 1 | [Start with oncology prognosis and the files](001-oncology-prognosis-from-files.md) | Define the patient, index, and endpoint; open CSVs; inspect a valid baseline view |
| 2 | [Reconstruct the patient's evidence](002-reconstruct-oncology-evidence.md) | Resolve revisions and availability; normalize units; trace clinical findings to rows |
| 3 | [Turn records into an agent environment](003-oncology-tools-and-environment.md) | Define queries, observations, state, history, budgets, stopping, and reset |
| 3A | [From schemas to actual model tokens](003a-tool-schemas-to-model-tokens.md) | Trace full declarations, parser/execution, complete results, messages, templates, and token IDs before continuing lesson 3 |
| 4 | [Define outcomes and feedback](004-oncology-outcomes-and-reward.md) | Understand censoring, probability scores, evidence fidelity, costs, and baselines |
| 5 | [From trajectory to policy gradient](005-oncology-trajectories-and-gradients.md) | Derive likelihoods, returns, baselines, loss, credit assignment, and token masks |
| 6 | [Connect to real verl and GRPO](006-oncology-verl-and-grpo.md) | Use the genuine tool and algorithm interfaces; inspect groups, ratios, clipping, and gradients |

## What the outputs establish

CSV and tool outputs are real executions over invented records. Rewards, action probabilities, and candidate trajectories in numerical examples are explicitly stipulated teaching values. They are not oncology model predictions or clinically validated scores. The only optimizer step in the current lessons updates two toy logits; it does not train a clinical model.

`scripts/check_lesson_snippets.py` executes every Python block in the current sequence and compares its stdout with the adjacent Output block. Context is shared within a lesson and reset between lessons. See [curriculum verification](../../experiments/006-oncology-curriculum.md) and the [expanded 47-block verification](../../experiments/007-tool-schema-token-trace.md).

## Supporting references

- [Exact data dictionary](../design/003-oncology-data-contract.md): consult as needed, not a separate starting route.
- [Source references](../references/README.md): clinical and framework provenance.
- [Historical materials](../archive/README.md): earlier teaching editions and the original GPU update explanation, outside the required order.

After lesson 6, the next engineering milestone is a bounded untrained model rollout over the oncology tables. Full autonomous RL training and clinical validation remain pending.
