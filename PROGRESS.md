# Progress and decisions

## User direction

- Become proficient in practical and theoretical Agentic RL; existing expertise is foundation-model training.
- Focus on biomedical agents, initially clinical reasoning; keep materials, discussions, and code in nanoAstra.
- Use existing public frameworks and concrete code; remote `ssh vp` demonstrations are authorized.
- Explain each concept with detailed examples, intermediate steps, and snippets.
- Latest change: all documentation in English; Chinese conversation is acceptable. Replace the active emergency case with cancer prognosis starting from clinical `csv.gz` files.

## Completed historical milestones

- Local repository, curriculum, discussion archive, and six original domain tests.
- Pinned verl source, genuine BaseTool adapter/registry, and CPU GRPO probe.
- One H200 Qwen3-0.6B constrained A/B update: P(B) 0.260893 to 0.269492, no saved checkpoint. Queries were scripted.
- Four detailed mathematical/framework lessons; 42 Python blocks parsed and 29 selected blocks executed in the original verification.
- A staged synthetic emergency case, eleven tests, and three additional verified snippets. Retained as historical evidence, superseded as the active application.

## Current oncology work

- Thirteen gzip CSV clinical tables plus a separately loaded mortality/follow-up table, generated entirely from invented records.
- Main illustrative patient: pretreatment metastatic NSCLC; provisional endpoint is 180-day all-cause mortality at a fixed index.
- As-of filtering on event and availability clocks, record revisions, cancellation handling, patient isolation, and explicit missing follow-up labels.
- English data dictionary, detailed file-first lesson, scripted queries through the existing adapter, and evaluator-only labels.
- Verification evidence and final checks are in [experiment 005](experiments/005-oncology-csv.md).

## Next milestones

1. Review the endpoint, eligibility, real source schema, and outcome ascertainment before using actual data.
2. Run a bounded untrained model's table-query trajectory and verify original tokens, observations, masks, and stopping.
3. Establish evidence-quality and censoring-aware predictive evaluation, then connect a full trainer.

No autonomous oncology model, calibrated survival predictor, clinical validation, or complete multi-turn RL training has been demonstrated. The remote directory is an execution mirror, not a published Git repository.

## Curriculum ordering revision

The current course now starts at oncology lesson 1 and proceeds through evidence, environment, outcomes/reward, trajectories/gradients, and real verl/GRPO. Historical editions moved to `docs/archive/`. Every current Python block has adjacent recorded output; verification is in [experiment 006](experiments/006-oncology-curriculum.md).

## 2026-09-29: interactive tutoring

Started lesson 3 with one record query and the environment/model visibility boundary. A short understanding check is pending before proceeding to runtime state and transitions.

## 2026-09-29: detailed schema and token chapter

Added Lesson 3A within the oncology-first route: 13 executed examples trace tool definitions, backend registration, schema serialization, cached Qwen tokens, parser dispatch, complete clinical responses, and the next conversation context. All 47 snippets across six lessons plus chapter 3A passed execution. Full record IDs survive the response; the previous score list was a diagnostic projection. See [experiment 007](experiments/007-tool-schema-token-trace.md). No model sampling or full AgentLoop run was performed.
