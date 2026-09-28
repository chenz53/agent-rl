# Historical teaching materials

These are preserved earlier editions, not the current curriculum. Begin at [current lesson 1](../lessons/001-oncology-prognosis-from-files.md) and follow the six lessons in order.

| Historical document | Reason retained |
| --- | --- |
| [Original environment lesson](001-clinical-reasoning-as-an-environment.md) | A/B task definitions behind early experiments |
| [Original policy-gradient lesson](002-one-clinical-trajectory.md) | Earlier trajectory derivations |
| [Original verl walkthrough](003-verl-code-walkthrough.md) | Original adapter and scorer call path |
| [Original H200 update explanation](004-first-real-gpu-update.md) | Measured Qwen update and detailed numerical reconstruction |
| [Emergency case](005-realistic-clinical-case.md) | Superseded staged clinical example |
| [First oncology draft](006-oncology-prognosis-from-csv.md) | Earlier combined file-to-reward lesson, now split into the current sequence |

Historical numbering and outputs describe the original experiments. `scripts/check_historical_lesson_snippets.py` preserves the earlier explicit selection of 29 runnable blocks from 42 parsed blocks. The current checker instead verifies all current Python blocks against their adjacent recorded outputs.

Original experiment evidence remains in `experiments/results/`. Moving teaching files did not rerun or reinterpret the old GPU experiment as oncology training.
