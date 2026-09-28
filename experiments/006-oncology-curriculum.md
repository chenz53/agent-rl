# Experiment 006: Coherent curriculum and adjacent executable outputs

Date: 2026-09-28. Scope: reorganize teaching, preserve historical results, and verify code/output agreement. No clinical model inference or GPU training was run.

## Current sequence

The six lessons begin with the oncology files and then introduce evidence reconstruction, an agent environment, outcome/reward design, trajectories/gradients, and real verl/GRPO. Each lesson uses the same patient workflow. Earlier editions now live in `docs/archive/` and have a separate historical checker.

## Execution results

All **34 Python blocks across six lessons** ran in the existing vp environment. The outputs are stored immediately beneath their code and in [the execution record](results/006-curriculum-snippets.json). A subsequent check without `--refresh` compares the document against newly executed stdout and passes.

The checks cover real CSV reads, point-in-time evidence, units and trends, a teaching episode with reset/stopping, evaluator-only censoring labels, illustrative scoring, policy-gradient/mask calculations, the genuine verl registry/BaseTool, group advantage, clipped loss, and a two-logit optimizer step. The optimizer step changes only a small teaching tensor; it does not update a language model.

During verification, remote filesystem metadata sidecars were included by an initial directory glob. The lesson inventory now explicitly excludes hidden files and reports 13 clinical tables. The clinical reader already uses an explicit allowlist and was unaffected. Captured final outputs reflect the corrected inventory.

## Reproduce

In the existing vp checkout, run `PYTHONPATH=src .venv/bin/python scripts/check_lesson_snippets.py`. By default, mismatched output raises an error without modifying documents. Use `--refresh` only when intentionally regenerating outputs after reviewing a lesson edit. Code runs sequentially within each lesson and in a fresh namespace between lessons.

The checker executes trusted repository Python, including deliberately illustrated failures caught by the snippets. It is not a sandbox for untrusted documents. `scripts/check_historical_lesson_snippets.py` preserves the old 42-parsed/29-executed selection; historical result files remain unchanged.

Local Markdown checks confirm internal file links, English content, six numbered current lessons, and an adjacent output for every Python block. Software/teaching checks do not establish clinical validity or autonomous agent performance.
