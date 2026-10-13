# Editorial guide and evidence policy

## What we learn from D2L

The [D2L repository](https://github.com/d2l-ai/d2l-zh) and its [softmax implementation chapter](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_linear-networks/softmax-regression-scratch.md) were inspected on 2026-10-13. We adopt its progression from a motivated mechanism to visible tensor operations, implementation, evaluation, and exercises. We do not copy its text, examples, or multi-framework architecture. All new numerical examples are authored here and use PyTorch.

## A chapter must answer a question

Begin with a practical ambiguity. Define variables and tensor axes before manipulating them. Derive enough of the objective to justify indexing, masks, stop-gradients, and reductions. Print interpretable values and verify identities. Explain the output immediately. End with an exercise that changes an assumption and exposes a failure, rather than asking readers to merely repeat code.

## A design discussion must compare contracts

For each option state its objective, data-generating policy, support assumptions, compute cost, estimator bias/variance, engineering implications, evaluation, and unknowns. Distinguish a reference from a teacher and a behavior policy; distinguish sampled value from sampled gradient; distinguish a fixed-prefix result from a full sequence objective. A named algorithm is not a complete recipe.

## Research and sources

Use original papers, official documentation, and exact source commits. Label preprints and author-reported results; do not promote them to universal defaults. Recheck the research chapter before a new edition. Do not claim that all vendors share a last-stage pipeline or infer private details from an API format. Maintain a source ledger with date and evidence boundaries.

## Reproducibility

Canonical Markdown is the source of truth; generated MDX must be rebuilt after edits. Execute trusted snippets with `scripts/check_tutorial_snippets.py`; every block needs adjacent captured Output. Use fixed CPU seeds and explicit dtype where useful. Keep experiment metadata and stdout in `experiments/`. A successful tiny gradient test is not a large-model benchmark reproduction.

## Coding-first revision

Use coding questions for every example, with the clamp issue as the continuous authored thread and DeepSWE as the public system reference. The dependency order is task, environment, trajectory ownership, token probability, learning signal, update. Start each chapter by answering why its mechanism is now needed.

Define every teaching function visibly. Do not import project helpers. Explain argument shapes, intermediate tensors, validation, detachment, and reduction choices before using the function. Repeated local definitions preserve standalone chapter execution. PyTorch and standard-library primitives are allowed; implementation shortcuts must not conceal the mechanism.

Display math must use `$$` on its own opening and closing lines with blank lines around the block. Use Mermaid for simple interaction or dependency diagrams; use numerical experiments to validate the mechanism rather than decorative diagrams. The checker must enforce helper-import and math-format rules.
