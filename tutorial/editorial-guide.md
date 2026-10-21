# Editorial guide

This guide is the contract for every chapter of *Agentic RL: From Policy Gradients to Coding Agents*. Read it before writing or revising a chapter. [The outline](outline.md) assigns the scope and the `agentrl` API of each chapter.

## Who the reader is

The reader trains or fine-tunes large models and knows transformers, cross-entropy, optimizers, and distributed training. Do not explain those topics. The reader may never have derived a policy gradient, debugged an RL run, or built a multi-turn tool environment. Explain RL rigorously and from first principles. Whenever a choice is a matter of design, explain the trade-off the way a senior post-training engineer would explain it in a design review.

The book's promise is that a reader who works through it can join a frontier post-training team and contribute without a long ramp-up. That standard decides what goes in: a reader should be able to derive each estimator, implement it from scratch, recognize its failure on a dashboard, and argue for or against it in a design review.

## What we take from D2L

[D2L](https://github.com/d2l-ai/d2l-zh) (inspected 2026-10-13) contributes its teaching method. We do not copy its text, code, or multi-framework layout.

1. **One build.** A chapter derives a mechanism, implements it from scratch in a short block, and saves the result with `#@save`. Later chapters call `arl.name(...)` instead of repeating the code. By the capstone, the reader has built the whole training stack.
2. **Problem first.** Each section opens with the problem the previous section left open, then states the mechanism that resolves it.
3. **From scratch, then concise.** Build the transparent version first. Use the optimized or library version only after the reader can check it against the transparent one.
4. **Run everything.** Every printed number in the book comes from executing the block above it.
5. **Close with summary and exercises.** Every chapter ends with `## Summary` (3–6 bullet points) and `## Exercises` (4–8 items). The exercises include at least one change-an-assumption experiment and one design-review question.

## Chapter anatomy

```text
# N. Title                           <- H1 is the only H1; numbered
Opening paragraph(s): the question carried over from the previous chapter, why it matters
in practice, and what this chapter builds.

## N.1 Section title                 <- numbered sections, 4-8 per chapter
   motivation -> definition/derivation -> code -> output -> interpretation
   ...
## N.k Design problems in practice   <- REQUIRED. Real decisions, symptoms, fixes.
## Summary
## Exercises
```

Use `###` for subsections when a section needs them. Cross-reference by link, for example `[Section 7.3](07-advantages-and-credit.md#73-generalized-advantage-estimation)`, or by chapter.

### The "Design problems in practice" section

This section is what separates this book from a course on RL algorithms. Write each problem as a block in this form:

> **Problem: <a short name>.** *Symptom:* what an engineer sees (a metric, a log line, a behavior). *Cause:* the mechanism, linked to the chapter's math. *Options:* two or more fixes, each with its cost. *Default:* what to try first and why, labeled as the book's recommendation rather than a universal rule.

Include two to five problems per chapter. Show the symptom with code whenever possible, using a small reproduction from the chapter's own machinery.

### Callouts

Use these blockquote labels sparingly:

- `> **Pitfall.**` for a mistake that compiles, runs, and gives the wrong answer.
- `> **In production.**` for how large systems implement or scale the idea. Cite a source, or state that the claim is a general engineering observation.
- `> **Evidence.**` for the status of a claim: measured here, reported by a paper, or a hypothesis.

## Code

Code quality is the largest complaint about the previous edition. Code blocks must be **short and focused on a single idea**.

- **Size.** Aim for 5–20 lines per block and no more than 30. When a function is long, split it across blocks: define helpers first and explain each one. Saved blocks may reach 40 lines only when splitting would hide the mechanism.
- **One idea per block.** Define something, or run something, or show something. A block that defines a function and then demonstrates it is acceptable when both parts are short.
- **Visible mechanism.** Use explicit tensor operations for the concept being taught (`gather`, masks, `exp` of log-ratios). PyTorch built-ins are allowed for everything else (`nn.Linear`, `F.cross_entropy`, optimizers).
- **Shapes.** State tensor shapes in comments using the book's axis names: `B` (batch of trajectories), `G` (group size), `T` (sequence length), `V` (vocabulary), `D` (model width). For example, `# logp: [B, T]`.
- **Small, interpretable output.** Print a few clearly labeled numbers rather than whole tensors. Round to 3–4 significant figures. Print tensors only when they are small enough to read, at most about 4x8.
- **Determinism.** Seed everything with `torch.manual_seed`, or pass a `torch.Generator`. CPU float reductions depend on the intra-op thread count, so RL trajectories do too. A chapter that trains should call `torch.set_num_threads(4)` in its first unsaved block (the checker resets the count before each chapter), and its figure module should do the same. Pin `dtype` where precision matters. Everything runs on CPU. A chapter should finish in under 60 seconds on the check host. Chapters 9 and 19 may take up to 4 minutes.
- **Naming.** Use descriptive names (`old_logp`, `advantages`, `action_mask`), not single letters, except for standard math (`r`, `eps`, `gamma`, `lam`).
- **No hidden helpers.** A block may use only Python, PyTorch, the standard library, earlier blocks in the same chapter, and `agentrl` functions saved by **earlier** chapters. The checker enforces this.

### The `#@save` mechanism

A block whose **first line** is `#@save` becomes part of a topic module in the generated `agentrl` package. `scripts/book_library.py` owns the chapter-to-module map, import resolution, and the shared loader used by the checker. `__init__.py` contains explicit public re-exports only. Rules:

- A saved block contains only imports, `def`, `class`, and constant assignments. It does not print or call anything at top level.
- A saved block prints nothing, so it has no `Output:` block.
- A later chapter starts with `import agentrl as arl` (plus `import torch` and other imports) and calls `arl.name`.
- Within the chapter that defines a function, call it by its bare name.
- Save only functions the outline lists in that chapter's API, or that a later chapter will plausibly reuse. A demo-only helper stays unsaved.
- Never change the signature of a function another chapter saved. If a chapter needs to extend one, define a new function and explain why.
- A saved function needs a one-line docstring that states shapes.
- The generator preserves saved implementations and comments, resolves `arl.name` to the module that defines it, and adds source pointers. Topic modules use explicit dependencies on earlier chapters rather than importing the public facade. Regenerate with `python3 scripts/build_library.py`; use `--check` to detect stale generated files.
- Mutable settings belong to their defining module. Rebinding a public re-export does not update a function's module globals. For example, Chapter 17 temporarily changes `arl.grpo.FN_NAMES`, not `arl.FN_NAMES`.

### Output blocks

Each block that prints is followed by a captured output block:

````text
```python
print(1 + 1)
```

Output:

```text
2
```
````

Never type output by hand. Run `scripts/remote_check.sh <name> --refresh <chapter-prefix>` and then **read** the refreshed outputs. Prose that interprets numbers must match the actual output; check it again after every refresh.

## Mathematics

Inline math uses `$...$`. Display math puts `$$` on its own line, with blank lines around the block:

```text
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, A_t\Big]
$$
```

Number important equations in prose only ("Equation (6.3)"). Use this notation across the whole book. [The notation chapter](chapters/00-notation.md) lists it.

| Symbol | Meaning |
| --- | --- |
| $\pi_\theta$ | policy (the language model) with parameters $\theta$ |
| $\pi_{\text{old}}$ / $\mu$ | the behavior policy that generated the data; $\mu$ when it differs from a stale $\pi_\theta$ (for example, the inference engine) |
| $\pi_{\text{ref}}$ | a frozen reference policy for KL regularization |
| $s_t, a_t, o_t$ | state, action, observation at step $t$ (turn level) |
| $y_i$, $c_i$ | the $i$-th generated token and its prefix (token level) |
| $\tau$ | trajectory; $R(\tau)$ its return |
| $r_t$ or $\rho_t$ | the probability ratio $\pi_\theta/\pi_{\text{old}}$ (use $\rho$ when $r$ means reward) |
| $A_t$, $\hat A_t$ | advantage and its estimate |
| $V^\pi, Q^\pi$ | value and action-value functions |
| $G$ | group size (samples per prompt) |
| $\gamma, \lambda$ | discount, GAE parameter |
| $\epsilon$ | clipping half-width ($\epsilon_{\text{low}}, \epsilon_{\text{high}}$ when asymmetric) |
| $\beta$ | KL coefficient |

## Figures

Figures should explain mechanisms that prose and code cannot show as well. Use four kinds:

1. **Data figures** are computed from book code. Write a module `scripts/figures/chNN_<slug>.py` that exposes `FIGURES = {"name": fn}`. Each function computes its data with torch and `agentrl`, then draws with `scripts/svgplot.py`. Render them on the check host with `scripts/remote_figures.sh <name> chNN`. Examples: the variance of an estimator versus the baseline, clipped-objective landscapes, the bias and variance of KL estimators, training curves, and pass@k curves.
2. **Mechanism diagrams** use the hairline style in `scripts/hairline.py`, adapted from Lucas Marques' [Hairline](https://github.com/lucasmarkes/hairline) (MIT) and re-implemented independently. Systems, state machines, and loops become **isometric scenes**: rounded solids drawn as a bright outline with dim inner creases on a dotted floor, placed on a screen-aligned grid so flows read horizontally and vertically, with tracks routed through the visible aisle in front of the solids. Text-heavy diagrams (tables, triage maps, component maps, token strips) use the **flat hairline** primitives (`card`, `farrow`, `wrapped_strip`). Both use one thin stroke and ink tones. Exactly one stroke family carries the accent color, and it marks the thing the figure is about, such as policy tokens that carry gradient, rewarded transitions, or the path that computes reward. Text never sits on a solid. Numbered callouts with thin leaders sit outside the drawing, short readouts sit in a corner, and each figure has a `FIG N.K` caption. Ownership in token strips is shown by outline: accent for policy, solid for tool, dotted for prompt, and no box for template tokens. Keep every piece of information the figure needs; the restraint is visual, not editorial.
3. **Data plots** stay in `svgplot` (axes, heatmaps, Gantt timelines, value-encoded strips), because isometric projection distorts the quantities they show.
4. **Mermaid** is for simple flowcharts and sequence diagrams that the site renders natively.

Embed a figure as `![Figure N.k: caption sentence.](../figures/name.svg)`, then add a paragraph that tells the reader what to look at. Aim for 2–4 figures per chapter, plus Mermaid where useful. Every figure must be inspected after rendering: run `scripts/preview_figure.sh tutorial/figures/name.svg` and view the PNG. Fix overlapping labels before finishing.

Palette rules, encoded in `svgplot`: categorical slots are assigned in order and never cycled. Use one y-axis per panel and small multiples instead of dual axes. Each panel has at most 4 series with direct end labels. A legend is required when a panel has at least 2 series. Use semantic slots consistently across the book:

| Slot | Meaning everywhere in the book |
| --- | --- |
| 1 (blue) | policy-generated tokens; the main method; "trained" |
| 2 (orange) | environment and tool tokens; the comparison method; "baseline" |
| 3 (aqua) | prompt and template tokens; a third method |
| 8 (red) | reserved for failures, hacks, and errors |

## Evidence and honesty

- Separate **measured here** (printed by a block or saved to `tutorial/labs/results/`), **reported by a source** (with a link and date), and **hypothesis or recommendation**.
- Never report an experiment that was not run. Tiny CPU experiments demonstrate mechanisms. They are not benchmark results, so say so where a reader could confuse them.
- Cite primary sources (papers, official repositories at pinned commits, official docs). Record every external source in [sources.md](sources.md) with the date inspected and the boundary of what was checked. Vendor-private recipes are unknown unless they are publicly documented. Never write "all labs do X".
- The running environment executes only trusted, authored code inside the process. State this whenever execution appears, and explain the isolation that real model-generated code requires.

## Prose style

- Write in English with a direct, technical voice. Use paragraphs for explanation and bullets for enumerations, not as a substitute for explanation.
- Define every term before using it. When a term has competing meanings in the literature (for example, "GRPO", "on-policy", "KL penalty"), state which one the book uses.
- After each output, explain what the numbers show and why they had to come out that way.
- Avoid hype words such as "simply", "powerful", "magic", "crucial", and "note that".
- Use one running example: the **BugFix gym** from Chapter 4 (a family of small Python bug-fix tasks with tools `read`, `edit <op>`, `test`, and `submit`), learned by **TinyAgent**, the small transformer policy from Chapter 3. The `clamp` bug from Chapter 1 is task 0 of that family.

## Verification checklist (before you call a chapter done)

1. Run `scripts/remote_check.sh <name> --refresh NN`, then run it again without `--refresh` so the check passes cleanly.
2. Check that every block is at most 30 lines (40 for saved blocks). The checker reports `longest_block_lines`.
3. Check that every number quoted in prose matches the refreshed output.
4. Check that every figure is rendered, previewed, and referenced from the text.
5. Check that the chapter has `Design problems in practice`, `Summary`, and `Exercises` sections.
6. Check that new sources are added to `sources.md`.
7. Check that `python3 scripts/build_library.py` succeeds locally.

## The Chinese edition

`tutorial/zh/` holds an optional Simplified Chinese reading edition with the same file layout. It translates prose only. Every fenced block (code, `Output:` blocks, Mermaid), every display-math block, and every figure stays byte-identical to the English page, and `scripts/check_translation.py` enforces this together with matching heading structure and valid links. Translators keep English `#anchors` in cross-chapter links, and `--fix-anchors` maps each one to the Chinese heading at the same position. Terms and conventions are fixed in `tutorial/zh/glossary.md`. Any change to an English page must be carried into its translation in the same change.

