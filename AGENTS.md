# nanoAstra working agreements

- All repository documentation must be in English. Conversation with the user may be in Chinese. Historical Chinese dialogue is represented by clearly labeled English translations or summaries, never falsely labeled verbatim English quotes; prior originals remain in Git history.
- The learner has years of foundation-model training experience. Skip generic deep-learning introductions; explain necessary RL foundations rigorously.
- The active application is coding tasks, starting from DeepSWE as a public reference and an authored clamp repair as the continuous teaching example. Earlier domain examples remain historical source only, outside the active tutorial and site.
- Use concrete code and existing public infrastructure. DeepSWE/rLLM/R2E-Gym provide the coding-system reference; the fixed verl SHA remains available for objective inspection. Teaching functions must be constructed visibly in each chapter rather than imported from nanoAstra.
- Lessons must explain definitions, motivation, mechanisms, worked examples, code, outputs/tensor shapes, and failure modes. Distinguish measurements, reconstructed numerical demonstrations, and hypotheses. Do not replace explanations with outlines.
- Mark standalone versus contextual snippets. Audit `scripts/check_lesson_snippets.py` when changing lesson block order. Keep conversational updates concise, while retaining detailed teaching materials.
- Remote demonstrations via `ssh vp` are authorized: one H200, `/home/user/nanoAstra`, isolated `.venv`. A previous local dependency installation was rejected; do not retry local installation. Use existing remote dependencies.
- The real verl tool/GRPO probe and one constrained Qwen3-0.6B A/B update passed. This is not complete multi-turn training or clinical capability improvement. See experiment 002.
- Archive substantive discussions under `docs/discussions/`, clearly separating translated user statements, tutor summaries, and experiment evidence. Do not invent historical conversation.
- Keep related lessons, design, implementation, and PROGRESS synchronized. Preserve the history and rationale of changed decisions.
- Distinguish paper claims, tutor interpretation, and project assumptions. Clinical knowledge and external standards require source checks and dates.
- Never report unrun experiments as completed. No promise of expertise on a fixed schedule.
- Check source permissions before using actual data. Do not commit identifiable patient data, credentials, weights, or bulk rollouts. Current fixtures are entirely synthetic and not clinically validated.
- Do not modify the parent project's read-only `sources/`. Do not create or publish a remote Git repository without a user request.
- The current record tools are read-only projections, not dynamic patient simulators. Do not imply that querying orders a test, advances time, or implements a validated clinical reward.

- The current reading route is `tutorial/chapters/01-coding-task.md` through chapter 15. Keep the dependency order: coding task, environment/tools, trajectory ownership, token probabilities, rewards/advantages, policy updates, values, sample reuse, other post-training signals, systems/evaluation, and capstone. Every executable Python block needs adjacent captured output verified by `scripts/check_tutorial_snippets.py`. Older lessons remain historical; use their original checkers for reproduction.

- Tool examples must label full objects, display projections, template excerpts, and hand-authored continuations explicitly. Trace the actual model template/token path before asserting model visibility; do not equate a captured rendering with an executed model rollout.

- The user expanded scope to broader post-training and then requested coding-only examples on 2026-10-13. `tutorial/chapters/` is the English PyTorch-only route. Use D2L as a pedagogical reference without copying its prose/code or adding MXNet/Paddle paths. Define every teaching function locally and explain its steps; project helper imports are prohibited.
- `website/` is the generated Mintlify reading edition. Edit canonical Markdown and rebuild with `scripts/build_tutorial_site.py`. Verify new chapter outputs with `scripts/check_tutorial_snippets.py`. Frontier entries need dated primary sources and clear evidence boundaries; vendor-private implementations remain unknown unless publicly documented.

- Display math uses `$$` on separate opening/closing lines. Preserve user formatting edits. Use simple Mermaid visuals where they clarify interaction, dependencies, or update flow. Generated site exports only the coding edition; do not expose historical domain pages in its navigation or downloads.
