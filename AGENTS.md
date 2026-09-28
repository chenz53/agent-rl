# nanoAstra working agreements

- All repository documentation must be in English. Conversation with the user may be in Chinese. Historical Chinese dialogue is represented by clearly labeled English translations or summaries, never falsely labeled verbatim English quotes; prior originals remain in Git history.
- The learner has years of foundation-model training experience. Skip generic deep-learning introductions; explain necessary RL foundations rigorously.
- The active application is cancer prognosis from clinical `csv.gz` exports. Start with data contracts, patient/time boundaries, longitudinal evidence, missingness, and censoring. Keep the old A/B and emergency examples only for historical reproducibility.
- Use concrete code and existing public infrastructure. The current base is the fixed verl SHA; nanoAstra implements domain data, tools, and evaluation rather than another general RL framework.
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

- The required curriculum starts at oncology lesson 1 and introduces each concept from the same patient workflow. Keep historical lessons in `docs/archive/`, outside the numbered current route. Every current executable Python block must have an immediately adjacent captured Output block verified by `scripts/check_lesson_snippets.py`; use the historical checker for old material.
