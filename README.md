# nanoAstra

Learn modern post-training and Agentic RL through coding tasks and visible PyTorch implementations. Documentation is English; tutoring conversation may be Chinese. DeepSWE supplies the public system reference, and one authored clamp repair connects the executable examples.

## Start with the coding task

Read [chapter 1](tutorial/chapters/01-coding-task.md), then follow the [dependency-based learning route](tutorial/README.md). The fifteen chapters contain twenty-two executed Python blocks, local definitions of every teaching function, centered display equations, and interaction/training diagrams.

**Coding issue → tools/environment → trajectory → token probability → feedback → policy update.** Later chapters develop values/GAE, PPO/GRPO, SFT/DPO, on-policy distillation, KL/entropy, rollout systems, evaluation, current alternatives, and an executable coding-patch capstone.

## Run and read

```sh
# Existing remote environment, from /home/user/nanoAstra:
.venv/bin/python scripts/check_tutorial_snippets.py

# Local reading interface:
python3 scripts/build_tutorial_site.py
cd website
mint dev
```

The checker rejects project helper imports and inline display-math delimiters. Source Markdown is canonical; the Mintlify edition is generated. [Experiment 009](experiments/009-coding-tutorial.md) records the revised verification. [Sources](tutorial/sources.md) and [editorial conventions](tutorial/editorial-guide.md) distinguish public reports, mathematical assumptions, and repository measurements.

The capstone learns a finite patch-selection policy from actual test execution. It is not a complete language-model or DeepSWE training reproduction. Earlier source, fixtures, and experiments remain in the repository as historical records and are excluded from the active reading site. No public hosted site or remote GitHub repository has been created.
