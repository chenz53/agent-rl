# nanoAstra

Learn Agentic RL through code and build a small clinical reasoning agent. The active case is **cancer prognosis from longitudinal compressed CSV exports**. Documentation is in English; tutoring conversation may be in Chinese.

## Start here: lesson 1

Read [Lesson 1: oncology prognosis from files](docs/lessons/001-oncology-prognosis-from-files.md), then follow the [six-lesson sequence](docs/lessons/README.md) in order:

**Files and task -> evidence reconstruction -> agent environment -> outcomes and reward -> policy gradient -> verl and GRPO.**

The same synthetic patient connects every lesson. Each executable Python block has its actual output directly underneath. There is no need to read the historical A/B or emergency case first.

The task assumption is pretreatment metastatic NSCLC with a 180-day all-cause mortality endpoint. All records and outcomes are invented. The examples do not constitute a calibrated survival model or clinical benchmark.

The [data dictionary](docs/design/003-oncology-data-contract.md), [clinical files](fixtures/oncology/clinical), and [generator](scripts/generate_oncology_fixture.py) are supporting references. The main lessons 1-4 need only the standard library; [chapter 3A](docs/lessons/003a-tool-schemas-to-model-tokens.md) and lessons 5-6 use the existing vp environment, with no new local dependencies.

## RL curriculum and project records

- [Lesson and concept index](docs/lessons/README.md)
- [Learning roadmap](docs/roadmap.md)
- [Infrastructure choice](docs/design/002-infrastructure.md)
- [H200 update and reproduction](experiments/002-remote-h200.md)
- [Current curriculum and output verification](experiments/006-oncology-curriculum.md)
- [Discussion archive](docs/discussions/README.md)
- [References](docs/references/README.md)
- [Progress and decisions](PROGRESS.md)

Earlier lessons are in the [historical archive](docs/archive/README.md), separate from the current reading order. Original experiment measurements remain unchanged.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `fixtures/oncology/clinical/` | Small invented gzip CSV exports |
| `fixtures/oncology/evaluator_only/` | Outcome data, excluded from the agent tool |
| `src/nanoastra/` | Domain ingestion, tools, and evaluation helpers |
| `configs/` | Real verl tool schemas |
| `scripts/` | Reproduction commands and demonstrations |
| `tests/` | Behavioral and information-boundary checks |
| `docs/` | Lessons, data contracts, discussions, references |
| `experiments/` | Recorded results and limitations |

Current demonstrations validate data projection and framework interfaces. Complete autonomous multi-turn training and clinical evaluation remain unfinished. The remote machine is an execution mirror; no hosted Git repository has been created.
