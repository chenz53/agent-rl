# nanoAstra

Learn Agentic RL through code and build a small clinical reasoning agent. The active case is **cancer prognosis from longitudinal compressed CSV exports**. Documentation is in English; tutoring conversation may be in Chinese.

## Start with the data

1. [Cancer prognosis: from CSV files to an as-of patient view](docs/lessons/006-oncology-prognosis-from-csv.md).
2. [Exact data contract and dictionary](docs/design/003-oncology-data-contract.md).
3. [Synthetic clinical files](fixtures/oncology/clinical), [manifest](fixtures/oncology/manifest.json), and [generator](scripts/generate_oncology_fixture.py).
4. [Verification and limitations](experiments/005-oncology-csv.md).

```sh
python3 scripts/demo_oncology.py --tables diagnosis labs vitals notes
python3 scripts/demo_oncology.py --tables pathology staging biomarkers performance_status
# Evaluator only: never expose this output to the agent.
python3 scripts/build_oncology_labels.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

These commands need only the standard library. In the existing vp environment, `.venv/bin/python scripts/demo_oncology.py --verl` uses the real verl registry and BaseTool. No new local training dependencies are required.

The teaching assumption is metastatic NSCLC, before first systemic therapy, with a 180-day all-cause mortality endpoint. The main fixture contains a detailed fictional patient; two sparse additional patients demonstrate isolation and censoring. It is not a training cohort, a calibrated survival model, or a clinically reviewed benchmark.

## RL curriculum and project records

- [Lesson and concept index](docs/lessons/README.md)
- [Learning roadmap](docs/roadmap.md)
- [Infrastructure choice](docs/design/002-infrastructure.md)
- [H200 update and reproduction](experiments/002-remote-h200.md)
- [Mathematics and snippet verification](experiments/003-teaching-verification.md)
- [Discussion archive](docs/discussions/README.md)
- [References](docs/references/README.md)
- [Progress and decisions](PROGRESS.md)

The original A/B task remains for reproducing the recorded single-step update. The emergency case is superseded as the active application and retained as historical material. Neither should be mistaken for oncology training.

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
