> Historical material. Follow the [current oncology curriculum](../lessons/README.md) instead.

# Lesson 6: Cancer prognosis starts with the clinical exports

This is the active nanoAstra case. The earlier emergency diagnosis example is retained only as historical teaching material. All patients, dates, events, and outcomes below are invented. There are no real patient records in this repository.

We start with files, not a polished vignette or an already computed patient vector. The first job is to reconstruct what an agent could legitimately know at a particular time. A powerful model cannot repair a prediction task whose input already contains its outcome.

## 1. Define the prediction before opening the files

Our provisional teaching task is **180-day all-cause mortality prognosis for a patient with metastatic non-small cell lung cancer, before first systemic treatment**. The index time is `2025-03-01T09:00:00Z`. This choice is a project assumption, not a clinical recommendation or a claim that 180 days is universally appropriate.

The target probability is:

$$
p_{180}(x)=P(T\le180\text{ days}\mid X_{\le t_0}=x,\text{ eligible and alive at }t_0).
$$

Here `T` is time from the index to death, not time from diagnosis, admission, biopsy, or first treatment. `X` contains only information available by the index. The intended eventual estimate describes outcomes under the care patterns in the development population; it is not untreated survival and not the causal benefit of a particular treatment.

For now, the agent's useful output is an evidence-grounded prognostic assessment: disease context, functional/nutritional trajectory, treatment-related uncertainties, cited record IDs, and missing information. **No calibrated individual probability or exact life expectancy is implemented.** The fixture cannot supply one: three invented patients are not a development cohort.

Eligibility for a real cohort would need explicit rules: confirmed histology, documented metastatic disease, an independently defined pretreatment assessment encounter, age criteria, sufficient observation history, and evidence about previous systemic treatment. Do not require eventual treatment to include a patient: that would condition inclusion on a future event and can introduce selection and immortal-time problems. Our main fictional patient's note explicitly says treatment has not started; an empty administration table alone would not establish this in real data.

## 2. What arrives from a clinical system

The files live in [fixtures/oncology/clinical](../../fixtures/oncology/clinical). Each is a real gzip-compressed UTF-8 CSV. The names are a readable local export contract, using `snake_case`; they are not claimed to be a universal hospital schema.

| File | Clinical content | Typical connection to real systems |
| --- | --- | --- |
| `patients.csv.gz` | Patient key, age at index, sex, birth year | Patient master/demographics |
| `encounters.csv.gz` | Visit type, department, reason | Ambulatory and hospital encounters |
| `diagnosis.csv.gz` | Code system, source code, description, verification | Problem lists and encounter diagnoses |
| `labs.csv.gz` | Specimen, analyte, numeric/text result, unit, reference range | Laboratory information system |
| `vitals.csv.gz` | Weight, heart rate, oxygen saturation, measurement context | Nursing flowsheets and outpatient observations |
| `notes.csv.gz` | Note type, author role, multiline text | Signed clinical documentation |
| `pathology.csv.gz` | Specimen site, histology, report | Anatomic pathology |
| `staging.csv.gz` | Stage group, basis, disease sites | Oncology assessment/tumor registry abstraction |
| `biomarkers.csv.gz` | Assay, marker, result, pending/final status | Molecular pathology and immunohistochemistry |
| `medication_orders.csv.gz` | Intended drug/regimen, intent, order status | Orders and treatment plans |
| `medication_administrations.csv.gz` | Actual administration, dose, route, status | Medication administration record/infusion system |
| `procedures.csv.gz` | Imaging/procedure status and report summary | Radiology and procedural systems |
| `performance_status.csv.gz` | ECOG score, assessor, context | Structured oncology assessments |

MIMIC-IV uses names such as `patients`, `admissions`, `diagnoses_icd`, `labevents`, and `prescriptions`; its lab records distinguish `charttime` and `storetime`. Our `labs` is a simplified local contract, not a renamed MIMIC table with identical semantics. [MIMIC schema](https://mimic.mit.edu/docs/iv/about/schema-overview.html), [lab timestamps](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html).

OMOP uses domains such as `condition_occurrence`, `measurement`, `drug_exposure`, and `note`. Its oncology work links disease/treatment episodes to lower-level events. A mapping to that model would require explicit terminology and event relationships; changing filenames would not make this fixture OMOP-compliant. [OHDSI oncology model](https://ohdsi.github.io/OncologyWG/model.html).

The separate [evaluator_only/mortality.csv.gz](../../fixtures/oncology/evaluator_only/mortality.csv.gz) is never loaded by the agent-facing projection. It contains death and follow-up information for label construction. This is a code-level boundary for the read-only tool, not an OS sandbox: do not give a future model an unrestricted shell with access to that directory.

## 3. Read the compressed CSV correctly

Run all Python examples from the repository root with `PYTHONPATH=src`. No third-party dependencies are needed for these examples.

```python
import csv
import gzip
from nanoastra.oncology import DEFAULT_DATA

with gzip.open(DEFAULT_DATA / "labs.csv.gz", "rt", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
print(len(rows))                    # 9 raw rows, including versions and a late result
print(rows[0]["test_name"])         # hemoglobin
print(type(rows[0]["value_num"]).__name__)  # str
```

CSV does not preserve numeric types. `"3.1"` must be parsed after the relevant concept and unit are identified. Empty strings mean absent values under this export contract; they are not numeric zero or a negative test.

Use a CSV parser rather than splitting lines or commas. `NOTE-001` contains both commas and a newline inside a quoted field. The test suite checks that it survives gzip/CSV round-tripping as one note. A naive line split can corrupt the text or shift data into the wrong columns.

The deterministic generator is [generate_oncology_fixture.py](../../scripts/generate_oncology_fixture.py). It fixes gzip metadata so regeneration yields identical bytes. The JSON manifest describes the complete synthetic export for maintainers; raw row counts in that manifest are not agent observations, because they include future records.

## 4. The minimum temporal schema

Every clinical table includes:

| Column | Meaning in this fixture |
| --- | --- |
| `record_id` | Stable identity of a clinical record within a table |
| `version` | Positive integer revision of that record |
| `patient_id` | Join key across patient records |
| `encounter_id` | Related encounter when available; not every row must have one |
| `event_time` | Time of measurement, assessment, specimen, or order as appropriate to the table |
| `available_time` | Time this particular record version became visible |
| `record_status` | Active or cancelled record version |
| `source_system` | Provenance; always synthetic in this dataset |

For this fixture, all timestamps are UTC, timezone-aware, and `available_time >= event_time`. A real ETL must document source-specific meanings rather than impose this convention without checking it. An order for a future appointment has an order time and a scheduled time: neither should silently substitute for the other.

A row is eligible only if both clocks are at or before the index:

```python
from nanoastra.oncology import INDEX_TIME, timestamp
cutoff = timestamp(INDEX_TIME)
late_result = {
    "event_time": "2025-02-28T08:00:00Z",
    "available_time": "2025-03-01T11:00:00Z",
}
visible = (timestamp(late_result["event_time"]) <= cutoff
           and timestamp(late_result["available_time"]) <= cutoff)
print(visible)  # False: specimen predates index, result does not
```

Our LDH result is exactly this trap. A filter on specimen date alone would leak it. The late-signed note is another: its encounter happened before the index, but the text was not available until March 2. Retrospective text may also mention future events; timestamp filtering cannot rescue incorrectly backdated source records. Real ingestion requires provenance review and note version history.

## 5. Corrections must be resolved as of the index

`ALB-002` first reports albumin as 4.9 g/dL, then corrects it to 3.1 g/dL at 16:00 on February 28. Both versions precede the index. Keeping both would double-count one specimen; keeping the first would use a superseded value.

The implemented sequence is:

1. Validate schema, row shape, identifiers, versions, status, and clocks.
2. Restrict to the server-selected patient and the two time cutoffs.
3. For each record, select the latest available version, with version as a tie-breaker.
4. Remove records whose selected version is cancelled.
5. Sort and expose the surviving records with their original provenance.

Cancellation happens after revision selection. Otherwise a cancelled test could be resurrected by accidentally falling back to its older active version. Missing clocks fail closed with an explicit error; the demo does not guess an availability date. A production ETL could instead quarantine invalid rows and publish a quality report, but should not silently include them.

```python
from nanoastra.oncology import snapshot
view = snapshot()
albumin = [r for r in view["labs"] if r["test_name"] == "albumin"]
print([(r["record_id"], r["version"], r["value_num"], r["unit"]) for r in albumin])
# [('ALB-001', '1', '35', 'g/L'), ('ALB-002', '2', '3.1', 'g/dL')]
print([r["record_id"] for r in view["notes"]])  # ['NOTE-001']
print(view["medication_administrations"])       # []
```

There are 28 visible rows across 13 tables for the main patient, after filtering and revision resolution. This is an ingestion result, not a feature vector or a clinical risk score.

## 6. Reconstruct the patient from evidence

The main patient is a fictional 67-year-old woman with newly documented metastatic lung adenocarcinoma. Pathology supports a lung primary. The clinician records stage IV disease with liver and bone involvement; the dataset does not invent a precise TNM substage from incomplete information. Brain MRI has been ordered but not performed.

The pretreatment assessment records fatigue, reduced appetite, bone pain, and declining function. Weight is 72 kg on January 5, 68 kg on February 12, and 64 kg on February 28. ECOG is recorded as 1 and later 2. Albumin is 35 g/L and later corrected to 3.1 g/dL; hemoglobin falls from 11.2 to 10.1 g/dL. Creatinine is 0.9 mg/dL in the available report. The patient remains independent in basic self-care and receives help with shopping.

PD-L1 TPS is reported as 60%. The molecular panel is pending at the index. A later negative panel report exists in the export but is excluded. The treatment plan is a draft awaiting molecular results and shared decision-making; an actual systemic administration appears only after the index. These facts affect what can be said about the anticipated treatment context. They do not determine an individual survival probability.

Stage, performance status, weight loss, and molecular/immunologic features are clinically relevant context in NSCLC; treatment selection and prognosis must not be reduced to a single field. [NCI NSCLC PDQ](https://www.cancer.gov/types/lung/hp/non-small-cell-lung-treatment-pdq). The specific trajectory above is authored teaching data, not a patient from that source.

### Compute a transparent derived quantity

```python
from nanoastra.oncology import weight_change
change = weight_change(view["vitals"])
print(change)
# {'percent_loss': 11.111, 'baseline_record': 'WT-001', 'current_record': 'WT-003'}
normalized_albumin = [
    float(r["value_num"]) / 10 if r["unit"] == "g/L" else float(r["value_num"])
    for r in albumin  # These two verified units are g/L and g/dL.
]
print(normalized_albumin)  # [3.5, 3.1], both in g/dL
```

Weight loss is `(72-64)/72 * 100 = 11.111%`, over 54 days. The dates and source IDs matter: an identical percentage over three years is a different observation. For real patients, fluid shifts and measurement conditions can affect interpretation. [NCI nutrition PDQ](https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss/nutrition-hp-pdq).

This snippet explicitly handles the two units present in the fixture; it is not a universal unit converter. A production pipeline should reject unknown units and preserve the original result, conversion rule, and source row. Do not compare raw 35 with 3.1 and conclude there was a ninety-percent decline.

### An appropriate instructor-written assessment

> The available baseline evidence shows metastatic lung adenocarcinoma (PATH-001, STG-001) with deteriorating recorded performance status (PS-001 to PS-002), weight loss (WT-001 to WT-003), and changes in albumin and hemoglobin. These findings warrant careful prognostic assessment. Molecular results and the final systemic treatment plan remain unresolved (BIO-002, ORD-001); the existing PD-L1 result does not remove that uncertainty. Brain staging remains incomplete (PROC-002). An individualized 180-day mortality probability cannot be justified from this teaching fixture alone.

This is authored teaching text, not an LLM output or a validated scoring rule. Albumin and hemoglobin are reported as observations; we have not estimated their independent effects or assigned invented risk weights.

## 7. Build labels without calling censored patients survivors

A fixed-horizon classification label is not always observed. Let `C` denote the last time at which follow-up establishes survival, and let the observed duration be `min(T,C)`. A patient with no death record and only 60 days of verified follow-up does not have a known 180-day outcome.

Our evaluator applies these rules:

- Observed death on or before day 180: label 1.
- Known survival through day 180, including an observed death after day 180: label 0.
- No observed death and last known alive before day 180: label missing (`None`), not 0.

`ascertainment_end_time` is the end of the data collection window. It is not proof that every patient remained alive until then. In this fixture, censoring uses the last known alive timestamp when no death is observed.

```python
from nanoastra.oncology import ROOT, INDEX_TIME
from nanoastra.oncology_outcomes import load_labels
labels = load_labels(ROOT / "fixtures/oncology/evaluator_only/mortality.csv.gz", INDEX_TIME)
print([(r["patient_id"], r["followup_days"], r["death_by_horizon"]) for r in labels])
# [('SYN-ONC-001', 100.0, 1), ('SYN-ONC-002', 214.0, 0), ('SYN-ONC-003', 60.0, None)]
```

This snippet is for the evaluator, not the agent. It is deliberately in a separate module and command. The outcomes are invented to demonstrate three label states; the main patient's death is not a deduction from her baseline records.

For a real project, excluding all censored cases can bias results. Survival estimators or censoring-aware evaluation require explicit assumptions about censoring and ascertainment. No such estimator is implemented here. All-cause mortality also differs from progression-free survival, cancer-specific mortality, and response; each needs a separate outcome definition and source validation.

## 8. Where the agent and verl fit

The current agent-facing tool is still `get_clinical_record`, reusing the existing BaseTool adapter. Its `field` now selects an allowed table such as `labs` or `notes`. Patient ID, index time, and the projected records are supplied by the environment; the model cannot override them through its tool arguments.

```python
import json
from nanoastra.oncology import tool_view, TABLES
from nanoastra.clinical import read_record
record = tool_view()
answer = read_record(record, "labs", TABLES)
lab_rows = json.loads(answer["value"])["rows"]
print(len(lab_rows))                            # 7
print(read_record(record, "mortality", TABLES))  # {'status': 'invalid_field'}
```

The as-of snapshot is built once before scripted calls. Its table contents are serialized into the adapter's per-call record storage. This works for a small fixture; it is not a scalable query engine for millions of rows. A production reader would use a validated ETL plus a database/columnar store and bounded pagination. We reuse verl for execution/training interfaces rather than implement an RL framework here.

```sh
python3 scripts/demo_oncology.py --tables diagnosis labs vitals notes
python3 scripts/demo_oncology.py --tables pathology staging biomarkers performance_status
python3 scripts/build_oncology_labels.py
# In the existing vp environment only:
.venv/bin/python scripts/demo_oncology.py --verl --tables labs notes biomarkers mortality
```

The demo is scripted record retrieval. It does not generate a model forecast, calculate a clinical reward, or train an agent. The old A/B scorer must not be applied to this task.

## 9. What should eventually receive reward?

Before RL, define the output contract and compare non-RL baselines. We will need both prognostic performance and evidence fidelity. A score that rewards only predicting the observed death can favor an unsupported guess; a score that rewards many citations can favor irrelevant retrieval.

For an observed binary outcome, a candidate statistical component is negative Brier loss, `-(p-y)^2`. As a purely numerical example, if `y=1`, predictions 0.7 and 0.2 incur losses 0.09 and 0.64. This does not make either number a clinically supported forecast for this patient. Censored outcomes require appropriate treatment rather than being assigned `y=0`, and proper scoring rules assess probability quality across cases, not the truth of a written rationale.

Separate checks should cover valid evidence citations, temporal compliance, uncertainty about missing tests, and order-versus-administration distinctions. Any combined reward weights need justification and validation. Initially, report these components separately. Split real data by patient; also assess temporal and site shifts. Never let two landmarks from one patient cross training/test boundaries unnoticed.

The next milestone is an untrained model's bounded table-query rollout with a structured evidence summary. Only after verifying actual token IDs, tool observations, masks, termination, and evaluation should this become a multi-turn RL experiment.

## 10. Exercise and verification

Explain why each item is unavailable or uncertain at the index: LDH, the final molecular result, the late-signed note, actual systemic administration, and brain MRI findings. Then show how a naive join on patient ID followed only by an event-date filter would create leakage.

The [data dictionary](../design/003-oncology-data-contract.md) lists the exact columns. The [experiment record](../../experiments/005-oncology-csv.md) documents executable checks and their limits. Clinical and data-model sources were consulted on 2026-09-28; source systems and code mappings still need independent validation before real-data use.
