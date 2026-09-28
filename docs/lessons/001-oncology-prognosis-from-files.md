# Lesson 1: Start with oncology prognosis and the files

We will build the entire course around one question: **given the records available before treatment, what can an agent responsibly conclude about a cancer patient's prognosis, and what should it read next?** We start with actual compressed files. Tools, environments, rewards, trajectories, and policy gradients will be introduced when this workflow needs them.

The provisional endpoint is 180-day all-cause mortality for a patient with metastatic non-small cell lung cancer (NSCLC), assessed before first systemic treatment. Every record and outcome in this repository is invented. This is a teaching workflow, not a calibrated survival model or a clinically validated benchmark.

## 1. What are we trying to produce?

The prediction time, or **index**, is `2025-03-01T09:00:00Z`. The horizon is 180 days after that time. The eventual statistical target is:

\[
P(T\le180\text{ days}\mid \text{information available at the index}).
\]

T is time from this index to death, not time from biopsy, diagnosis, hospital admission, or treatment. Those are different clocks and would define different tasks.

For now, we will reconstruct evidence and write an auditable assessment: disease context, functional and nutritional trajectory, unresolved tests, and source record IDs. We will not invent an individual probability. Three fictional patients cannot establish calibration or clinical accuracy.

The patient identifier and index below are supplied by the environment, not chosen by the model. Run snippets from the repository root with `PYTHONPATH=src python3`. Within each lesson, execute blocks in order in one Python session; the first block of every new lesson starts fresh. Lessons 1-4 require only the standard library. Every Output block is captured execution output, not an imagined result.

```python
from nanoastra.oncology import INDEX_TIME
patient_id = "SYN-ONC-001"
horizon_days = 180
print("patient:", patient_id)
print("index:", INDEX_TIME)
print("endpoint:", f"death within {horizon_days} days")
```

Output:

```text
patient: SYN-ONC-001
index: 2025-03-01T09:00:00Z
endpoint: death within 180 days
```

The endpoint is all-cause mortality. It is not progression, response, cancer-specific mortality, or the benefit of a particular drug. A prognostic prediction under observed care is also not a causal estimate of untreated survival.

## 2. Inspect the directory before interpreting the patient

```python
from nanoastra.oncology import DEFAULT_DATA
files = sorted(path for path in DEFAULT_DATA.glob("*.csv.gz")
               if not path.name.startswith("."))
print("clinical files:", len(files))
for path in files:
    print(path.name)
```

Output:

```text
clinical files: 13
biomarkers.csv.gz
diagnosis.csv.gz
encounters.csv.gz
labs.csv.gz
medication_administrations.csv.gz
medication_orders.csv.gz
notes.csv.gz
pathology.csv.gz
patients.csv.gz
performance_status.csv.gz
procedures.csv.gz
staging.csv.gz
vitals.csv.gz
```

The inventory excludes hidden filesystem metadata sidecars; these are not clinical CSV tables. The names follow a readable local clinical-export convention. They are not a universal EHR schema. Real systems may use `diagnoses_icd` and `labevents`, or standard domains such as `condition_occurrence` and `measurement`. A real adapter must map semantics and terminology, not simply rename files. [MIMIC schema overview](https://mimic.mit.edu/docs/iv/about/schema-overview.html), [OHDSI oncology model](https://ohdsi.github.io/OncologyWG/model.html).

Think of the files as complementary evidence sources:

| Question | Records to inspect |
| --- | --- |
| Who is the patient and which encounter defines the assessment? | patients, encounters |
| What malignancy has actually been established? | diagnosis, pathology |
| What disease extent was documented? | staging, procedures |
| How is the patient functioning over time? | performance_status, vitals, notes |
| What laboratory and molecular information is available? | labs, biomarkers |
| Was treatment planned or actually given? | medication_orders, medication_administrations |

The mortality file lives outside this directory, under `evaluator_only/`. We will use it later to explain labels. An agent reading baseline evidence must not see it.

## 3. Read one file rather than assuming it is already a dataframe

```python
import csv
import gzip
with gzip.open(DEFAULT_DATA / "labs.csv.gz", "rt", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    columns = reader.fieldnames
    raw_labs = list(reader)
print("columns:", ", ".join(columns))
print("raw rows:", len(raw_labs))
first = raw_labs[0]
print(first["record_id"], first["test_name"], first["value_num"], first["unit"])
print("numeric value stored as:", type(first["value_num"]).__name__)
```

Output:

```text
columns: record_id, version, patient_id, encounter_id, event_time, available_time, record_status, source_system, specimen_id, test_code, test_name, value_num, value_text, unit, ref_low, ref_high
raw rows: 9
HB-001 hemoglobin 11.2 g/dL
numeric value stored as: str
```

`gzip.open(..., "rt")` decompresses and decodes text. `DictReader` associates values with header names and handles quoted commas/newlines. CSV does not preserve numeric types: the reported number is still a string. An empty numeric field is missing data, not zero.

Nine raw lab rows do not mean nine distinct baseline results. Some are revisions; another result becomes visible only after the index. This is why we inspect the storage model before deriving features.

## 4. Two timestamps answer two different questions

`event_time` tells us when the underlying event occurred. `available_time` tells us when this version became visible. A specimen can be collected before a prediction even though the result is reported afterward. The distinction has a real analogue in laboratory event and storage times. [MIMIC laboratory documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html).

```python
from nanoastra.oncology import timestamp
ldh = next(row for row in raw_labs if row["record_id"] == "LDH-001")
cutoff = timestamp(INDEX_TIME)
print("collected:", ldh["event_time"])
print("available:", ldh["available_time"])
print("event before index:", timestamp(ldh["event_time"]) <= cutoff)
print("result visible at index:", timestamp(ldh["available_time"]) <= cutoff)
```

Output:

```text
collected: 2025-02-28T08:00:00Z
available: 2025-03-01T11:00:00Z
event before index: True
result visible at index: False
```

The collection date alone would admit this result incorrectly. This is **temporal leakage**: a model receives information unavailable at the intended decision time. Even perfect optimization would then solve the wrong task.

The tutor can inspect the raw export to explain the trap. The agent must receive only the filtered view. That difference between environment-owned information and agent-visible information will become the state/observation distinction in lesson 3.

## 5. Build the first valid patient view

The existing domain reader validates schemas, isolates the patient, filters both clocks, resolves record revisions, and removes cancelled records. We will unpack those operations in lesson 2; here we inspect their result.

```python
from nanoastra.oncology import snapshot
view = snapshot(patient_id=patient_id, index_time=INDEX_TIME)
print("visible rows:", sum(len(rows) for rows in view.values()))
for table in ["patients", "diagnosis", "labs", "notes", "medication_administrations"]:
    print(table, len(view[table]))
patient = view["patients"][0]
print("age:", patient["age_at_index"], "sex:", patient["sex"])
for row in view["diagnosis"]:
    print(row["record_id"], row["description"])
```

Output:

```text
visible rows: 28
patients 1
diagnosis 3
labs 7
notes 1
medication_administrations 0
age: 67 sex: female
DX-003 Chronic obstructive pulmonary disease
DX-001 Lung adenocarcinoma
DX-002 Metastatic disease involving liver and bone
```

We now have one fictional 67-year-old woman with documented lung adenocarcinoma and metastatic disease, plus a historical comorbidity. We have not inferred disease extent from age or from a future death. We read explicit records.

The seven visible labs differ from the nine raw rows because a late result was excluded and two versions of one specimen result were collapsed. No visible administration means there is no administration record in this view; by itself, that does not prove treatment never occurred elsewhere. The note supplies additional context.

## 6. Clinical notes are data with their own timing and structure

```python
note = view["notes"][0]
print("record:", note["record_id"])
print("event:", note["event_time"])
print("available:", note["available_time"])
print(note["text"])
```

Output:

```text
record: NOTE-001
event: 2025-02-28T14:00:00Z
available: 2025-02-28T17:00:00Z
New metastatic lung adenocarcinoma. Fatigue, reduced appetite, approximately 8 kg weight loss since January. ECOG 2 today, previously 1. Lives with partner; daughter helps with shopping. Independent in basic self-care. Bone pain treated with as-needed analgesia. Systemic treatment not started; final regimen awaits molecular results and shared decision-making.
Brain MRI remains pending. Discuss prognosis with uncertainty; no individualized survival estimate documented.
```

Notice what the note adds: changing function, reduced appetite, weight loss, support at home, a pending molecular panel, incomplete brain imaging, and treatment not yet started. It also says that no individualized survival estimate is documented. None of these facts should be replaced by a fabricated survival number.

The embedded newline is intentional. Splitting this file by newline or comma manually can corrupt the note. Later lessons will preserve its record ID when citing evidence and will distinguish absent results from negative tests.

## 7. How the rest of the course follows from this patient

We now have data, but not yet an agent. Each next lesson addresses the next missing piece:

| Lesson | New problem | Concept introduced because we need it |
| --- | --- | --- |
| 2 | Which rows and trends are trustworthy? | As-of joins, revisions, units, provenance |
| 3 | Which evidence should the model read next? | Tools, actions, observations, state, history, reset |
| 4 | How do we judge the final assessment? | Outcomes, censoring, evaluation, reward versus utility |
| 5 | How can final feedback train earlier queries? | Trajectory likelihood, returns, baselines, policy gradient, masks |
| 6 | How does that become a framework computation? | Real verl, group advantage, old probabilities, clipping, gradients |

We will keep the same patient and index throughout. Later numerical forecasts and rewards are explicitly labeled teaching assumptions, never presented as fitted oncology results. The earlier A/B experiment is historical evidence, not a required detour in this sequence.

Before continuing, explain why a February 28 specimen might be unavailable on March 1 and why an empty administration table does not alone establish a complete pretreatment history.

**Next:** [Lesson 2: reconstruct the evidence](002-reconstruct-oncology-evidence.md). Exact columns are in the [data dictionary](../design/003-oncology-data-contract.md).
