# Lesson 2: Reconstruct the patient's evidence before reasoning about prognosis

In [lesson 1](001-oncology-prognosis-from-files.md), we opened the clinical export and built a baseline view. Now we examine why each row survives, how to compute a trend, and what uncertainty remains. No model or RL is needed yet: these are properties of the task's evidence.

Start a fresh Python session from the repository root with `PYTHONPATH=src`. Execute blocks in order. Outputs are recorded executions of the invented fixture.

## 1. A patient join is necessary but not sufficient

Joining every row with the same `patient_id` would include future events and late-entered documentation. The snapshot requires both `event_time <= index` and `available_time <= index`, then resolves revisions. First inspect a corrected result in the raw file:

```python
from nanoastra.oncology import DEFAULT_DATA, COMMON, TABLES, read_csv_gz, snapshot
raw = list(read_csv_gz(DEFAULT_DATA / "labs.csv.gz", COMMON + TABLES["labs"]))
for row in raw:
    if row["record_id"] == "ALB-002":
        print("version", row["version"], "value", row["value_num"], row["unit"], "available", row["available_time"])
view = snapshot()
selected = next(row for row in view["labs"] if row["record_id"] == "ALB-002")
print("selected:", selected["version"], selected["value_num"], selected["unit"])
```

Output:

```text
version 1 value 4.9 g/dL available 2025-02-28T08:00:00Z
version 2 value 3.1 g/dL available 2025-02-28T16:00:00Z
selected: 2 3.1 g/dL
```

These are two versions of one record, not two specimens. Keeping both double-counts a result. Keeping the first ignores a correction already available at the index. Choosing the latest export version before applying a time cutoff would be another error: a later correction can overwrite what was knowable earlier.

The reader's order is deliberate: validate, filter patient and clocks, select latest available version, remove selected cancellations, then sort. Filtering cancelled rows before selecting a version could resurrect an older active record. Missing clocks fail explicitly instead of being guessed. These rules are implemented in [oncology.py](../../src/nanoastra/oncology.py), with behavioral tests in [test_oncology.py](../../tests/test_oncology.py).

## 2. Pending does not mean negative

A molecular panel has a pending baseline version and a final version available later:

```python
raw_biomarkers = list(read_csv_gz(DEFAULT_DATA / "biomarkers.csv.gz", COMMON + TABLES["biomarkers"]))
for row in raw_biomarkers:
    if row["record_id"] == "BIO-002":
        print("raw:", row["version"], row["available_time"], row["result_status"])
for row in view["biomarkers"]:
    print("visible:", row["record_id"], row["marker"], row["result_status"], repr(row["result"]))
```

Output:

```text
raw: 1 2025-02-20T12:00:00Z pending
raw: 2 2025-03-04T12:00:00Z final
visible: BIO-001 PD-L1 final '60%'
visible: BIO-002 tumor_NGS_panel pending ''
```

The panel's eventual result cannot be used for baseline prognosis. Nor should the empty pending result be normalized to a negative assay. The existing PD-L1 result is a separate observation and does not make the molecular panel complete.

This distinguishes three concepts: the assay was ordered, its result is available, and the result is positive or negative. Conflating them creates false evidence even without an explicit future date.

## 3. Normalize units before comparing values

```python
albumin = [row for row in view["labs"] if row["test_name"] == "albumin"]
def albumin_g_dl(row):
    value = float(row["value_num"])
    if row["unit"] == "g/L":
        return value / 10
    if row["unit"] == "g/dL":
        return value
    raise ValueError("Unsupported albumin unit")
for row in albumin:
    print(row["record_id"], row["value_num"], row["unit"], "->", albumin_g_dl(row), "g/dL")
print("change in g/dL:", round(albumin_g_dl(albumin[-1]) - albumin_g_dl(albumin[0]), 2))
```

Output:

```text
ALB-001 35 g/L -> 3.5 g/dL
ALB-002 3.1 g/dL -> 3.1 g/dL
change in g/dL: -0.4
```

The comparison is 3.5 to 3.1 g/dL, a change of -0.4, not 35 to 3.1 in the same unit. The explicit rejection branch matters: unknown units must not silently receive a guessed conversion. Preserve the original value, unit, source ID, and conversion alongside any normalized feature.

A numerical decline is an observed trend. It is not, by itself, an estimated independent mortality effect or permission to assign an invented risk weight.

## 4. A longitudinal feature needs dates and provenance

```python
from nanoastra.oncology import weight_change, timestamp
weights = [row for row in view["vitals"] if row["measurement"] == "weight"]
for row in weights:
    print(row["record_id"], row["event_time"][:10], row["value_num"], row["unit"])
change = weight_change(view["vitals"])
days = (timestamp(weights[-1]["event_time"]) - timestamp(weights[0]["event_time"])).total_seconds()/86400
print("weight loss percent:", change["percent_loss"])
print("elapsed days:", round(days, 3))
print("source records:", change["baseline_record"], change["current_record"])
```

Output:

```text
WT-001 2025-01-05 72 kg
WT-002 2025-02-12 68 kg
WT-003 2025-02-28 64 kg
weight loss percent: 11.111
elapsed days: 54.167
source records: WT-001 WT-003
```

The calculation is `(72-64)/72 * 100 = 11.111%`. The exact elapsed interval includes the recorded times, not only calendar dates. The same percentage over years would have a different interpretation. In actual clinical data, fluid balance and measurement conditions also matter. [NCI nutrition PDQ](https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss/nutrition-hp-pdq).

A feature without provenance is difficult to audit. If a model cites this decline, we should be able to recover both rows, their dates, and the formula. This is why later outputs will cite record IDs rather than only a polished summary.

## 5. Bring disease, function, and treatment context together

```python
for row in view["pathology"]:
    print("pathology:", row["record_id"], row["histology"])
for row in view["staging"]:
    print("stage:", row["stage_group"], "sites:", row["disease_sites"])
for row in view["performance_status"]:
    print("function:", row["record_id"], row["event_time"][:10], row["scale"], row["score"])
for row in view["medication_orders"]:
    print("order:", row["record_id"], row["order_status"])
print("administrations available:", len(view["medication_administrations"]))
for row in view["procedures"]:
    print("procedure:", row["procedure_name"], row["procedure_status"])
```

Output:

```text
pathology: PATH-001 adenocarcinoma
stage: IV sites: lung; liver; bone
function: PS-001 2025-02-12 ECOG 1
function: PS-002 2025-02-28 ECOG 2
order: ORD-001 draft_awaiting_molecular_results
order: ORD-002 active
administrations available: 0
procedure: CT chest abdomen pelvis completed
procedure: Brain MRI ordered_not_performed
```

The clinician recorded stage IV disease; we do not manufacture a precise TNM substage from this summary. ECOG was recorded as 1 and then 2. A draft systemic plan is not proof of administration. Ordered brain imaging is not a negative brain scan.

Stage, performance status, weight loss, and molecular/immunologic features matter to the clinical context of NSCLC prognosis and treatment selection. This source supports their relevance, not a quantitative forecast for our invented patient. [NCI NSCLC PDQ](https://www.cancer.gov/types/lung/hp/non-small-cell-lung-treatment-pdq).

## 6. Separate findings from claims

An instructor-written assessment can now say:

> Baseline records support metastatic lung adenocarcinoma with worsening recorded function and weight loss. The albumin and hemoglobin trajectories are observable findings. Molecular results, completion of brain imaging, and the final systemic treatment plan remain unresolved. These records do not support a calibrated individual 180-day mortality probability.

That paragraph is authored teaching material, not model inference. It cites categories here for readability; the later structured output should attach the exact record IDs. Do not insert the evaluator's future death into this assessment.

The as-of view is still a set of tables, not the model's entire context. A model with a finite query budget must choose which table to read and when to stop. That is the first genuinely agentic decision.

**Next:** [Lesson 3: turn evidence access into an environment](003-oncology-tools-and-environment.md).
