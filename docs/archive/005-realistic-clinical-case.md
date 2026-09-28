> Historical material. Follow the [current oncology curriculum](../lessons/README.md) instead.

# Lesson 5: Historical emergency case and staged diagnostic evidence

**Superseded application:** the active case is now [oncology prognosis from clinical CSVs](006-oncology-prognosis-from-csv.md). This English teaching record preserves the earlier case's rationale and reproducibility. The fixture and original experiment outputs retain their original language and are historical data, not current documentation.

All people, dates, values, and reports are invented. The case was checked for consistency against public guidelines but has not undergone independent clinical review or benchmark validation.

## 1. Initial handoff, D0 09:40

A 58-year-old woman presents with approximately one day of dyspnea and worsening right-sided chest pain since 06:30. She has had dry cough and fatigue for three days. Pain worsens with deep inspiration and coughing. A community clinic initially suspected lower respiratory infection, gave one dose of amoxicillin/clavulanate, and referred her for persistent breathlessness.

At 09:10: temperature 37.8 C, heart rate 112/min, respiratory rate 26/min, blood pressure 124/76 mmHg, and room-air oxygen saturation 91%. The clinical team started nasal oxygen at 2 L/min at 09:14; saturation reached 95% at 09:25. She remains alert and able to speak in full sentences, with rapid breathing.

Oxygen administration is an existing clinical event, not an action performed by the record-reading agent. The initial model view contains the handoff; detailed records must be queried.

## 2. Evidence available through records

| Record | Findings |
| --- | --- |
| Detailed history, 09:20 | No syncope, hemoptysis, orthopnea, or substantial purulent sputum. Left calf tightness for three days, initially attributed to postoperative recovery. |
| Risk history, 09:22 | Left knee arthroscopic meniscal surgery 18 days earlier; mostly bed/chair activity for five days, then reduced walking. BMI 31, hypertension; no known prior VTE, active malignancy, heart failure, chronic lung disease, or estrogen use. |
| Medication review | Amlodipine and occasional acetaminophen; no current anticoagulant. No known contrast allergy or recorded active bleeding. |
| Examination, 09:25 | Right basal crackles and tachycardia. Left calf 3 cm larger than right, mild pitting edema, and tenderness along the deep venous distribution. Dry surgical wound. |
| ECG, 09:17 | Sinus tachycardia at 111/min, no definite acute ST elevation, no prior comparison. |
| Labs returned 09:35 | WBC 12.4x10^9/L, neutrophils 78%, CRP 48 mg/L, Hb 12.1 g/dL, platelets 267x10^9/L. Creatinine 0.84 mg/dL, eGFR 82, lactate 1.4. hs-cTnI 5 ng/L (upper reference 18), NT-proBNP 112 pg/mL (report reference <125); no serial values. |
| Chest X-ray, 09:37 | Peripheral right basal opacity and small right pleural effusion; no pneumothorax or obvious pulmonary edema. Nonspecific appearance. |
| D-dimer, 09:38 | 2.8 mg/L FEU, routine threshold <0.50; ordered before detailed thrombotic risk history. |
| Missing/limited tests | Leg ultrasound not performed. No CTPA report at 09:40. Influenza and SARS-CoV-2 tests negative; no blood culture or suitable sputum sample. |

## 3. How to update the assessment

Infection is a plausible initial hypothesis given cough, low fever, inflammatory markers, crackles, and opacity. These findings are not uniquely diagnostic. PE can present with pleuritic pain, hypoxemia, and nonspecific radiographic changes. [ESC/ERS guideline](https://academic.oup.com/eurheartj/article/41/4/543/5556136).

The added surgery, reduced mobility, tachycardia, and unilateral tender swelling change the priority of thromboembolic evaluation. They do not themselves prove PE or an ultrasound-confirmed DVT. Persistence after one antibiotic dose is not evidence of treatment failure.

Under the NICE two-level PE Wells pathway, clinical DVT signs contribute 3, heart rate over 100 contributes 1.5, and recent surgery or immobilization contributes 1.5. The subtotal is 6 without using the subjective alternative-diagnosis item. Surgery and immobilization are one item, not two. A score above 4 enters the PE-likely pathway; it is not a diagnosis or a percent probability. [NICE NG158](https://www.nice.org.uk/guidance/ng158/chapter/recommendations).

```python
items = {"clinical_dvt_signs": 3.0, "hr_over_100": 1.5,
         "recent_surgery_or_immobility": 1.5}
print(sum(items.values()))  # 6.0
```

D-dimer elevation alone is not diagnostic, and postoperative patients frequently have positive results. [ASH diagnosis guideline](https://ashpublications.org/bloodadvances/article/2/22/3226/16134/American-Society-of-Hematology-2018-guidelines-for).

| Fact | Supported statement | Unsupported inference |
| --- | --- | --- |
| Elevated D-dimer | Compatible with concern; not definitive | PE already confirmed |
| No ultrasound | DVT not imaged | Positive or negative ultrasound |
| Selected viral tests negative | Those assays negative | All infection excluded |
| One normal-range troponin | That measurement not elevated | Every acute coronary condition excluded |
| Stable blood pressure, oxygen requirement | Current observations | Automatically safe outpatient disposition |

An appropriate 09:40 assessment prioritizes suspected PE, cites the acquired evidence, retains infection in the differential, and calls for timely clinical diagnostic/risk evaluation. It does not invent an imaging result or prescribe a simulated treatment.

## 4. Follow-up at D0 10:35

The clinical team has arranged CTPA outside the record-reader. It was performed at 10:18 and reported at 10:32: adequate study, bilateral lower-lobe segmental pulmonary arterial filling defects, right subpleural wedge-shaped opacity interpreted as infarct-related change, small right effusion, RV/LV about 0.85, and no obvious RV enlargement. Echocardiography remains unperformed.

Current observations are BP 122/78, HR 108, RR 24, saturation 95% on the same oxygen. The segmental CTPA findings provide diagnostic evidence for acute PE; stable blood pressure and lack of obvious RV enlargement do not alone complete risk assessment. [ESC/ERS guideline](https://academic.oup.com/eurheartj/article/41/4/543/5556136).

The key update is from suspicion to imaging-supported diagnosis. Leg DVT remains unconfirmed. Certainty about one conclusion does not automatically transfer to related conclusions.

## 5. Preserve the information boundary in code

The [fixture](../../cases/ed-0001.json) has shared records, stage-specific records, and instructor-only annotations. [case_replay.py](../../src/nanoastra/case_replay.py) projects only the selected stage into tool context. Never pass the complete fixture to the model and ask it to pretend it has not seen the answer.

With `PYTHONPATH=src python3`:

```python
from nanoastra.case_replay import load_case, project_case
from nanoastra.clinical import read_record
view = project_case(load_case(), "initial")
records = view["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
print(view["clock"])  # D0 09:40
print(read_record(records, "instructor_only", view["record_fields"])["status"])
# invalid_field
```

The CTPA field initially contains a pending-status message. A successful record lookup (`status=ok`) does not itself mean the clinical test was performed or positive. Reading the same field again does not advance the clock.

```python
later = project_case(load_case(), "followup")
later_records = later["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
assert records["ctpa_report"] != later_records["ctpa_report"]
assert "instructor_only" not in later_records
```

```sh
python3 scripts/demo_realistic_case.py --stage initial
python3 scripts/demo_realistic_case.py --stage followup --fields ctpa_report leg_ultrasound
```

In vp, add `--verl` to use the real registry/BaseTool. Queries are scripted; there is no LLM inference, test ordering, automatic clinical score, or new training.

## 6. What this taught us about reward

A reward for merely mentioning PE could treat unsupported guessing, fabricated ultrasound evidence, and properly updated reasoning as equivalent. Candidate evaluation dimensions are stage consistency, evidence fidelity, recognition of important risks, useful information acquisition, and remaining uncertainty. A particular arbitrary query order should not be required if another order acquires adequate evidence.

The fixture is two fixed record snapshots. It does not simulate waiting, costs, deterioration, treatment effects, or resource constraints. Multiple contrasting cases and independently reviewed evaluation criteria would be needed before serious assessment.

Exercise: compare initial and follow-up conclusions, listing new evidence, changed conclusions, and conclusions that remain unsupported. The objective is auditable evidence-to-claim reasoning rather than a long private reasoning transcript. Original verification is in [experiment 004](../../experiments/004-realistic-case.md).
