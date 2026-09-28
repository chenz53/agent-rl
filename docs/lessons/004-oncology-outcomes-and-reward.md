# Lesson 4: Define outcomes and feedback before training the query policy

[Lesson 3](003-oncology-tools-and-environment.md) gave the policy a way to obtain evidence. We now need to decide what counts as a good final output. A clinical endpoint, a statistical score, and evidence quality are separate concepts.

This lesson includes evaluator-only code. Its mortality outputs must not be inserted into the agent's baseline prompt. Run snippets sequentially in a fresh `PYTHONPATH=src python3` session.

## 1. A missing death record is not proof of survival

```python
from nanoastra.oncology import ROOT, INDEX_TIME
from nanoastra.oncology_outcomes import load_labels
labels = load_labels(ROOT / "fixtures/oncology/evaluator_only/mortality.csv.gz", INDEX_TIME)
for row in labels:
    print(row["patient_id"], "follow-up days:", row["followup_days"],
          "event observed:", row["event_observed"], "180-day label:", row["death_by_horizon"])
```

Output:

```text
SYN-ONC-001 follow-up days: 100.0 event observed: 1 180-day label: 1
SYN-ONC-002 follow-up days: 214.0 event observed: 0 180-day label: 0
SYN-ONC-003 follow-up days: 60.0 event observed: 0 180-day label: None
```

The first fictional patient has an observed death at day 100, so the 180-day label is 1. The second is known alive through day 214, so the label is 0. The third was last known alive at day 60 with no observed death; its 180-day outcome is unknown, represented by `None`.

This is **right censoring**: observation ends before the event time is established. The export's ascertainment window may extend much later, but that does not prove each individual remained alive until the window ended. Our label builder uses affirmative follow-up, not the file's extraction date, to establish survival.

These outcomes were invented to teach three label states. The main patient's future death was not derived from her baseline features.

## 2. A horizon is a precise boundary

```python
from nanoastra.oncology_outcomes import horizon_label
row = {"patient_id": "boundary-example", "last_known_alive_time": INDEX_TIME,
       "ascertainment_end_time": "2025-12-31T09:00:00Z",
       "death_time": "2025-08-28T09:00:00Z"}
print("death on day 180:", horizon_label(row, INDEX_TIME)["death_by_horizon"])
row["death_time"] = "2025-08-29T09:00:00Z"
print("death on day 181:", horizon_label(row, INDEX_TIME)["death_by_horizon"])
```

Output:

```text
death on day 180: 1
death on day 181: 0
```

Our contract includes death exactly at day 180. A later observed death establishes survival through the horizon and therefore a horizon label of 0, even though an event is eventually observed. Event-observation status and death-by-horizon are not interchangeable fields.

A real study must validate death ascertainment, cohort eligibility, prior-treatment coverage, follow-up, and index construction. Selecting only patients who eventually receive treatment conditions inclusion on future events. Excluding all early-censored patients can also bias evaluation. No censoring-aware survival estimator is implemented here.

## 3. A proper probability score differs from classification accuracy

For a known binary outcome y, the Brier loss is `(p-y)^2`. It scores the issued probability rather than only the side of a classification threshold. The following forecasts are **invented arithmetic examples**, not model predictions for this patient:

```python
y = labels[0]["death_by_horizon"]
for p in [0.2, 0.7]:
    loss = (p-y)**2
    print(f"p={p:.1f} y={y} Brier={loss:.2f} candidate_reward={-loss:.2f}")
print("early-censored label:", labels[2]["death_by_horizon"])
```

Output:

```text
p=0.2 y=1 Brier=0.64 candidate_reward=-0.64
p=0.7 y=1 Brier=0.09 candidate_reward=-0.09
early-censored label: None
```

Against this one event, 0.7 incurs less error than 0.2. That does not show calibration: calibration is a population-level relationship between forecasts and event frequencies. Nor does it justify always predicting one. Across repeated events, expected squared error is minimized at the true event probability under the relevant distribution.

Never replace the censored `None` with zero to make this formula run. Survival modeling and censoring-adjusted metrics need explicit methods and assumptions. A naïve mean over only observed labels need not estimate the desired population performance.

## 4. Reward is a training proxy, not the full scientific objective

Suppose two outputs issue the same probability, but one cites available records and the other fabricates a completed brain MRI. Brier loss alone cannot distinguish them. We therefore need separate checks for evidence fidelity, temporal compliance, uncertainty, and task validity.

A small provenance check can detect a nonexistent citation:

```python
from nanoastra.oncology import snapshot
view = snapshot()
available = {row["record_id"] for rows in view.values() for row in rows}
for citations in [["PATH-001", "PS-002", "WT-003"], ["PATH-001", "MRI-RESULT-999"]]:
    missing = sorted(set(citations) - available)
    print("citations:", citations, "unavailable:", missing)
```

Output:

```text
citations: ['PATH-001', 'PS-002', 'WT-003'] unavailable: []
citations: ['PATH-001', 'MRI-RESULT-999'] unavailable: ['MRI-RESULT-999']
```

Existence is only the first check. A cited record can exist while failing to support the statement. `PROC-002`, for example, is an ordered, unperformed MRI; citing it as proof of a negative scan would pass an ID-existence check and still be wrong. We have not implemented a validated semantic clinical judge.

Report candidate components separately before combining them. Arbitrary weights can change what the policy optimizes, and large citation-count bonuses can reward irrelevant reading. The old artificial A/B scorer is not an oncology reward function.

## 5. Query cost makes the stopping objective explicit

The next calculation uses hypothetical **score improvements**, not clinical probabilities or measured utility. Suppose stopping has expected score 0.60 and acquiring a useful table raises it to 0.75:

```python
stop_score, informed_score = 0.60, 0.75
for cost in [0.05, 0.20]:
    query_score = informed_score - cost
    print(f"cost={cost:.2f} stop={stop_score:.2f} query={query_score:.2f} prefer_query={query_score > stop_score}")
```

Output:

```text
cost=0.05 stop=0.60 query=0.70 prefer_query=True
cost=0.20 stop=0.60 query=0.55 prefer_query=False
```

The same information gain is worth acquiring under the smaller cost and not under the larger one. This is a decision-theoretic consequence of the objective. If an agent stops too early, inspect the objective and costs before concluding that it cannot reason.

Tool returns currently have zero reward; these cost examples are not connected to a trainer. Later integration must verify that intended costs reach the actual objective rather than remain only diagnostic metadata.

## 6. Establish baselines before RL

Use a fixed initial-information baseline, a full-allowed-information baseline, and an untrained querying agent under matched budgets. They distinguish limited evidence, limited underlying predictive competence, and a poor acquisition policy. A trained agent is then compared under the same task definition and independent splits.

Split by patient, not note row. Multiple index times from one patient must not leak across training and test sets. Temporal and site validation matter because care patterns, documentation, and outcome ascertainment can shift. The tiny synthetic fixture is for checking these mechanics, not estimating performance.

We now have a conceptual terminal score and query actions. The next question is how a score at the end can affect a query generated earlier.

**Next:** [Lesson 5: from the oncology trajectory to policy gradient](005-oncology-trajectories-and-gradients.md).
