# English documentation and cancer prognosis from clinical exports

Date: 2026-09-28.

## User instruction, translated from Chinese

All documentation should be in English; conversation may remain Chinese. Change the case to cancer prognosis reasoning, starting from files ending in `csv.gz`, such as diagnosis, labs, vitals, notes, and other realistically named clinical-system records.

## Implementation decisions and assumptions

The initial teaching setup is pretreatment metastatic NSCLC with a fixed index and a 180-day all-cause mortality horizon. This was presented as a provisional choice; no real data were supplied. All fixture contents and outcomes are invented. Thirteen clinical tables and an evaluator-only mortality table provide the starting point.

The reader validates schemas and both event/availability clocks, selects record versions, handles cancellations, and isolates patients and outcomes. The data deliberately include a corrected lab, mixed albumin units, a late-signed note, pending molecular results, a draft treatment order, and post-index administration. Separate label code demonstrates observed events, known non-events, and early censoring.

All Markdown documentation was rewritten in English, including working instructions and historical discussion material. Historical Chinese statements are labeled as translations/summaries; earlier originals remain in Git history. Raw historical experiment outputs are unchanged. The old emergency case remains reproducible but is no longer the active application.

The current demo is scripted evidence retrieval, not an autonomous model, calibrated survival predictor, or clinical reward. Code and verification details are in [lesson 6](../archive/006-oncology-prognosis-from-csv.md) and [experiment 005](../../experiments/005-oncology-csv.md).
