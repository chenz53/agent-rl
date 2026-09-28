# A more realistic clinical case

Date: 2026-09-28. Translated user feedback: the generated case was too simple and abstract; create one closer to the real world.

## Historical response summary

Added an entirely synthetic 58-year-old woman presenting with chest pain and dyspnea. Infection-like symptoms coexist with recent surgery, reduced mobility, and unilateral leg findings. The initial stage supports suspicion, while a later CTPA report changes diagnostic certainty; leg DVT remains unconfirmed.

The [historical lesson](../lessons/005-realistic-clinical-case.md) includes a timeline, measurements, evidence interpretation, executable projection, and reward-design pitfalls. Records reuse the BaseTool adapter; the caller explicitly selects the stage. Lookup does not order a test or advance time.

Eleven tests, three teaching snippets, the real remote tool, and the original GRPO probe passed. No new training or clinical capability assessment occurred. [Experiment 004](../../experiments/004-realistic-case.md) retains the original evidence. The later user instruction replaced this as the active case with oncology prognosis.
