# Code-first teaching and infrastructure selection

Date: 2026-09-28. Translated user instruction: explain concepts with code and reuse public repositories/frameworks rather than rebuild infrastructure. The user also stated that a remote GPU was available.

## Historical actions and limits

Selected verl, pinned its source revision, and implemented artificial records, lookup, a data row, and terminal scoring. Added a real BaseTool subclass, YAML, and CPU probe, initially unrun. The standard-library demo and six tests passed. A local dependency installation request was rejected and was not retried.

Source inspection showed per-call create/execute/release in the selected ToolAgentLoop path, so tool instance state could not be assumed persistent across turns. At this milestone remote hardware details and access were not yet supplied; no remote connection or training had occurred.

See [lesson 3](../lessons/003-verl-code-walkthrough.md) and the [infrastructure decision](../design/002-infrastructure.md). Subsequent authorization and results are in the remote demonstration record.
