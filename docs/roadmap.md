# Learning roadmap

Each unit follows task definition, mechanism, implementation or worked trajectory, failure analysis, and an understanding check. Documentation is English; discussion may be Chinese.

| Unit | Core question | Practical milestone |
| --- | --- | --- |
| 1. Oncology data and environment | What is knowable at the index? What can the agent query? | CSV contract, as-of view, outcome isolation |
| 2. Baselines and evaluation | Is the limitation knowledge, information, or strategy? | Non-RL baselines and independent evaluation |
| 3. Trajectories and gradients | How do tool interactions affect policy gradients? | Original tokens, likelihoods, masks, returns |
| 4. PPO/GRPO and credit | What do relative advantage, ratio, clipping, and KL do? | Auditable updates and diagnostics |
| 5. Data and exploration | How do we collect informative successes and failures? | Sampling and task-difficulty ablations |
| 6. Training systems | How do latency, retries, and policy lag change learning? | Reproducible small training loop |
| 7. Scientific validity | Does reward improvement represent better prognosis? | Censoring-aware evaluation, calibration, leakage audit |
| 8. Research reproduction | Which mechanism actually explains a gain? | Controlled replication and failure report |

Prioritize the objective and data flow over API familiarity. Progress follows understanding and verified experiments, not a fixed promise of expertise.
