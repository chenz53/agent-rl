# Domain implementation

- `oncology.py`: gzip CSV contract and patient/index-specific clinical views.
- `oncology_outcomes.py`: evaluator-only follow-up and horizon labels.
- `clinical.py`: original artificial case, generic allowed-field lookup, and toy verl row.
- `verl_tool.py`: genuine BaseTool adapter with a configurable field allowlist.
- `case_replay.py`: historical emergency-case stages.
- `reward.py`: artificial A/B scorer; not an oncology scoring function.
- `upstream.py`: validate and activate the pinned upstream checkout.

The implementation provides data boundaries and probes. It does not yet implement an autonomous clinical agent or complete training loop.

`posttraining.py` provides compact PyTorch teaching primitives for selected log-probabilities, explicit group estimators and reductions, PPO surrogate, fixed-prefix reverse KL, and DPO. It is not a production trainer; use existing framework infrastructure for model-backed rollouts.
