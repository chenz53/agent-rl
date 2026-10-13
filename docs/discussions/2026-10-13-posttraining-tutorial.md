# Post-training tutorial expansion

Recorded 2026-10-13. English summaries of the learner's Chinese requests; these are not verbatim English quotations.

## Recent tutoring context

The learner asked to slow down and explain trajectories, generated reasoning, action masks, terminal rewards, losses, advantages, and repeated-rollout tensor shapes from first principles. They correctly distinguished inference context changes from parameter updates. The teaching discussion used B for tasks, G for repeated trajectories, T for positions, and V for vocabulary, with explicit flattening to B*G.

The next written edition corrects oversimplifications: sampled-token selection still differentiates all softmax logits; group means containing the sample are not action-independent baselines; value and gradient estimators require separate derivations; token reductions change weighting. No unrecorded learner answers are invented.

## New request

The learner requested an Agentic RL tutorial using D2L as a reference for explanatory style, PyTorch as the sole teaching tensor framework, code-backed theory and practice, deep discussion of design decisions and emerging alternatives, and on-policy distillation. They expanded the scope to broader large-model post-training and suggested Mintlify. Existing English-documentation and nanoAstra repository preferences remain applicable.

## Delivered first edition

Fourteen new chapters with fifteen executed Python blocks, a source ledger, editorial guide, and generated Mintlify edition. Existing oncology lessons remain the domain track. Current research was checked on 2026-10-13 and labeled as public proposals/results rather than presumed private vendor implementations. The small capstone performs actual on-policy categorical sampling and updates; it is not a complete clinical language-model trainer.

Verification and preview results are recorded in experiment 008. The learner did not request a published GitHub repository or hosted deployment, so the output is a local reading site and repository content.
