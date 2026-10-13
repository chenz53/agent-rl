# nanoAstra: Hands-on Post-training

An English, PyTorch-only tutorial for practitioners who know foundation-model training and want to understand modern post-training through agent interaction. D2L is a pedagogical reference; the prose and laboratories are newly authored. The existing synthetic oncology case remains the continuous domain example.

## Read and run

Canonical chapters are in `tutorial/chapters/`; `website/` is a generated Mintlify reading edition. Read the numbered chapters in order, including 5A after policy gradients and 8A after distillation. Each Python block has captured output and runs in sequence within its chapter. The compact laboratories use PyTorch CPU tensors and no model downloads. The existing remote environment is the tested execution environment.

```sh
# From /home/user/nanoAstra on vp:
.venv/bin/python scripts/check_tutorial_snippets.py
```

From a checkout with the Mintlify CLI available:

```sh
python3 scripts/build_tutorial_site.py
cd website
mint dev
```

The current official CLI package is `mint`; a local installation already exists, so this change installs no local Python packages. Hosting is a separate step and has not been published.

## Learning route

| Stage | Chapters | Observable learning outcome |
| --- | --- | --- |
| Define the training problem | 1 | Identify prefix generator, target signal, and evaluation boundary |
| Understand tokens and trajectories | 2-3 | Derive selected log-probabilities, their gradients, and action ownership |
| Turn feedback into updates | 4-6 and 5A | Inspect grouping, baseline bias, returns, GAE, loss reductions, and clipping |
| Compare post-training signals | 7-8 and 8A | Contrast SFT, DPO, exact/sampled OPD, reference KL, and entropy |
| Build reliable experiments | 9-10 | Audit serving/training mismatch, policy lag, resource budgets, and independent evaluation |
| Test research alternatives | 11 | Separate vocabulary sparsity from timestep selection and design matched-cost ablations |
| Complete a bounded training loop | 12 | Run an actual sampled categorical-policy update before agent integration |
| Apply to domain data | Existing oncology lessons 1-6 plus 3A | Read gzip tables, reconstruct evidence, define tools, and inspect pinned verl |

## Teaching contract

Every chapter follows a concrete question, assumptions, derivation, small implementation, captured output, interpretation, failure modes, design trade-offs, and exercises. “From scratch” means implementing the educational mechanism; production scheduling and training remain delegated to existing frameworks.

Use three evidence labels consistently: mathematical identity under stated assumptions, observed repository execution, and external author-reported result. Frontier entries include a review date and a falsifiable experiment. Private vendor implementation details remain unknown unless public evidence is inspected.

This first edition contains fourteen new chapters with fifteen executable blocks and the existing case-study track. It does not claim exhaustive coverage of every post-training method or completed autonomous clinical training. The capstone updates a categorical policy; the next scale-up milestone is a bounded tokenizer-backed model/tool rollout followed by independently evaluated training. Future expansion should deepen model-backed experiments, preference-data construction, teacher serving, distributed training, and domain validation.
