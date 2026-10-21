# Generated agentrl package

Implementations are grouped by topic; `__init__.py` only re-exports the public API. For example, `agentrl.TinyAgent` and `agentrl.policy.TinyAgent` are the same class.

Edit `#@save` blocks in the English chapters, then run `python3 scripts/build_library.py`. Do not edit generated files. The generator resolves `arl.name` references into explicit imports from the defining topic module; it preserves the implementation and comments. Each saved block has a source pointer.

| Module | Source chapter |
| --- | --- |
| [examples.py](examples.py) | [01-introduction.md](../tutorial/chapters/01-introduction.md) |
| [mdp.py](mdp.py) | [02-rl-fundamentals.md](../tutorial/chapters/02-rl-fundamentals.md) |
| [policy.py](policy.py) | [03-language-models-as-policies.md](../tutorial/chapters/03-language-models-as-policies.md) |
| [environments.py](environments.py) | [04-environments-and-tools.md](../tutorial/chapters/04-environments-and-tools.md) |
| [trajectories.py](trajectories.py) | [05-trajectories-and-templates.md](../tutorial/chapters/05-trajectories-and-templates.md) |
| [policy_gradient.py](policy_gradient.py) | [06-policy-gradient.md](../tutorial/chapters/06-policy-gradient.md) |
| [advantages.py](advantages.py) | [07-advantages-and-credit.md](../tutorial/chapters/07-advantages-and-credit.md) |
| [ppo.py](ppo.py) | [08-ppo.md](../tutorial/chapters/08-ppo.md) |
| [grpo.py](grpo.py) | [09-grpo-and-variants.md](../tutorial/chapters/09-grpo-and-variants.md) |
| [regularization.py](regularization.py) | [10-kl-entropy-regularization.md](../tutorial/chapters/10-kl-entropy-regularization.md) |
| [rewards.py](rewards.py) | [11-rewards-and-verifiers.md](../tutorial/chapters/11-rewards-and-verifiers.md) |
| [multiturn.py](multiturn.py) | [12-multi-turn-and-long-horizons.md](../tutorial/chapters/12-multi-turn-and-long-horizons.md) |
| [rollout_systems.py](rollout_systems.py) | [13-rollout-systems.md](../tutorial/chapters/13-rollout-systems.md) |
| [efficiency.py](efficiency.py) | [14-scaling-and-efficiency.md](../tutorial/chapters/14-scaling-and-efficiency.md) |
| [sft.py](sft.py) | [15-sft-and-rejection-sampling.md](../tutorial/chapters/15-sft-and-rejection-sampling.md) |
| [distillation.py](distillation.py) | [16-distillation.md](../tutorial/chapters/16-distillation.md) |
| [evaluation.py](evaluation.py) | [17-evaluation.md](../tutorial/chapters/17-evaluation.md) |
| [debugging.py](debugging.py) | [18-debugging-rl-runs.md](../tutorial/chapters/18-debugging-rl-runs.md) |
| [math_utils.py](math_utils.py) | [A-math.md](../tutorial/chapters/A-math.md) |

Dependencies point to earlier chapters. The checker builds the same module graph in memory, with only chapters preceding the one being checked.

Mutable module settings belong to their defining module. For example, change `agentrl.grpo.FN_NAMES` when temporarily restricting the renaming pool; rebinding the re-export `agentrl.FN_NAMES` does not change that module's setting.
