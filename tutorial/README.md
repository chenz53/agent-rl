# Agentic RL: From Policy Gradients to Coding Agents

*[中文版](zh/README.md) is available: the prose is translated; code, outputs, and figures are identical.*

This book teaches reinforcement learning for language-model agents. That means models that call tools, read observations, act over many turns, and are trained against verifiers. It starts from the definition of a policy gradient and ends with a complete, tested training stack for a small coding agent. Along the way it covers the design decisions that post-training teams make every day: how to aggregate a loss over variable-length tool trajectories, when a critic is worth its cost, how to keep the inference engine and the trainer consistent, how to detect a reward hack before it reaches an evaluation, and how to read a training dashboard that is going wrong.

## How the book works

The book builds **one system**, one chapter at a time:

- **The BugFix gym** (Chapter 4) is a family of small Python repair tasks. The agent fixes them with the tools `read`, `test`, `edit`, and `submit`. Tests execute real code, a hidden test set guards against overfitting, and an exploit operation lets reward hacking happen for real.
- **TinyAgent** (Chapter 3) is a small transformer policy that is trained with RL on a CPU in about a minute. It shows the phenomena that large runs show: format learning, entropy collapse, length effects, credit assignment across turns, and reward hacking.
- **The real-model thread** uses the actual Qwen3-0.6B tokenizer and model wherever model-specific behavior matters, such as chat templates, tool-call tokens, retokenization drift, and the log-probability mismatch between the inference engine and the trainer.

Each mechanism is first derived, then implemented from scratch in a short code block, then run. Its printed output appears directly below the code. In the style of [D2L](https://github.com/d2l-ai/d2l-zh), a function built in one chapter is marked `#@save` and collected into the `agentrl` package, so later chapters call `arl.ppo_policy_loss(...)` instead of repeating it. By the capstone, the reader has written every piece of the stack. The generated package groups implementations into topic modules such as `policy.py`, `environments.py`, `trajectories.py`, and `grpo.py`; `__init__.py` only re-exports the API. See the [module index](../agentrl/README.md) for the chapter-to-module map.

Every chapter ends with **Design problems in practice**. This section takes problems engineers actually hit, and for each one it gives the symptom, the cause, the options, and a recommended default. Every chapter then gives a summary and exercises. [Appendix B](chapters/B-design-review.md) collects open-ended design-review problems, with model answers.

## Contents

| Part | Chapter | The question it answers |
| --- | --- | --- |
| | [0. Notation and how to use this book](chapters/00-notation.md) | How do I read and run this book? |
| **I. Foundations** | [1. What is agentic RL?](chapters/01-introduction.md) | What changes when a model acts in an environment? |
| | [2. Reinforcement learning fundamentals](chapters/02-rl-fundamentals.md) | What are policies, returns, and values, and how is an LLM agent an MDP? |
| | [3. Language models as policies](chapters/03-language-models-as-policies.md) | What exactly is the probability of an action? |
| | [4. Environments and tools: the BugFix gym](chapters/04-environments-and-tools.md) | What does the agent act on, and what counts as success? |
| | [5. Trajectories, templates, and masks](chapters/05-trajectories-and-templates.md) | How does a multi-turn episode become training tensors? |
| **II. Policy optimization** | [6. Policy gradients](chapters/06-policy-gradient.md) | How does a reward become a gradient? |
| | [7. Advantages and credit assignment](chapters/07-advantages-and-credit.md) | Which actions deserve the credit? |
| | [8. PPO: reusing samples safely](chapters/08-ppo.md) | How can expensive rollouts be reused for several updates? |
| | [9. GRPO and the critic-free family](chapters/09-grpo-and-variants.md) | What do GRPO, Dr. GRPO, DAPO, RLOO, and GSPO each change? |
| | [10. KL, entropy, and regularization](chapters/10-kl-entropy-regularization.md) | How do we keep the policy from drifting or collapsing? |
| **III. Rewards and agents** | [11. Rewards, verifiers, and reward hacking](chapters/11-rewards-and-verifiers.md) | What should be rewarded, and how will the agent cheat? |
| | [12. Multi-turn agents and long horizons](chapters/12-multi-turn-and-long-horizons.md) | What breaks when episodes get long? |
| **IV. Systems** | [13. Rollout systems and off-policy correction](chapters/13-rollout-systems.md) | How do inference engines, trainers, and environments fit together? |
| | [14. Scaling and efficiency](chapters/14-scaling-and-efficiency.md) | Where do the tokens, memory, and GPU-hours go? |
| **V. Beyond pure RL** | [15. SFT, rejection sampling, and preferences](chapters/15-sft-and-rejection-sampling.md) | When are demonstrations or comparisons the better signal? |
| | [16. Distillation for agents](chapters/16-distillation.md) | How can a teacher score the student's own trajectories? |
| **VI. Practice** | [17. Evaluation](chapters/17-evaluation.md) | Did the agent actually improve? |
| | [18. Debugging RL runs](chapters/18-debugging-rl-runs.md) | What do common failures look like on a dashboard? |
| | [19. Capstone: train a coding agent end to end](chapters/19-capstone.md) | Can we assemble and evaluate the whole stack? |
| | [20. Research frontier and open problems](chapters/20-frontier.md) | What is unsettled, and how would we test it? |
| **Appendices** | [A. Mathematical tools](chapters/A-math.md) | Which identities and estimators do the chapters rely on? |
| | [B. Design review problems](chapters/B-design-review.md) | Can I reason through a real post-training design question? |

## Running the code

Every Python block runs. The checker executes the chapters in order, rebuilds `agentrl` from the `#@save` blocks of earlier chapters, and compares each block's stdout with the recorded output:

```sh
scripts/remote_check.sh <workdir> [chapter-prefix ...]   # runs on the GPU host; CPU is enough
python3 scripts/build_library.py                         # regenerate agentrl/ from the chapters
scripts/remote_figures.sh <workdir> [chNN ...]           # regenerate data figures
python3 scripts/build_tutorial_site.py                   # regenerate the Mintlify edition in website/
```

[The real-model lab](labs/README.md) trains Qwen3-0.6B on the BugFix gym on one H200, and Chapter 19 reports its measured results.

[The editorial guide](editorial-guide.md) defines the book's conventions, and [the source ledger](sources.md) records every external source with the date it was inspected and what was checked.

## What the experiments do and do not show

TinyAgent experiments are real and run on a CPU, but they are small. They demonstrate mechanisms and make failure modes reproducible. They are not benchmark results. Results reported by papers are labeled as reported, with dates. Recommendations are labeled as the book's recommendations. The environment executes only trusted code written for the book; Chapter 4 explains the isolation that real model-generated code requires.
