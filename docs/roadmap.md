# Current learning roadmap

The active route is the [coding-first PyTorch post-training tutorial](../tutorial/README.md). Begin with a concrete bug and executable checks, construct read/edit/test/submit tools, identify generated trajectory segments, and only then introduce token probabilities and learning weights.

The sequence is task → environment → trajectory → probabilities → reward/advantage → policy update → values/returns → PPO/GRPO → SFT/DPO → on-policy distillation → KL/entropy → systems → evaluation → research alternatives → executable patch-selection capstone.

DeepSWE is the public system reference. Every teaching function is defined in its chapter, and every executable Python block has captured output. The next scale-up milestone is a tokenizer-backed coding-agent rollout in an isolated repository environment, followed by independently evaluated training. Earlier lessons remain project history and are outside the active site.
