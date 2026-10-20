# Real-model labs: GRPO on BugFix tasks with Qwen3-0.6B

Two optional labs run the recipe of [Chapter 19](../chapters/19-capstone.md) on a real pretrained model. They share the tasks, the hidden-test reward, the group advantages, the clipped loss, and the token-in/token-out rollout, and they differ in how the model acts:

- `qwen_grpo_bugfix.py` keeps the gym's action menu: the model calls `edit` with one of the authored edit ops, and the gym writes the change.
- `qwen_grpo_tools.py` gives the model generic coding tools named after Claude Code's (Read, Edit, Write, Bash), and the model writes the change itself.

We kept both scripts rather than rewriting the first. Chapter 19 compares their results, and each result file must stay reproducible by the script that produced it.

## The edit-menu lab

`qwen_grpo_bugfix.py` takes the gym, tasks, hidden-test reward, group advantages, and clipped loss from the book's generated `agentrl` library, and adds only what a real model needs:

- **Qwen3's own chat template with tool schemas**, in non-thinking mode. The model writes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`; a parser maps the JSON to the gym's `<call> tool [op] </call>`, and anything else counts as a malformed call that uses up a turn.
- **Token-in/token-out rollouts** (Chapter 5). Sampled ids are kept as sampled. Each tool result is appended as the token ids of the template's tool-response block, and at startup the script asserts that those ids equal the template's own rendering of the same conversation.
- **Batched Hugging Face `generate`** at temperature 1 with top-k and top-p disabled, so that the sampler's distribution is the one the trainer differentiates. vLLM is not used.
- **A memory-aware update.** The LM head runs only on policy positions, and gradients accumulate over microbatches with a single token-mean normalizer for each minibatch (Chapter 14). Zero-advantage rows are skipped in the backward pass, but their tokens still count in the normalizer, which is exact.

Tool calls execute only the gym's authored edit operations inside the process. The model never writes code that is executed. A system that runs model-written code needs the isolation described in Chapter 4; the tools lab below takes the first step.

### How to run

You need a CUDA GPU (the measured run peaked at 32.8 GB), a local Hugging Face cache that already contains `Qwen/Qwen3-0.6B` (nothing is downloaded), `torch`, and `transformers`. From the repository root:

```bash
python3 scripts/build_library.py                      # generates agentrl/ from the chapters
HF_HUB_OFFLINE=1 python tutorial/labs/qwen_grpo_bugfix.py --steps 30 --lr 1e-6 --eval-n 8 \
    --out tutorial/labs/results/qwen_grpo_bugfix.json
```

The options are `--steps`, `--group-size` (G, default 8), `--prompts` (tasks per step, default 8), `--lr`, `--max-new-tokens` per turn (default 64), `--micro` (microbatch rows, default 8), `--eval-n` (held-out episodes per task before and after training), and `--seed`. Everything else is `agentrl.RLConfig()`: renamed functions, one edit per episode, at most 6 turns, the hidden-test reward, `adv_mode='grpo'`, a symmetric clip of 0.2, 1 epoch, 2 minibatches, no KL term, and a gradient-norm clip of 1.0. The optimizer is AdamW without weight decay, and the model runs in fp32.

The book's checker does not execute this script. Chapter 19 reads its results file.

### What was measured

There is one run, on 2026-10-14, on the project's shared H200 host (torch 2.9.1, transformers 5.5.3), with the command above. The results are in `results/qwen_grpo_bugfix.json` (per-step metrics, before and after evaluations with per-task counts, decoded example turns, and the configuration), and the console log is in `results/qwen_grpo_bugfix.log`.

| | Held-out pass@1 (10 tasks × 8) | Malformed calls / episode | Submitted |
| --- | --- | --- | --- |
| Before | 0.200 [0.062, 0.388] | 1.69 | 0.61 |
| After 30 steps | 0.800 [0.500, 1.000] | 0.00 | 1.00 |

The intervals are task-level bootstrap intervals. The paired difference is +0.600 [+0.325, +0.875]. Wall clock was 9.5 minutes in total.

How to read it: the gain is mostly the model learning the gym's tool-call format and the read, edit, test, submit procedure. The policy's entropy reached 0.000 by step 15, after which most groups had zero advantage. The final policy is deterministic: it solves 8 of the 10 held-out tasks every time and fails both `range` tasks every time by applying `fix_init` to them. This is one seed with no tuning, so it demonstrates the mechanism and is not a benchmark result. A rerun on the same host reproduced the first steps' sampled rewards exactly, while the sampler–trainer log-prob gaps differed in the second significant digit, which shows nondeterminism in GPU kernels.

## The tools lab

The edit menu is not how coding agents act. `qwen_grpo_tools.py` replaces it with the tool interface of Claude Code. The tool names and parameters follow Claude Code's [tools reference](https://code.claude.com/docs/en/tools-reference) (checked 2026-10-14). The descriptions are our own short versions, and the script prints nothing about Claude Code to the model.

### The interface

Each episode gets an in-memory repository built from one `arl.make_tasks` task. The task is renamed with `arl.randomize_names` to an unrelated gym function name, because that is the book's default RL setting (Section 9.4): the name carries no hint about the bug. The repository holds two files: the buggy source `<fn>.py`, and `test_<fn>.py`, which holds one pytest function per public test. The hidden tests never enter the repository. The system prompt says the repository is at `/repo` and that a reply without a tool call ends the session. The user message names the function, its file, and the test file.

| Tool | Parameters | Behavior |
| --- | --- | --- |
| `Read` | `file_path`, `offset`, `limit` | The file in `cat -n` format: the line number, a tab, then the line. Marks the file as read. |
| `Edit` | `file_path`, `old_string`, `new_string`, `replace_all` | Exact string replacement. Fails if the file was not read in this episode, if `old_string` is absent, if it appears more than once and `replace_all` is false, or if the two strings are equal. On success it returns the edited file with line numbers. |
| `Write` | `file_path`, `content` | Creates or overwrites a file in the repository root. Overwriting an existing file requires a Read first. |
| `Bash` | `command`, `timeout` | An allowlisted emulation, not a shell: `pytest [file]` and `python -m pytest [file]` run the public tests, `python test_<fn>.py` runs that test file's tests, `python <file>.py` runs a script, and `ls`, `cat <file>`, and `grep [-n] [-i] <pattern> [file]` read the in-memory repository. `cd /repo` and `&&` chains are accepted. Anything else returns `command not available in this sandbox`. |

Requiring a Read before an Edit or an overwriting Write is our choice. Claude Code's reference applies it to some models only. Grep and Glob are left out because the reference says they are absent by default on Linux, where Claude Code searches through Bash.

The model may make up to four tool calls in one turn. They run in order, and their results come back as consecutive `<tool_response>` blocks, which is how Qwen3's template renders several tool messages. A call whose JSON does not parse or does not match a tool schema is **malformed**. A well-formed call that the tool rejects, such as an Edit whose `old_string` is absent, is **failed**. Both return an error observation and use up the turn. An assistant turn without `<tool_call>` is the final answer and ends the episode, as in real agents. Otherwise the episode ends after 6 turns. In both cases the reward is 1 if the final `<fn>.py` passes every hidden test. At startup the script asserts that its token path matches Qwen3's template for a prompt with the four tool schemas and for observations with one and with two tool results.

### Execution and its limits

In this lab the model writes code that runs, which is the case Chapter 4 says needs isolation. The script never runs model output with `exec` in the trainer process. Every test run, every `python <file>.py`, and the hidden-test grading go through `sandbox_run`:

- a fresh temporary directory that holds a copy of the episode's files and is deleted afterward (the grader copies only `<fn>.py`);
- `python -I -S -B` with an environment of only `PATH`, `HOME` (the temporary directory), and `LANG`, so no variables or secrets are inherited;
- a 10-second wall-clock timeout (Bash's `timeout` can lower it);
- `resource` limits that the child sets before it imports any repository code: 5 CPU-seconds, 1 GiB of address space, 1 MiB per written file, and `RLIMIT_NPROC=0`, so it cannot start processes.

The tool layer was checked on the run host before training: a fork attempt raised `BlockingIOError`, a 10^10-element list raised `MemoryError`, an infinite loop was killed by the CPU limit, and the child's environment held only the three variables above.

**This is process isolation with resource limits on a controlled host, not a security sandbox.** The child runs as the same user, so it can read any file that user can read, and it can open network connections. The grader trusts the output of a process that runs model-written code, so a malicious module could print a forged result line. It would have to guess a per-call random marker, which raises the bar but does not remove the risk. Production systems run model-written code in containers or VMs with no network access and grade in a fresh container (Chapter 4). The same `RLIMIT_NPROC` limit does not apply to root, so do not run the script as root.

### How to run

You need Linux (for `resource` limits), a CUDA GPU, a local Hugging Face cache that already contains `Qwen/Qwen3-0.6B`, `torch`, and `transformers`. From the repository root:

```bash
python3 scripts/build_library.py
HF_HUB_OFFLINE=1 python tutorial/labs/qwen_grpo_tools.py --steps 18 --lr 1e-6 --eval-n 8 \
    --out tutorial/labs/results/qwen_grpo_tools.json
```

The options are those of the edit-menu lab, plus `--max-turns` (default 6), `--max-new-tokens` per turn (default 256, because an Edit must repeat a line of code), `--gen-batch` (episodes per `generate` call, default 16, which bounds KV-cache memory), and `--smoke` (check the template and the tools, run 40 held-out episodes, print some transcripts, and exit). The defaults of `--micro` (4) and the use of gradient checkpointing differ from the edit-menu lab because episodes are longer. The script records sampler log-probabilities with a logits processor that keeps one step of log-softmax instead of storing every step's logits. The `max_edits` field of `RLConfig` appears in the printed configuration but does not apply here: the model may edit as often as its turns allow.

### What was measured

There is one run, on 2026-10-14, on the project's shared H200 host (torch 2.9.1, transformers 5.5.3), with the command above. Another job held about 115 GB of the GPU's memory during the run. The results are in `results/qwen_grpo_tools.json` (per-step metrics, before and after evaluations with per-task counts and tool statistics, one full transcript per held-out task before and after training and per training task at step 0, and the configuration, tool schemas, and limits), and the console log is in `results/qwen_grpo_tools.log`.

| | Held-out pass@1 (10 tasks × 8) | Malformed / episode | Failed / episode | Bash calls / episode | Ended with an answer |
| --- | --- | --- | --- | --- | --- |
| Before | 0.100 [0.025, 0.212] | 0.09 | 0.85 | 0.07 | 0.99 |
| After 18 steps | 0.488 [0.237, 0.750] | 0.00 | 0.21 | 0.00 | 1.00 |

The intervals are task-level bootstrap intervals. The paired difference is +0.388 [+0.150, +0.650]. Per held-out task, the counts went from [1, 0, 0, 0, 2, 1, 0, 0, 4, 0] to [8, 8, 1, 2, 8, 8, 0, 0, 4, 0] (two tasks each of minmax, range, compare, zero, and init, in that order). Wall clock was 20.0 minutes in total, and peak GPU memory was 14.2 GB.

How to read it:

- **Before training**, the failures are mostly tool misuse: Edits whose `old_string` is not in the file (0.30 per episode), Edits before a Read (0.20), Edits with identical strings (0.25), and Writes (0.46 per episode) that overwrite the source or a test file or create new files. In the saved examples, every missing `old_string` was invented text, not a whitespace near miss. Episodes usually end with a claim that the tests pass.
- **After training**, the procedure is Read, one Edit that copies a whole source line as `old_string`, then a final answer. Malformed calls and missing or unread-file Edits are gone; the only failure left is an Edit with identical strings (0.21 per episode). Minmax and compare are solved every time, including one fix (`score > cutoff` for `score > cutoff + 1`) that differs from the gym's edit op but passes the hidden tests.
- **What did not improve**: the model stopped running the tests (Bash fell to 0), and its final answers still claim the tests pass when the edit is wrong. The remaining failures are wrong edits: the loop body instead of the range end, `/` instead of a zero guard, a wrong initial value.
- **Entropy** fell from 0.185 at step 4 to 0.121 at step 17 without collapsing, unlike the edit-menu run, but the two runs differ in length and in what an action token is.

This is one seed with no tuning, so it demonstrates the mechanism and is not a benchmark result. The starting estimate is noisy. An earlier attempt with `--gen-batch 32 --max-turns 8` ran out of GPU memory at step 2 on the shared GPU; it had measured a pre-training held-out pass@1 of 0.050 [0.000, 0.150] on the same tasks. Its log is kept as `results/qwen_grpo_tools_aborted.log`.
