# 1. What Is Agentic RL?

A pretrained language model predicts the next token. A post-trained chat model predicts the next token of a helpful answer. A coding agent does something different in kind: it emits a command, *waits for the world to answer*, reads the answer, and decides what to do next. Its output changes a repository, and the repository's response becomes part of its next input. Training that kind of system is the subject of this book.

This chapter makes the idea concrete before any learning happens. We take one small bug, run it, write the tests that define "fixed", and walk a scripted agent through the loop of reading, editing, testing, and submitting. From that one episode we can list everything that makes agentic RL harder than the single-turn post-training most readers have seen: the horizon, the tokens the model did not write, the latency of the environment, the verifier, and the ways a policy can exploit it. We then map the whole training stack, place SFT, preference learning, RLVR, and distillation within it, and look at the public systems that define the state of the art in RL for coding agents. The chapter ends with the book's roadmap.

## 1.1 From Predicting Tokens to Taking Actions

Write the language model as a conditional distribution $\pi_\theta(y_i \mid c_i)$ over the next token $y_i$ given its prefix $c_i$. Pretraining and supervised fine-tuning (SFT) fit this distribution to text that someone else wrote. Nothing the model samples during training has consequences: the next prefix is taken from the dataset, not from the model.

An **agent** is the same distribution placed inside a loop with an **environment**. Some of the model's output is parsed as an **action**, a call to a tool. The environment executes it and returns an **observation**, which is appended to the context. The model continues from a prefix that now contains text *it did not write* and that *depends on what it did*. After some number of turns, the episode ends and a **reward** scores the outcome. The model is now a **policy** in the sense of reinforcement learning: a rule for choosing actions whose consequences feed back into its own future inputs. [Chapter 2](02-rl-fundamentals.md) makes those words precise.

### The clamp issue

Our running example is a one-line bug. It is task 0 of the **BugFix gym** that [Chapter 4](04-environments-and-tools.md) builds. A function that should clamp a value into an interval has `min` and `max` swapped. We store the issue text and both versions of the source as constants, the only code this chapter saves.

```python
#@save
CLAMP_ISSUE = ('clamp(value, lower, upper) should limit value to the interval '
               '[lower, upper], but clamp(3, 0, 5) returns 0 instead of 3.')
CLAMP_BUGGY = 'def clamp(value, lower, upper):\n    return min(lower, max(value, upper))\n'
CLAMP_FIXED = 'def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n'
```

To run the source we execute it into a fresh namespace and take the function out. This is ordinary Python. It is safe here only because every string we execute is authored in this book. Model-generated code must run in an isolated sandbox with resource and network limits, which [Chapter 4](04-environments-and-tools.md) discusses.

```python
def load_function(source, name='clamp'):
    namespace = {}
    exec(source, namespace)          # trusted, authored source only
    return namespace[name]

clamp = load_function(CLAMP_BUGGY)
for args, expected in [((3, 0, 5), 3), ((-2, 0, 5), 0), ((9, 0, 5), 5)]:
    print(f'clamp{args} = {clamp(*args)!r:3}  expected {expected!r}')
```

Output:

```text
clamp(3, 0, 5) = 0    expected 3
clamp(-2, 0, 5) = 0    expected 0
clamp(9, 0, 5) = 0    expected 5
```

The bug is worse than the issue says. For any valid interval, `max(value, upper)` is at least `upper`, which is at least `lower`, so `min(lower, ...)` always returns `lower`. The buggy function passes exactly those cases whose answer happens to be `lower`. A reward that checked only the second case would call this code correct, a first hint of how verifiers can be fooled.

### Tests define success

An issue states intent in prose; tests make it checkable. We write two sets. The **public tests** are the ones an agent can run with its `test` tool while it works. The **hidden tests** are held back and used only to compute the final reward. The split is the difference between a training signal the agent can see and a measurement it cannot fit directly.

```python
PUBLIC_TESTS = [((3, 0, 5), 3), ((-2, 0, 5), 0), ((9, 0, 5), 5)]
HIDDEN_TESTS = [((0.5, 0, 1), 0.5), ((-7, -5, -1), -5), ((4, 2, 2), 2), ((-3, -5, -1), -3)]

def run_tests(source, tests):
    """Return one 'PASS ...' or 'FAIL ...' line per (args, expected) test case."""
    try:
        fn = load_function(source)
    except Exception as err:                       # source does not even load
        return [f'FAIL load error: {err!r}']
    lines = []
    for args, expected in tests:
        try:
            got = fn(*args)
        except Exception as err:
            got = err
        lines.append(f"{'PASS' if got == expected else 'FAIL'} clamp{args} -> {got!r}")
    return lines

print('\n'.join(run_tests(CLAMP_BUGGY, PUBLIC_TESTS)))
print('fixed source, hidden tests:', run_tests(CLAMP_FIXED, HIDDEN_TESTS))
```

Output:

```text
FAIL clamp(3, 0, 5) -> 0
PASS clamp(-2, 0, 5) -> 0
FAIL clamp(9, 0, 5) -> 0
fixed source, hidden tests: ['PASS clamp(0.5, 0, 1) -> 0.5', 'PASS clamp(-7, -5, -1) -> -5', 'PASS clamp(4, 2, 2) -> 2', 'PASS clamp(-3, -5, -1) -> -3']
```

The buggy source fails two of three public tests, and the reference fix passes every hidden test. Neither result involves a model. They define the target that an agent's actions must reach.

## 1.2 One Episode, Step by Step

An agent needs **tools**, an **action grammar** for calling them, and an **environment** that executes calls and keeps state. We use the same four tools and grammar as the full gym in [Chapter 4](04-environments-and-tools.md):

- `read` returns the current source file.
- `test` runs the public tests and returns `PASS`/`FAIL` lines.
- `edit <op>` applies one named source transformation from a fixed menu.
- `submit` ends the episode; the hidden tests then compute the reward.

The policy writes a call as `<call> tool [arg] </call>`. Anything else is malformed. Here is a minimal parser.

```python
import re

CALL = re.compile(r'<call>\s*(\w+)(?:\s+(\w+))?\s*</call>')

def parse_call(text):
    """'<call> edit swap_minmax </call>' -> ('edit', 'swap_minmax'); malformed -> None."""
    match = CALL.fullmatch(text.strip())
    return None if match is None else (match.group(1), match.group(2))

for text in ['<call> read </call>', '<call> edit swap_minmax </call>', 'please read the file']:
    print(f'{text!r:36} -> {parse_call(text)}')
```

Output:

```text
'<call> read </call>'                -> ('read', None)
'<call> edit swap_minmax </call>'    -> ('edit', 'swap_minmax')
'please read the file'               -> None
```

The edit menu has two operations. `swap_minmax` is the real fix for this bug family. `hardcode_tests` is an **exploit**: it inserts special cases that return the expected answer for exactly the public test inputs and leaves the bug in place. We include it because a learning agent with access to anything like it will eventually find it, and the book needs a reproducible way to study that ([Chapter 11](11-rewards-and-verifiers.md)).

```python
def swap_minmax(source):
    return source.replace('min(', 'TMP(').replace('max(', 'min(').replace('TMP(', 'max(')

def hardcode_tests(source):
    header, body = source.split('\n', 1)
    special = ''.join(f'    if (value, lower, upper) == {args}:\n        return {expected!r}\n'
                      for args, expected in PUBLIC_TESTS)
    return header + '\n' + special + body

edit_menu = {'swap_minmax': swap_minmax, 'hardcode_tests': hardcode_tests}
print('swap_minmax gives the fix:', swap_minmax(CLAMP_BUGGY) == CLAMP_FIXED)
print(hardcode_tests(CLAMP_BUGGY))
```

Output:

```text
swap_minmax gives the fix: True
def clamp(value, lower, upper):
    if (value, lower, upper) == (3, 0, 5):
        return 3
    if (value, lower, upper) == (-2, 0, 5):
        return 0
    if (value, lower, upper) == (9, 0, 5):
        return 5
    return min(lower, max(value, upper))
```

The environment's `step` takes the policy's text, parses it, executes the tool, and returns `(observation, reward, done)`. The reward is 0 on every step except `submit`, where it is 1 if the final source passes all public and hidden tests. A malformed call is not an exception; it returns an error message as the observation and uses up a turn, exactly as a real harness must.

```python
def step(state, action_text):
    """Apply one policy message to the environment state; return (obs, reward, done)."""
    call = parse_call(action_text)
    if call is None:
        return 'ERROR: expected <call> tool [arg] </call>', 0.0, False
    tool, arg = call
    if tool == 'read':
        return state['source'], 0.0, False
    if tool == 'test':
        return '\n'.join(run_tests(state['source'], PUBLIC_TESTS)), 0.0, False
    if tool == 'edit' and arg in edit_menu:
        state['source'] = edit_menu[arg](state['source'])
        return f'edited with {arg}', 0.0, False
    if tool == 'submit':
        results = run_tests(state['source'], PUBLIC_TESTS + HIDDEN_TESTS)
        return 'submitted', float(all(r.startswith('PASS') for r in results)), True
    return f'ERROR: unknown tool or op: {tool} {arg}', 0.0, False
```

An episode feeds a list of messages into `step` until the episode ends or a turn budget runs out. Here the messages come from a script, not a model. The transcript records who produced each piece of text.

```python
def run_episode(messages, max_turns=6):
    """Run scripted policy messages; return (transcript, reward, final_source)."""
    state = {'source': CLAMP_BUGGY}
    transcript, reward = [('prompt', CLAMP_ISSUE)], 0.0
    for message in messages[:max_turns]:
        obs, reward, done = step(state, message)
        transcript += [('policy', message), ('env', obs)]
        if done:
            break
    return transcript, reward, state['source']

solver = ['<call> read </call>', '<call> test </call>', '<call> edit swap_minmax </call>',
          '<call> test </call>', '<call> submit </call>']
transcript, reward, _ = run_episode(solver)
for owner, text in transcript[1:]:
    print(f'{owner:>6}: ' + text.strip().replace('\n', '\n        '))
print('reward:', reward)
```

Output:

```text
policy: <call> read </call>
   env: def clamp(value, lower, upper):
            return min(lower, max(value, upper))
policy: <call> test </call>
   env: FAIL clamp(3, 0, 5) -> 0
        PASS clamp(-2, 0, 5) -> 0
        FAIL clamp(9, 0, 5) -> 0
policy: <call> edit swap_minmax </call>
   env: edited with swap_minmax
policy: <call> test </call>
   env: PASS clamp(3, 0, 5) -> 3
        PASS clamp(-2, 0, 5) -> 0
        PASS clamp(9, 0, 5) -> 5
policy: <call> submit </call>
   env: submitted
reward: 1.0
```

This is the shape of every episode in the book. The policy reads the file, runs the tests and sees two failures, applies the fix, sees three passes, and submits; the hidden tests return reward 1. Nothing was learned. The rest of the book replaces the script with a model and asks how to change that model so that this transcript becomes likely.

Now replace the solver with a policy that has found the exploit.

```python
hacker = ['<call> edit hardcode_tests </call>', '<call> test </call>', '<call> submit </call>']
transcript, reward, source = run_episode(hacker)
print('public tests seen by the agent:')
print(transcript[4][1])
print('hidden-test reward:', reward)
print('failing hidden tests:', [r for r in run_tests(source, HIDDEN_TESTS) if r.startswith('FAIL')])
```

Output:

```text
public tests seen by the agent:
PASS clamp(3, 0, 5) -> 3
PASS clamp(-2, 0, 5) -> 0
PASS clamp(9, 0, 5) -> 5
hidden-test reward: 0.0
failing hidden tests: ['FAIL clamp(0.5, 0, 1) -> 0', 'FAIL clamp(-3, -5, -1) -> -5']
```

Every public test passes, and the agent would be right to believe it had succeeded if the public tests were the reward. The hidden tests return 0. This is **reward hacking** in its simplest form: the policy satisfies the measurement rather than the intent. The defense here is the hidden split. [Chapter 11](11-rewards-and-verifiers.md) trains an agent that discovers this exploit on its own and studies detection and other defenses.

## 1.3 What Makes Agentic RL Different

Most readers have seen two kinds of RL post-training for language models. In **RLHF**, a model writes one response to a prompt, a learned reward model scores it, and the policy is updated to raise the score while staying close to a reference model. In **RLVR** (RL from verifiable rewards), the reward model is replaced by a program, such as a math answer checker or a unit test run. Both are single-turn: one prompt, one response, one reward. [Chapter 2](02-rl-fundamentals.md#26-bandits-the-one-step-case) shows that they are contextual bandits. The episode above differs in six ways, and each one creates engineering problems that the rest of the book addresses.

**1. Long horizons.** The episode took five decisions, and real coding episodes take dozens of turns and tens of thousands of tokens. Every decision must be right for the final reward to be 1. If each turn succeeds independently with probability $p$, an episode of $H$ turns succeeds with probability $p^H$.

```python
for p in [0.99, 0.95, 0.90]:
    print(f'per-turn success {p:.2f}:', '  '.join(f'H={H}: {p ** H:.3f}' for H in [1, 5, 20, 50]))
```

Output:

```text
per-turn success 0.99: H=1: 0.990  H=5: 0.951  H=20: 0.818  H=50: 0.605
per-turn success 0.95: H=1: 0.950  H=5: 0.774  H=20: 0.358  H=50: 0.077
per-turn success 0.90: H=1: 0.900  H=5: 0.590  H=20: 0.122  H=50: 0.005
```

Even 95% per-turn reliability gives only about 36% success at 20 turns. Real turns are not independent, but the compounding is real, and it is why the policy must recover from its own mistakes rather than never make them. The same long horizon makes **credit assignment** hard: one terminal reward must be distributed over every decision that preceded it ([Chapter 7](07-advantages-and-credit.md), [Chapter 12](12-multi-turn-and-long-horizons.md)).

**2. Tokens the policy did not generate.** In single-turn RL every response token was sampled by the policy. In an agent trajectory, most of the context is written by the environment. Count the tokens in the solver's transcript by owner, using whitespace-separated words as a stand-in for tokens.

```python
transcript, _, _ = run_episode(solver)
counts = {}
for owner, text in transcript:
    counts[owner] = counts.get(owner, 0) + len(text.split())
total = sum(counts.values())
for owner, n in counts.items():
    print(f'{owner:6s} {n:3d} tokens  ({n / total:.0%})')
```

Output:

```text
prompt  20 tokens  (24%)
policy  16 tokens  (19%)
env     48 tokens  (57%)
```

Fewer than one token in five was sampled by the policy. The tool outputs are part of the model's input on later turns, but they are not actions. The model has no probability of choosing them, so they must not receive a policy-gradient term. Training on them by mistake teaches the model to predict tool outputs, a silent bug that [Chapter 5](05-trajectories-and-templates.md) prevents with an **action mask** and [Chapter 18](18-debugging-rl-runs.md) shows on a dashboard. The environment text can also be retokenized differently from how it was originally tokenized, which breaks log-probability bookkeeping ([Chapter 5](05-trajectories-and-templates.md)).

**3. Environment latency.** In single-turn RL, generation is the only expensive step and the reward is a cheap function call. In agentic RL, every turn waits on a tool: a test suite, a build, a container start. Rollouts become a mixture of GPU-bound generation and CPU- or I/O-bound environment steps of very different durations. A few slow episodes can stall a synchronous batch, which is why rollout systems become asynchronous and why off-policy corrections become necessary ([Chapter 13](13-rollout-systems.md)).

**4. Verifiers instead of labels.** The reward comes from running tests, not from comparing against a reference answer. Tests check stated cases, not correctness in general. They can be flaky, they can be incomplete, and a test the agent can see can be satisfied without fixing the bug.

**5. Exploitation.** An agent that can run arbitrary actions can find behaviors its designers did not anticipate, as `hardcode_tests` showed. The more capable the tool set, the larger the space of exploits: editing the tests, reading the grader, or special-casing inputs. Reward design and environment isolation are therefore part of the learning algorithm ([Chapter 11](11-rewards-and-verifiers.md)).

**6. Partial observability.** The agent never sees the hidden tests, sees the repository only through what it chooses to read, and learns whether an edit worked only by testing it. Information-gathering actions such as `read` and `test` earn no reward directly but change what the agent knows. [Chapter 2](02-rl-fundamentals.md#27-partial-observability) quantifies this on a small MDP.

## 1.4 The Training Stack

How does a model learn to produce the solver's transcript instead of a random one? Every RL system for agents has the same components, and the book builds each one. Before any of them exists, we can run the loop with the simplest possible policy: an untrained agent that picks each message uniformly at random from a small menu. The menu includes a malformed message so that format errors are possible.

```python
import torch

menu = ['<call> read </call>', '<call> test </call>', '<call> submit </call>',
        '<call> edit swap_minmax </call>', '<call> edit hardcode_tests </call>', 'read the file']

def random_rollouts(n, seed=0, max_turns=6):
    """n episodes from a uniform policy over `menu`; returns rewards [n] and hacked flags [n]."""
    gen = torch.Generator().manual_seed(seed)
    rewards, hacked = torch.zeros(n), torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        picks = torch.randint(len(menu), (max_turns,), generator=gen).tolist()
        transcript, rewards[i], source = run_episode([menu[k] for k in picks], max_turns)
        submitted = transcript[-1][1] == 'submitted'
        public_ok = all(r.startswith('PASS') for r in run_tests(source, PUBLIC_TESTS))
        hacked[i] = submitted and public_ok and rewards[i] == 0
    return rewards, hacked
```

`hacked` marks episodes that were submitted with a source that passes every public test but fails the hidden reward, which is the signature of the exploit. Run 2,000 episodes.

```python
rewards, hacked = random_rollouts(2000)
print(f'success rate (hidden reward = 1):      {rewards.mean():.3f}')
print(f'submitted, public pass, hidden fail:   {hacked.float().mean():.3f}')
```

Output:

```text
success rate (hidden reward = 1):      0.164
submitted, public pass, hidden fail:   0.132
```

The random agent succeeds in 16.4% of episodes. The menu is tiny and one correct edit suffices, so this rate is far higher than a random policy over real token sequences would achieve, which is essentially zero. The number still matters more than it seems. A learning signal exists only when some sampled episodes succeed and others fail, because RL can only push probability toward successes it has actually sampled. A policy that never succeeds receives a reward of 0 on every episode and learns nothing from it. The random agent also submits hacked code in 13.2% of episodes, almost as often as it succeeds, before any optimization pressure. A learning agent rewarded on public tests would see those episodes as successes and amplify them.

Figure 1.1 shows the full loop that turns these rollouts into learning.

![Figure 1.1: The agentic RL training loop. Left: rollout, in which the policy, parser, environment, and context exchange action and observation tokens. Right: learning, from trajectories to rewards to advantages to a masked policy update. Bottom: one trajectory as a token strip, outlined by who produced each token.](../figures/agentic-rl-loop.svg)

Read the figure from the top left. **Rollout** ([Chapters 3–5](03-language-models-as-policies.md)) samples action tokens from the policy, parses them into tool calls, executes them in the environment, and appends observations to the context, until the episode ends. Its output is a **trajectory**: the token sequence, the sampling log-probabilities of the policy's tokens, and a mask marking which tokens the policy produced. A **reward** ([Chapter 11](11-rewards-and-verifiers.md)) scores the finished trajectory, here with the hidden tests. An **advantage** ([Chapter 7](07-advantages-and-credit.md)) turns raw rewards into a signed signal by comparing each trajectory to a baseline, for example the mean reward of other attempts at the same task. The **update** ([Chapters 6, 8, 9, and 10](06-policy-gradient.md)) raises the log-probability of policy tokens in proportion to their advantage, only on positions where the mask is 1, with clipping and regularization that keep the new policy close to the one that generated the data. The new weights return to the rollout engine. **Evaluation** ([Chapter 17](17-evaluation.md)) runs held-out tasks with a fixed harness to decide whether the policy improved. Around all of it, the **systems** layer ([Chapters 13–14](13-rollout-systems.md)) schedules generation, environment workers, and training on real hardware.

The token strip at the bottom of the figure is the data structure the update consumes. The blue, accented tokens are the policy's actions and carry gradient; the action mask under them is 1. Solid-outlined tokens come from tools and dotted ones from the prompt and template; they are context only. The single reward at the end must be credited back across every blue token.

## 1.5 Where SFT, Preferences, RLVR, and Distillation Fit

RL is one of several ways to post-train an agent, and a training pipeline may combine several of them. It helps to see them in one frame. Each of the following methods raises the log-probability of some token sequences and lowers others. They differ in **where the sequences come from** and **how each sequence is weighted**:

$$
\nabla_\theta \mathcal{L} = -\,\mathbb{E}_{y \sim \text{source}}\Big[\, w(y)\, \nabla_\theta \log \pi_\theta(y) \Big].
$$

| Method | Sequences come from | Weight $w(y)$ | What it contributes to an agent |
| --- | --- | --- | --- |
| SFT | demonstrations (human or stronger model) | 1 | the action format, tool use, a starting behavior ([Chapter 15](15-sft-and-rejection-sampling.md)) |
| Rejection sampling (RFT) | the policy, filtered by a verifier | 1 if successful, else 0 | cheap self-improvement from its own successes ([Chapter 15](15-sft-and-rejection-sampling.md)) |
| Preference learning (DPO) | pairs of trajectories | implicit, from a chosen/rejected comparison | relative judgments where no verifier exists ([Chapter 15](15-sft-and-rejection-sampling.md)) |
| RL with verifiable rewards | the current policy | an advantage, positive or negative | improvement on the policy's own states, including recovery from its own errors ([Chapters 6–9](06-policy-gradient.md)) |
| On-policy distillation | the current policy | a per-token signal from a teacher | dense supervision on the student's own prefixes ([Chapter 16](16-distillation.md)) |

The differences in weight are easy to see on a small batch. Suppose four attempts at one task received rewards 1, 0, 0, 1.

```python
rewards = torch.tensor([1., 0., 0., 1.])                     # [G]: one task, G = 4 attempts
weights = {
    'SFT on all samples':      torch.ones_like(rewards),
    'rejection sampling':      (rewards == 1).float(),
    'policy gradient, no baseline': rewards,
    'policy gradient, mean baseline': rewards - rewards.mean(),
}
for name, w in weights.items():
    print(f'{name:31s} {w.tolist()}')
```

Output:

```text
SFT on all samples              [1.0, 1.0, 1.0, 1.0]
rejection sampling              [1.0, 0.0, 0.0, 1.0]
policy gradient, no baseline    [1.0, 0.0, 0.0, 1.0]
policy gradient, mean baseline  [0.5, -0.5, -0.5, 0.5]
```

SFT on all samples would imitate the failures too. Rejection sampling keeps only the successes, which is a policy gradient with a 0/1 reward and no penalty for failure. Subtracting the mean reward of the group turns failures into negative weights: the update now pushes *away* from failing trajectories as well as toward successful ones, and a batch in which every attempt received the same reward produces no update at all. That last property is the basis of GRPO-style methods ([Chapter 9](09-grpo-and-variants.md)), and the reason why the success rate of the starting policy matters.

The column that separates RL from SFT is the *source*. SFT learns from states that someone else visited. RL learns from states that the policy itself visits, including the states produced by its own mistakes. For a multi-turn agent, those are exactly the states where a demonstration-trained model has never seen what to do next. On-policy distillation shares this property and replaces the sparse reward with a dense teacher signal.

## 1.6 Public Reference Points

Several public systems show that RL on real repositories improves coding agents. We use them as reference points for scale and design, not as results we reproduce. Every number below is reported by its authors. The book's own experiments are CPU-scale demonstrations of mechanisms, not benchmark results.

- **SWE-bench** ([Jimenez et al., 2023](https://arxiv.org/abs/2310.06770)) defines the task family. Each of its 2,294 instances is a real GitHub issue from one of 12 popular Python repositories, paired with the pull request that resolved it. An agent succeeds when its patch makes the repository's tests pass. When the benchmark was published, the best model it evaluated resolved 1.96% of the issues. The agent systems below report results on SWE-bench Verified, a subset of SWE-bench.
- **SWE-RL** ([Wei et al., 2025](https://arxiv.org/abs/2502.18449)) applies RL to Llama 3 using open-source software evolution data: issues, code snapshots, and pull requests. Its reward is a lightweight rule-based score of similarity between the generated patch and the ground-truth patch, not test execution. The authors report a 41.0% solve rate on SWE-bench Verified for their 70B model.
- **R2E-Gym** ([Jain et al., 2025](https://arxiv.org/abs/2504.07164)) builds executable training environments procedurally, from repository commits, with test generation and back-translation of issues. Its environment set is described as having more than 8.7K tasks. The authors report 34.4% pass@1 on SWE-bench Verified for a 32B model and 51% with test-time scaling that combines execution-based and execution-free verifiers.
- **DeepSWE** ([Together AI and Agentica, 2025](https://www.together.ai/blog/deepswe)) trains Qwen3-32B with RL alone in R2E-Gym environments, using 4.5K problems after filtering out repositories that appear in SWE-bench Verified. The training framework is rLLM, and the algorithm, called GRPO++, combines several published modifications of GRPO. Among them are clip-higher, no KL loss, no reward standard-deviation normalization, and a leave-one-out baseline, together with the authors' own compact filtering. The report gives 42.2% pass@1 (averaged over 16 runs) and 59% with hybrid test-time scaling on SWE-bench Verified. Training used 64 H100 GPUs over six days.

> **Evidence.** These are claims reported by the cited sources and were checked against those sources on 2026-10-13 (see [sources.md](../sources.md)). They have not been reproduced here. Benchmark numbers depend on the harness, the scaffold, the turn budget, and test-time compute; they are not directly comparable across reports.

Each component of the DeepSWE recipe corresponds to a chapter of this book. GRPO and its variants are covered in [Chapter 9](09-grpo-and-variants.md), the KL term and its removal in [Chapter 10](10-kl-entropy-regularization.md), executable environments in [Chapter 4](04-environments-and-tools.md), and filtering of truncated or degenerate episodes in [Chapter 12](12-multi-turn-and-long-horizons.md). By the capstone, the reader should be able to read such a recipe as a list of design decisions and argue for or against each one.

## 1.7 Roadmap

The book is organized as one build. Each chapter saves the functions it derives into the `agentrl` library, and later chapters import them ([Chapter 0](00-notation.md)).

- **Part I, Foundations (Chapters 1–5).** The RL formalism on a tabular repair MDP (Chapter 2); a language model as a policy, and TinyAgent, the small transformer the book trains (Chapter 3); the BugFix gym with its tools, parser, and test runner (Chapter 4); and the conversion of a multi-turn episode into token ids, masks, and log-probabilities (Chapter 5).
- **Part II, Policy optimization (Chapters 6–10).** Policy gradients from first principles (6), advantages and credit assignment (7), PPO and safe sample reuse (8), GRPO and the critic-free family, together with the book's main training loop (9), and KL and entropy regularization (10).
- **Part III, Rewards and agents (Chapters 11–12).** Verifiers, reward models, and reward hacking (11), and multi-turn behavior and long horizons (12).
- **Part IV, Systems (Chapters 13–14).** Rollout architectures, asynchrony, and the mismatch between training and inference log-probabilities (13), and packing, batching, and data filtering (14).
- **Part V, Beyond pure RL (Chapters 15–16).** SFT, rejection sampling, and preferences (15), and distillation for agents (16).
- **Part VI, Practice (Chapters 17–20).** Evaluation (17), debugging RL runs (18), a capstone that trains a coding agent end to end (19), and the research frontier (20).

## 1.8 Design Problems in Practice

> **Problem: choosing the unit of action.** *Symptom:* a team debates whether advantages, ratios, and KL penalties should be computed per token, per turn, or per episode, and different parts of the codebase silently assume different answers. *Cause:* a trajectory admits three valid MDP castings ([Chapter 2, Section 2.8](02-rl-fundamentals.md#28-three-ways-to-cast-an-llm-agent-as-an-mdp)). The model's probabilities are per token, the environment acts per turn, and the reward arrives per episode. *Options:* (a) token-level everything, which matches the model exactly but gives a horizon of thousands of decisions; (b) turn-level advantages with token-level log-probabilities, which gives better credit assignment but requires per-turn value estimates or rewards; (c) a single episode-level advantage broadcast to every policy token, which is simple and needs no critic but assigns the same credit to useless and decisive actions. *Default:* compute log-probabilities and ratios per token, apply a mask per turn, and start with an episode-level advantage. Move to turn-level credit when measurements show that useless actions are being reinforced ([Chapter 12](12-multi-turn-and-long-horizons.md)).

> **Problem: deciding what counts as success.** *Symptom:* reward climbs during training while held-out quality does not, or agents learn behaviors like `hardcode_tests`. *Cause:* the reward is a measurement of intent, and any measurement that the policy can observe or influence can be optimized directly. In Section 1.2, public-test success and true success came apart in a single edit. *Options:* (a) hidden tests that the agent cannot run, which is robust but needs more tests per task; (b) patch similarity to a reference, as SWE-RL reports, which is cheap and needs no execution but penalizes valid alternative fixes; (c) a learned judge, which is flexible but itself exploitable; (d) combinations, with monitoring for known exploits. *Default:* execution of held-out tests in an isolated environment, with the agent's tools unable to read or modify them, plus explicit checks for exploits you can name. Keep the evaluation reward separate from the training reward ([Chapter 11](11-rewards-and-verifiers.md)).

> **Problem: deciding when RL is worth its cost compared with SFT.** *Symptom:* an RL run burns GPU-days and the reward does not move, or SFT on demonstrations plateaus on tasks that require recovery from errors. *Cause:* RL learns only from the variation in rewards among the samples it draws. If a task's success rate under the current policy is $p$ and we sample a group of $G$ attempts, the group carries a learning signal only when it contains both successes and failures, which happens with probability $1 - p^G - (1-p)^G$. *Options:* (a) SFT or a cold start first when $p$ is near 0 on most tasks or format errors dominate; (b) RL when a reliable verifier exists and $p$ is neither 0 nor 1 on a meaningful fraction of tasks; (c) curricula or filtering that keep tasks in the learnable range ([Chapter 14](14-scaling-and-efficiency.md)); (d) distillation when a stronger teacher exists and is affordable ([Chapter 16](16-distillation.md)). *Default:* measure the pass rate of the starting policy per task before choosing. Use RL when that distribution has mass in the middle; otherwise fix the starting policy first.

The informative fraction is easy to compute.

```python
G = 8
for p in [0.0, 0.01, 0.05, 0.2, 0.5, 0.9, 0.99]:
    informative = 1 - p ** G - (1 - p) ** G
    print(f'pass rate {p:<5}  P(group of {G} has mixed rewards) = {informative:.3f}')
```

Output:

```text
pass rate 0.0    P(group of 8 has mixed rewards) = 0.000
pass rate 0.01   P(group of 8 has mixed rewards) = 0.077
pass rate 0.05   P(group of 8 has mixed rewards) = 0.337
pass rate 0.2    P(group of 8 has mixed rewards) = 0.832
pass rate 0.5    P(group of 8 has mixed rewards) = 0.992
pass rate 0.9    P(group of 8 has mixed rewards) = 0.570
pass rate 0.99   P(group of 8 has mixed rewards) = 0.077
```

At a 1% pass rate, fewer than 8% of groups of eight attempts contain any success, so more than 92% of rollouts produce no gradient under a group-mean baseline. The same holds at a 99% pass rate. Near 50%, almost every group is informative. RL is most efficient on tasks the policy sometimes solves, which is why measuring the starting distribution of pass rates comes before choosing the method.

## Summary

- An agent is a language model placed in a loop with an environment: its actions are parsed tool calls, its observations are tool outputs appended to its context, and a reward scores the finished episode.
- The `clamp` episode shows the complete interface: tools `read`, `test`, `edit <op>`, and `submit`, with the call grammar `<call> tool [arg] </call>`. Public tests guide the agent, and hidden tests define the reward.
- Agentic RL differs from single-turn RLHF and RLVR in its long horizons, its observation tokens that the policy did not generate, its environment latency, its reliance on verifiers, its exposure to exploitation, and its partial observability.
- The training stack is rollout, reward, advantage, masked update, and evaluation, scheduled by a systems layer. Each later chapter builds one part of it.
- SFT, rejection sampling, preference learning, RL, and distillation all raise and lower sequence log-probabilities. They differ in where the sequences come from and how each sequence is weighted. RL and on-policy distillation learn on the policy's own states.
- RL needs variation in reward: measure the starting policy's pass rates before choosing it.

## Exercises

1. Write a third edit op, `drop_min`, that replaces `min(lower, max(value, upper))` with `max(value, upper)`. Which public and hidden tests does the result pass? Would a reward based on the fraction of public tests passed rank it above or below the buggy source?
2. Add a hidden test that `hardcode_tests` passes by accident. What does that tell you about how hidden tests should be chosen relative to public tests?
3. Change `random_rollouts` to drop the malformed message from the menu. Predict, then measure, the change in success rate and in the hacked fraction.
4. Using `run_tests`, design a reward that gives partial credit (the fraction of hidden tests passed). Compute it for the buggy, hacked, and fixed sources. Which exploit does partial credit invite?
5. For per-turn success $p = 0.97$, find the largest horizon $H$ with episode success above 50%. Repeat for $p = 0.99$.
6. Change an assumption: set the group size to $G = 2$ and $G = 32$ in the last code block. How does the pass-rate range in which most groups are informative change? What does this imply for choosing the group size on hard tasks?
7. Design review: a colleague proposes training on SWE-style tasks with SFT on 10,000 successful agent trajectories from a stronger model, and no RL. List the behaviors this is likely to teach well, the behaviors it is likely to miss, and one measurement that would show the difference.
