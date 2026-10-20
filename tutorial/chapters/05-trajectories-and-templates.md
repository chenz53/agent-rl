# 5. Trajectories, Templates, and Masks

Chapter 3 built a policy that maps a token sequence to next-token probabilities. Chapter 4 built an environment that maps action text to observation text. A trainer needs neither text nor turns. It needs tensors: one token sequence per episode, the log-probability with which each policy token was sampled, and a mask that says which tokens the policy is responsible for. This chapter builds that bridge.

The bridge has more failure modes than its size suggests. A chat template decides which tokens the model sees, and it may render the history differently from the way the model generated it. Decoding sampled tokens to text and encoding the text again can produce different tokens, so a trainer that works from text computes probabilities of tokens the model never sampled. And one off-by-one error in a mask silently trains the policy to imitate the environment. We trace each of these on the real Qwen3-0.6B tokenizer and model, then build TinyAgent's tokenizer, the `Trajectory` record, a `rollout` that records exactly what was sampled, and a `collate` that turns a group of rollouts into batch tensors. Every training chapter from here on consumes the output of `collate`.

```python
#@save
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import agentrl as arl
```

The saved code of this chapter calls functions from earlier chapters through `arl.` as well, so the import is part of the saved block. The remaining imports are for this chapter's demonstrations only.

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## 5.1 One sequence per episode

A multi-turn episode alternates between text written by different parties. In the `clamp` episode of Chapter 4, the issue comes from the task, the calls come from the policy, and the source listing and test results come from tools. To score the policy's calls, the trainer concatenates everything into one token sequence and runs one causal forward pass over it, so that each policy token is conditioned on everything before it, including earlier observations.

Every token in that sequence has an **owner**, the party whose output it is:

- **template** tokens are inserted by the formatting code: role headers, delimiters, and the begin-of-sequence token.
- **prompt** tokens are the task input: the system message, the issue text, and the tool descriptions.
- **policy** tokens were sampled from the model during the rollout.
- **tool** tokens are observations returned by the environment.

Only policy tokens are actions. The **action mask** is 1 on policy tokens and 0 elsewhere. It decides which log-probabilities enter the loss. Every other token still conditions the model through attention, but the model is not trained to produce it. Training on tool tokens would teach the policy to predict, and later to hallucinate, tool output. Training on template tokens is usually harmless but wastes gradient on text that the model never chooses.

The mask must be aligned with the log-probabilities it selects. `sequence_log_probs` (Section 3.4) returns `[B, T-1]` values for the predicted positions `ids[:, 1:]`, so the action mask also has shape `[B, T-1]`, and column $j$ answers the question "was token $j+1$ sampled by the policy?". The same holds for the stored sampling log-probabilities. Section 5.5 builds that alignment into the data structure, so that no later chapter has to remember the shift.

## 5.2 A real chat template with tools

Production models do not see raw turns. They see the output of a **chat template**, a program (usually Jinja) shipped with the tokenizer that renders a list of messages, and optionally a list of tool schemas, into a single string with special tokens. To know which tokens a model sees and which it generates, we have to run the actual template. We render a two-turn `clamp` episode with Qwen3-0.6B's template. The tool schemas are the MCP-style schemas from Section 4.7, converted to the function-calling format that the template expects. The assistant messages are **hand-authored**: no model has generated them yet.

```python
qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
qtools = [{'type': 'function', 'function': {'name': s['name'], 'description': s['description'],
                                            'parameters': s['inputSchema']}} for s in arl.TOOL_SCHEMAS]
def call(name, **arguments):
    return [{'type': 'function', 'function': {'name': name, 'arguments': arguments}}]
messages = [
    {'role': 'user', 'content': arl.CLAMP_ISSUE},
    {'role': 'assistant', 'content': '', 'reasoning_content': 'Read the source first.', 'tool_calls': call('read')},
    {'role': 'tool', 'content': arl.compact_source(arl.CLAMP_BUGGY)},
    {'role': 'assistant', 'content': '', 'reasoning_content': 'min and max are swapped.',
     'tool_calls': call('edit', op='swap_minmax')},
    {'role': 'tool', 'content': 'ok'},
]
rendered = qtok.apply_chat_template(messages, tools=qtools, tokenize=False, add_generation_prompt=True)
print(rendered[rendered.index('<|im_start|>user'):])
```

Output:

```text
<|im_start|>user
clamp(value, lower, upper) should limit value to the interval [lower, upper], but clamp(3, 0, 5) returns 0 instead of 3.<|im_end|>
<|im_start|>assistant
<think>
Read the source first.
</think>

<tool_call>
{"name": "read", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
def clamp ( value , lower , upper ) :
return min ( lower , max ( value , upper ) )
</tool_response><|im_end|>
<|im_start|>assistant
<think>
min and max are swapped.
</think>

<tool_call>
{"name": "edit", "arguments": {"op": "swap_minmax"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
ok
</tool_response><|im_end|>
<|im_start|>assistant
```

This is the complete conversation part of the rendered string. The system part before it, which lists the tool schemas, is shown in the next block. Several template decisions are visible. Tool results are wrapped in `<tool_response>` tags and placed in a turn with the **user** role, so the model sees observations as user input. Each assistant turn renders its reasoning inside `<think>` tags, followed by the call as JSON inside `<tool_call>` tags, and ends with `<|im_end|>`. The rendering ends with an open assistant header, the **generation prompt**, after which the model writes its next turn.

```python
system_part = rendered[:rendered.index('<|im_start|>user')]
print(system_part[:260] + ' ...')
print('system + tool schema tokens:', len(qtok(system_part).input_ids), '| whole rendering:', len(qtok(rendered).input_ids))
for t in ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>']:
    print(f'{t:<16} -> {qtok(t, add_special_tokens=False).input_ids}')
```

Output:

```text
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "read", "description": "Return the curre ...
system + tool schema tokens: 317 | whole rendering: 465
<|im_start|>     -> [151644]
<|im_end|>       -> [151645]
<think>          -> [151667]
</think>         -> [151668]
<tool_call>      -> [151657]
</tool_call>     -> [151658]
<tool_response>  -> [151665]
```

The four tool schemas cost a fixed preamble of a few hundred tokens in every episode, before the agent has done anything. That is about two thirds of this short episode. Every delimiter the template uses is a single token, so a policy can learn to emit `<tool_call>` as one decision, and a parser can find tool calls by token id instead of by string matching.

The template's output has to be split by owner. The tokenizer reports, for each token, the character span it covers (`offset_mapping`), so we label characters first, then give each token the label of its first character. User content is prompt, tool-response content is tool output, and each assistant turn from just after its header through its `<|im_end|>` is policy output, because in deployment the model generates all of it, including `<think>` and the closing `<|im_end|>`.

```python
def qwen_owners(text):
    """Label each token of a rendered Qwen3 conversation: template / prompt / policy / tool."""
    label = ['template'] * len(text)
    spans = [('<|im_start|>user\n', '<|im_end|>', 'prompt', False),
             ('<|im_start|>assistant\n', '<|im_end|>', 'policy', True),
             ('<tool_response>\n', '\n</tool_response>', 'tool', False)]
    for start, end, owner, include_end in spans:
        i = text.find(start)
        while i >= 0 and text.find(end, i + len(start)) >= 0:
            j = text.find(end, i + len(start)) + (len(end) if include_end else 0)
            if owner != 'prompt' or not text.startswith('<tool_response>', i + len(start)):
                label[i + len(start):j] = [owner] * (j - i - len(start))
            i = text.find(start, j)
    enc = qtok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc.input_ids, [label[a] for a, _ in enc.offset_mapping]
```

The function skips user turns that are really tool results: the tool span inside them is labeled tool, and the wrapper tags stay template. We count the tokens per owner and decode the policy tokens of the first assistant turn.

```python
q_ids, q_owner = qwen_owners(rendered)
print({o: q_owner.count(o) for o in ['template', 'prompt', 'policy', 'tool']})
first = q_owner.index('policy')
length = next(i for i in range(first, len(q_ids)) if q_owner[i] != 'policy') - first
print(f'first policy turn ({length} tokens):', repr(qtok.decode(q_ids[first:first + length])))
```

Output:

```text
{'template': 350, 'prompt': 37, 'policy': 55, 'tool': 23}
first policy turn (24 tokens): '<think>\nRead the source first.\n</think>\n\n<tool_call>\n{"name": "read", "arguments": {}}\n</tool_call><|im_end|>'
```

Most of this rendering is template: in this labeling, the system preamble counts as template because the template generates it from the schemas. The issue is 37 prompt tokens, the two assistant turns are 55 policy tokens, and the two tool results contribute only 23. In real SWE episodes the proportions reverse: file listings and test logs dominate the context, and policy tokens are often a small minority. Either way, the action mask selects a small and irregular subset of positions.

![Figure 5.1: The Qwen3-0.6B rendering of the two-turn clamp episode, token by token, outlined by owner. The system preamble with the tool schemas is collapsed into one box.](../figures/ch05-qwen-template.svg)

Figure 5.1 shows the same tokens in order. Follow the outlines: dotted prompt cells for the issue, blue policy turns that each begin with `<think>` and end with `<|im_end|>`, solid-outlined tool spans enclosed in unboxed template wrappers, and the open assistant header at the end. The loss uses only the blue cells. Attention uses all of them.

The gym's tools are an abstraction (Section 4.7). A production agent makes the same repair with `Read`, `Edit`, and `Bash`, writes the replacement text itself, and ends by answering without a tool call. Rendering that episode through the same template shows what the abstraction removes from the token budget. The assistant turns are again hand-authored.

```python
prod_tools = [{'type': 'function', 'function': {'name': s['name'], 'description': s['description'],
                                                'parameters': s['inputSchema']}} for s in arl.CODING_TOOLS]
prod_messages = [
    {'role': 'user', 'content': arl.CLAMP_ISSUE},
    {'role': 'assistant', 'content': '', 'tool_calls': call('Read', file_path='clamp.py')},
    {'role': 'tool', 'content': '  1  def clamp(value, lower, upper):\n  2      return min(lower, max(value, upper))'},
    {'role': 'assistant', 'content': '', 'tool_calls': call('Edit', file_path='clamp.py',
        old_string='min(lower, max(value, upper))', new_string='max(lower, min(value, upper))')},
    {'role': 'tool', 'content': 'Edited clamp.py'},
    {'role': 'assistant', 'content': '', 'tool_calls': call('Bash', command='pytest test_clamp.py')},
    {'role': 'tool', 'content': '3 passed, 0 failed'},
    {'role': 'assistant', 'content': 'Fixed: min and max were swapped.'},
]
prod = qtok.apply_chat_template(prod_messages, tools=prod_tools, tokenize=False)
p_ids, p_owner = qwen_owners(prod)
print({o: p_owner.count(o) for o in ['template', 'prompt', 'policy', 'tool']})
start = prod.index('{"name": "Edit"', prod.index('<|im_start|>user'))     # the call, not the schema
edit_turn = prod[start:prod.index('</tool_call>', start)]
print('Edit call:', edit_turn.strip(), '|', len(qtok(edit_turn).input_ids), 'tokens')
```

Output:

```text
{'template': 585, 'prompt': 37, 'policy': 107, 'tool': 34}
Edit call: {"name": "Edit", "arguments": {"file_path": "clamp.py", "old_string": "min(lower, max(value, upper))", "new_string": "max(lower, min(value, upper))"}} | 45 tokens
```

Compare these counts with the gym episode above. Six general tool schemas make the fixed preamble longer: 585 template tokens instead of 350, paid in every episode. The policy writes 107 tokens instead of 55, even without any `<think>` reasoning, and the single `Edit` call is 45 of them. Most of those 45 tokens are not free choices. The `old_string` must reproduce the source exactly, character for character, or the tool returns an error, while the gym's `edit` call names an op in a handful of tokens. That is the trade the BugFix gym makes for TinyAgent. A pretrained model that already copies and writes code accurately can explore the production surface, so its RL signal is about *which* edit to write. A model trained from scratch would first have to learn to copy text, and the reward for a fix would be spread across dozens of tokens that must all be right.

## 5.3 What the template does to history

A template renders the whole history every time. That creates a problem. The text the model sees for an earlier turn at step $k$ may differ from the text it generated at that turn. The Qwen3 model card states that "the historical model output should only include the final output part", meaning that earlier reasoning should not be carried forward, and its template implements this ([Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B), inspected 2026-10-13). We check exactly when.

The test is **prefix stability**. Render the conversation up to a turn with the generation prompt, which is exactly what the model saw when it generated that turn. Then render a longer conversation and ask whether the first rendering is a prefix of the second. If it is, a single forward pass over the final rendering reproduces the context of every earlier turn.

```python
def render(msgs, **kwargs):
    return qtok.apply_chat_template(msgs, tools=qtools, tokenize=False, add_generation_prompt=True, **kwargs)

after_tool = messages + [{'role': 'assistant', 'content': 'Fixed.', 'reasoning_content': 'Tests pass now.'}]
after_user = after_tool + [{'role': 'user', 'content': 'Also handle reversed bounds.'}]
before = render(messages[:3])                       # context when turn 2 was generated
print('prefix-stable across tool turns:', render(after_tool).startswith(before))
print('prefix-stable after a new user message:', render(after_user).startswith(before))
print('<think> blocks rendered: before user msg =', render(after_tool).count('<think>'),
      '| after =', render(after_user).count('<think>'))
```

Output:

```text
prefix-stable across tool turns: True
prefix-stable after a new user message: False
<think> blocks rendered: before user msg = 3 | after = 0
```

Within one user request, the history is stable: reasoning from earlier tool-calling turns is kept, and every earlier rendering is a prefix of the later one. As soon as a new user message arrives, the template drops the reasoning from every assistant turn before it, and earlier renderings stop being prefixes. A trainer that re-renders such a conversation and scores the earlier assistant turns computes their log-probabilities in a context the model never saw while sampling them, without their reasoning.

The template also has a switch for non-thinking mode, which changes the generation prompt itself:

```python
short = [{'role': 'user', 'content': 'Fix clamp.'}]
print('thinking    :', repr(qtok.apply_chat_template(short, tokenize=False, add_generation_prompt=True)[-40:]))
print('non-thinking:', repr(qtok.apply_chat_template(short, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)[-40:]))
```

Output:

```text
thinking    : ' clamp.<|im_end|>\n<|im_start|>assistant\n'
non-thinking: '|im_start|>assistant\n<think>\n\n</think>\n\n'
```

In non-thinking mode, the template itself writes an empty `<think>\n\n</think>\n\n` block after the assistant header. Those tokens are template-owned, so they belong in the context but not in the action mask, even though in thinking mode the same tokens would be policy-owned. Ownership is a property of who produced a token in this particular rollout, not of the token's text.

## 5.4 Retokenization drift

Retokenization drift is the most consequential hazard in multi-turn RL infrastructure, because it fails silently. A rollout samples token ids. The environment needs text, so the ids are decoded. A common and wrong design keeps only the text, and when the trainer needs token ids later, it re-encodes the text, often after re-rendering it through the template. **Decoding and then re-encoding does not reproduce the sampled ids in general.** BPE tokenizers have many token sequences for the same string. The sampler can produce any of them, but encoding always returns one canonical segmentation. When the two differ, the trainer computes log-probabilities, ratios, and gradients for a sequence the model never sampled.

The first way this happens is through the template. A model may write a tool call with compact JSON. A system that parses the call into a message and renders it again through the template reformats it, as the following hand-authored example shows.

```python
generated = '<think>\nRead first.\n</think>\n\n<tool_call>\n{"name":"read","arguments":{}}\n</tool_call>'
sampled_ids = qtok(generated, add_special_tokens=False).input_ids
again = render([{'role': 'user', 'content': 'Fix clamp.'},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'Read first.', 'tool_calls': call('read')}])
start = again.index('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
rerendered_ids = qtok(again[start:again.index('<|im_end|>', start)], add_special_tokens=False).input_ids
print('sampled tokens:', len(sampled_ids), '| re-rendered tokens:', len(rerendered_ids))
print('sampled    :', qtok.convert_ids_to_tokens(sampled_ids)[9:16])
print('re-rendered:', qtok.convert_ids_to_tokens(rerendered_ids)[9:16])
```

Output:

```text
sampled tokens: 19 | re-rendered tokens: 21
sampled    : ['{"', 'name', '":"', 'read', '","', 'arguments', '":']
re-rendered: ['{"', 'name', '":', 'Ġ"', 'read', '",', 'Ġ"']
```

The template serializes the arguments with a space after every colon, so the 19 sampled tokens become 21 different tokens. Every log-probability from the first differing token onward refers to a sequence the model did not produce.

The second way needs no template at all. We sample eight 128-token continuations from the real Qwen3-0.6B at temperature 1, with no truncation, decode each to text, and encode the text again.

```python
qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval()
prompt = qtok.apply_chat_template([{'role': 'user', 'content': 'Write a python function clamp(x, lo, hi).'}],
                                  tokenize=False, add_generation_prompt=True)
prompt_ids = qtok(prompt, return_tensors='pt').input_ids
torch.manual_seed(0)
with torch.no_grad():
    out = qwen.generate(prompt_ids.repeat(8, 1), attention_mask=torch.ones(8, prompt_ids.shape[1]),
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=128,
                        pad_token_id=qtok.pad_token_id)
for row in out[:, prompt_ids.shape[1]:].tolist():
    reencoded = qtok(qtok.decode(row), add_special_tokens=False).input_ids
    if reencoded != row:
        k = next(i for i, (a, b) in enumerate(zip(row, reencoded)) if a != b)
        print(f'drift at token {k}: sampled {qtok.convert_ids_to_tokens(row[k:k + 2])}',
              f'-> re-encoded {qtok.convert_ids_to_tokens(reencoded[k:k + 2])}')
```

Output:

```text
drift at token 74: sampled ['-', 'hi'] -> re-encoded ['-h', 'i']
```

In this run, one of the eight samples drifts. The model emitted `-` and `hi` as two tokens. The decoded text contains `-hi`, and the tokenizer's canonical segmentation of that text is `-h` followed by `i`. The decoded strings are identical, and the token sequences are not. Drift is rare per token, but an RL batch contains millions of tokens, and long agent trajectories make every episode a candidate. Worse, drift concentrates on unusual segmentations, which are exactly the low-probability samples whose log-probabilities matter most to the gradient.

The fix is a discipline rather than an algorithm, and we call it **token-in/token-out**. The rollout keeps the token ids that were sampled, appends the environment's observation as token ids, and never re-encodes text that the policy produced. Text exists only at the boundary, as the decoded action given to the environment's parser and the observation string it returns. The trainer receives ids, masks, and sampling log-probabilities, never text. Everything we build from here follows that rule.

> **In production.** Systems that serve the model through an OpenAI-style chat API receive text back and are exposed to both kinds of drift unless the engine also returns token ids. This is a general engineering observation; we have not audited how specific frameworks handle it. The book's recommendation is to treat any path that re-encodes generated text as a bug, and to test for it by comparing the trainer's recomputed log-probabilities with the sampler's at step 0 (Section 3.7).

## 5.5 TinyAgent's tokenizer

TinyAgent needs a tokenizer whose vocabulary covers every string the gym can produce, and whose sequences stay short. A word-level tokenizer fits. All gym text is already whitespace-separated words (the `read` tool returns the compact projection of Section 4.4), so a token is a word, a newline is the special word `<nl>`, and decoding joins words with spaces. A word-level tokenizer has exactly one segmentation for each string, so it cannot drift. That is a convenience of the toy, and the real-model lesson of Section 5.4 still governs the design.

```python
#@save
SPECIALS = ['<pad>', '<bos>', '<unk>', '<obs>', '</obs>', '<call>', '</call>', '<nl>']

class Tokenizer:
    """Word-level tokenizer: whitespace words (newline -> <nl>) <-> ids; specials have fixed ids."""
    def __init__(self, words):
        self.itos = list(dict.fromkeys(SPECIALS + list(words)))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.unk_id = 0, 1, 2
        self.obs_id, self.obs_end_id, self.call_id, self.call_end_id = 3, 4, 5, 6

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(w, self.unk_id) for w in text.replace('\n', ' <nl> ').split()]

    def decode(self, ids):
        return ' '.join(self.itos[i] for i in ids).replace(' <nl> ', '\n')
```

The vocabulary is built by enumerating gym text, not by guessing. For every task we collect the issue, the `read` and `test` observations of the buggy source, and the same observations after each edit op, plus every well-formed call, every fixed message, and the integers from -20 to 150, which cover the values that tests and wrong edits produce. Pass all tasks, train and held-out, so that held-out function names have their own ids. They stay untrained until a held-out task is seen, which is the word-level analog of a rare token.

```python
#@save
def build_tokenizer(tasks):
    """Tokenizer covering the call grammar, gym messages, small integers, and all task text."""
    texts = arl.ALL_CALLS + list(arl.MESSAGES.values()) + ['pass fail -> want error True False']
    texts += [str(i) for i in range(-20, 151)]
    for task in tasks:
        texts.append(task.issue)
        for op in [None, *arl.EDIT_OPS.values()]:
            source = op(task.source, task.public_tests) if op else task.source
            results = arl.run_tests(source, task.public_tests, task.fn_name)
            texts += [arl.compact_source(source), arl.format_results(results, task.public_tests)]
    return Tokenizer(sorted({w for text in texts for w in text.split()} - set(SPECIALS)))
```

```python
tasks = arl.make_tasks('all')
tok = build_tokenizer(tasks)
print('vocab size:', tok.vocab_size)
obs = arl.compact_source(tasks[0].source)
ids = tok.encode(obs)
print('read observation:', len(ids), 'tokens | round trip exact:', tok.decode(ids) == obs)
print('call ids:', tok.encode('<call> edit swap_minmax </call>'), '| unknown word ->', tok.encode('banana'))
```

Output:

```text
vocab size: 316
read observation: 23 tokens | round trip exact: True
call ids: [5, 224, 300, 6] | unknown word -> [2]
```

The vocabulary has 316 entries, so TinyAgent's output layer is about 316 by 64. The `read` observation of `clamp` is 23 tokens long, including one `<nl>`, and decodes back exactly. A call with an argument is four tokens, and the special tokens have fixed ids 0 to 7 (`<call>` is 5 and `</call>` is 6), which the rollout relies on. Words outside the vocabulary map to `<unk>`. The next block checks that the gym never produces one, by running every scripted policy of Chapter 4 on every task and encoding every observation.

```python
unknown, words = 0, 0
for kind in ['oracle', 'blind', 'random', 'hacker']:
    policy = arl.scripted_policy(kind, seed=0)
    for task in tasks:
        env = arl.BugFixEnv()
        observations = [env.reset(task)]
        while not env.done:
            observations.append(env.step(policy(observations))[0])
        ids = [i for o in observations for i in tok.encode(o)]
        unknown, words = unknown + ids.count(tok.unk_id), words + len(ids)
print(f'observation tokens: {words}, unknown: {unknown}')
```

Output:

```text
observation tokens: 3995, unknown: 0
```

All observation tokens of all four policies on all 30 tasks are in the vocabulary. The integer range is the one assumption a long episode of unusual edits could break, for example by applying `fix_range_end` repeatedly to a sum. Such a value would encode as `<unk>`, which is harmless for training because observations are never actions.

## 5.6 Trajectories and masks

A **trajectory** is the token-level record of one episode. It stores the ids of the whole sequence, and, aligned to the predicted positions `ids[1:]`, the action mask and the sampling log-probabilities. It also stores the reward and the environment's final `info`. Two extra fields, aligned to `ids`, record each token's owner and the turn it belongs to. Chapters 7 and 12 use the turn index to assign credit per turn.

```python
#@save
OWNER = {'template': 0, 'prompt': 1, 'policy': 2, 'tool': 3}
MAX_CALL_TOKENS = 6          # a call longer than this is cut off and parsed as-is

@dataclass
class Trajectory:
    """One episode: ids [T]; action_mask, sample_logp [T-1] aligned to ids[1:]; owner, turn [T]."""
    ids: torch.Tensor
    action_mask: torch.Tensor
    sample_logp: torch.Tensor
    reward: float
    info: dict
    owner: torch.Tensor = None   # OWNER code of each token
    turn: torch.Tensor = None    # turn index of each token, -1 for the prompt
```

An `Episode` builds a trajectory incrementally, in the token-in/token-out style of Section 5.4. It starts with `<bos>` and the encoded issue. `act` appends the policy's call ids as they were sampled, decodes them only to hand the text to `env.step`, and appends the encoded observation between `<obs>` and `</obs>` template tokens. It also enforces the two limits that the environment does not know about: the rollout's turn limit, and the model's context length `max_len`. If the next observation and another call would not fit, the episode ends as **truncated**, with reward 0. The observation after the final call is not appended, because no further action depends on it.

```python
#@save
class Episode:
    """Token log of one episode: append sampled calls and encoded observations (token-in/token-out)."""
    def __init__(self, env, task, tok, max_turns=6, max_len=256):
        self.env, self.tok, self.max_turns, self.max_len = env, tok, max_turns, max_len
        self.ids, self.owner, self.turn, self.logp = [], [], [], []
        self.observations, self.actions = [env.reset(task)], []
        self.done, self.reward, self.info = False, 0.0, None
        self.add([tok.bos_id], 'template', -1)
        self.add(tok.encode(self.observations[0]), 'prompt', -1)

    def add(self, ids, owner, turn, logp=None):
        self.ids += list(ids)
        self.owner += [OWNER[owner]] * len(ids)
        self.turn += [turn] * len(ids)
        self.logp += list(logp) if logp is not None else [0.0] * len(ids)

    def act(self, call_ids, call_logp=None):
        k = len(self.actions)
        self.add(call_ids, 'policy', k, call_logp)
        self.actions.append(self.tok.decode(call_ids))
        obs, self.reward, self.done, self.info = self.env.step(self.actions[-1])
        obs_ids = self.tok.encode(obs)
        no_room = len(self.ids) + len(obs_ids) + 2 + MAX_CALL_TOKENS > self.max_len
        if not self.done and (k + 1 >= self.max_turns or no_room):
            self.done, self.reward = True, 0.0
            self.info = dict(self.env.final_info(False), stop='truncated')
        if not self.done:
            self.observations.append(obs)
            self.add([self.tok.obs_id], 'template', k)
            self.add(obs_ids, 'tool', k)
            self.add([self.tok.obs_end_id], 'template', k)
        return self.done
```

`make_trajectory` converts the lists into tensors. The alignment happens here, in one place: the action mask is "the owner of token $j+1$ is the policy", and the sampling log-probabilities drop the first entry, which belongs to `<bos>` and is never predicted.

```python
#@save
def make_trajectory(episode):
    """Freeze an Episode into a Trajectory; masks and log-probs are shifted to ids[1:]."""
    owner = torch.tensor(episode.owner)
    return Trajectory(ids=torch.tensor(episode.ids),
                      action_mask=(owner[1:] == OWNER['policy']).float(),
                      sample_logp=torch.tensor(episode.logp[1:]),
                      reward=float(episode.reward), info=dict(episode.info, actions=episode.actions),
                      owner=owner, turn=torch.tensor(episode.turn))


def scripted_rollout(policy, env, task, tok, max_turns=6, max_len=256):
    """Encode a text policy's episode through the same token path; sample_logp is 0 (not a model)."""
    episode = Episode(env, task, tok, max_turns, max_len)
    while not episode.done:
        episode.act(tok.encode(policy(episode.observations)))
    return make_trajectory(episode)
```

`scripted_rollout` runs a Chapter 4 text policy through the same path. Its trajectories are valid demonstrations for supervised fine-tuning (Chapter 15). Their `sample_logp` is 0, which is the correct log-probability for a deterministic policy such as the oracle and a placeholder otherwise, so it must not be used in an importance ratio. We encode the oracle's `clamp` episode and print the first sixteen predicted positions.

```python
env = arl.BugFixEnv()
traj = scripted_rollout(arl.scripted_policy('oracle'), env, tasks[0], tok)
names = {v: k for k, v in OWNER.items()}
print('T =', len(traj.ids), '| action tokens:', int(traj.action_mask.sum()), '| reward:', traj.reward)
print(f'{"col j":>5}  {"token j+1":<12}{"owner":<10}{"mask":>4}')
for j in range(16):
    tok_text = tok.itos[int(traj.ids[j + 1])]
    print(f'{j:>5}  {tok_text:<12}{names[int(traj.owner[j + 1])]:<10}{int(traj.action_mask[j]):>4}')
```

Output:

```text
T = 76 | action tokens: 13 | reward: 1.0
col j  token j+1   owner     mask
    0  issue       prompt       0
    1  :           prompt       0
    2  clamp       prompt       0
    3  returns     prompt       0
    4  wrong       prompt       0
    5  results     prompt       0
    6  <call>      policy       1
    7  read        policy       1
    8  </call>     policy       1
    9  <obs>       template     0
   10  def         tool         0
   11  clamp       tool         0
   12  (           tool         0
   13  value       tool         0
   14  ,           tool         0
   15  lower       tool         0
```

The issue occupies columns 0 to 5, so the first action token, `<call>`, is at column 6, and the first observation begins after `</call>`. Each column describes the token that the logits at that position predict, which is the convention `sequence_log_probs` uses. The oracle's episode is 76 tokens long, of which 13 are actions: four calls, three of three tokens and the edit of four.

![Figure 5.2: Token ownership in the oracle's clamp trajectory, with the action mask below. Only policy tokens (blue) have mask 1.](../figures/ch05-ownership.svg)

Figure 5.2 draws the whole trajectory. The blue policy spans are short and regular, and the solid-outlined tool spans vary from one token (`ok`) to about twenty (the source listing). The blue rule under the strip, the action mask, marks exactly the policy cells. That is the invariant every later loss relies on: gradients flow only through tokens the policy sampled.

We check the invariant and the lengths across all four scripted policies.

```python
for kind in ['oracle', 'blind', 'random', 'hacker']:
    policy = arl.scripted_policy(kind, seed=0)
    trajs = [scripted_rollout(policy, arl.BugFixEnv(), t, tok) for t in tasks for _ in range(3)]
    ok = all(torch.equal(t.action_mask.bool(), t.owner[1:] == OWNER['policy']) for t in trajs)
    lengths = torch.tensor([len(t.ids) for t in trajs]).float()
    actions = torch.tensor([t.action_mask.sum() for t in trajs])
    print(f'{kind:<7} length mean {lengths.mean():5.1f} max {lengths.max():4.0f} | action tokens mean {actions.mean():4.1f}',
          f'| mask == policy owner: {ok}')
```

Output:

```text
oracle  length mean  67.9 max   76 | action tokens mean 13.0 | mask == policy owner: True
blind   length mean  17.6 max   18 | action tokens mean  7.0 | mask == policy owner: True
random  length mean  66.2 max  156 | action tokens mean 17.1 | mask == policy owner: True
hacker  length mean  40.8 max   48 | action tokens mean 10.0 | mask == policy owner: True
```

Episodes are short: the oracle uses about 70 tokens, and even the random policy, which often reads the file several times, stays well below 200. Action tokens are a fifth to two fifths of each sequence. These lengths keep TinyAgent's training fast, and they fit comfortably in its 256-position context.

## 5.7 Sampling rollouts

Now the policy writes the calls. `generate_call` samples one call token by token with the KV-free loop of Section 3.5, stopping at `</call>` or after `MAX_CALL_TOKENS`. It records each token's log-probability under the sampling distribution. `rollout` repeats that until the episode ends. Both work with any model that maps ids `[B, T]` to logits `[B, T, V]`.

```python
#@save
def generate_call(model, ids, tok, temperature=1.0, generator=None):
    """Sample one call after prefix ids (list) -> (call ids, sampling log-probs)."""
    call_ids, call_logp = [], []
    with torch.no_grad():
        for _ in range(MAX_CALL_TOKENS):
            logits = model(torch.tensor([ids + call_ids]))[:, -1]           # [1, V]
            next_id, logp = arl.sample_next(logits, temperature, generator=generator)
            call_ids.append(int(next_id))
            call_logp.append(float(logp))
            if call_ids[-1] == tok.call_end_id:
                break
    return call_ids, call_logp


def rollout(model, env, task, tok, max_turns=6, temperature=1.0, generator=None):
    """Sample one episode token-in/token-out -> Trajectory (ids, action_mask, sample_logp, reward, info)."""
    episode = Episode(env, task, tok, max_turns, getattr(model, 'max_len', 256))
    while not episode.done:
        episode.act(*generate_call(model, episode.ids, tok, temperature, generator))
    return make_trajectory(episode)
```

What does a freshly initialized TinyAgent do in the gym? Chapter 3 found its next-token distribution is close to uniform over the vocabulary, so we expect it to write nonsense. We roll out one episode and print it.

```python
torch.manual_seed(0)
model = arl.TinyAgent(tok.vocab_size)
gen = torch.Generator().manual_seed(0)
traj = rollout(model, arl.BugFixEnv(), tasks[0], tok, generator=gen)
print(tok.decode(traj.ids[:40].tolist()).replace('\n', ' | '), '...')
print({k: traj.info[k] for k in ['stop', 'n_turns', 'n_invalid']}, '| reward:', traj.reward)
```

Output:

```text
<bos> issue : clamp returns wrong results 117 38 qualifies b 129 returns <obs> error : malformed call </obs> 72 goal lower t bound 31 <obs> error : malformed call </obs> submit test -15 112 steps submitted <obs> error : ...
{'stop': 'budget', 'n_turns': 6, 'n_invalid': 6} | reward: 0.0
```

The untrained policy emits six random words per turn, never closes a call, and receives `error : malformed call` six times until the budget ends the episode. We measure the same behavior over 64 episodes, and check that the stored sampling log-probabilities equal what a trainer recomputes from the finished sequences.

```python
gen = torch.Generator().manual_seed(1)
trajs = [rollout(model, arl.BugFixEnv(), tasks[i % len(tasks)], tok, generator=gen) for i in range(64)]
turns = sum(t.info['n_turns'] for t in trajs)
print('malformed calls:', sum(t.info['n_invalid'] for t in trajs), 'of', turns, '| mean reward:',
      sum(t.reward for t in trajs) / 64, '| mean length:', sum(len(t.ids) for t in trajs) / 64)
gaps = []
for t in trajs[:8]:
    with torch.no_grad():
        recomputed = arl.sequence_log_probs(model, t.ids[None])[0]           # [T-1]
    gaps.append(((recomputed - t.sample_logp) * t.action_mask).abs().max())
print(f'max |recomputed - sampled| on action tokens: {max(gaps):.1e}')
```

Output:

```text
malformed calls: 384 of 384 | mean reward: 0.0 | mean length: 72.546875
max |recomputed - sampled| on action tokens: 9.5e-07
```

All 384 calls of the 64 episodes are malformed, and the reward is 0 everywhere. A policy-gradient method learns from differences in reward between samples, and here there are none: every advantage is zero and the gradient is exactly zero. This is the motivation for the rest of the book in one number. RL needs a policy that already succeeds sometimes. Chapter 6 starts from a TinyAgent that first imitates well-formed scripted calls, the toy counterpart of starting RL from a pretrained model, and Chapter 15 treats supervised warm starts in depth. The last line confirms token-in/token-out: on action tokens, the trainer's recomputation matches the sampler to fp32 rounding.

The single-episode rollout runs one forward pass per generated token. RL samples many episodes per step, so we also save a **batched** rollout that advances many episodes in lockstep. Each step pads the active sequences on the right, runs one forward pass, and reads each row's logits at its own last position. Right padding is safe under a causal mask, because padding positions come after the position we read. An episode that finishes a call steps its environment, and an episode that ends leaves the batch.

```python
#@save
def pad_rows(rows, length, value):
    """Right-pad 1-D tensors to a common length -> [B, length]."""
    return torch.stack([F.pad(r, (0, length - len(r)), value=value) for r in rows])


def rollout_batch(model, envs, tasks, tok, max_turns=6, temperature=1.0, generator=None):
    """Sample len(tasks) episodes in lockstep (one forward per token step) -> list of Trajectory."""
    max_len = getattr(model, 'max_len', 256)
    episodes = [Episode(env, task, tok, max_turns, max_len) for env, task in zip(envs, tasks)]
    calls = [([], []) for _ in episodes]
    active = list(range(len(episodes)))
    while active:
        rows = [torch.tensor(episodes[b].ids + calls[b][0]) for b in active]
        last = torch.tensor([len(r) - 1 for r in rows])
        with torch.no_grad():
            logits = model(pad_rows(rows, int(last.max()) + 1, tok.pad_id))  # [A, T, V]
        next_ids, logps = arl.sample_next(logits[torch.arange(len(rows)), last], temperature, generator=generator)
        for b, i, lp in zip(active, next_ids.tolist(), logps.tolist()):
            calls[b][0].append(i)
            calls[b][1].append(lp)
            if i == tok.call_end_id or len(calls[b][0]) == MAX_CALL_TOKENS:
                episodes[b].act(*calls[b])
                calls[b] = ([], [])
        active = [b for b in active if not episodes[b].done]
    return [make_trajectory(e) for e in episodes]
```

The batched rollout samples from the same policy, but it consumes the random stream in a different order, so its episodes differ from the sequential ones for the same seed. We compare speed and check the log-probabilities again.

```python
batch_tasks = [tasks[i % len(tasks)] for i in range(64)]
start = time.time()
seq = [rollout(model, arl.BugFixEnv(), t, tok, generator=torch.Generator().manual_seed(2)) for t in batch_tasks]
t_seq = time.time() - start
start = time.time()
bat = rollout_batch(model, [arl.BugFixEnv() for _ in batch_tasks], batch_tasks, tok,
                    generator=torch.Generator().manual_seed(2))
t_bat = time.time() - start
print('batched faster than sequential:', t_bat < t_seq, '| same episode count:', len(bat) == len(seq))
with torch.no_grad():
    lp = arl.sequence_log_probs(model, bat[5].ids[None])[0]
print(f'batched: max |recomputed - sampled| = {((lp - bat[5].sample_logp) * bat[5].action_mask).abs().max():.1e}')
```

Output:

```text
batched faster than sequential: True | same episode count: True
batched: max |recomputed - sampled| = 4.8e-07
```

On the check host, 64 sequential rollouts of the untrained model took about 1.5 seconds and the batched version about 0.2 seconds while this chapter was written. Those timings vary by machine and are not checked by the build. The batched rollout's log-probabilities agree with recomputation as well. The tiny residual comes from running the same weights on differently shaped batches, the toy version of the batch-size effects that Chapter 13 measures in inference engines.

The last step turns a list of trajectories into batch tensors. `collate` right-pads every field to the longest episode: ids with the pad id, masks with 0, log-probabilities with 0, and turn indices with -1. `attention_mask` marks real tokens. Under a causal model, right padding never affects real positions, so TinyAgent ignores it, but losses and metrics need it. Every `[B, T-1]` field is aligned with `sequence_log_probs`.

```python
#@save
def collate(trajs, pad_id):
    """Pad trajectories -> ids, attention_mask [B, T]; action_mask, sample_logp, turn_ids [B, T-1]; rewards [B]."""
    T = max(len(t.ids) for t in trajs)
    return dict(ids=pad_rows([t.ids for t in trajs], T, pad_id),
                attention_mask=pad_rows([torch.ones(len(t.ids)) for t in trajs], T, 0.0),
                action_mask=pad_rows([t.action_mask for t in trajs], T - 1, 0.0),
                sample_logp=pad_rows([t.sample_logp for t in trajs], T - 1, 0.0),
                turn_ids=pad_rows([t.turn[1:] for t in trajs], T - 1, -1),
                rewards=torch.tensor([float(t.reward) for t in trajs]))
```

A GRPO step (Chapter 9) samples a **group** of $G$ episodes on the same task and compares their rewards. We collate a group of four oracle demonstrations of different tasks with four random-model rollouts, so the lengths differ, and verify that the masked log-probabilities match a batched recomputation.

```python
group = [scripted_rollout(arl.scripted_policy('oracle'), arl.BugFixEnv(), t, tok) for t in tasks[:4]] + bat[:4]
batch = collate(group, tok.pad_id)
for name, value in batch.items():
    print(f'{name:<15} {str(tuple(value.shape)):<10} {value.dtype}')
with torch.no_grad():
    logp = arl.sequence_log_probs(model, batch['ids'])                       # [B, T-1]
gap = ((logp - batch['sample_logp']) * batch['action_mask'])[4:].abs().max()
print(f'rewards: {batch["rewards"].tolist()} | sampled rows max gap: {gap:.1e}')
print('padding positions with mask 1:', int((batch['action_mask'] * (1 - batch['attention_mask'][:, 1:])).sum()))
```

Output:

```text
ids             (8, 76)    torch.int64
attention_mask  (8, 76)    torch.float32
action_mask     (8, 75)    torch.float32
sample_logp     (8, 75)    torch.float32
turn_ids        (8, 75)    torch.int64
rewards         (8,)       torch.float32
rewards: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] | sampled rows max gap: 4.8e-07
padding positions with mask 1: 0
```

The batch has the shapes the training chapters expect: `[8, 76]` for ids and attention, and `[8, 75]` for the three predicted-position fields. The oracle rows have reward 1 and the random rows 0. The oracle rows' `sample_logp` is the scripted placeholder, so the log-probability check uses only the sampled rows, where recomputation over the padded batch still matches. No padding position has action mask 1.

## 5.8 Design problems in practice

> **Problem: context growth across turns.** *Symptom:* step time and memory grow over training while the number of episodes is unchanged, and late turns of long episodes are slow to sample. *Cause:* every turn appends the full observation to the context. The sequence length is the sum of all turns, attention cost grows quadratically with it, and without a KV cache sampling cost grows quadratically too (Section 3.5). Agent RL tends to lengthen episodes as policies learn to gather more information. *Options:* (a) cap observations (Section 4.8), which loses information. (b) Cap turns, which changes the task. (c) Summarize or drop old observations, which changes the policy's input distribution (Chapter 12). (d) Pay for it with token-budget batching (Chapter 14). *Default:* log the length per turn from the start, and set the context limit from measured lengths, not guesses. The next block shows how the context grows turn by turn in the oracle's episode.

```python
traj_oracle = scripted_rollout(arl.scripted_policy('oracle'), arl.BugFixEnv(), tasks[0], tok)
prompt_len = int((traj_oracle.turn == -1).sum())
ends = [int((traj_oracle.turn == k).nonzero().max()) + 1 for k in range(4)]
print('prompt:', prompt_len, '| context length after each turn:', ends,
      '| positions processed without a cache:', sum(range(prompt_len, ends[-1])))
```

Output:

```text
prompt: 7 | context length after each turn: [35, 42, 73, 76] | positions processed without a cache: 2829
```

Each turn adds its call and its observation, so the context grows from a prompt of 7 tokens to 76. Without a cache, scoring each new token reprocesses everything before it.

> **Problem: truncation policy shapes the training distribution.** *Symptom:* a growing fraction of episodes ends with `stop='truncated'`, and the policy learns to be terse or to submit early. *Cause:* a truncated episode has no outcome, and the rollout assigns it reward 0. If long episodes are systematically truncated, RL learns that length itself is bad, whatever the episode was doing. *Options:* (a) reward 0, which is simple and biased toward short episodes. (b) Drop truncated episodes from the batch, which is unbiased for the episodes kept but filters the data toward short ones. (c) Bootstrap a value estimate at the cut (Chapter 7), which needs a critic. *Default:* track the truncation rate as a first-class metric, keep it near zero by sizing the context, and treat a rising rate as a bug. The next block forces truncation with a model whose context is only 48 positions.

```python
small = arl.TinyAgent(tok.vocab_size, max_len=48)
cut = [rollout(small, arl.BugFixEnv(), t, tok, generator=torch.Generator().manual_seed(3)) for t in tasks[:10]]
print('stops:', sorted({t.info['stop'] for t in cut}), '| max length:', max(len(t.ids) for t in cut),
      '| rewards:', sum(t.reward for t in cut))
```

Output:

```text
stops: ['truncated'] | max length: 37 | rewards: 0.0
```

Every episode stops as truncated before the turn budget runs out, and none exceeds the context.

> **Problem: template changes between SFT and RL.** *Symptom:* reward collapses in the first RL steps after a successful SFT stage, or the step-0 importance ratio is far from 1. *Cause:* the SFT data was rendered with one template (or one thinking mode, Section 5.3) and RL rollouts with another, so the RL policy sees prefixes that differ from those it was trained on. Template details that look cosmetic, such as JSON spacing or an empty `<think>` block, are distribution shift at the token level. *Options:* (a) pin the tokenizer and template versions in the run configuration, which is cheap. (b) Re-render SFT data with the RL template, which costs a data pass. (c) Add a check that renders a fixed conversation with both pipelines and compares token ids. *Default:* (a) and (c). A one-line diff of rendered ids catches these errors before any training.

> **Problem: stripping reasoning from history.** *Symptom:* multi-turn runs where the step-0 ratio is fine for the last turn and not for earlier turns, or where learning on long conversations is unstable. *Cause:* templates that remove earlier reasoning (Section 5.3) make the final rendering differ from the context each turn was sampled in. Scoring all turns in one forward pass over the final rendering evaluates earlier turns in the wrong context. *Options:* (a) store each turn's tokens and context as sampled and score each turn against its own context, which costs one forward pass per turn. (b) Keep the reasoning in history during training, which makes the training context differ from deployment. (c) Train only on the last turn, which discards data. *Default:* score turns against the context they were sampled in. Within one tool-calling request, Qwen3's template is prefix-stable, so a single pass is correct there.

> **Problem: mask bugs that silently train on tool output.** *Symptom:* the policy starts to write text that looks like observations, such as test results or file contents, after its calls. The loss looks normal. *Cause:* the action mask is misaligned with the log-probabilities, usually because one of them was shifted and the other was not. *Options:* (a) build masks from per-token ownership in one place (as `make_trajectory` does), which is cheap. (b) Assert in the trainer that every masked token was sampled, using the ownership tensor. *Default:* both. The next block shows what an unshifted mask selects.

```python
wrong = (traj_oracle.owner[:-1] == OWNER['policy']).float()        # forgot the shift
picked = traj_oracle.owner[1:][wrong.bool()]
print('correct mask selects owners:', sorted({names[int(o)] for o in traj_oracle.owner[1:][traj_oracle.action_mask.bool()]}))
print('unshifted mask selects owners:', {names[int(o)]: int((picked == o).sum()) for o in picked.unique()})
```

Output:

```text
correct mask selects owners: ['policy']
unshifted mask selects owners: {'template': 3, 'policy': 9}
```

The correctly shifted mask selects the 13 policy tokens and nothing else. The unshifted mask selects 12 positions, and 3 of them are wrong: the first token of every call (`<call>`) drops out of the loss, the final `</call>` falls off the end, and for each of the first three calls the slot after `</call>` now trains the `<obs>` template token. Nothing crashes, and the policy is trained to predict that an observation begins after its own call. With longer observations and a mask built from tool spans, the same off-by-one trains on tool output.

> **Problem: stop-token handling.** *Symptom:* calls that never end, episodes that always hit the call-length limit, or a model that keeps generating after its tool call. *Cause:* the stop token (`</call>` here, `<|im_end|>` for Qwen3) is generated by the policy and must be in the action mask. If the rollout stops on it but does not record it, or records it with mask 0, the policy never receives a gradient for stopping. *Options:* (a) always include the sampled stop token in the trajectory with its log-probability, as `generate_call` does. (b) Inject the stop token as a template token when the call-length limit is hit, which ends the call cleanly but adds tokens the policy did not sample. *Default:* (a). If a limit cuts a call off, keep it unterminated and let the parser reject it, as the BugFix gym does, so the policy learns that unterminated calls fail.

## Summary

- A multi-turn episode becomes one token sequence whose tokens are owned by the template, the prompt, the policy, or tools. The action mask selects policy tokens. It is aligned with the predicted positions `ids[:, 1:]`, as are the stored sampling log-probabilities.
- Qwen3-0.6B's template renders tool schemas as a system preamble of a few hundred tokens, observations as user turns in `<tool_response>` tags, and calls as JSON in `<tool_call>` tags. It keeps reasoning within a tool-calling request and drops it once a new user message arrives, which breaks prefix stability.
- Decoding and re-encoding generated tokens can change them. Measured here: one of eight 128-token Qwen3 samples drifted, and re-rendering a compact JSON call changed 19 tokens into 21. Rollouts must be token-in/token-out.
- TinyAgent's word-level tokenizer has 316 entries and covers all gym text. Episodes are about 70 tokens for the oracle and stay under 200 for every scripted policy.
- `rollout` and `rollout_batch` record sampled ids, sampling log-probabilities, and masks. `collate` pads them into `ids`, `attention_mask` `[B, T]`, `action_mask`, `sample_logp`, `turn_ids` `[B, T-1]`, and `rewards` `[B]`. An untrained TinyAgent emits only malformed calls and earns reward 0, so RL needs a starting policy that sometimes succeeds.

## Exercises

1. Render the episode of Section 5.2 with `enable_thinking=False` and without `reasoning_content`. Count tokens per owner. Which template tokens would wrongly enter the action mask if you labeled everything after the assistant header as policy?
2. Repeat the drift experiment of Section 5.4 with 32 samples and temperatures 0.6 and 1.0. Does lowering the temperature reduce drift? Explain in terms of which tokens drift.
3. Change an assumption: let the rollout append the observation after the final call as well. Which fields of the trajectory change, and does any loss in Part II change its value? What changes in the cost?
4. Write `assert_token_in_token_out(traj, tok)` that fails if any action span, decoded and re-encoded, differs from the stored ids. Why can it never fail for TinyAgent's tokenizer, and what would you check for Qwen3 instead?
5. Modify `collate` to pad on the left instead of the right. What else must change for TinyAgent to compute the same log-probabilities (consider positions and the causal mask)? Why do inference engines often prefer left padding for generation?
6. Design review: your inference service returns completions as text through a chat API, and the trainer re-tokenizes them. A colleague argues that drift is too rare to matter. Design a measurement that bounds its effect on the gradient, and propose the minimum change to the serving interface that removes the problem.
