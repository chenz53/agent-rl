# 5. 轨迹、模板与掩码

第 3 章构建了一个策略（policy），它把 token 序列映射为下一个 token 的概率。第 4 章构建了一个环境（environment），它把动作文本映射为观测文本。训练器（trainer）既不需要文本，也不需要“轮”的概念。它需要的是张量：每个回合（episode）一条 token 序列、每个策略 token 被采样时的对数概率（log-probability），以及一个掩码，标明哪些 token 由策略负责。本章就来搭建这座桥梁。

这座桥的故障模式比它的规模所暗示的要多。对话模板（chat template）决定模型看到哪些 token，而它渲染历史的方式可能与模型生成时不同。把采样得到的 token 解码为文本、再重新编码，可能得到不同的 token，于是一个基于文本工作的训练器会去计算模型从未采样过的 token 的概率。而掩码中一个差一（off-by-one）错误，就会悄无声息地训练策略去模仿环境。我们先在真实的 Qwen3-0.6B 分词器（tokenizer）和模型上逐一追踪这些问题，然后为 TinyAgent 构建分词器、`Trajectory` 记录、一个精确记录采样内容的 `rollout`，以及一个把一组 rollout 转换为批次张量的 `collate`。此后每一个训练章节都以 `collate` 的输出为输入。

```python
#@save
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import agentrl as arl
```

本章保存的代码同样通过 `arl.` 调用前面章节的函数，因此这条导入属于保存块的一部分。其余导入只用于本章的演示。

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## 5.1 每个回合一条序列

多轮（multi-turn）回合交替包含由不同参与方写出的文本。在第 4 章的 `clamp` 回合中，问题描述（issue）来自任务，调用来自策略，源代码清单和测试结果来自工具。为了给策略的调用打分，训练器把所有内容拼接成一条 token 序列，并对其做一次因果前向计算，使每个策略 token 都以它之前的全部内容为条件，包括更早的观测（observation）。

序列中的每个 token 都有一个**所有者**（owner），即产生它的参与方：

- **template**（模板）token 由格式化代码插入：角色头、分隔符和序列起始 token。
- **prompt**（提示）token 是任务输入：系统消息、问题文本和工具描述。
- **policy**（策略）token 是 rollout 期间从模型中采样得到的。
- **tool**（工具）token 是环境返回的观测。

只有策略 token 才是动作（action）。**动作掩码**（action mask）在策略 token 上为 1，其余位置为 0。它决定哪些对数概率进入损失。其他所有 token 仍然通过注意力影响模型的条件分布，但模型不会被训练去生成它们。在工具 token 上训练，会教策略去预测工具输出，进而在之后幻觉出工具输出。在模板 token 上训练通常无害，但会把梯度浪费在模型从不做选择的文本上。

掩码必须与它所选取的对数概率对齐。`sequence_log_probs`（3.4 节）对被预测的位置 `ids[:, 1:]` 返回 `[B, T-1]` 个值，因此动作掩码的形状也是 `[B, T-1]`，第 $j$ 列回答的问题是“token $j+1$ 是否由策略采样？”。存储的采样对数概率同样如此。5.5 节会把这种对齐固化到数据结构中，使后面的章节都不必再记住这个偏移。

## 5.2 带工具的真实对话模板

生产模型看不到原始的轮次。它们看到的是**对话模板**的输出：对话模板是随分词器一起发布的一段程序（通常是 Jinja），把一个消息列表（以及可选的工具 schema 列表）渲染成一个带特殊 token 的字符串。要知道模型看到哪些 token、生成哪些 token，就必须运行真实的模板。我们用 Qwen3-0.6B 的模板渲染一个两轮的 `clamp` 回合。工具 schema 采用 4.7 节中 MCP 风格的 schema，并转换为模板所期望的函数调用格式。助手消息是**手工编写的**：还没有任何模型生成过它们。

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

这是渲染字符串中完整的对话部分。它前面列出工具 schema 的系统部分在下一个代码块中展示。从中可以看到模板做出的若干决定。工具结果被包裹在 `<tool_response>` 标签中，并放在 **user** 角色的轮次里，因此模型把观测当作用户输入来看待。每个助手轮次先在 `<think>` 标签内渲染推理，然后在 `<tool_call>` 标签内以 JSON 形式给出调用，并以 `<|im_end|>` 结束。渲染结果以一个打开的助手头结尾，即**生成提示**（generation prompt），模型在其后写出下一轮。

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

四个工具 schema 在每个回合中都带来几百个 token 的固定前导内容，而此时智能体（agent）还什么都没做。在这个很短的回合中，它约占三分之二。模板使用的每个分隔符都是单个 token，因此策略可以把输出 `<tool_call>` 学成一次决策，解析器也可以按 token id 而非字符串匹配来找到工具调用。

模板的输出必须按所有者切分。分词器会报告每个 token 覆盖的字符区间（`offset_mapping`），因此我们先给字符打标签，再让每个 token 取其第一个字符的标签。用户内容是 prompt，工具响应内容是工具输出，而每个助手轮次从其头部之后一直到它的 `<|im_end|>` 都是策略输出，因为在部署中这些内容全部由模型生成，包括 `<think>` 和结尾的 `<|im_end|>`。

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

该函数会跳过那些实际上是工具结果的用户轮次：其中的工具区间被标为 tool，包裹标签仍为 template。我们统计每个所有者的 token 数，并解码第一个助手轮次的策略 token。

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

这次渲染的大部分是模板：在这种标注下，系统前导内容算作模板，因为它由模板根据 schema 生成。问题描述是 37 个 prompt token，两个助手轮次是 55 个策略 token，两个工具结果只贡献了 23 个。在真实的 SWE 回合中，比例恰好相反：文件清单和测试日志占据了上下文的主体，策略 token 往往只是一小部分。无论哪种情况，动作掩码选取的都是一个小而不规则的位置子集。

![图 5.1：两轮 clamp 回合的 Qwen3-0.6B 渲染结果，逐 token 展示并按所有者用不同边框区分。包含工具 schema 的系统前导内容折叠为一个方框。](../../figures/ch05-qwen-template.svg)

图 5.1 按顺序展示了同样的 token。沿着边框看：虚线单元格是问题描述的 prompt，蓝色是策略轮次，每个都以 `<think>` 开始、以 `<|im_end|>` 结束，实线边框的是被无框模板 token 包裹的工具区间，末尾是打开的助手头。损失只使用蓝色单元格，注意力使用全部单元格。

gym 的工具是一种抽象（第 4.7 节）。生产环境中的智能体用 `Read`、`Edit` 和 `Bash` 完成同样的修复，自己写出替换文本，并以不调用工具直接回答来结束。用同一个模板渲染这个回合，可以看出这种抽象从 token 预算中去掉了什么。助手轮次同样是手写的。

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

把这些数字与上面的 gym 回合对比。六个通用工具的 schema 让固定前导更长：每个回合都要付出 585 个模板 token，而不是 350 个。策略写出 107 个 token 而不是 55 个，即使没有任何 `<think>` 推理；其中仅一次 `Edit` 调用就占 45 个。这 45 个 token 大多不是自由选择：`old_string` 必须逐字符精确复现源码，否则工具返回错误；而 gym 的 `edit` 调用只用寥寥几个 token 指定一个操作名。这就是 BugFix gym 为 TinyAgent 做出的取舍。已经能准确复制和编写代码的预训练模型可以在生产环境的工具接口上探索，因此它的强化学习信号关乎写*哪一个*编辑。从零训练的模型则必须先学会复制文本，而一次修复的奖励会分摊到几十个必须全部正确的 token 上。

## 5.3 模板对历史做了什么

模板每次都会渲染整段历史。这带来一个问题：在第 $k$ 步，模型看到的某个较早轮次的文本，可能与它在那一轮生成的文本不同。Qwen3 的模型卡写道“the historical model output should only include the final output part”，意思是较早的推理不应被带入后续上下文，而它的模板也是这样实现的（[Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)，2026-10-13 查阅）。我们来精确检查这在什么时候发生。

检验方法是**前缀稳定性**（prefix stability）。先把对话渲染到某一轮并带上生成提示，这恰好是模型生成该轮时看到的内容。然后渲染一段更长的对话，看第一次渲染是否是第二次渲染的前缀。如果是，那么对最终渲染结果做一次前向计算，就能复现每个较早轮次的上下文。

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

在同一个用户请求之内，历史是稳定的：较早的工具调用轮次中的推理会被保留，每次较早的渲染都是之后渲染的前缀。一旦出现新的用户消息，模板就会丢弃此前所有助手轮次中的推理，较早的渲染也就不再是前缀。一个重新渲染这种对话并为较早助手轮次打分的训练器，会在一个模型采样时从未见过、且缺少其推理的上下文中计算这些轮次的对数概率。

模板还有一个非思考模式的开关，它会改变生成提示本身：

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

在非思考模式下，模板自己会在助手头之后写入一个空的 `<think>\n\n</think>\n\n` 块。这些 token 归模板所有，因此它们属于上下文，但不属于动作掩码，尽管在思考模式下同样的 token 会归策略所有。所有权取决于在这一次具体的 rollout 中是谁产生了该 token，而不取决于 token 的文本。

## 5.4 重分词漂移

重分词漂移（retokenization drift）是多轮强化学习基础设施中后果最严重的隐患，因为它会静默失败。rollout 采样的是 token id。环境需要文本，于是这些 id 被解码。一种常见但错误的设计只保留文本，等到训练器之后需要 token id 时，再对文本重新编码，而且往往是先通过模板重新渲染。**一般而言，先解码再重新编码并不能复现采样得到的 id。** BPE 分词器对同一个字符串有许多种 token 序列。采样器可以产生其中任何一种，但编码总是返回一种规范切分。当两者不同时，训练器计算的对数概率、比率和梯度都针对一条模型从未采样过的序列。

第一种发生途径是模板。模型可能以紧凑 JSON 写出工具调用。如果系统把该调用解析成一条消息、再通过模板重新渲染，就会改变它的格式，如下面这个手工编写的例子所示。

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

模板在每个冒号后加一个空格来序列化参数，于是 19 个采样 token 变成了 21 个不同的 token。从第一个不同的 token 开始，之后的每个对数概率所指的都是一条模型并未产生的序列。

第二种途径根本不需要模板。我们从真实的 Qwen3-0.6B 以温度 1、不做截断，采样 8 条 128 token 的续写，把每条解码为文本，再对文本重新编码。

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

在这次运行中，8 个样本中有 1 个发生了漂移。模型把 `-` 和 `hi` 作为两个 token 输出。解码后的文本包含 `-hi`，而分词器对该文本的规范切分是 `-h` 后接 `i`。解码出的字符串完全相同，token 序列却不同。按 token 计，漂移很少见，但一个强化学习批次包含数百万个 token，而长的智能体轨迹使每个回合都可能中招。更糟的是，漂移集中在不常见的切分上，而这恰恰是那些对数概率对梯度影响最大的低概率样本。

解决办法是一种纪律而非一种算法，我们称之为 **token-in/token-out**。rollout 保留采样得到的 token id，把环境的观测以 token id 形式追加，并且从不重新编码策略产生的文本。文本只存在于边界上：解码后交给环境解析器的动作，以及环境返回的观测字符串。训练器接收的是 id、掩码和采样对数概率，从不接收文本。从这里开始我们构建的一切都遵循这条规则。

> **生产实践。** 通过 OpenAI 风格的对话 API 提供模型服务的系统拿回的是文本，除非推理引擎也返回 token id，否则两种漂移都会受到影响。这是一条一般性的工程观察；我们没有审查具体框架如何处理它。本书的建议是：把任何重新编码生成文本的路径都视为 bug，并通过在第 0 步比较训练器重新计算的对数概率与采样器的对数概率来检测它（3.7 节）。

## 5.5 TinyAgent 的分词器

TinyAgent 需要一个分词器，其词表（vocabulary）覆盖 gym 可能产生的每个字符串，同时序列要保持较短。词级分词器正合适。所有 gym 文本本来就是以空白分隔的单词（`read` 工具返回 4.4 节的紧凑投影），因此一个 token 就是一个单词，换行是特殊单词 `<nl>`，解码时用空格连接单词。词级分词器对每个字符串只有一种切分，因此不会发生漂移。这是玩具模型带来的便利，5.4 节中关于真实模型的教训仍然支配着设计。

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

词表通过枚举 gym 文本来构建，而不是靠猜测。对每个任务，我们收集问题描述、有 bug 源代码的 `read` 和 `test` 观测、每种编辑操作之后的相同观测，再加上每个格式正确的调用、每条固定消息，以及从 -20 到 150 的整数，这些整数覆盖了测试和错误编辑会产生的数值。传入所有任务，包括训练任务和留出任务（held-out tasks），使留出任务的函数名也有自己的 id。在见到留出任务之前，这些 id 一直未被训练，这就是罕见 token 在词级上的对应物。

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

词表有 316 个条目，因此 TinyAgent 的输出层大约是 316 乘 64。`clamp` 的 `read` 观测长 23 个 token（含一个 `<nl>`），并能精确解码还原。一个带参数的调用是四个 token，特殊 token 的 id 固定为 0 到 7（`<call>` 是 5，`</call>` 是 6），rollout 依赖这一点。词表之外的单词映射为 `<unk>`。下一个代码块在每个任务上运行第 4 章的每个脚本化策略，并编码每条观测，以检查 gym 从不产生 `<unk>`。

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

四个策略在全部 30 个任务上产生的所有观测 token 都在词表之内。整数范围是唯一一个可能被打破的假设：一个包含异常编辑的长回合可能做到这一点，例如对一个求和函数反复应用 `fix_range_end`。这样的值会被编码为 `<unk>`，但这对训练无害，因为观测从来不是动作。

## 5.6 轨迹与掩码

**轨迹**（trajectory）是一个回合在 token 层面的记录。它存储整条序列的 id，以及与被预测位置 `ids[1:]` 对齐的动作掩码和采样对数概率。它还存储奖励（reward）和环境最终的 `info`。另有两个与 `ids` 对齐的字段，记录每个 token 的所有者及其所属的轮次。第 7 章和第 12 章用轮次索引来做按轮的信用分配（credit assignment）。

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

`Episode` 以 5.4 节的 token-in/token-out 方式逐步构建一条轨迹。它以 `<bos>` 和编码后的问题描述开头。`act` 按采样原样追加策略调用的 id，只在把文本交给 `env.step` 时才解码它们，并把编码后的观测追加在 `<obs>` 和 `</obs>` 两个模板 token 之间。它还执行两条环境并不知道的限制：rollout 的轮数上限，以及模型的上下文长度 `max_len`。如果下一条观测和再一次调用放不下，回合就以**截断**（truncated）结束，奖励为 0。最后一次调用之后的观测不会被追加，因为不再有后续动作依赖于它。

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

`make_trajectory` 把这些列表转换为张量。对齐在这里集中完成：动作掩码就是“token $j+1$ 的所有者是策略”，采样对数概率则丢掉第一项，因为它属于 `<bos>`，而 `<bos>` 从不被预测。

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

`scripted_rollout` 让第 4 章的文本策略走同一条路径。它产生的轨迹是监督微调（SFT，第 15 章）的有效示范数据。它们的 `sample_logp` 为 0：对 oracle 这样的确定性策略，这是正确的对数概率，对其他策略则只是占位值，因此不能用在重要性比率（importance ratio）中。我们编码 oracle 的 `clamp` 回合，并打印前十六个被预测的位置。

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

问题描述占据第 0 到 5 列，因此第一个动作 token `<call>` 位于第 6 列，第一条观测从 `</call>` 之后开始。每一列描述的是该位置的 logits 所预测的 token，这正是 `sequence_log_probs` 采用的约定。oracle 的回合长 76 个 token，其中 13 个是动作：四次调用，三次各三个 token，编辑调用四个 token。

![图 5.2：oracle 的 clamp 轨迹中的 token 所有权，下方是动作掩码。只有策略 token（蓝色）的掩码为 1。](../../figures/ch05-ownership.svg)

图 5.2 画出了整条轨迹。蓝色的策略区间短而规整，实线边框的工具区间从一个 token（`ok`）到约二十个 token（源代码清单）不等。条带下方的蓝色横线即动作掩码，恰好标出了策略单元格。这就是后面每个损失都依赖的不变量：梯度只流经策略采样的 token。

我们在全部四个脚本化策略上检查这一不变量以及序列长度。

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

回合都很短：oracle 用约 70 个 token，即使是常常多次读取文件的随机策略，也远低于 200。动作 token 占每条序列的五分之一到五分之二。这样的长度使 TinyAgent 的训练很快，并能轻松放入其 256 个位置的上下文中。

## 5.7 采样 rollout

现在由策略来写调用。`generate_call` 用 3.5 节的无 KV 缓存循环逐 token 采样一次调用，在遇到 `</call>` 或达到 `MAX_CALL_TOKENS` 时停止。它记录每个 token 在采样分布下的对数概率。`rollout` 重复这一过程直到回合结束。两者都适用于任何把 id `[B, T]` 映射为 logits `[B, T, V]` 的模型。

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

刚初始化的 TinyAgent 在 gym 中会做什么？第 3 章发现它的下一 token 分布在词表上接近均匀，因此我们预期它会写出无意义的内容。我们 rollout 一个回合并打印出来。

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

未训练的策略每轮输出六个随机单词，从不闭合调用，六次收到 `error : malformed call`，直到预算耗尽结束回合。我们在 64 个回合上测量同样的行为，并检查存储的采样对数概率是否等于训练器从完成的序列重新计算出的值。

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

64 个回合的全部 384 次调用都格式错误，奖励处处为 0。策略梯度方法从样本之间的奖励差异中学习，而这里没有任何差异：每个优势（advantage）都为零，梯度恰好为零。这一个数字就概括了本书其余部分的动机：强化学习需要一个已经能偶尔成功的策略。第 6 章从一个先模仿格式正确的脚本化调用的 TinyAgent 出发，这是“从预训练模型开始做强化学习”的玩具版对应物，第 15 章会深入讨论监督式热启动。最后一行确认了 token-in/token-out：在动作 token 上，训练器的重新计算与采样器在 fp32 舍入误差内一致。

单回合 rollout 每生成一个 token 就做一次前向计算。强化学习每一步要采样很多回合，因此我们还保存一个**批量**（batched）rollout，让许多回合同步推进。每一步在右侧填充（padding）活跃序列，做一次前向计算，并在每一行自己的最后位置读取 logits。在因果掩码下右填充是安全的，因为填充位置位于我们读取的位置之后。完成一次调用的回合推进其环境，结束的回合离开批次。

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

批量 rollout 从同一个策略采样，但它消耗随机数流的顺序不同，因此在相同随机种子（seed）下，它的回合与逐个采样的回合不同。我们比较速度，并再次检查对数概率。

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

撰写本章时，在检查主机上，未训练模型的 64 次逐个 rollout 约用时 1.5 秒，批量版本约 0.2 秒。这些计时因机器而异，构建过程不检查它们。批量 rollout 的对数概率同样与重新计算一致。微小的残差来自用相同权重处理形状不同的批次，这是第 13 章在推理引擎中测量的批大小效应的玩具版。

最后一步把轨迹列表转换为批次张量。`collate` 把每个字段右填充到最长回合的长度：id 用 pad id 填充，掩码用 0，对数概率用 0，轮次索引用 -1。`attention_mask` 标出真实 token。在因果模型下，右填充从不影响真实位置，因此 TinyAgent 忽略它，但损失和指标需要它。每个 `[B, T-1]` 字段都与 `sequence_log_probs` 对齐。

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

一个 GRPO 步骤（第 9 章）在同一任务上采样一**组**（group）$G$ 个回合并比较它们的奖励。我们把四条不同任务上的 oracle 示范与四条随机模型的 rollout 组合在一起，使长度各不相同，并验证掩码后的对数概率与批量重新计算的结果一致。

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

这个批次具有训练章节所期望的形状：id 和注意力为 `[8, 76]`，三个被预测位置字段为 `[8, 75]`。oracle 行的奖励为 1，随机行为 0。oracle 行的 `sample_logp` 是脚本化占位值，因此对数概率检查只使用采样行，在这些行上，对填充后批次的重新计算仍然一致。没有任何填充位置的动作掩码为 1。

## 5.8 实践中的设计问题

> **问题：上下文随轮次增长。** *症状：* 回合数不变，但步骤耗时和显存随训练增长，长回合后期轮次的采样很慢。*原因：* 每一轮都把完整的观测追加到上下文。序列长度是所有轮次之和，注意力开销随之二次增长，而没有 KV 缓存时采样开销也是二次增长（3.5 节）。随着策略学会收集更多信息，智能体强化学习往往会使回合变长。*可选方案：* (a) 限制观测长度（4.8 节），这会丢失信息。(b) 限制轮数，这会改变任务。(c) 摘要或丢弃旧观测，这会改变策略的输入分布（第 12 章）。(d) 用按 token 预算组批来承担开销（第 14 章）。*默认做法：* 从一开始就记录每轮的长度，并根据实测长度而非猜测来设定上下文上限。下一个代码块展示 oracle 回合中上下文如何逐轮增长。

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

每一轮加入它的调用和观测，因此上下文从 7 个 token 的提示增长到 76 个。没有缓存时，为每个新 token 打分都要重新处理它之前的全部内容。

> **问题：截断策略塑造训练分布。** *症状：* 以 `stop='truncated'` 结束的回合比例不断上升，策略学会变得简短或过早提交。*原因：* 被截断的回合没有结果，rollout 给它奖励 0。如果长回合被系统性地截断，强化学习就会学到长度本身是坏的，无论该回合当时在做什么。*可选方案：* (a) 奖励 0，简单，但偏向短回合。(b) 从批次中丢弃截断回合，这对保留下来的回合是无偏的，但会把数据过滤得偏向短回合。(c) 在截断处用价值估计做自举（bootstrap）（第 7 章），这需要一个评论家（critic）。*默认做法：* 把截断率作为一等指标来跟踪，通过合理设定上下文大小使其保持在接近零的水平，并把截断率上升视为 bug。下一个代码块用一个上下文只有 48 个位置的模型强制触发截断。

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

每个回合都在轮数预算耗尽之前以截断结束，且没有一个超出上下文。

> **问题：SFT 与强化学习之间的模板变化。** *症状：* 在一个成功的 SFT 阶段之后，奖励在强化学习的最初几步就崩溃，或者第 0 步的重要性比率远离 1。*原因：* SFT 数据用一种模板（或一种思考模式，5.3 节）渲染，而强化学习的 rollout 用另一种，于是强化学习中的策略看到的前缀与它训练时见过的不同。看似只是表面上的模板细节，例如 JSON 空格或一个空的 `<think>` 块，在 token 层面都是分布偏移。*可选方案：* (a) 在运行配置中固定分词器和模板的版本，成本很低。(b) 用强化学习的模板重新渲染 SFT 数据，需要过一遍数据。(c) 增加一项检查：用两条流水线渲染同一段固定对话并比较 token id。*默认做法：* (a) 和 (c)。对渲染后的 id 做一行 diff，就能在任何训练开始前发现这类错误。

> **问题：从历史中剥离推理。** *症状：* 多轮运行中，第 0 步的比率对最后一轮正常、对较早轮次却不正常，或者在长对话上的学习不稳定。*原因：* 删除较早推理的模板（5.3 节）使最终渲染与每一轮采样时的上下文不同。在最终渲染上做一次前向计算来给所有轮次打分，会在错误的上下文中评估较早的轮次。*可选方案：* (a) 按采样时的原样存储每一轮的 token 和上下文，并针对各自的上下文给每一轮打分，代价是每轮一次前向计算。(b) 训练时在历史中保留推理，这会使训练上下文与部署不同。(c) 只在最后一轮上训练，这会丢弃数据。*默认做法：* 针对每一轮采样时的上下文为其打分。在一次工具调用请求之内，Qwen3 的模板是前缀稳定的，因此在那里单次前向计算是正确的。

> **问题：掩码 bug 静默地在工具输出上训练。** *症状：* 策略开始在调用之后写出看起来像观测的文本，例如测试结果或文件内容。损失看起来正常。*原因：* 动作掩码与对数概率没有对齐，通常是因为其中一个做了偏移而另一个没有。*可选方案：* (a) 在一个地方根据逐 token 的所有权构建掩码（如 `make_trajectory` 所做），成本很低。(b) 在训练器中利用所有权张量断言每个被掩码选中的 token 都是采样得到的。*默认做法：* 两者都做。下一个代码块展示一个未偏移的掩码会选中什么。

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

正确偏移的掩码恰好选中 13 个策略 token，别无其他。未偏移的掩码选中 12 个位置，其中 3 个是错的：每次调用的第一个 token（`<call>`）从损失中消失，最后一个 `</call>` 落到了末尾之外，而对前三次调用中的每一次，`</call>` 之后的那个位置现在在训练 `<obs>` 模板 token。程序不会崩溃，而策略被训练去预测在它自己的调用之后会开始一条观测。如果观测更长、掩码又是根据工具区间构建的，同样的差一错误就会在工具输出上训练。

> **问题：停止 token 的处理。** *症状：* 调用永不结束，回合总是触到调用长度上限，或者模型在工具调用之后继续生成。*原因：* 停止 token（这里是 `</call>`，Qwen3 是 `<|im_end|>`）由策略生成，必须在动作掩码中。如果 rollout 在遇到它时停止却没有记录它，或者记录了但掩码为 0，策略就永远得不到关于“停止”的梯度。*可选方案：* (a) 总是把采样得到的停止 token 连同其对数概率包含在轨迹中，如 `generate_call` 所做。(b) 在触到调用长度上限时把停止 token 作为模板 token 注入，这能干净地结束调用，但加入了策略并未采样的 token。*默认做法：* (a)。如果上限截断了一次调用，就让它保持未终止状态，交给解析器拒绝，就像 BugFix gym 所做的那样，使策略学到未终止的调用会失败。

## 小结

- 一个多轮回合变成一条 token 序列，其中的 token 分别归模板、提示、策略或工具所有。动作掩码选取策略 token。它与被预测位置 `ids[:, 1:]` 对齐，存储的采样对数概率也是如此。
- Qwen3-0.6B 的模板把工具 schema 渲染为几百个 token 的系统前导内容，把观测渲染为 `<tool_response>` 标签中的用户轮次，把调用渲染为 `<tool_call>` 标签中的 JSON。它在一次工具调用请求之内保留推理，一旦出现新的用户消息就将其丢弃，这破坏了前缀稳定性。
- 对生成的 token 解码再重新编码可能改变它们。本书测得：8 条 128 token 的 Qwen3 样本中有 1 条发生漂移，而重新渲染一个紧凑 JSON 调用把 19 个 token 变成了 21 个。rollout 必须遵循 token-in/token-out。
- TinyAgent 的词级分词器有 316 个条目，覆盖全部 gym 文本。oracle 的回合约 70 个 token，所有脚本化策略的回合都在 200 以内。
- `rollout` 和 `rollout_batch` 记录采样的 id、采样对数概率和掩码。`collate` 把它们填充为 `ids`、`attention_mask` `[B, T]`，`action_mask`、`sample_logp`、`turn_ids` `[B, T-1]`，以及 `rewards` `[B]`。未训练的 TinyAgent 只输出格式错误的调用，奖励为 0，因此强化学习需要一个有时能成功的初始策略。

## 练习

1. 用 `enable_thinking=False` 并去掉 `reasoning_content`，渲染 5.2 节的回合。统计每个所有者的 token 数。如果把助手头之后的所有内容都标为策略，哪些模板 token 会被错误地纳入动作掩码？
2. 用 32 个样本、温度 0.6 和 1.0 重复 5.4 节的漂移实验。降低温度会减少漂移吗？从哪些 token 发生漂移的角度加以解释。
3. 改变一个假设：让 rollout 在最后一次调用之后也追加观测。轨迹的哪些字段会改变？第二部分中的任何损失的值会改变吗？开销会有什么变化？
4. 编写 `assert_token_in_token_out(traj, tok)`：如果任何动作区间在解码再重新编码后与存储的 id 不同，就让它失败。为什么对 TinyAgent 的分词器它永远不会失败？对 Qwen3 你会改为检查什么？
5. 修改 `collate`，改为在左侧而非右侧填充。为了让 TinyAgent 计算出相同的对数概率，还必须改变什么（考虑位置和因果掩码）？为什么推理引擎在生成时常常偏好左填充？
6. 设计评审：你的推理服务通过对话 API 以文本形式返回补全结果，训练器再对其重新分词。一位同事认为漂移太少见，无关紧要。设计一个能界定其对梯度影响的测量方法，并提出能消除该问题的、对服务接口的最小改动。
