"""Figures for Chapter 5: Qwen3 template ownership and a TinyAgent trajectory ownership strip."""
import agentrl as arl
import hairline

OWNER_LEGEND = [('template', 'template'), ('cell-prompt', 'prompt'), ('cell-policy', 'policy'), ('cell', 'tool')]


def _save(sc, name, title):
    (hairline.OUT / f'{name}.svg').write_text(sc.svg(title))


def qwen_template():
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    tools = [{'type': 'function', 'function': {'name': s['name'], 'description': s['description'],
                                               'parameters': s['inputSchema']}} for s in arl.TOOL_SCHEMAS]
    call = lambda name, **a: [{'type': 'function', 'function': {'name': name, 'arguments': a}}]
    messages = [
        {'role': 'user', 'content': arl.CLAMP_ISSUE},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'Read the source first.', 'tool_calls': call('read')},
        {'role': 'tool', 'content': arl.compact_source(arl.CLAMP_BUGGY)},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'min and max are swapped.',
         'tool_calls': call('edit', op='swap_minmax')},
        {'role': 'tool', 'content': 'ok'},
    ]
    text = qtok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    label = ['template'] * len(text)
    for start, end, owner, include_end in [('<|im_start|>user\n', '<|im_end|>', 'prompt', False),
                                           ('<|im_start|>assistant\n', '<|im_end|>', 'policy', True),
                                           ('<tool_response>\n', '\n</tool_response>', 'tool', False)]:
        i = text.find(start)
        while i >= 0 and text.find(end, i + len(start)) >= 0:
            j = text.find(end, i + len(start)) + (len(end) if include_end else 0)
            if owner != 'prompt' or not text.startswith('<tool_response>', i + len(start)):
                label[i + len(start):j] = [owner] * (j - i - len(start))
            i = text.find(start, j)
    enc = qtok(text, return_offsets_mapping=True, add_special_tokens=False)
    owners = [label[a] for a, _ in enc.offset_mapping]
    pieces = [p.replace('Ġ', '·').replace('Ċ', '⏎') for p in qtok.convert_ids_to_tokens(enc.input_ids)]
    first_user = text.index('<|im_start|>user')
    k0 = next(k for k, (a, _) in enumerate(enc.offset_mapping) if a >= first_user)
    sc = hairline.Scene(1000, 420)
    sc.head('FIG 5.1 · QWEN3-0.6B CHAT TEMPLATE: WHO OWNS EACH TOKEN',
            'two-turn clamp episode with hand-authored assistant turns, rendered by apply_chat_template(tools=...)')
    sc.legend_cells(24, 66, OWNER_LEGEND)
    sc.card(24, 92, 380, 26, lines=[f'<|im_start|>system ... tool schemas ... ({k0} tokens)'], style='card-soft', mono=True, align='start')
    y = sc.wrapped_strip(24, 130, 952, list(zip(pieces[k0:], owners[k0:])))
    sc.label(24, y + 28, 'Loss uses only policy tokens (accent), which include <think>, the JSON call, and <|im_end|>; '
             'attention uses all of them.', cls='desc')
    _save(sc, 'ch05-qwen-template', 'Qwen3-0.6B chat template: who owns each token')


def ownership():
    tasks = arl.make_tasks('all')
    tok = arl.build_tokenizer(tasks)
    traj = arl.scripted_rollout(arl.scripted_policy('oracle'), arl.BugFixEnv(), tasks[0], tok)
    names = {v: k for k, v in arl.OWNER.items()}
    items = [(tok.itos[int(i)].replace('<nl>', '⏎'), names[int(o)]) for i, o in zip(traj.ids, traj.owner)]
    mask = [0] + traj.action_mask.int().tolist()          # mask of token j is action_mask[j-1]
    sc = hairline.Scene(1000, 270)
    sc.head(f'FIG 5.2 · TINYAGENT TRAJECTORY: ORACLE ON CLAMP ({len(traj.ids)} TOKENS)',
            'cells by owner; the accent rule under a cell marks action_mask = 1 (stored at column j-1)')
    sc.legend_cells(24, 66, OWNER_LEGEND)
    y = sc.wrapped_strip(24, 92, 952, items, mask=mask)
    sc.label(24, y + 30, f'{int(traj.action_mask.sum())} of {len(traj.ids)} tokens are actions; '
             f'reward {traj.reward:g} after submit.', cls='desc')
    _save(sc, 'ch05-ownership', 'TinyAgent trajectory ownership')


FIGURES = {'ch05-qwen-template': qwen_template, 'ch05-ownership': ownership}
