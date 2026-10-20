"""Figures for Chapter 13: simulated sync/async timelines, staleness vs throughput, log-prob mismatch.

The simulator figures call arl.simulate_pipeline with the chapter's settings, so drawn intervals are
the simulated events themselves.
"""
import torch

torch.set_num_threads(4)

import agentrl as arl
from svgplot import Figure, esc

OPACITY = [0.95, 0.6, 0.32]


def _gantt(fig, events, x0, y0, width, n_slots, t_max, label):
    lane, gap = 22, 4
    scale = width / t_max
    fig.text(x0, y0 - 10, label, cls='t', size=13, weight=600)
    for s in range(n_slots):
        fig.text(x0 - 8, y0 + s * lane + 15, f'slot {s}', cls='t2', size=11, anchor='end')
    ty = y0 + n_slots * lane + gap
    fig.text(x0 - 8, ty + 15, 'trainer', cls='t2', size=11, anchor='end')
    fig.body.append(f'<rect x="{x0}" y="{y0 - 2}" width="{width}" height="{n_slots * lane + lane + gap + 4}" '
                    f'fill="var(--panel)" rx="4"/>')
    for kind, slot, start, end, version in events:
        if start >= t_max:
            continue
        x, w = x0 + start * scale, max(1.5, (min(end, t_max) - start) * scale)
        if kind == 'sync':
            fig.body.append(f'<rect x="{x:.1f}" y="{y0 - 2}" width="{w:.1f}" height="{n_slots * lane + lane + gap + 4}" '
                            f'fill="var(--s4)" fill-opacity="0.35"><title>weight sync to v{version + 1}</title></rect>')
        elif kind == 'train':
            fig.body.append(f'<rect x="{x:.1f}" y="{ty + 2}" width="{w:.1f}" height="{lane - 4}" rx="3" fill="var(--s3)">'
                            f'<title>train step on v{version}</title></rect>')
            if w > 40:
                fig.text(x + w / 2, ty + 15, f'train v{version}', cls='t', size=10, anchor='middle')
        else:
            op = OPACITY[version % 3]
            tip = esc(f'slot {slot}: episode started on v{version}, {start:.0f}-{end:.0f} s')
            fig.body.append(f'<rect x="{x:.1f}" y="{y0 + slot * lane + 2}" width="{w:.1f}" height="{lane - 4}" rx="3" '
                            f'fill="var(--s1)" fill-opacity="{op}" stroke="var(--surface)" stroke-width="1"><title>{tip}</title></rect>')
            if w > 26:
                fig.text(x + w / 2, y0 + slot * lane + 15, f'v{version}', cls='t' if op < 0.5 else 'bg',
                         size=10, anchor='middle')
    for k in range(0, int(t_max) + 1, 100):
        px = x0 + k * scale
        fig.body.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{ty + lane + 2}" y2="{ty + lane + 6}"/>')
        fig.text(px, ty + lane + 18, f'{k}', cls='m', size=10, anchor='middle')


def timelines():
    durations = arl.agent_episode_durations(65_536, torch.Generator().manual_seed(0))
    sync = arl.simulate_pipeline(durations, 8, 8, 60.0, 10.0, max_staleness=0, steps=4)
    asyn = arl.simulate_pipeline(durations, 8, 8, 60.0, 10.0, max_staleness=1, steps=4)
    t_max = sync['time']
    fig = Figure(1000, 610, title='Synchronous vs asynchronous rollout (simulated, same episode durations)',
                 subtitle=f'8 slots, 8 episodes per step, 60 s train, 10 s sync. Sync finishes 4 steps at '
                          f'{sync["time"]:.0f} s; async (eta = 1, partial rollouts) at {asyn["time"]:.0f} s.')
    _gantt(fig, sync['events'], 90, 100, 880, 8, t_max, 'Synchronous: wait for the slowest episode, then train')
    _gantt(fig, asyn['events'], 90, 360, 880, 8, t_max, 'Asynchronous, eta = 1: slots refill immediately; '
                                                         'trainer takes the oldest 8 finished')
    fig.text(530, 600, 'time (s)', cls='lab', size=12, anchor='middle')
    end = 90 + asyn['time'] / t_max * 880
    fig.text(end + 8, 390, f'4 steps done at {asyn["time"]:.0f} s', cls='t2', size=12)
    fig.save('ch13-timelines')


def staleness_throughput():
    durations = arl.agent_episode_durations(65_536, torch.Generator().manual_seed(0))
    etas = [0, 1, 2, 3, 4, 6, 8]
    runs = {mode: [arl.simulate_pipeline(durations, 64, 64, 60.0, 10.0, max_staleness=e, interrupt=mode == 'partial',
                                         steps=20) for e in etas] for mode in ['partial', 'drain']}
    fig = Figure(1000, 420, title='What a staleness bound buys and costs (simulated pipeline)',
                 subtitle='64 slots, 64 episodes per step, 60 s train, 10 s sync, 20 steps; eta = 0 is synchronous.')
    ax = fig.axes(0, 0, 0.5, 1, xlabel='staleness bound eta', ylabel='episodes trained per hour', xlim=(0, 9.6))
    ax.line(etas, [r['throughput'] for r in runs['partial']], label='partial', series=1, markers=True)
    ax.line(etas, [r['throughput'] for r in runs['drain']], label='drain', series=2, markers=True)
    ax.legend()
    bx = fig.axes(0.5, 0, 0.5, 1, xlabel='staleness bound eta', ylabel='staleness of trained episodes (versions)',
                  xlim=(0, 9.6))
    bx.line(etas, [r['staleness_max'] for r in runs['partial']], label='max', series=2, markers=True, dashed=True)
    bx.line(etas, [r['staleness_mean'] for r in runs['partial']], label='mean', series=1, markers=True)
    bx.line(etas, etas, label='bound', series=3, end_label=True)
    bx.legend()
    fig.save('ch13-staleness-throughput')


FIGURES = {'timelines': timelines, 'staleness_throughput': staleness_throughput}


def _qwen_scores():
    """The chapter's Section 13.6 measurement: engine-style sampling vs trainer-style recomputation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qtok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    q16 = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16).to(device).eval()
    q32 = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).to(device).eval()
    prompts = [f'{t.issue}\n{t.source}# the bug is' for t in arl.make_tasks('all')]
    enc = qtok(prompts, return_tensors='pt', padding=True, padding_side='left').to(device)
    attn, gen = enc.attention_mask, torch.Generator(device=device).manual_seed(0)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    new_ids, new_logp, past, step = [], [], None, enc.input_ids
    with torch.no_grad():
        for _ in range(32):
            out = q16(input_ids=step, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
            logp = torch.log_softmax(out.logits[:, -1].float(), -1)
            step = torch.multinomial(logp.exp(), 1, generator=gen)
            new_ids.append(step)
            new_logp.append(logp.gather(-1, step))
            past, pos = out.past_key_values, pos[:, -1:] + 1
            attn = torch.cat([attn, torch.ones_like(step)], -1)
    new_ids, engine = torch.cat(new_ids, -1), torch.cat(new_logp, -1)
    prompt_ids = [row[m.bool()] for row, m in zip(enc.input_ids, enc.attention_mask)]

    def trainer(model, bs):
        seqs, out = [torch.cat([p, n]) for p, n in zip(prompt_ids, new_ids)], []
        for s0 in range(0, len(seqs), bs):
            chunk = seqs[s0:s0 + bs]
            T = max(len(x) for x in chunk)
            ids = torch.stack([torch.nn.functional.pad(x, (0, T - len(x)), value=qtok.pad_token_id) for x in chunk])
            am = torch.stack([torch.arange(T, device=device) < len(x) for x in chunk]).long()
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=am).logits
            for r, x in enumerate(chunk):
                P = len(x) - 32
                out.append(arl.selected_log_probs(logits[r, P - 1:len(x) - 1], x[P:]))
        return torch.stack(out)
    return dict(engine=engine.cpu(), bf16_1=trainer(q16, 1).cpu(), bf16_30=trainer(q16, 30).cpu(),
                fp32_1=trainer(q32, 1).cpu())


def mismatch():
    s = _qwen_scores()
    pairs = [('bf16 vs engine', s['bf16_1'] - s['engine'], 1),
             ('fp32 vs engine', s['fp32_1'] - s['engine'], 2),
             ('bf16 b1 vs b30', s['bf16_1'] - s['bf16_30'], 3)]
    fig = Figure(1000, 420, title='Training-inference mismatch on Qwen3-0.6B, identical weights',
                 subtitle='960 sampled tokens (30 gym prompts x 32 tokens). Engine: bf16, KV cache, left-padded batch. '
                          'Trainer: full forward pass.')
    ax = fig.axes(0, 0, 0.5, 1, xlabel='|log-prob difference| (nats, log scale)', ylabel='share of tokens below',
                  xlog=True, xlim=(1e-4, 1), ylim=(0, 1))
    for name, d, k in pairs:
        a = d.abs().flatten().clamp(min=1e-4).sort().values
        ax.line(a.tolist(), [(i + 1) / len(a) for i in range(len(a))], label=name, series=k, end_label=False)
    ax.legend()
    edges = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01]
    p = s['engine'].exp().flatten()
    bx = fig.axes(0.5, 0, 0.5, 1, xlabel='engine probability of the sampled token', ylabel='mean |difference| (nats)',
                  xlim=(0, 1))
    for name, d, k in pairs[:2]:
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            xs.append((lo + min(hi, 1)) / 2)
            ys.append(d.abs().flatten()[m].mean().item())
        bx.line(xs, ys, label=name, series=k, markers=True, end_label=False)
    bx.legend()
    fig.save('ch13-mismatch')


FIGURES['mismatch'] = mismatch
