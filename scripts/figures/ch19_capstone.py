"""Figures for Chapter 19: the training dashboard, the ablation forest plot, and the component map.

The runs repeat the chapter's code with the same seeds and thread count, so plotted numbers match the
printed ones.
"""
import copy
import random

import torch

import agentrl as arl
from svgplot import Figure
import hairline

torch.set_num_threads(4)
_C = {}
STEPS, EVERY, N_VAL, SEEDS, ABL_STEPS = 60, 10, 16, [0, 1, 2], 30


def _setup():
    if not _C:
        tok = arl.build_tokenizer(arl.make_tasks('all'))
        train, held = arl.make_tasks('train'), arl.make_tasks('heldout')
        cfg = arl.RLConfig()
        base = arl.make_rl_base(tok)
        rng = random.Random(0)
        demos = [arl.scripted_rollout(arl.scripted_policy('oracle'), arl.make_env(cfg), arl.randomize_names(t, rng), tok)
                 for t in train]
        sft = copy.deepcopy(base)
        arl.sft_train(sft, demos, tok, steps=10, lr=1e-3, batch_size=16)
        _C.update(tok=tok, train=train, held=held, val=held[0::2], cfg=cfg, base=base, sft=sft)
    return _C


def _train_run(model, cfg, steps, every=0, on_checkpoint=None):
    c = _setup()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env_factory = lambda: arl.make_env(cfg)
    ref_model = arl.make_ref(model, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for step in range(steps):
        kept, sampled = arl.sample_step(model, env_factory, c['train'], c['tok'], cfg, gen)
        batch = arl.collate(kept, c['tok'].pad_id)
        batch['n_turns'] = torch.tensor([float(t.info['n_turns']) for t in kept])
        update = arl.debug_update(model, optimizer, batch, cfg, None, ref_model, gen)
        history.append(dict(arl.log_metrics(sampled, cfg, update), step=step))
        if every and (step + 1) % every == 0:
            on_checkpoint(step + 1, model)
    return history


def _main_run():
    c = _setup()
    if 'history' not in c:
        val = lambda m: arl.sample_counts(m, c['val'], c['tok'], N_VAL, c['cfg'], torch.Generator().manual_seed(3))
        val_counts = {0: val(c['base'])}
        policy = copy.deepcopy(c['base'])
        c['history'] = _train_run(policy, c['cfg'], STEPS, EVERY, lambda s, m: val_counts.__setitem__(s, val(m)))
        c['val_counts'] = val_counts
    return c['history'], c['val_counts']


ARMS = {'default': ('base', {}), 'KL k3 0.05': ('base', dict(kl_coef=0.05)),
        'SFT cold start': ('sft', {}), 'public-test reward': ('base', dict(reward='public'))}


def _ablations():
    c = _setup()
    if 'runs' not in c:
        runs = {}
        for arm, (start, fields) in ARMS.items():
            for seed in SEEDS:
                model = copy.deepcopy(c[start])
                _train_run(model, arl.RLConfig(seed=seed, **fields), ABL_STEPS)
                runs[arm, seed] = arl.sample_counts(model, c['held'], c['tok'], 16, c['cfg'],
                                                    torch.Generator().manual_seed(100 + seed))
        c['runs'] = runs
    return c['runs']


def _smooth(xs, k=5):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def dashboard():
    history, val_counts = _main_run()
    steps = [h['step'] for h in history]
    get = lambda key: [h[key] for h in history]
    fig = Figure(1080, 640, title='Training dashboard of the capstone run',
                 subtitle='Default GRPO from the RL base agent, 60 steps of 64 episodes, one seed. Thin lines: per '
                          'step; thick: 5-step moving average.')
    a = fig.axes(0, 0, 1 / 3, 0.5, xlabel='step', ylabel='pass rate', ylim=(0, 1), title='Outcome')
    a.line(steps, get('reward'), series=1, end_label=False, width=1)
    a.line(steps, _smooth(get('reward')), label='training batch', series=1, end_label=False)
    xs = list(val_counts)
    rates = [val_counts[s].sum().item() / (N_VAL * len(val_counts[s])) for s in xs]
    for s, r in zip(xs, rates):
        lo, hi = arl.bootstrap_ci(val_counts[s] / N_VAL)
        a.line([s, s], [lo, hi], series=2, end_label=False, width=2)
    a.scatter(xs, rates, label='validation (95% CI)', series=2, r=5)
    a.hline(0.2, label='blind single guess')
    a.legend()
    b = fig.axes(1 / 3, 0, 1 / 3, 0.5, xlabel='step', ylabel='calls per episode', title='Behavior')
    for key, slot in [('reads', 1), ('tests', 2), ('edits', 3)]:
        b.line(steps, _smooth(get(key)), label=key, series=slot, end_label=False)
    b.legend()
    e = fig.axes(2 / 3, 0, 1 / 3, 0.5, xlabel='step', ylabel='nats per token', title='Policy entropy', ylim=(0, 0.6))
    e.line(steps, get('entropy'), series=1, end_label=False)
    z = fig.axes(0, 0.5, 1 / 3, 0.5, xlabel='step', ylabel='share of groups', ylim=(0, 1),
                 title='Zero-advantage groups')
    z.line(steps, get('zero_adv_frac'), series=1, end_label=False, width=1)
    z.line(steps, _smooth(get('zero_adv_frac')), series=1, end_label=False)
    k = fig.axes(1 / 3, 0.5, 1 / 3, 0.5, xlabel='step', ylabel='approx KL (log scale)', ylog=True,
                 ylim=(1e-6, 1e-2), title='Policy change per step')
    k.line(steps, [max(v, 1e-6) for v in get('approx_kl')], series=1, end_label=False)
    g = fig.axes(2 / 3, 0.5, 1 / 3, 0.5, xlabel='step', ylabel='nats (log scale)', ylog=True, ylim=(1e-10, 1e-3),
                 title='Sampler-trainer log-prob gap')
    g.line(steps, [max(v, 1e-10) for v in get('logp_gap')], series=1, end_label=False)
    g.hline(1e-4, label='sanity-check threshold')
    fig.save('ch19-dashboard')


def forest():
    c = _setup()
    runs = _ablations()
    names = list(ARMS)
    rates = {a: torch.stack([runs[a, s] / 16 for s in SEEDS]) for a in names}        # [seeds, tasks]
    fig = Figure(1000, 420, title='Ablations: three seeds per arm, 30 steps, held-out pass@1',
                 subtitle='Left: each dot is one seed (160 episodes); bar: arm mean with a task-bootstrap 95% CI of '
                          'the seed-averaged rates. Right: paired difference from the default over 10 tasks.')
    xt = list(range(len(names)))
    left = fig.axes(0, 0, 0.5, 1, ylabel='held-out pass@1', ylim=(0, 1), xlim=(-0.5, len(names) - 0.5),
                    xticks=xt, xticklabels=names, title='Pass rate by arm')
    right = fig.axes(0.5, 0, 0.5, 1, ylabel='difference from default', ylim=(-0.9, 0.3),
                     xlim=(-0.5, len(names) - 0.5), xticks=xt, xticklabels=names, title='Effect vs default')
    right.hline(0.0)
    for i, arm in enumerate(names):
        slot = 8 if arm == 'public-test reward' else (2 if arm == 'default' else 1)
        mean = rates[arm].mean(0)
        lo, hi = arl.bootstrap_ci(mean)
        left.line([i, i], [lo, hi], series=slot, end_label=False)
        left.scatter([i], [mean.mean().item()], series=slot, r=6,
                     tips=[f'{arm}: {mean.mean().item():.3f} [{lo:.3f}, {hi:.3f}]'])
        per_seed = rates[arm].mean(-1).tolist()
        left.scatter([i + 0.18] * 3, per_seed, series=slot, r=3, tips=[f'{arm}, seed {s}: {v:.3f}'
                                                                       for s, v in zip(SEEDS, per_seed)])
        if arm != 'default':
            d, dlo, dhi = arl.paired_bootstrap_diff(mean, rates['default'].mean(0))
            right.line([i, i], [dlo, dhi], series=slot, end_label=False)
            right.scatter([i], [d], series=slot, r=6, tips=[f'{arm} - default: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]'])
            right.text(i, d, f'{d:+.3f}', dx=10, dy=4)
    left.hline(0.2, label='blind single guess')
    fig.save('ch19-forest')


def component_map():
    """Figure 19.3 in the hairline style. Ink: the arl functions; mid tone: production systems at scale.
    The accent marks the dashed weight path that closes the loop back to the sampler."""
    sc = hairline.Scene(1100, 620)
    sc.head("FIG 19.3 · FROM THE BOOK'S FUNCTIONS TO PRODUCTION INFRASTRUCTURE",
            'title: the stage · monospace: the arl functions · below the rule: public systems at scale (Section 19.8)')
    W, H, X = 214, 140, [24, 306, 588, 870]
    stages = [
        (X[0], 84, '01', 'Tasks and environment', ['make_tasks', 'EditBudgetEnv, run_tests', 'TOOL_SCHEMAS'],
         ['R2E-Gym, SWE-Gym, SWE-smith', 'Docker image per task']),
        (X[1], 84, '02', 'Rollout', ['sample_groups', 'rollout_batch, Tokenizer', 'collate'],
         ['vLLM / SGLang engines', 'rLLM agent loop']),
        (X[2], 84, '03', 'Reward', ['BugFixEnv reward', 'make_reward, VERIFIERS', 'hidden tests'],
         ['hidden tests in containers', 'hybrid verifiers (R2E-Gym)']),
        (X[3], 84, '04', 'Advantage', ['compute_advantages', 'group_advantages', 'rloo_advantages'],
         ['verl core_algos', 'DeepSWE GRPO++ choices']),
        (X[3], 306, '05', 'Update', ['debug_update', 'ppo_policy_loss', 'aggregate_loss'],
         ['verl / OpenRLHF / slime', 'FSDP, DeepSpeed, Megatron']),
        (X[2], 306, '06', 'Weight sync, staleness', ['train_offpolicy', 'tis_weights', 'simulate_pipeline'],
         ['verl one-step / fully async', 'AReaL, OpenRLHF async']),
        (X[1], 306, '07', 'Logging and checks', ['log_metrics', 'sanity_checks', 'train_logged'],
         ['verl metric_utils', 'rollout_probs_diff']),
        (X[0], 306, '08', 'Evaluation', ['sample_counts, pass_at_k', 'bootstrap_ci', 'paired_bootstrap_diff'],
         ['SWE-bench harness (Docker)', 'pass@1 averaged over runs']),
    ]
    for x, y, num, title, book, prod in stages:
        sc.card(x, y, W, H, style='card')
        sc.label(x + 14, y + 22, num + '  ', cls='num')
        sc.label(x + 34, y + 22, title, cls='ctitle')
        for i, line in enumerate(book):
            sc.label(x + 14, y + 42 + 14 * i, line, cls='cmono')
        sc.overlay.append(f'<path class="crease" d="M{x + 14},{y + 92} H{x + W - 14}"/>')
        for i, line in enumerate(prod):
            sc.label(x + 14, y + 110 + 14 * i, line, cls='cline')
    top, bot = 84 + 70, 306 + 70
    sc.farrow([(X[0] + W + 4, top), (X[1] - 6, top)])
    sc.farrow([(X[1] + W + 4, top), (X[2] - 6, top)])
    sc.farrow([(X[2] + W + 4, top), (X[3] - 6, top)])
    sc.farrow([(X[3] + W / 2, 84 + H + 4), (X[3] + W / 2, 306 - 6)])
    sc.farrow([(X[3] - 4, bot), (X[2] + W + 6, bot)])
    sc.farrow([(X[2] - 4, bot), (X[1] + W + 6, bot)])
    sc.farrow([(X[1] - 4, bot), (X[0] + W + 6, bot)])
    for x, y, text in [((X[1] + W + X[2]) / 2, top - 8, 'episodes'), ((X[2] + W + X[3]) / 2, top - 8, 'rewards'),
                       ((X[3] + W + X[2] + W) / 2 - 100, bot - 8, 'weights')]:
        sc.label(x, y, text, anchor='middle')
    sc.label(X[3] + W / 2 + 8, (84 + H + 306) / 2 + 4, 'advantages [B]')
    sc.farrow([(X[2] + W / 2, 306 - 4), (X[2] + W / 2, 260), (X[1] + W / 2 + 40, 260), (X[1] + W / 2 + 40, 84 + H + 6)],
              cls='accent')
    sc.overlay[-1] = sc.overlay[-1].replace('class="accent"', 'class="accent" stroke-dasharray="5 4"')
    sc.label(X[1] + W / 2 + 52, 254, 'new weights → sampler')
    sc.card(24, 480, 1052, 112, style='card-soft')
    sc.label(40, 506, 'WHAT CHANGES AT SCALE', cls='chead')
    notes = ['episodes: 70 tokens → tens of thousands; tool output dominates the context; KV caches and packing matter',
             'environments: authored in-process code → untrusted model code in containers with timeouts and resource limits',
             'rollout and training run on different engines: log-prob mismatch, staleness, and weight sync become first-class']
    for i, n in enumerate(notes):
        sc.label(40, 530 + 18 * i, n, cls='cline')
    (hairline.OUT / 'ch19-components.svg').write_text(sc.svg('From book functions to production'))


FIGURES = {'ch19-dashboard': dashboard, 'ch19-forest': forest, 'ch19-components': component_map}
