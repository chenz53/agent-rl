"""One real Qwen parameter update using verl GRPO advantage and policy loss.

Queries are scripted. Only a one-token A/B decision is sampled and optimized.
The policy is explicitly constrained to these two tokens at a fixed temperature;
sampling AND training log-probabilities use the same normalization. This is a
small integration experiment, not the distributed verl trainer or clinical eval.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoastra.upstream import ROOT, activate_verl

activate_verl()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.tools.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
from verl.workers.config import ActorConfig

from nanoastra.clinical import DEMO_CASE
from nanoastra.reward import compute_score


async def fetch_observations():
    tool = initialize_tools_from_config(str(ROOT / "configs/clinical_tools.yaml"))[0]
    observations = []
    for field in ("exposure", "test_x"):
        instance, _ = await tool.create(create_kwargs={"record": DEMO_CASE.tool_record()})
        try:
            response, _, _ = await tool.execute(instance, {"field": field})
            observations.append(json.loads(response.text))
        finally:
            await tool.release(instance)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gpu-update.json")
    args = parser.parse_args()
    if args.samples < 2 or args.temperature <= 0 or args.learning_rate <= 0:
        parser.error("Need samples >= 2, positive temperature and learning rate.")
    if not torch.cuda.is_available():
        raise RuntimeError("This demonstration requires a CUDA GPU.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    observations = asyncio.run(fetch_observations())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    # eval disables dropout, but does NOT disable gradient computation.
    model.eval()
    model.config.use_cache = False
    labels = ["A", "B"]
    encodings = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in encodings):
        raise RuntimeError("This demo requires single-token A/B labels.")
    allowed_ids = torch.tensor([ids[0] for ids in encodings], device="cuda")
    messages = [
        {"role": "system", "content": (
            "This is a synthetic interface test, not medical advice. "
            "Choose the case label A or B. Output only the letter."
        )},
        {"role": "user", "content": DEMO_CASE.presentation + "\nRecords: " + json.dumps(observations)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    def policy_log_probs():
        logits = model(**inputs).logits[0, -1, allowed_ids]
        return torch.log_softmax(logits / args.temperature, dim=-1)

    with torch.no_grad():
        old_distribution = policy_log_probs()
        actions = torch.multinomial(old_distribution.exp(), args.samples, replacement=True)
        old_log_probs = old_distribution[actions].unsqueeze(1).clone()
        before = old_distribution.exp().tolist()

    # The wrapper is deterministic task formatting, not another sampled action.
    chosen = [labels[i] for i in actions.tolist()]
    scores = [compute_score(
        "nanoastra_synthetic", '<final>' + json.dumps({"diagnosis": label}) + '</final>', DEMO_CASE.label
    )["score"] for label in chosen]
    rewards = torch.tensor(scores, dtype=torch.float32, device="cuda").unsqueeze(1)
    mask = torch.ones_like(rewards)
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, mask, np.array([DEMO_CASE.case_id] * args.samples)
    )
    config = ActorConfig(
        strategy="fsdp", rollout_n=args.samples, ppo_micro_batch_size_per_gpu=1
    )
    # strategy is a config requirement; no FSDP engine is launched here.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    current_log_probs = policy_log_probs()[actions].unsqueeze(1)
    ratio_error = (current_log_probs.detach() - old_log_probs).abs().max().item()
    if ratio_error > 1e-4:
        raise RuntimeError(f"Sampling/training log-prob mismatch: {ratio_error}")
    loss, metrics = compute_policy_loss_vanilla(
        old_log_probs, current_log_probs, advantages, mask, config=config
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite gradient; refusing parameter update.")
    optimizer.step()
    with torch.no_grad():
        after = policy_log_probs().exp().tolist()

    report = {
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "dtype": "float32",
        "policy": "softmax of A/B logits only; queries scripted; no full-vocabulary generation",
        "prompt": prompt,
        "token_ids": allowed_ids.tolist(),
        "sampled_labels": chosen,
        "rewards": scores,
        "advantages": advantages[:, 0].tolist(),
        "loss_before_update": loss.item(),
        "sampling_training_logprob_max_error": ratio_error,
        "gradient_norm_before_clipping": grad_norm.item(),
        "clip_gradient_norm": 1.0,
        "probabilities_before": dict(zip(labels, before)),
        "probabilities_after": dict(zip(labels, after)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "upstream_metrics": metrics,
        "mixed_reward_group": len(set(scores)) > 1,
        "updates": 1,
        "checkpoint_saved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
