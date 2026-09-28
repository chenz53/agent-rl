"""Check lesson examples against real verl and PyTorch; no model/GPU required.

This is a small numerical teaching check, not a replacement training algorithm.
Run with the existing vp environment from the nanoAstra root.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoastra.upstream import ROOT, activate_verl

activate_verl()

import numpy as np
import torch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
from verl.workers.config import ActorConfig


def main():
    evidence = json.loads((ROOT / "experiments/results/002-gpu-update.json").read_text())
    config = ActorConfig(strategy="fsdp", rollout_n=16, ppo_micro_batch_size_per_gpu=1)
    rewards = torch.tensor(evidence["rewards"], dtype=torch.float64).unsqueeze(1)
    mask = torch.ones_like(rewards)
    groups = np.array(["same-case"] * len(rewards))
    advantage, _ = compute_grpo_outcome_advantage(rewards, mask, groups)
    assert torch.allclose(advantage[:, 0], torch.tensor(evidence["advantages"], dtype=torch.float64), atol=1e-6)

    # Equivalent logits reconstructed from measured probabilities, NOT raw Qwen logits.
    temperature = evidence["temperature"]
    probs = torch.tensor(list(evidence["probabilities_before"].values()), dtype=torch.float64)
    logits = (temperature * probs.log()).requires_grad_()
    logp = torch.log_softmax(logits / temperature, dim=-1)
    actions = torch.tensor([0 if a == "A" else 1 for a in evidence["sampled_labels"]])
    chosen = logp[actions].unsqueeze(1)
    loss, _ = compute_policy_loss_vanilla(chosen.detach(), chosen, advantage, mask, config=config)
    loss.backward()
    expected_b_gradient = -((advantage[:, 0] * ((actions == 1).double() - logp.exp()[1].detach())).mean()) / temperature
    assert torch.allclose(logits.grad[1], expected_b_gradient)
    assert abs(loss.item()) < 1e-6 and logits.grad[1] < 0

    # Deliberately wrong: the same graph in numerator and denominator cancels.
    broken_logits = logits.detach().clone().requires_grad_()
    broken = torch.log_softmax(broken_logits / temperature, dim=-1)[actions].unsqueeze(1)
    broken_loss, _ = compute_policy_loss_vanilla(broken, broken, advantage, mask, config=config)
    broken_loss.backward()
    assert torch.count_nonzero(broken_logits.grad) == 0

    # Standard clipping examples; dual-clip only changes the last one here.
    clipping = []
    for adv, ratio, expected in [(2., 1.5, -2.4), (2., .6, -1.2), (-1., .5, .8), (-1., 1.5, 1.5), (-1., 5., 3.)]:
        current = torch.tensor([[math.log(ratio)]], dtype=torch.float64, requires_grad=True)
        value, _ = compute_policy_loss_vanilla(
            torch.zeros_like(current), current, torch.tensor([[adv]], dtype=torch.float64),
            torch.ones_like(current), config=config,
        )
        value.backward()
        assert math.isclose(value.item(), expected, abs_tol=1e-9)
        clipping.append({"advantage": adv, "ratio": ratio, "loss": value.item(), "d_loss_d_logp": current.grad.item()})

    # Response-mask demonstration uses synthetic positions, not a tokenizer.
    toy_mask = torch.tensor([[1., 1., 0., 0., 1., 1.]] * 4)
    toy_rewards = torch.zeros_like(toy_mask)
    toy_rewards[:, -1] = torch.tensor([1., 1., 0., 0.])
    toy_adv, _ = compute_grpo_outcome_advantage(toy_rewards, toy_mask, np.array(["case"] * 4))
    assert torch.count_nonzero(toy_adv[:, 2:4]) == 0
    for constant in (0., 1.):
        homogeneous, _ = compute_grpo_outcome_advantage(torch.full((4, 1), constant), torch.ones(4, 1), np.array(["case"] * 4))
        assert torch.count_nonzero(homogeneous) == 0

    # All failed within each group vs incorrectly centering across cases.
    two_cases = torch.tensor([[1.], [1.], [0.], [0.]])
    correct_groups, _ = compute_grpo_outcome_advantage(two_cases, torch.ones_like(two_cases), np.array(["easy", "easy", "hard", "hard"]))
    wrong_groups, _ = compute_grpo_outcome_advantage(two_cases, torch.ones_like(two_cases), np.array(["all"] * 4))
    assert torch.count_nonzero(correct_groups) == 0
    assert torch.count_nonzero(wrong_groups) == 4

    p = evidence["probabilities_before"]["B"]
    result = {
        "reward_mean": rewards.mean().item(),
        "sample_std": rewards.std(correction=1).item(),
        "population_std": rewards.std(correction=0).item(),
        "advantage_A": advantage[0, 0].item(),
        "advantage_B": advantage[11, 0].item(),
        "expected_B_count": 16 * p,
        "probability_of_exactly_two_B": math.comb(16, 2) * p**2 * (1-p)**14,
        "probability_of_no_B": (1-p)**16,
        "rare_success_p_001_group16_all_fail": .99**16,
        "equivalent_raw_logit_gap_A_minus_B": temperature * math.log((1-p)/p),
        "probabilities_by_temperature": {str(t): torch.softmax(logits.detach()/t, dim=0).tolist() for t in (1., 2., 4.)},
        "reconstructed_two_logit_loss": loss.item(),
        "two_logit_gradient": logits.grad.tolist(),
        "broken_old_logprob_gradient": broken_logits.grad.tolist(),
        "clipping_examples": clipping,
        "mask_example_positive": toy_adv[0].tolist(),
        "correct_grouping": correct_groups[:, 0].tolist(),
        "incorrect_grouping": wrong_groups[:, 0].tolist(),
        "gradient_scale_in_Qwen_run": 1.0/evidence["gradient_norm_before_clipping"],
        "B_probability_change_percentage_points": 100*(evidence["probabilities_after"]["B"]-p),
        "query_example": {"stop_expected_reward": .8, "query_expected_reward": .95-.05},
        "status": "All numerical assertions passed against pinned verl; no model update performed.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
