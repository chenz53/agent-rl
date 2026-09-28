"""Run real verl tool registry + CPU GRPO code when dependencies are installed.

This is an integration probe, not inference or a training run. No stub modules.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoastra.upstream import ROOT, activate_verl

activate_verl()

import numpy as np
import torch
from verl.tools.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from nanoastra.clinical import DEMO_CASE


async def main():
    tool = initialize_tools_from_config(str(ROOT / "configs/clinical_tools.yaml"))[0]
    # Match upstream's per-call lifecycle; this is not the full ToolAgentLoop.
    for field in ("exposure", "test_x", "label"):
        key, initial = await tool.create(create_kwargs={"record": DEMO_CASE.tool_record()})
        assert initial.is_empty()
        try:
            response, reward, metrics = await tool.execute(key, {"field": field})
            print(response.text, reward, metrics)
            assert "label" not in json.loads(response.text)
        finally:
            await tool.release(key)
    assert not tool._records

    # Four samples of ONE prompt. Synthetic token positions, not real tokenization.
    masks = torch.tensor([[1, 1, 0, 0, 1, 1]] * 4, dtype=torch.float32)
    rewards = torch.zeros_like(masks)
    rewards[:, -1] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, masks, np.array(["case-001"] * 4), norm_adv_by_std_in_grpo=True
    )
    expected = torch.tensor([1, 1, -1, -1], dtype=torch.float32) * (3 ** 0.5 / 2)
    assert torch.allclose(advantages[:, 0], expected, atol=1e-5)
    assert torch.count_nonzero(advantages[:, 2:4]) == 0
    print("GRPO advantages:", advantages.tolist())
    # All-failed groups of size >1 carry no outcome-based relative signal here.
    zero_advantages, _ = compute_grpo_outcome_advantage(
        torch.zeros_like(rewards), masks, np.array(["case-001"] * 4)
    )
    assert torch.count_nonzero(zero_advantages) == 0
    print("Actual verl CPU probe passed; no model or optimizer update was run.")


if __name__ == "__main__":
    asyncio.run(main())
