"""Run the shared domain functions using scripted actions, not an LLM."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanoastra.clinical import DEMO_CASE, read_record, to_verl_row
from nanoastra.reward import compute_score


def main():
    row = to_verl_row(DEMO_CASE)
    record = row["extra_info"]["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
    print("SCRIPTED DOMAIN DEMO — no model inference, no verl runtime, no training")
    print("Initial observation:", row["prompt"][-1]["content"])
    for field in ("exposure", "test_x", "label"):
        print(f"get_clinical_record({field!r}) ->", json.dumps(read_record(record, field)))
    for final in ('<final>{"diagnosis":"B"}</final>', '<final>{"diagnosis":"A"}</final>', "B"):
        reward = compute_score(row["data_source"], final, row["reward_model"]["ground_truth"])
        print(f"{final} -> {reward}")


if __name__ == "__main__":
    main()
