"""Terminal outcome scorer with verl's custom reward function signature."""

import json
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "nanoastra_synthetic":
        raise ValueError("This teaching scorer only supports nanoastra_synthetic.")
    if ground_truth not in ("A", "B"):
        raise ValueError("Invalid synthetic answer key.")

    # verl may pass decoded multi-turn text, not just the last assistant message.
    # Require exactly one final block and require it to terminate the response.
    match = re.search(r"<final>([^<>]*)</final>\s*$", solution_str)
    valid = False
    diagnosis = None
    if match and solution_str.count("<final>") == solution_str.count("</final>") == 1:
        try:
            answer = json.loads(match.group(1))
            valid = (
                isinstance(answer, dict)
                and set(answer) == {"diagnosis"}
                and isinstance(answer["diagnosis"], str)
                and answer["diagnosis"] in ("A", "B")
            )
            if valid:
                diagnosis = answer["diagnosis"]
        except (ValueError, TypeError):
            pass
    correct = bool(valid and diagnosis == ground_truth)
    return {"score": float(correct), "accuracy": float(correct), "format_valid": float(valid)}
