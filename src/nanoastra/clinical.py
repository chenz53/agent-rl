"""Synthetic task data and record access; no medical claims or RL engine."""

from dataclasses import dataclass
from typing import Mapping

ALLOWED_FIELDS = ("exposure", "test_x")


@dataclass(frozen=True)
class Case:
    case_id: str
    presentation: str
    exposure: str
    test_x: str
    label: str

    def tool_record(self) -> dict:
        # Explicit projection: the answer key never enters the tool instance.
        return {"exposure": self.exposure, "test_x": self.test_x}


DEMO_CASE = Case(
    case_id="synthetic-001",
    presentation="Synthetic patient: fever and cough. Candidate labels are A and B.",
    exposure="positive",
    test_x="negative",
    label="B",
)


def read_record(record: Mapping[str, str], field: object) -> dict:
    if not isinstance(field, str) or field not in ALLOWED_FIELDS:
        return {"status": "invalid_field"}
    if field not in record:
        return {"status": "unavailable", "field": field}
    return {"status": "ok", "field": field, "value": record[field]}


def to_verl_row(case: Case) -> dict:
    """Follow the pinned verl example's tool_agent dataset contract."""
    return {
        "data_source": "nanoastra_synthetic",
        "agent_name": "tool_agent",
        "prompt": [
            {
                "role": "system",
                "content": (
                    "You are solving a synthetic record-access task, not giving medical advice. "
                    "Use get_clinical_record if useful. Finish with exactly one "
                    '<final>{"diagnosis":"A"}</final> or '
                    '<final>{"diagnosis":"B"}</final>. '
                    "Only exposure and test_x are queryable; labels are artificial."
                ),
            },
            {"role": "user", "content": case.presentation},
        ],
        "reward_model": {"style": "rule", "ground_truth": case.label},
        "extra_info": {
            "case_id": case.case_id,
            "tools_kwargs": {
                "get_clinical_record": {"create_kwargs": {"record": case.tool_record()}}
            },
        },
    }
