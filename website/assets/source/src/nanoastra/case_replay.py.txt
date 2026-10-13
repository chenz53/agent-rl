"""Stage-specific input projection for a richer synthetic clinical case.

This replays authored records. It does not order tests, simulate treatment, or
provide a validated clinical reward. Teacher annotations never enter tool input.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_case(path=None):
    case = json.loads(Path(path or ROOT / "cases/ed-0001.json").read_text())
    if case.get("synthetic") is not True:
        raise ValueError("This demo accepts only explicitly synthetic case fixtures.")
    return case


def project_case(case, stage="initial"):
    if stage not in case["stages"]:
        raise ValueError(f"Unknown replay stage: {stage}")
    snapshot = case["stages"][stage]
    # Explicit projection rather than passing the entire fixture as metadata.
    records = {**case["shared_records"], **snapshot["records"]}
    fields = tuple(case["record_fields"])
    if set(records) - set(fields):
        raise ValueError("Snapshot contains fields outside the tool contract.")
    if not all(isinstance(value, str) for value in records.values()):
        raise ValueError("Tool records must be text.")
    return {
        "case_id": case["case_id"],
        "stage": stage,
        "clock": snapshot["clock"],
        "prompt": [
            {"role": "system", "content": (
                "这是合成临床记录回放。使用 get_clinical_record 读取当前阶段允许的信息。"
                "只根据已经获取的证据，区分怀疑与确诊，并明确缺失信息；不要伪造报告。"
                "输出鉴别诊断、证据来源、剩余不确定性及下一步评估重点。"
                "读取记录不等于开立检查；不得自行推进时钟。"
            )},
            {"role": "user", "content": snapshot["presentation"]},
        ],
        "record_fields": fields,
        "tools_kwargs": {"get_clinical_record": {"create_kwargs": {"record": records}}},
    }
