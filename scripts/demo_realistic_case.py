"""Query staged synthetic records, optionally through the real verl registry."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoastra.case_replay import ROOT, load_case, project_case
from nanoastra.clinical import read_record


async def run(args):
    view = project_case(load_case(), args.stage)
    print(json.dumps({"case_id": view["case_id"], "stage": view["stage"], "clock": view["clock"], "presentation": view["prompt"][-1]["content"]}, ensure_ascii=False))
    print("脚本化记录查询；没有模型推理、检查下单或自动临床评分。")
    records = view["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
    tool = None
    if args.verl:
        from nanoastra.upstream import activate_verl
        activate_verl()
        from verl.tools.tool_registry import initialize_tools_from_config
        tool = initialize_tools_from_config(str(ROOT / "configs/realistic_case_tools.yaml"))[0]

    for field in args.fields:
        if tool:
            key, initial = await tool.create(create_kwargs={"record": records})
            assert initial.is_empty()
            try:
                response, reward, metrics = await tool.execute(key, {"field": field})
                result = json.loads(response.text)
                assert reward == 0.0
            finally:
                await tool.release(key)
        else:
            result = read_record(records, field, view["record_fields"])
        if field not in view["record_fields"]:
            assert result["status"] == "invalid_field"
        if args.stage == "initial" and field == "ctpa_report":
            assert "腔内充盈缺损" not in json.dumps(result, ensure_ascii=False)
        print(json.dumps(result, ensure_ascii=False))
    if tool:
        assert not tool._records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial", "followup"], default="initial")
    parser.add_argument("--fields", nargs="+", default=["history_present_illness", "risk_history", "focused_exam", "initial_labs", "chest_xray", "d_dimer", "ctpa_report"])
    parser.add_argument("--verl", action="store_true", help="Use actual verl tool registry; requires vp dependencies.")
    asyncio.run(run(parser.parse_args()))
