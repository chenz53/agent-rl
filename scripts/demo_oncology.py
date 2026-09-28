"""Read gzip CSVs, construct an as-of view, and replay table queries."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nanoastra.oncology import DEFAULT_DATA, INDEX_TIME, ROOT, TABLES, tool_view
from nanoastra.clinical import read_record

async def run(args):
    record = tool_view(args.data, args.patient, args.index_time)
    print(json.dumps({'patient_id': args.patient, 'index_time': args.index_time,
        'task': 'Assess 180-day all-cause mortality prognosis using baseline evidence; cite record IDs and state uncertainty.',
        'mode': 'Scripted queries, not LLM inference or a calibrated survival prediction.'}))
    tool = None
    if args.verl:
        from nanoastra.upstream import activate_verl
        activate_verl()
        from verl.tools.tool_registry import initialize_tools_from_config
        tool = initialize_tools_from_config(str(ROOT / 'configs/oncology_tools.yaml'))[0]
    for table in args.tables:
        if tool:
            key, initial = await tool.create(create_kwargs={'record': record})
            assert initial.is_empty()
            try:
                response, reward, _ = await tool.execute(key, {'field': table})
                result = json.loads(response.text)
                assert reward == 0
            finally:
                await tool.release(key)
        else:
            result = read_record(record, table, TABLES)
        if result['status'] == 'ok':
            result['value'] = json.loads(result['value'])
        print(json.dumps(result, indent=2))
    if tool:
        assert not tool._records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--patient', default='SYN-ONC-001')
    parser.add_argument('--index-time', default=INDEX_TIME)
    parser.add_argument('--tables', nargs='+', default=['diagnosis', 'labs', 'vitals', 'notes'])
    parser.add_argument('--verl', action='store_true')
    asyncio.run(run(parser.parse_args()))
