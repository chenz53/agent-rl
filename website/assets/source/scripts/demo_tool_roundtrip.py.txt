"""Hand-authored Chat Completions-style call, real local lookup, no LLM/API."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nanoastra.oncology import tool_view, TABLES
from nanoastra.clinical import read_record


def main():
    records = tool_view()
    call = {
        'id': 'call_1', 'type': 'function',
        'function': {'name': 'get_clinical_record',
                     'arguments': '{"field":"performance_status"}'},
    }
    # This illustrative dispatcher exposes only an explicitly registered tool.
    def get_clinical_record(field):
        return read_record(records, field, TABLES)
    registry = {'get_clinical_record': get_clinical_record}
    args = json.loads(call['function']['arguments'])
    if not isinstance(args, dict) or set(args) != {'field'}:
        raise ValueError('Invalid tool arguments')
    function = registry[call['function']['name']]
    result = function(**args)
    result_message = {
        'role': 'tool', 'tool_call_id': call['id'],
        'content': json.dumps(result),
    }
    messages = [
        {'role': 'user', 'content': 'Assess prognosis from baseline records.'},
        {'role': 'assistant', 'tool_calls': [call]},
        result_message,
    ]
    rows = json.loads(result['value'])['rows']
    print('Parsed arguments:', args)
    print('Executed:', call['function']['name'])
    print('Evidence:', [(r['record_id'], r['score']) for r in rows])
    print('Result link:', result_message['tool_call_id'])
    print('Next request roles:', [m['role'] for m in messages])

if __name__ == '__main__':
    main()
