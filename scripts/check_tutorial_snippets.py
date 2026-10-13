"""Execute trusted tutorial chapters and verify their adjacent stdout blocks."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PAIR = re.compile(r'```python\n(?P<code>.*?)\n```\n\nOutput:\n\n```text\n(?P<output>.*?)\n```', re.S)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    records = []
    for path in sorted((ROOT/'tutorial/chapters').glob('*.md')):
        source = path.read_text()
        if len(PAIR.findall(source)) != len(re.findall(r'^```python$', source, re.M)):
            raise ValueError(f'{path}: every Python block needs adjacent output')
        namespace = {'__name__': 'tutorial_snippet'}
        block = 0
        def execute(match):
            nonlocal block
            block += 1
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(compile(match['code'], str(path), 'exec'), namespace)
            actual = capture.getvalue().rstrip('\n')
            if not actual:
                raise ValueError(f'{path}: empty output')
            if not args.refresh and actual != match['output']:
                raise AssertionError(f'{path}: block {block} output mismatch\n{actual}')
            records.append({'chapter': path.name, 'block': block, 'stdout': actual, 'matched': True})
            return f'```python\n{match["code"]}\n```\n\nOutput:\n\n```text\n{actual}\n```'
        updated = PAIR.sub(execute, source)
        if args.refresh:
            path.write_text(updated)
    print(json.dumps({'chapters': len(set(r['chapter'] for r in records)), 'executed_blocks': len(records), 'outputs': records}, indent=2))

if __name__ == '__main__':
    main()
