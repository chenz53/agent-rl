"""Execute trusted tutorial chapters and verify their adjacent stdout blocks."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
PAIR = re.compile(r'```python\n(?P<code>.*?)\n```\n\nOutput:\n\n```text\n(?P<output>.*?)\n```', re.S)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    records = []
    for path in sorted((ROOT/'tutorial/chapters').glob('*.md')):
        source = path.read_text()
        prose_parts = re.split(r"```.*?```", source, flags=re.S)
        for part in prose_parts:
            for line in part.splitlines():
                if "$$" in line and line.strip() != "$$":
                    raise ValueError(f"{path}: display math delimiters must be on separate lines")
        if len(PAIR.findall(source)) != len(re.findall(r'^```python$', source, re.M)):
            raise ValueError(f'{path}: every Python block needs adjacent output')
        namespace = {'__name__': 'tutorial_snippet'}
        block = 0
        def execute(match):
            nonlocal block
            block += 1
            tree = ast.parse(match['code'])
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nanoastra"):
                    raise ValueError(f"{path}: teaching functions must be defined in the chapter")
                if isinstance(node, ast.Import) and any(alias.name.startswith("nanoastra") for alias in node.names):
                    raise ValueError(f"{path}: project helper imports are forbidden")
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
