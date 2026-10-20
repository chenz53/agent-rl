"""Execute tutorial chapters in order and verify each block's captured stdout.

Rules enforced:
  * Every ```python block that prints must be followed by an ``Output:`` text block
    whose content matches the printed stdout exactly. A block that prints nothing
    must not be followed by an Output block.
  * A block whose first line is ``#@save`` is appended to the book library
    ``agentrl``. Saved blocks may only contain imports, function/class definitions,
    and constant assignments.
  * Chapter N sees ``import agentrl`` built only from the saved blocks of chapters
    that precede it. Each chapter runs in a fresh namespace.
  * Display math uses ``$$`` on its own lines.

Use --chapters to execute a subset (earlier chapters still contribute their saved
blocks to the library). Use --refresh to rewrite Output blocks from actual stdout.
"""
import argparse
import ast
import contextlib
import fnmatch
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / 'tutorial' / 'chapters'
BLOCK = re.compile(
    r'```python\n(?P<code>.*?)\n```(?P<tail>\n\nOutput:\n\n```text\n(?P<output>.*?)\n```)?',
    re.S,
)
DEFAULT_THREADS = None
SAVE_ALLOWED = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef,
                ast.Assign, ast.AnnAssign, ast.Expr)


def chapter_paths():
    return sorted(CHAPTERS.glob('*.md'))


def saved_blocks(path):
    """Return the code of every #@save block in a chapter, in order."""
    blocks = []
    for match in BLOCK.finditer(path.read_text()):
        code = match['code']
        if code.lstrip().startswith('#@save'):
            blocks.append(code)
    return blocks


def validate_saved(path, code):
    for node in ast.parse(code).body:
        if not isinstance(node, SAVE_ALLOWED):
            raise ValueError(f'{path.name}: #@save block may only define names, got {type(node).__name__}')
        if isinstance(node, ast.Expr) and not isinstance(getattr(node, 'value', None), ast.Constant):
            raise ValueError(f'{path.name}: #@save block may not call functions at top level')


def make_library(paths):
    module = types.ModuleType('agentrl')
    module.__file__ = '<agentrl built from tutorial chapters>'
    sys.modules['agentrl'] = module
    for path in paths:
        for code in saved_blocks(path):
            validate_saved(path, code)
            exec(compile(code, f'agentrl:{path.name}', 'exec'), module.__dict__)
    return module


def check_math(path, source):
    for part in re.split(r'```.*?```', source, flags=re.S):
        for line in part.splitlines():
            if '$$' in line and line.strip() != '$$':
                raise ValueError(f'{path.name}: display math delimiters must be on separate lines: {line!r}')


def reset_threads():
    """Give every chapter the host's default intra-op thread count.

    CPU float reductions depend on the thread count, so RL trajectories do too. A
    chapter that calls torch.set_num_threads must not leak that setting into the
    chapters that follow it in a full-book run.
    """
    # Only touch the pool when an earlier chapter changed it: calling set_num_threads
    # even with the default count alters CPU reduction order relative to never calling it.
    torch = sys.modules.get('torch')
    if torch is not None and torch.get_num_threads() != DEFAULT_THREADS:
        torch.set_num_threads(DEFAULT_THREADS)


def run_chapter(path, refresh):
    reset_threads()
    source = path.read_text()
    check_math(path, source)
    opened = len(re.findall(r'^```python$', source, re.M))
    if opened != len(BLOCK.findall(source)):
        raise ValueError(f'{path.name}: malformed python block')
    namespace = {'__name__': 'tutorial_snippet'}
    records = []
    started = time.time()

    def execute(match):
        code = match['code']
        index = len(records) + 1
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(code, f'{path.name}:block{index}', 'exec'), namespace)
        actual = capture.getvalue().rstrip('\n')
        records.append({'block': index, 'lines': len(code.splitlines()), 'stdout_lines': len(actual.splitlines()) if actual else 0})
        if not refresh:
            if actual and match['output'] is None:
                raise AssertionError(f'{path.name}: block {index} prints but has no Output block\n{actual}')
            if not actual and match['output'] is not None:
                raise AssertionError(f'{path.name}: block {index} prints nothing but has an Output block')
            if actual and actual != match['output']:
                raise AssertionError(f'{path.name}: block {index} output mismatch\n--- actual ---\n{actual}\n--- recorded ---\n{match["output"]}')
        if not actual:
            return f'```python\n{code}\n```'
        return f'```python\n{code}\n```\n\nOutput:\n\n```text\n{actual}\n```'

    updated = BLOCK.sub(execute, source)
    if refresh and updated != source:
        path.write_text(updated)
    return {'chapter': path.name, 'blocks': len(records), 'seconds': round(time.time() - started, 1),
            'longest_block_lines': max((r['lines'] for r in records), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refresh', action='store_true', help='rewrite Output blocks from actual stdout')
    parser.add_argument('--chapters', nargs='*', default=['*'], help='glob patterns of chapter files to execute')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    global DEFAULT_THREADS
    import torch
    DEFAULT_THREADS = torch.get_num_threads()
    paths = chapter_paths()
    results = []
    for i, path in enumerate(paths):
        if not any(fnmatch.fnmatch(path.name, pattern) or path.name.startswith(pattern) for pattern in args.chapters):
            continue
        make_library(paths[:i])
        results.append(run_chapter(path, args.refresh))
        print(json.dumps(results[-1]), file=sys.stderr)
    print(json.dumps({'chapters': len(results), 'blocks': sum(r['blocks'] for r in results), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
