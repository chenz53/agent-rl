"""Execute every Python block in the current curriculum and compare printed output.

Only execute trusted repository lessons. Each lesson has its own namespace;
blocks within it run in order. --refresh replaces adjacent recorded outputs.
No model weights are loaded and no GPU training is started.
"""
import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PAIR = re.compile(r'```python\n(?P<code>.*?)\n```\n\nOutput:\n\n```text\n(?P<output>.*?)\n```', re.S)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    results=[]
    for path in sorted((ROOT/'docs/lessons').glob('[0-9]*.md')):
        source=path.read_text()
        pairs=list(PAIR.finditer(source))
        if len(pairs)!=len(re.findall(r'^```python$',source,re.M)):
            raise ValueError(f'{path.name}: every Python block needs adjacent Output')
        ns={'__name__':'lesson_snippet'}
        def evaluate(match):
            code=match['code'];ast.parse(code)
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(code,str(path),'exec'),ns)
            actual=out.getvalue().rstrip('\n')
            if not actual:raise ValueError(f'{path.name}: snippet has no visible output')
            if not args.refresh and actual != match['output']:
                raise AssertionError(f'{path.name}: recorded output differs\nACTUAL:\n{actual}\nEXPECTED:\n{match["output"]}')
            results.append({'lesson':path.name,'block':1+sum(r['lesson']==path.name for r in results),'stdout':actual,'matched':True})
            return '```python\n'+code+'\n```\n\nOutput:\n\n```text\n'+actual+'\n```'
        updated=PAIR.sub(evaluate,source)
        if args.refresh:path.write_text(updated)
    print(json.dumps({'lessons':len(set(r['lesson'] for r in results)),'executed_blocks':len(results),'outputs':results},indent=2))

if __name__=='__main__':main()
