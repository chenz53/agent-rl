"""Execute explicitly selected teaching snippets; skip contextual model excerpts.

Only use on this trusted repository's lesson files. The selection is explicit so
that partial methods and deliberately wrong code are not mistaken for programs.
No model is loaded and no GPU training is started.
"""

import ast
import contextlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SELECTION = {
    "001-clinical-reasoning-as-an-environment.md": list(range(1, 14)),
    "002-one-clinical-trajectory.md": list(range(1, 6)),
    "003-verl-code-walkthrough.md": [1, 2, 6, 8, 9],
    "004-first-real-gpu-update.md": [3, 5, 9, 11, 12, 15],
}


def main():
    import os
    os.chdir(ROOT)
    outputs = []
    syntax_count = 0
    for name, selected in SELECTION.items():
        path = ROOT / "docs/archive" / name
        blocks = re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S)
        namespace = {"__name__": "lesson_snippet"}
        for number, code in enumerate(blocks, 1):
            ast.parse(code, filename=f"{name}:block-{number}")
            syntax_count += 1
            if number not in selected:
                continue
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(compile(code, f"{name}:block-{number}", "exec"), namespace)
            outputs.append({"lesson": name, "block": number, "stdout": capture.getvalue()})
        if max(selected) > len(blocks):
            raise ValueError(f"Snippet selection is stale for {name}")
    print(json.dumps({"parsed_blocks": syntax_count, "executed_blocks": len(outputs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
