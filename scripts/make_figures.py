"""Render the book's data figures into tutorial/figures/*.svg.

Each scripts/figures/chNN_*.py module exposes FIGURES = {name: function}. A figure
function computes its data (usually with torch and the generated agentrl library)
and saves an SVG through svgplot. Run on the vp host, where torch is installed:

    scripts/remote_figures.sh <workdir> [module-prefix ...]
"""
import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'figures'), str(HERE.parent)]


def main(selectors):
    """selectors: module prefixes (e.g. ch08) or module:figure pairs (e.g. ch12:ch12-turn-credit)."""
    wanted = [sel.split(':', 1) for sel in selectors]
    for path in sorted((HERE / 'figures').glob('ch*.py')):
        picks = [w for w in wanted if path.stem.startswith(w[0])]
        if wanted and not picks:
            continue
        names = {w[1] for w in picks if len(w) == 2}
        module = importlib.import_module(path.stem)
        for name, function in module.FIGURES.items():
            if names and name not in names:
                continue
            function()
            print(f'{path.stem}: {name}')


if __name__ == '__main__':
    main(sys.argv[1:])
