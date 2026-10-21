"""Generate topic modules from the book's #@save blocks and a small public facade.

Run with --check to verify generated files without modifying them.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_library import GENERATED, ROOT, build_modules, chapter_paths, package_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if generated files are stale')
    args = parser.parse_args()
    modules = build_modules(chapter_paths())
    files = package_files(modules)
    target = ROOT / 'agentrl'
    stale = [p for p in target.glob('*.py') if p.name not in files and GENERATED in p.read_text()]
    changed = [name for name, source in files.items()
               if not (target / name).exists() or (target / name).read_text() != source]
    if args.check:
        if changed or stale:
            parser.exit(1, 'Generated files are stale: ' + ', '.join(changed + [p.name for p in stale]) + '\n')
        print(f'Checked {len(modules)} topic modules and public API; generated files are current')
        return
    target.mkdir(exist_ok=True)
    for name in changed:
        (target / name).write_text(files[name])
    for path in stale:
        path.unlink()
    print(f'Generated {len(modules)} topic modules, public API and module index in agentrl/')


if __name__ == '__main__':
    main()
