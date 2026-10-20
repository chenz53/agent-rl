"""Verify that the Chinese edition (tutorial/zh/) stays in sync with the canonical English edition.

The translation changes prose only. For each English page that has a Chinese counterpart, this
checks that the two pages have:
  * the same fenced blocks (python code, Output, text, mermaid), byte-identical and in order;
  * the same display-math blocks ($$ ... $$), byte-identical and in order;
  * the same figures (image basenames), in order;
  * the same heading structure (count of each heading level);
and that every relative link and #anchor in the Chinese page resolves.

Code, outputs, math, and figures are executed and checked only in the English edition
(scripts/check_tutorial_snippets.py); this script guarantees the translation shows the same ones.

Usage: python3 scripts/check_translation.py [--require-all] [name-prefix ...]
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / 'tutorial'
ZH = ROOT / 'tutorial' / 'zh'
FENCE = re.compile(r'^(`{3,})[^\n]*\n.*?^\1[ \t]*$', re.S | re.M)
MATH = re.compile(r'^\$\$\n.*?\n\$\$$', re.S | re.M)
IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)\)')
LINK = re.compile(r'(?<!!)\[[^\]]+\]\(([^)\s]+)\)')


def slug(heading):
    """GitHub/Mintlify-style anchor: lower-case, drop punctuation, spaces to hyphens (keeps CJK)."""
    text = re.sub(r'[`*_]', '', heading.strip().lower())
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


def prose(text):
    return FENCE.sub('', text)


def pairs():
    english = [EN / 'README.md', EN / 'labs' / 'README.md'] + sorted((EN / 'chapters').glob('*.md'))
    for page in english:
        yield page, ZH / page.relative_to(EN)


def compare(en_path, zh_path):
    en, zh = en_path.read_text(), zh_path.read_text()
    errors = []
    en_fences, zh_fences = [m.group(0) for m in FENCE.finditer(en)], [m.group(0) for m in FENCE.finditer(zh)]
    if en_fences != zh_fences:
        for i, (a, b) in enumerate(zip(en_fences, zh_fences)):
            if a != b:
                errors.append(f'fenced block {i + 1} differs:\n--- en ---\n{a[:400]}\n--- zh ---\n{b[:400]}')
                break
        else:
            errors.append(f'fenced block count differs: en {len(en_fences)} vs zh {len(zh_fences)}')
    if MATH.findall(prose(en)) != MATH.findall(prose(zh)):
        errors.append('display-math blocks differ (math must be copied verbatim)')
    en_imgs = [Path(p).name for p in IMAGE.findall(prose(en))]
    zh_imgs = [Path(p).name for p in IMAGE.findall(prose(zh))]
    if en_imgs != zh_imgs:
        errors.append(f'figures differ: en {en_imgs} vs zh {zh_imgs}')
    for level in range(1, 5):
        pattern = re.compile(rf'^{"#" * level} ', re.M)
        if len(pattern.findall(prose(en))) != len(pattern.findall(prose(zh))):
            errors.append(f'number of level-{level} headings differs')
    return errors


def headings(path):
    return [slug(h) for h in re.findall(r'^#+ (.*)$', prose(path.read_text()), re.M)]


def fix_anchors(path):
    """Rewrite English anchors in a translated page to the Chinese heading at the same position.

    Translators keep English anchors (e.g. 07-advantages-and-credit.md#73-generalized-advantage-estimation);
    because heading order and count are identical across editions, the i-th English heading maps to
    the i-th Chinese heading of the same page.
    """
    text = path.read_text()

    def repl(match):
        label, target = match.group(1), match.group(2)
        file_part, sep, anchor = target.partition('#')
        if not sep or '://' in target:
            return match.group(0)
        zh_dest = (path.parent / file_part).resolve() if file_part else path.resolve()
        if not zh_dest.exists() or anchor in headings(zh_dest):
            return match.group(0)
        try:
            en_dest = EN.resolve() / zh_dest.relative_to(ZH.resolve())
        except ValueError:
            return match.group(0)
        en_heads, zh_heads = headings(en_dest), headings(zh_dest)
        if anchor in en_heads and len(en_heads) == len(zh_heads):
            return f'[{label}]({file_part}#{zh_heads[en_heads.index(anchor)]})'
        return match.group(0)

    pieces = re.split(r'(^(`{3,})[^\n]*\n.*?^\2[ \t]*$)', text, flags=re.S | re.M)
    out = []
    i = 0
    while i < len(pieces):
        out.append(re.sub(r'(?<!!)\[([^\]]+)\]\(([^)\s]+)\)', repl, pieces[i]))
        if i + 1 < len(pieces):
            out.append(pieces[i + 1])
        i += 3
    new = ''.join(out)
    if new != text:
        path.write_text(new)
        return True
    return False


def check_links(path):
    errors = []
    for target in LINK.findall(prose(path.read_text())) + IMAGE.findall(prose(path.read_text())):
        if '://' in target or target.startswith('mailto:'):
            continue
        file_part, _, anchor = target.partition('#')
        dest = (path.parent / file_part).resolve() if file_part else path.resolve()
        if not dest.exists():
            errors.append(f'broken link {target}')
            continue
        if anchor and dest.suffix == '.md':
            heads = {slug(h) for h in re.findall(r'^#+ (.*)$', prose(dest.read_text()), re.M)}
            if anchor not in heads:
                errors.append(f'broken anchor {target}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--require-all', action='store_true', help='fail if any English page lacks a translation')
    parser.add_argument('--fix-anchors', action='store_true', help='map English #anchors to Chinese headings in place')
    parser.add_argument('names', nargs='*', help='only check pages whose filename starts with one of these')
    args = parser.parse_args()
    failed, checked, missing = 0, 0, []
    for en_path, zh_path in pairs():
        if args.names and not any(en_path.name.startswith(n) for n in args.names):
            continue
        if not zh_path.exists():
            missing.append(str(zh_path.relative_to(ROOT)))
            continue
        checked += 1
        if args.fix_anchors and fix_anchors(zh_path):
            print(f'fixed anchors in {zh_path.relative_to(ROOT)}')
        errors = compare(en_path, zh_path) + check_links(zh_path)
        if errors:
            failed += 1
            print(f'FAIL {zh_path.relative_to(ROOT)}')
            for e in errors:
                print('   ', e.replace('\n', '\n    '))
    if missing:
        print(f'missing translations ({len(missing)}): ' + ', '.join(missing))
    print(f'checked {checked} translated pages, {failed} failed')
    sys.exit(1 if failed or (args.require_all and missing) else 0)


if __name__ == '__main__':
    main()
