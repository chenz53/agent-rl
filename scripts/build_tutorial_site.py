"""Generate the Mintlify reading edition (website/) from the canonical tutorial Markdown.

Canonical sources: tutorial/README.md, tutorial/chapters/*.md, tutorial/figures/ image assets,
tutorial/sources.md, tutorial/editorial-guide.md. Never edit website/ by hand.
"""
import ast
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / 'tutorial'
SITE = ROOT / 'website'
sys.path.insert(0, str(ROOT / 'scripts'))
from book_library import CHAPTER_MODULES, saved_blocks  # noqa: E402

PARTS = [
    ('Front matter', lambda n: n == '00'),
    ('Part I · Foundations', lambda n: n in {'01', '02', '03', '04', '05'}),
    ('Part II · Policy optimization', lambda n: n in {'06', '07', '08', '09', '10'}),
    ('Part III · Rewards and agents', lambda n: n in {'11', '12'}),
    ('Part IV · Systems', lambda n: n in {'13', '14'}),
    ('Part V · Beyond pure RL', lambda n: n in {'15', '16'}),
    ('Part VI · Practice', lambda n: n in {'17', '18', '19', '20'}),
    ('Appendices', lambda n: n.isalpha()),
]

IMAGE_SUFFIXES = {'.svg', '.png', '.jpg', '.jpeg', '.webp'}

ZH_PARTS = ['前言', '第一部分 · 基础', '第二部分 · 策略优化', '第三部分 · 奖励与智能体', '第四部分 · 系统',
            '第五部分 · 超越纯强化学习', '第六部分 · 实践', '附录']

FENCE_RE = re.compile(r'^(`{3,})[^\n]*\n.*?^\1[ \t]*$', re.S | re.M)


def github_slug(heading):
    """Anchor convention used in the Markdown sources (GitHub style; keeps CJK)."""
    text = re.sub(r'[`*_]', '', heading.strip().lower())
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


def mintlify_slug(heading):
    """Anchor Mintlify renders for a heading; validated against all 609 headings of the site.

    Markup and inline math are dropped, straight quotes become curly, parentheses vanish,
    and runs of whitespace or periods become one hyphen; other punctuation is kept.
    """
    text = re.sub(r'`([^`]*)`', r'\1', heading.strip())
    text = re.sub(r'\*\*|__|(?<!\w)[*_]|[*_](?!\w)', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\$[^$]*\$', '', text)
    text = re.sub(r'"([^"]*)"', '\u201c\\1\u201d', text).replace("'", '\u2019')
    text = re.sub(r'[()]', '', text)
    return re.sub(r'[\s.]+', '-', text.lower()).strip('-')


def site_anchor(md_path, anchor):
    """Translate a source anchor into the anchor Mintlify renders for the same heading."""
    for heading in re.findall(r'^#{1,6} (.*)$', FENCE_RE.sub('', md_path.read_text()), re.M):
        if github_slug(heading) == anchor:
            return mintlify_slug(heading)
    return anchor


def escape_prose(text):
    """Escape MDX-significant characters in prose while preserving code and math."""
    fragments = re.split(r'(`[^`\n]*`|\$\$.*?\$\$|\$[^\n$]+\$|!\[[^\]]*\]\([^)]*\))', text, flags=re.S)
    for j in range(0, len(fragments), 2):
        fragments[j] = fragments[j].replace('<', '&lt;').replace('{', '&#123;').replace('}', '&#125;')
    return ''.join(fragments)


def convert(source, url_of):
    text = source.read_text()
    lines = text.splitlines()
    title = lines[0].removeprefix('# ').strip()
    body = '\n'.join(lines[1:]).strip()

    def link(match):
        bang, label, target = match.group(1), match.group(2), match.group(3)
        if '://' in target or target.startswith('mailto:'):
            return match.group(0)
        path_part, sep, anchor = target.partition('#')
        if not path_part:
            return f'{bang}[{label}](#{site_anchor(source, anchor)})'
        resolved = (source.parent / path_part).resolve()
        if resolved.suffix.lower() in IMAGE_SUFFIXES:
            return f'{bang}[{label}](/tutorial/figures/{resolved.name})'
        if resolved in url_of:
            if sep:
                anchor = site_anchor(resolved, anchor)
            return f'{bang}[{label}]({url_of[resolved]}{sep}{anchor})'
        if resolved.is_relative_to(ROOT):
            # No public repository exists; name the repository path instead of inventing a URL.
            rel = resolved.relative_to(ROOT).as_posix()
            return f'{label} (`{rel}`)' if label.strip('`') != rel else f'`{rel}`'
        return match.group(0)

    pieces = re.split(r'(```.*?```)', body, flags=re.S)
    for i in range(0, len(pieces), 2):
        # Rewrite links only outside inline code spans, which may show link syntax as an example.
        spans = re.split(r'(`[^`\n]*`)', pieces[i])
        for j in range(0, len(spans), 2):
            spans[j] = re.sub(r'(!?)\[([^\]]+)\]\(([^)\s]+)\)', link, spans[j])
        pieces[i] = escape_prose(''.join(spans))
    return title, ''.join(pieces)


def library_page():
    """An API index of every #@save definition, grouped by defining chapter."""
    rows = ['The book builds `agentrl` one `#@save` block at a time. Each entry links to the chapter that derives it. '
            'Implementations live in topic modules; `agentrl/__init__.py` only re-exports the public API. '
            'Regenerate the package with `python3 scripts/build_library.py`.\n']
    for path in sorted((TUTORIAL / 'chapters').glob('*.md')):
        entries = []
        for code in saved_blocks(path):
            for node in ast.parse(code).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    doc = (ast.get_docstring(node) or '').split('\n')[0]
                    if isinstance(node, ast.FunctionDef):
                        sig = f'{node.name}({ast.unparse(node.args)})'
                    else:
                        sig = f'class {node.name}'
                    entries.append(f'| `{sig}` | {escape_prose(doc)} |')
        if entries:
            title = path.read_text().splitlines()[0].removeprefix('# ')
            module = CHAPTER_MODULES[path.stem.split('-')[0]]
            rows.append(f'\n## [{title}](/tutorial/chapters/{path.stem})\n\n'
                        f'Implementation: `agentrl/{module}.py`.\n\n'
                        '| Definition | Summary |\n| --- | --- |\n' + '\n'.join(entries))
    return '\n'.join(rows)


def main():
    chapters = sorted((TUTORIAL / 'chapters').glob('*.md'))
    pages = [TUTORIAL / 'README.md', TUTORIAL / 'sources.md', TUTORIAL / 'editorial-guide.md',
             TUTORIAL / 'labs' / 'README.md'] + chapters
    # Optional Chinese edition: same chapter files under tutorial/zh/, prose translated only.
    zh_root = TUTORIAL / 'zh'
    zh_chapters = sorted((zh_root / 'chapters').glob('*.md'))
    zh_pages = [p for p in [zh_root / 'README.md', zh_root / 'glossary.md', zh_root / 'labs' / 'README.md']
                if p.exists()] + zh_chapters
    pages += zh_pages
    url_of = {p.resolve(): '/' + p.relative_to(ROOT).with_suffix('').as_posix() for p in pages}
    url_of[(ROOT / 'README.md').resolve()] = '/index'
    if (SITE / 'tutorial').exists():
        shutil.rmtree(SITE / 'tutorial')
    for page in pages:
        title, body = convert(page, url_of)
        out = SITE / (url_of[page.resolve()].lstrip('/') + '.mdx')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('---\ntitle: ' + json.dumps(title) + '\n---\n\n' + body + '\n')
    (SITE / 'tutorial' / 'library.mdx').write_text('---\ntitle: "The agentrl library"\n---\n\n' + library_page() + '\n')
    figures = SITE / 'tutorial' / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    image_assets = sorted(p for p in (TUTORIAL / 'figures').iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for asset in image_assets:
        shutil.copyfile(asset, figures / asset.name)
    title, body = convert(TUTORIAL / 'README.md', url_of)
    (SITE / 'index.mdx').write_text('---\ntitle: ' + json.dumps(title) + '\n---\n\n' + body + '\n')

    def chapter_groups(chapter_files, names):
        groups = []
        for (_, match), name in zip(PARTS, names):
            members = [url_of[p.resolve()].lstrip('/') for p in chapter_files if match(p.stem.split('-')[0])]
            if members:
                groups.append({'group': name, 'pages': members})
        return groups

    en_groups = [{'group': 'Start here', 'pages': ['index']}]
    en_groups += chapter_groups(chapters, [name for name, _ in PARTS])
    en_groups.append({'group': 'Reference', 'pages': ['tutorial/library', 'tutorial/labs/README', 'tutorial/sources', 'tutorial/editorial-guide']})
    navigation = {'groups': en_groups}
    if zh_pages:
        start = [url_of[p.resolve()].lstrip('/') for p in [zh_root / 'README.md'] if p.exists()]
        zh_groups = [{'group': '开始', 'pages': start}] if start else []
        zh_groups += chapter_groups(zh_chapters, ZH_PARTS)
        reference = [url_of[p.resolve()].lstrip('/') for p in [zh_root / 'glossary.md', zh_root / 'labs' / 'README.md'] if p.exists()]
        zh_groups.append({'group': '参考', 'pages': reference + ['tutorial/library', 'tutorial/sources']})
        navigation = {'languages': [{'language': 'en', 'groups': en_groups}, {'language': 'cn', 'groups': zh_groups}]}
    config = {
        '$schema': 'https://mintlify.com/docs.json', 'theme': 'mint',
        'name': 'Agentic RL',
        'colors': {'primary': '#2a78d6', 'light': '#3987e5', 'dark': '#256abf'},
        'navigation': navigation,
    }
    (SITE / 'docs.json').write_text(json.dumps(config, indent=2) + '\n')
    for stale in ('about.mdx', 'build-manifest.json'):
        (SITE / stale).unlink(missing_ok=True)
    print(f'Generated {len(pages)} pages ({len(chapters)} chapters, {len(zh_chapters)} translated) and {len(image_assets)} figures')


if __name__ == '__main__':
    main()
