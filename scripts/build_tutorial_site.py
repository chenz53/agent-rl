"""Generate the Mintlify reading edition from canonical repository Markdown."""
import gzip
import json
from pathlib import Path
import re
import shutil
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'website'

def main():
    # Export only the active coding edition. Historical source remains untouched.
    generated = set()
    markdown = list((ROOT/'tutorial').rglob('*.md')) + [ROOT/'experiments/009-coding-tutorial.md']
    paths = {p.resolve(): p.relative_to(ROOT).with_suffix('').as_posix() for p in markdown}
    assets = set()
    for source in markdown:
        text = source.read_text()
        lines = text.splitlines()
        title = lines[0].removeprefix('# ')
        body = '\n'.join(lines[1:]).strip()
        def link(match):
            label, target = match.group(1), match.group(2)
            if '://' in target or target.startswith('#'):
                return match.group(0)
            path_part, separator, anchor = target.partition('#')
            resolved = (source.parent / path_part).resolve()
            if resolved in paths:
                url = '/' + paths[resolved]
                if separator:
                    url += '#' + anchor
            elif resolved.is_file() and resolved.is_relative_to(ROOT):
                relative = resolved.relative_to(ROOT)
                destination = SITE/'assets/source'/relative
                # Source files become explicit downloadable text; no execution.
                destination = destination.with_name(destination.name.removesuffix(".gz") + ".txt")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if resolved.suffix == ".gz":
                    with gzip.open(resolved, "rt", encoding="utf-8") as handle:
                        destination.write_text(handle.read())
                else:
                    shutil.copyfile(resolved, destination)
                assets.add(destination.relative_to(SITE).as_posix())
                generated.add(destination.resolve())
                url = '/' + destination.relative_to(SITE).as_posix()
            else:
                # Directory references open the repository-navigation guide.
                url = '/about'
            return f'[{label}]({url})'
        # Keep executable blocks byte-for-byte; translate only prose/math/links.
        pieces = re.split(r'(```.*?```)', body, flags=re.S)
        for i in range(0, len(pieces), 2):
            prose = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, pieces[i])
            prose = prose.replace('\\[', '$$').replace('\\]', '$$')
            prose = re.sub(r'\\\((.*?)\\\)', r'$\1$', prose, flags=re.S)
            # Escape prose-only MDX syntax; preserve inline code and math.
            fragments = re.split(r"(`[^`]*`|\$\$.*?\$\$|\$[^\n$]+\$)", prose, flags=re.S)
            for j in range(0, len(fragments), 2):
                fragments[j] = fragments[j].replace("<", "&lt;").replace("{", "&#123;").replace("}", "&#125;")
            pieces[i] = "".join(fragments)
        out = SITE/(paths[source.resolve()]+'.mdx')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('---\ntitle: '+json.dumps(title)+'\n---\n\n'+''.join(pieces)+'\n')
        generated.add(out.resolve())
    chapters = [paths[p.resolve()] for p in sorted((ROOT/'tutorial/chapters').glob('*.md'))]
    # Remove obsolete generated pages only after replacements have been written.
    for name in ('tutorial', 'docs', 'experiments', 'assets'):
        tree = SITE/name
        if tree.exists():
            for file in tree.rglob('*'):
                if file.is_file() and file.resolve() not in generated:
                    file.unlink()
    config = {
        '$schema': 'https://mintlify.com/docs.json', 'theme': 'mint',
        'name': 'nanoAstra · Hands-on Post-training',
        'colors': {'primary': '#0D9488', 'light': '#14B8A6', 'dark': '#0F766E'},
        'navigation': {'groups': [
            {'group': 'Start here', 'pages': ['index','tutorial/README']},
            {'group': 'Coding task and agent interaction', 'pages': chapters[:4]},
            {'group': 'From feedback to policy updates', 'pages': chapters[4:8]},
            {'group': 'Other post-training signals', 'pages': chapters[8:11]},
            {'group': 'Systems, evaluation, and experiments', 'pages': chapters[11:]},
            {'group': 'Evidence and reproducibility', 'pages': ['tutorial/sources','tutorial/editorial-guide','experiments/009-coding-tutorial','about']},
        ]},
    }
    (SITE/'docs.json').write_text(json.dumps(config, indent=2)+'\n')
    (SITE/'build-manifest.json').write_text(json.dumps({'generated_markdown_pages': len(markdown), 'new_chapters': len(chapters), 'active_case': 'coding repair', 'source_assets': sorted(assets)}, indent=2)+'\n')
    print(f'Generated {len(markdown)} pages, {len(chapters)} new chapters, coding-only reading route')

if __name__ == '__main__':
    main()
