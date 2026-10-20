#!/usr/bin/env bash
# Rasterise SVG figures to PNG with headless Chrome for visual inspection.
# Usage: scripts/preview_figure.sh [--dark] tutorial/figures/name.svg [...]
#   -> /tmp/figure-preview/name.png (or name.dark.png with --dark)
set -euo pipefail
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p /tmp/figure-preview
scheme=light; suffix=""
if [[ "${1:-}" == "--dark" ]]; then scheme=dark; suffix=".dark"; shift; fi
for f in "$@"; do
  base=$(basename "$f" .svg)
  read -r w h < <(python3 -c "import re,sys;m=re.search(r'viewBox=\"0 0 ([\d.]+) ([\d.]+)\"',open(sys.argv[1]).read());print(int(float(m[1])),int(float(m[2])))" "$f")
  out="/tmp/figure-preview/${base}${suffix}.png"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-dark-mode=$([[ $scheme == dark ]] && echo true || echo false) \
    --blink-settings=preferredColorScheme=$([[ $scheme == dark ]] && echo 0 || echo 1) \
    --window-size="${w},${h}" --screenshot="$out" "file://$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" >/dev/null 2>&1
  echo "$out"
done
