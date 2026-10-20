#!/usr/bin/env bash
# Render data figures on vp (torch available there) and copy the SVGs back.
# Usage: scripts/remote_figures.sh <workdir-name> [module-prefix ...]   e.g. ch08
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
remote="/home/user/nanoAstra/.bookcheck/${name}"
ssh vp "mkdir -p ${remote}"
python3 scripts/build_library.py >/dev/null
rsync -a --delete --exclude '__pycache__' tutorial scripts agentrl vp:"${remote}/"
ssh vp "cd ${remote} && HF_HUB_OFFLINE=1 /home/user/nanoAstra/.venv/bin/python scripts/make_figures.py $*"
rsync -a vp:"${remote}/tutorial/figures/" tutorial/figures/
