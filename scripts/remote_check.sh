#!/usr/bin/env bash
# Run the tutorial checker on the vp H200 host in a private work directory.
#
# Usage: scripts/remote_check.sh <workdir-name> [--refresh] [chapter-pattern ...]
#   workdir-name   isolates concurrent authors (e.g. "ch06"); files land in
#                  /home/user/nanoAstra/.bookcheck/<workdir-name>
#   --refresh      rewrite Output blocks remotely, then copy the executed chapters back
#   patterns       chapter filename prefixes, e.g. 06 07 (default: all)
#
# Uses the existing remote .venv only; never installs packages.
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
refresh=""
if [[ "${1:-}" == "--refresh" ]]; then refresh="--refresh"; shift; fi
patterns=("$@")
remote="/home/user/nanoAstra/.bookcheck/${name}"
ssh vp "mkdir -p ${remote}"
rsync -a --delete --exclude '__pycache__' tutorial scripts vp:"${remote}/"
args=""
if (( ${#patterns[@]} )); then args="--chapters ${patterns[*]}"; fi
ssh vp "cd ${remote} && HF_HUB_OFFLINE=1 /home/user/nanoAstra/.venv/bin/python scripts/check_tutorial_snippets.py ${refresh} ${args}"
if [[ -n "${refresh}" ]]; then
  if (( ${#patterns[@]} )); then
    for p in "${patterns[@]}"; do
      rsync -a vp:"${remote}/tutorial/chapters/${p}*.md" tutorial/chapters/
    done
  else
    rsync -a vp:"${remote}/tutorial/chapters/" tutorial/chapters/
  fi
fi
