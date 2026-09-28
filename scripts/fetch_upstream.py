"""Fetch exactly the recorded upstream commit; never overwrite local changes."""

import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
entry = json.loads((root / "upstream.lock.json").read_text())["verl"]
destination = root / entry["path"]
if destination.exists():
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != entry["commit"]:
        raise SystemExit("Existing checkout differs from lock; inspect it before changing revisions.")
    print(f"Already fetched: {actual}")
else:
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", str(destination)], check=True)
    for args in (
        ["remote", "add", "origin", entry["url"]],
        ["fetch", "--depth", "1", "origin", entry["commit"]],
        ["checkout", "--detach", "FETCH_HEAD"],
    ):
        subprocess.run(["git", "-C", str(destination), *args], check=True)
    print(f"Fetched: {entry['commit']}")
