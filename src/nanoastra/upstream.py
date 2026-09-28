"""Expose the pinned, unmodified checkout to local teaching scripts."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def activate_verl() -> Path:
    entry = json.loads((ROOT / "upstream.lock.json").read_text())["verl"]
    checkout = ROOT / entry["path"]
    if not (checkout / ".git").exists():
        raise RuntimeError("Missing upstream checkout. Run: python scripts/fetch_upstream.py")
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != entry["commit"]:
        raise RuntimeError(f"Upstream revision mismatch: {actual}")
    changes = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    ).strip()
    if changes:
        raise RuntimeError("Upstream checkout has local changes; use a clean pinned checkout.")
    sys.path.insert(0, str(checkout))
    return checkout
