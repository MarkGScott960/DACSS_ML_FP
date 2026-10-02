"""Repo-relative paths, usable from any notebook or script in the repo."""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: cwd) until a .git folder is found."""
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / ".git").exists():
            return p
    raise FileNotFoundError(f"No .git found above {here}; open the notebook from inside the repo.")


ROOT = find_repo_root(Path(__file__).parent)
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFEST = RAW / "MANIFEST.json"
