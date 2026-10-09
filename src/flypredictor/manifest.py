"""SHA-256 manifests for every raw file.

A manifest is a small JSON file in data/manifests/ (committed to git) that
records, for each raw file: its path relative to the repository root, size,
SHA-256 hash, where it came from and when. Results are only reproducible if
the inputs are byte-identical, and the manifest is how that is checked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK = 1 << 20  # 1 MiB


def sha256(path: Path | str) -> str:
    """Hex SHA-256 of a file, read in chunks so large files do not load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path, root: Path, **meta: Any) -> dict[str, Any]:
    """Manifest entry for one file. Extra keyword arguments (url, commit, ...) are stored as-is."""
    path = Path(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }


def write_manifest(name: str, entries: list[dict[str, Any]], manifests_dir: Path, **meta: Any) -> Path:
    """Write data/manifests/<name>.json. Entries are sorted by path so diffs are stable."""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out = manifests_dir / f"{name}.json"
    payload = {"name": name, **meta, "files": sorted(entries, key=lambda e: e["path"])}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def read_manifest(name: str, manifests_dir: Path) -> dict[str, Any] | None:
    p = manifests_dir / f"{name}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verify_manifest(name: str, manifests_dir: Path, root: Path) -> list[str]:
    """Return a list of problems (missing files, size or hash mismatches). Empty list = all good."""
    man = read_manifest(name, manifests_dir)
    if man is None:
        return [f"manifest {name}.json not found in {manifests_dir}"]
    problems = []
    for e in man["files"]:
        p = root / e["path"]
        if not p.is_file():
            problems.append(f"missing: {e['path']}")
            continue
        if p.stat().st_size != e["bytes"]:
            problems.append(f"size changed: {e['path']}")
            continue
        if sha256(p) != e["sha256"]:
            problems.append(f"hash changed: {e['path']}")
    return problems
