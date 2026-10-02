"""Download helpers: fetch raw files into data/raw/ and record provenance.

Imported by notebooks/01_download.ipynb — this module holds functions only;
the notebook runs them.

Re-running is safe: a file whose SHA-256 already matches
data/raw/MANIFEST.json is skipped.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

# UCDP Candidate (monthly) data is deliberately absent: it is used only for
# the live November 2026 check, never for train/validate/test.
SOURCES: dict[str, dict] = {
    "ged261-csv.zip": {
        "url": "https://ucdp.uu.se/downloads/ged/ged261-csv.zip",
        "dataset": "UCDP Georeferenced Event Dataset (GED) Global",
        "version": "26.1",
    },
    "ged261.pdf": {
        "url": "https://ucdp.uu.se/downloads/ged/ged261.pdf",
        "dataset": "UCDP GED codebook",
        "version": "26.1",
    },
    "gw_iisystem.dat": {
        "url": "http://ksgleditsch.com/data/iisystem.dat",
        "dataset": "Gleditsch-Ward list of independent states",
        "version": "as published",
    },
    "gw_microstatessystem.dat": {
        "url": "http://ksgleditsch.com/data/microstatessystem.dat",
        "dataset": "Gleditsch-Ward list of microstates",
        "version": "as published",
    },
    "DirectContiguity320.zip": {
        "url": "https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip",
        "dataset": "COW Direct Contiguity (1816-2016)",
        "version": "3.2",
    },
}

# Files that cannot be fetched by script (download form on the website).
# The notebook checks they are present and records their hash.
MANUAL_SOURCES: dict[str, dict] = {
    "V-Dem-CY-Core-v16.csv": {
        "url": "https://www.v-dem.net/data/the-v-dem-dataset/country-year-v-dem-core-v16/",
        "dataset": "V-Dem Country-Year Core (CC BY-SA 4.0)",
        "version": "16",
        "how": "Download by hand: choose CSV on the page above, save into data/raw/ unrenamed.",
    },
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def download(name: str, meta: dict, raw_dir: Path, manifest: dict) -> dict:
    """Fetch one file into raw_dir; return its manifest entry."""
    dest = raw_dir / name
    recorded = manifest.get(name, {})

    if dest.exists() and recorded.get("sha256") == sha256(dest):
        print(f"[skip] {name} — already present, hash matches manifest")
        return recorded

    print(f"[get ] {name} <- {meta['url']}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(meta["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    tmp.replace(dest)  # atomic: no half-written file if interrupted

    digest = sha256(dest)
    if recorded.get("sha256") and recorded["sha256"] != digest:
        print(f"  !! {name}: hash differs from previous download — source was updated")

    entry = {
        **meta,
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"  ok {entry['bytes']:,} bytes  sha256={digest[:12]}…")
    return entry


def register_manual(name: str, meta: dict, raw_dir: Path, manifest: dict) -> dict:
    """Hash a hand-downloaded file and return its manifest entry."""
    dest = raw_dir / name
    if not dest.exists():
        raise FileNotFoundError(f"{name} is missing. {meta['how']}\n  {meta['url']}")
    recorded = manifest.get(name, {})
    digest = sha256(dest)
    if recorded.get("sha256") == digest:
        print(f"[skip] {name} — manual file, hash matches manifest")
        return recorded
    if recorded.get("sha256"):
        print(f"  !! {name}: hash differs from the recorded file — a different copy was saved")
    entry = {
        **{k: v for k, v in meta.items() if k != "how"},
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acquired": "manual download",
    }
    print(f"[man ] {name}  {entry['bytes']:,} bytes  sha256={digest[:12]}…")
    return entry
