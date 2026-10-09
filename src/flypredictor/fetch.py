"""Download the raw data at a pinned commit and convert it to Parquet.

Sources (config/project.yaml -> sources.ert):
  * inst/ert.csv               ERT v16 country-year file (outcome labels)
  * data/vdem.RData            V-Dem v16 Full+Others country-year (features), as bundled in ERT v16
  * inst/ERT_codebook_v16.pdf  codebook, kept with the data for reference

Files are fetched from raw.githubusercontent.com at the exact commit in the
config, so a re-download is byte-identical; the manifest records each hash.
If a file already exists and matches the manifest, it is not downloaded again.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import file_entry, read_manifest, sha256, write_manifest
from .paths import Paths
from .rbridge import run_r

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
MANIFEST_NAME = "raw_ert_v16"


def _download(url: str, dest: Path, timeout: int = 300) -> None:
    """Stream a URL to dest via a temporary file, so a failed download never leaves a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "flypredictor-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent, suffix=".part"
    ) as tmp:
        shutil.copyfileobj(resp, tmp, length=1 << 20)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def fetch_raw(cfg: dict[str, Any], paths: Paths, force: bool = False) -> Path:
    """Download the ERT/V-Dem files listed in the config and write the raw manifest.

    Returns the manifest path. With ``force=False`` a file is skipped when it is
    already present and its hash matches the existing manifest.
    """
    src = cfg["sources"]["ert"]
    old = read_manifest(MANIFEST_NAME, paths.manifests) or {"files": []}
    known = {e["path"]: e["sha256"] for e in old["files"]}
    entries = []
    for key, rel in src["files"].items():
        dest = paths.raw / "ert" / Path(rel).name
        url = RAW_URL.format(repo=src["repo"], commit=src["commit"], path=rel)
        rel_dest = dest.resolve().relative_to(paths.root.resolve()).as_posix()
        if dest.is_file() and not force and known.get(rel_dest) == sha256(dest):
            print(f"[skip] {rel_dest} (hash matches manifest)")
        else:
            print(f"[get ] {url}")
            _download(url, dest)
        entries.append(file_entry(dest, paths.root, key=key, url=url, commit=src["commit"]))
    out = write_manifest(MANIFEST_NAME, entries, paths.manifests, repo=src["repo"], commit=src["commit"])
    print(f"manifest written: {out}")
    return out


def ert_to_parquet(paths: Paths) -> Path:
    """ERT CSV -> Parquet with explicit dtypes (the CSV's unnamed row-number column is dropped)."""
    df = pd.read_csv(paths.ert_csv, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for col in ("country_id", "year"):
        df[col] = df[col].astype("int64")
    paths.interim.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.ert_parquet, index=False)
    return paths.ert_parquet


def _vdem_via_python(paths: Paths) -> Path:
    import pyreadr  # optional dependency

    result = pyreadr.read_r(str(paths.vdem_rdata))
    if len(result) != 1:
        raise ValueError(f"Expected one object in {paths.vdem_rdata}, found {list(result)}")
    df = next(iter(result.values()))
    df["country_id"] = df["country_id"].astype("int64")
    df["year"] = df["year"].astype("int64")
    paths.interim.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.vdem_parquet, index=False)
    return paths.vdem_parquet


def _vdem_via_r(cfg: dict[str, Any], paths: Paths) -> Path:
    run_r(
        paths.r_scripts / "convert_vdem.R",
        [paths.vdem_rdata, paths.vdem_parquet],
        cwd=paths.root,
        expected_outputs=[paths.vdem_parquet],
        rscript=cfg.get("rscript"),
    )
    return paths.vdem_parquet


def vdem_to_parquet(cfg: dict[str, Any], paths: Paths) -> Path:
    """V-Dem .RData -> Parquet, using pyreadr or R according to config ``rdata_conversion``."""
    mode = cfg.get("rdata_conversion", "auto")
    if mode == "python":
        return _vdem_via_python(paths)
    if mode == "r":
        return _vdem_via_r(cfg, paths)
    try:
        return _vdem_via_python(paths)
    except ImportError:
        print("pyreadr not installed: converting with R instead (R/convert_vdem.R)")
        return _vdem_via_r(cfg, paths)


def check_vdem(paths: Paths, expected_last_year: int) -> dict[str, Any]:
    """Light sanity checks on the converted V-Dem file (shape, years, unique keys)."""
    df = pd.read_parquet(paths.vdem_parquet, columns=["country_id", "year", "v2x_polyarchy"])
    info = {
        "rows": len(df),
        "countries": df["country_id"].nunique(),
        "first_year": int(df["year"].min()),
        "last_year": int(df["year"].max()),
        "duplicate_keys": int(df.duplicated(["country_id", "year"]).sum()),
    }
    if info["last_year"] != expected_last_year:
        raise ValueError(f"V-Dem last year is {info['last_year']}, config says {expected_last_year}")
    if info["duplicate_keys"]:
        raise ValueError("V-Dem has duplicate (country_id, year) rows")
    return info
