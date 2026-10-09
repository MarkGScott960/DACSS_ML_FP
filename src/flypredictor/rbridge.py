"""The single place where Python calls R.

R is used only for what only R can do here: the ERT package's ``get_eps``
(real-time check, threshold variants, v10 relabelling) and, as a fallback,
converting V-Dem's .RData file to Parquet. Every call goes through
:func:`run_r`, which checks the exit code and that the promised output files
exist, so a silent R failure cannot leak into the analysis.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


class RError(RuntimeError):
    """Raised when an R script fails or does not produce its expected outputs."""


def find_rscript(configured: str | None = None) -> str:
    """Locate Rscript: config value, then PATH, then the standard Windows install folders."""
    if configured:
        if Path(configured).is_file():
            return str(configured)
        raise RError(f"Configured Rscript not found: {configured}")
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path
    if os.name == "nt":
        hits = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
        if hits:
            return hits[-1]  # newest version
    raise RError(
        "Rscript not found. Install R (https://cran.r-project.org) or set `rscript:` in config/project.yaml "
        "to the full path, e.g. C:/Program Files/R/R-4.5.2/bin/Rscript.exe"
    )


def run_r(
    script: Path | str,
    args: Sequence[str] = (),
    *,
    cwd: Path | str,
    expected_outputs: Iterable[Path | str] = (),
    rscript: str | None = None,
    timeout: int | None = None,
    echo: bool = True,
) -> subprocess.CompletedProcess:
    """Run an R script with Rscript and fail loudly on any problem.

    Parameters
    ----------
    script : path to the .R file
    args : command-line arguments (read in R with commandArgs(trailingOnly = TRUE))
    cwd : working directory, normally the repository root (so renv activates)
    expected_outputs : files that must exist after a successful run
    rscript : explicit Rscript path (default: find_rscript())
    timeout : seconds before the run is killed (None = no limit)
    echo : print R's stdout/stderr
    """
    exe = find_rscript(rscript)
    # Not --vanilla: that would skip .Rprofile, which is how renv activates the pinned library.
    cmd = [exe, "--no-save", str(script), *map(str, args)]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if echo:
        if proc.stdout.strip():
            print(proc.stdout.rstrip())
        if proc.stderr.strip():
            print(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RError(f"R script failed (exit {proc.returncode}): {script}\n{proc.stderr[-3000:]}")
    missing = [str(p) for p in expected_outputs if not Path(p).is_file()]
    if missing:
        raise RError(f"R script finished but expected outputs are missing: {missing}")
    return proc
