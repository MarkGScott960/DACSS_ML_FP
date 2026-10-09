"""Project paths, resolved from the repository root.

The root is found by walking up from the current directory (or from this file)
until a folder containing both ``pyproject.toml`` and ``config/project.yaml``
is found. Notebooks can therefore be run from any sub-folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Return the repository root (the folder holding pyproject.toml and config/)."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())
    for base in candidates:
        for folder in [base, *base.parents]:
            if (folder / "pyproject.toml").is_file() and (folder / "config" / "project.yaml").is_file():
                return folder
    raise FileNotFoundError(
        "Could not find the project root (a folder with pyproject.toml and config/project.yaml). "
        "Run from inside the repository, or pass root= explicitly."
    )


@dataclass(frozen=True)
class Paths:
    """All folders the pipeline reads or writes. Create with ``Paths.from_root()``."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "Paths":
        return cls(find_root(Path(root)) if root is not None else find_root())

    # --- configuration -------------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "config"

    # --- data ------------------------------------------------------------------
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    # --- code and outputs ------------------------------------------------------
    @property
    def r_scripts(self) -> Path:
        return self.root / "R"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    # --- canonical files -------------------------------------------------------
    @property
    def ert_csv(self) -> Path:
        return self.raw / "ert" / "ert.csv"

    @property
    def vdem_rdata(self) -> Path:
        return self.raw / "ert" / "vdem.RData"

    @property
    def vdem_parquet(self) -> Path:
        return self.interim / "vdem_v16.parquet"

    @property
    def ert_parquet(self) -> Path:
        return self.interim / "ert_v16.parquet"

    @property
    def episodes_parquet(self) -> Path:
        return self.processed / "democratic_episodes.parquet"

    @property
    def states_parquet(self) -> Path:
        return self.processed / "country_year_states.parquet"

    @property
    def labels_parquet(self) -> Path:
        return self.processed / "labels.parquet"

    @property
    def panel_parquet(self) -> Path:
        return self.processed / "panel.parquet"

    def ensure(self) -> "Paths":
        """Create every data/results folder that may not exist yet."""
        for p in (self.raw, self.interim, self.processed, self.manifests, self.results, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self
