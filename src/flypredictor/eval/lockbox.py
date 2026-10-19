"""The lockbox: the only way to fit on or score locked-window origins (plan 7.3).

Rules enforced here
-------------------
1. The pre-registration must be frozen: git tag ``prereg-v1`` exists.
2. ``config/`` and ``prereg/`` must be identical to the tagged version, unless the caller
   names a deviation that is recorded in prereg/preregistration.md section 10.
3. Each (model, feature set, event, k, refit) is scored once. A second run needs a
   ``rerun_reason``; both runs are kept in ``results/locked/ledger.json`` with commit and time.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .splits import Fold, locked_folds

TAG = "prereg-v1"


class LockboxError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True)


def prereg_frozen(root: Path) -> bool:
    r = _git(root, "tag", "-l", TAG)
    return r.returncode == 0 and r.stdout.strip() == TAG


def head_commit(root: Path) -> str:
    r = _git(root, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def frozen_files_changed(root: Path) -> list[str]:
    r = _git(root, "diff", "--name-only", TAG, "--", "config", "prereg")
    if r.returncode != 0:
        raise LockboxError(f"git diff against {TAG} failed: {r.stderr.strip()}")
    return [x for x in r.stdout.splitlines() if x.strip()]


class Lockbox:
    def __init__(self, root: Path, cfg: dict[str, Any], deviation: str | None = None):
        self.root = Path(root)
        self.cfg = cfg
        if not prereg_frozen(self.root):
            raise LockboxError(f"git tag {TAG} not found: freeze the pre-registration before touching the locked window")
        changed = frozen_files_changed(self.root)
        if changed:
            prereg = (self.root / "prereg" / "preregistration.md").read_text(encoding="utf-8")
            if not deviation or deviation not in prereg:
                raise LockboxError(
                    f"files changed since {TAG}: {changed}. Log the deviation in prereg section 10 and pass its text as deviation="
                )
        self.deviation = deviation
        self.ledger_path = self.root / "results" / "locked" / "ledger.json"

    def _ledger(self) -> list[dict[str, Any]]:
        if self.ledger_path.is_file():
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        return []

    def folds(self, k: int, event: str, refit: str | None = None) -> list[Fold]:
        return locked_folds(self.cfg, k, event, refit)

    def record(self, model: str, feature_set: str, event: str, k: int, refit: str, rerun_reason: str | None = None,
               **extra: Any) -> dict[str, Any]:
        """Register one locked scoring run. Call BEFORE computing the scores."""
        key = {"model": model, "feature_set": feature_set, "event": event, "k": int(k), "refit": refit}
        ledger = self._ledger()
        previous = [e for e in ledger if all(e.get(a) == b for a, b in key.items())]
        if previous and not rerun_reason:
            raise LockboxError(f"already scored once at {previous[0]['time']}: pass rerun_reason to score again")
        entry = {**key, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "commit": head_commit(self.root), "rerun_reason": rerun_reason, "deviation": self.deviation, **extra}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(ledger + [entry], indent=2), encoding="utf-8")
        return entry

    def masks(self, rows, fold: Fold):
        """(train, eval) masks for a locked fold; the only sanctioned use of allow_locked."""
        from .splits import fold_masks

        return fold_masks(rows, fold, self.cfg, allow_locked=True)
