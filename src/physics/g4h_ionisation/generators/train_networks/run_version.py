"""TensorBoard run naming and run metadata, shared by the two WGAN-GP
trainers.

Locating a resume checkpoint is each stage's own responsibility
(`train.py`'s `_resume_ckpt_path`). What both stages share is only what
sits here: which TensorBoardLogger `version` a run logs under, and the
meta.json provenance record.

The RUN_ID travels in the job environment (`PHINGAN_RUN_ID`, set by
`deploy_and_train.sh` through the submitter's `--run-id` flag) rather than a
shared file on disk: a shared file would be overwritten by the next deploy,
so a still-queued job would wake up with the wrong id.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from common.config import RESUME_FROM_CHECKPOINT


def resume_requested() -> bool:
    """`RESUME_FROM_CHECKPOINT` (a config constant) OR `PHINGAN_RESUME`.

    The env var exists because the cluster submitter cannot edit
    `common/config.py` -- and must not: `LEARNING_RATE` keys off
    `RESUME_FROM_CHECKPOINT` at import, so flipping the constant to resume a
    run silently also drops the LR 10x. The env var resumes without that
    coupling; set the constant when you also want that LR drop applied.
    """
    if RESUME_FROM_CHECKPOINT:
        return True
    return os.environ.get("PHINGAN_RESUME", "").strip().lower() in {"1", "true", "yes", "on"}


# Cached so repeated calls within one process agree on the run directory
# (the logger and meta.json must not name two different dirs).
_GENERATED_RUN_ID: str | None = None


def _generate_local_run_id() -> str:
    global _GENERATED_RUN_ID
    if _GENERATED_RUN_ID is None:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        _GENERATED_RUN_ID = f"{ts}_local_{uuid.uuid4().hex[:8]}"
    return _GENERATED_RUN_ID


def get_run_version() -> str:
    """The TensorBoardLogger `version` for this run. Resolution order:

    1. resume: `PHINGAN_RESUME_VERSION` (default `version_0`, matching each
       stage's `_resume_ckpt_path`) -- the resumed leg logs into the SAME
       directory, so the curves continue instead of splitting;
    2. `PHINGAN_RUN_ID` -- a fresh cluster run submitted through
       `deploy_and_train.sh`;
    3. a freshly minted `<ts>_local_<uniq>` -- a run that never went through
       the deploy script.

    Always a string, so TensorBoardLogger never falls back to its numeric
    `version_N` auto-increment.
    """
    if resume_requested():
        return os.environ.get("PHINGAN_RESUME_VERSION", "").strip() or "version_0"
    run_id = os.environ.get("PHINGAN_RUN_ID", "").strip()
    return run_id or _generate_local_run_id()


def write_run_meta(log_dir: str | Path, meta: dict) -> None:
    """Persist provenance (run_id / git_sha / stage) as meta.json in the run
    directory, creating it if needed.

    A resumed run walks forward in legs (MAX_EPOCHS 100_000 against a 72 h
    walltime), and each leg can carry a different git_sha -- so this reads
    the existing file back and appends rather than overwriting: top-level
    keys always reflect the latest leg, and `legs` is the append-only history
    of every leg's meta dict. Mostly for the human who finds a run directory
    six months later, but it is now also read back by this function itself on
    the next resume."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        old_legs = old.get("legs")
        if old_legs is None:
            old_legs = [{k: v for k, v in old.items() if k != "legs"}]
        legs = [*old_legs, meta]
    else:
        legs = [meta]
    combined = {**meta, "legs": legs}
    meta_path.write_text(json.dumps(combined, indent=2) + "\n")
