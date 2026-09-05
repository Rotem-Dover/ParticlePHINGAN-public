"""Discover trained generator checkpoints under the runs tree.

One root is scanned (see `common/paths.py::RUNS_DIR`), holding local training
runs plus anything imported through the cluster tunnel
(`playground/services/cluster.py::fetch_checkpoint` mirrors into it):

    <RUNS_DIR>/
      logger_l/<RUN_ID>/checkpoints/*.ckpt
      logger_cnt/<RUN_ID>/checkpoints/*.ckpt
      logger_sec/<RUN_ID>/checkpoints/*.ckpt

A `version_N` run-id directory is picked up the same as any other, since
`glob('*/checkpoints/*.ckpt')` doesn't care about the run-id format.
"""
from __future__ import annotations

from common.paths import RUNS_DIR


# Maps stage → logger subdir → eligible generator class names.
STAGE_TO_LOGGER = {
    "length":      "logger_l",
    "continuous":  "logger_cnt",
    "secondary":   "logger_sec",
}

# Each stage exposes a single generator class.
STAGE_TO_CLASSES = {
    "length":      ["LengthGenerator"],
    "continuous":  ["ContinuousGenerator"],
    "secondary":   ["SecondaryGenerator"],
}


def list_checkpoints(stage: str) -> list[dict]:
    sub = STAGE_TO_LOGGER.get(stage)
    if sub is None:
        return []
    items = []
    for source, tb_root in (("runs", RUNS_DIR),):
        root = tb_root / sub
        if not root.exists():
            continue
        for ckpt in sorted(root.glob("*/checkpoints/*.ckpt")):
            run_dir = ckpt.parent.parent
            items.append({
                "run_id": run_dir.name,
                "filename": ckpt.name,
                "path": str(ckpt),
                "source": source,
                "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
                # No class hint is read off disk — the frontend falls back
                # to substring matching on `run_id` / `filename`.
                "class_hint": None,
            })
    return items


def all_checkpoints() -> dict[str, list[dict]]:
    return {stage: list_checkpoints(stage) for stage in STAGE_TO_LOGGER}
