"""Build per-stage training datasets from a GEANT4 truth ROOT file, with a
provenance manifest.

Reads the truth file via `data_handling.truth_extract` (never
`data_handling.reader` -- that module caches a `.pkl` beside its source and
materialises the whole frame in memory, and truth files can be far larger
than that is worth) and writes the extracted `(X, Y)` pairs under
`DATASETS_DIR` in the exact on-disk format `physics.g4h_ionisation.
generators.train_networks.secondary_generator.datamodule.DataModule` reads:
one `.npy` per array, named `<stem>_X.npy` / `<stem>_Y.npy`.

**The secondary target's width is an explicit, recorded choice, not a
hardcoded one.** `physics.g4h_ionisation.generators.train_networks.
secondary_generator.datamodule.DataModule`'s constructor defaults to
`drop_theta=True`, used unmodified by
`secondary_generator/train.py::train_gan()`, so it expects
`secondary_Y_drop_theta.npy` to be `(N, 1)` -- `e_sec` only, matching the
secondary generator's single-output checkpoint contract. The two-column
`[theta, e_sec]` form (`truth_extract.SECONDARY_2COL_COLUMNS`; NOT
`SecondaryFeats` order, which describes the secondary generator's single
output column and does not describe this two-column truth array) stays
useful for comparison and for `DataModule(drop_theta=False)` callers.
Neither form is "wrong" -- they serve different callers. So `build()`
writes BOTH forms by default, to two distinct filenames:

- `secondary_Y.npy` -- `(N, 2)`, `[theta, e_sec]`. The name `DataModule`
  reads with `drop_theta=False`. Built from `extract_secondary_xy(...,
  drop_theta=False)`.
- `secondary_Y_drop_theta.npy` -- `(N, 1)`, `e_sec` only, matching the
  secondary generator's checkpoint contract. Derived by slicing the
  2-column array's `truth_extract.SECONDARY_2COL_E_SEC_COL` column rather
  than re-streaming the ROOT file a second time -- both forms share the same
  X and the same row order, so a second `extract_secondary_xy(...,
  drop_theta=True)` pass would just reproduce a column already in hand, at
  the cost of a second full-file scan. (Slicing with
  `SecondaryFeats.E_SEC_IDX` here would silently select theta instead of the
  secondary loss -- see the constant's comment in `truth_extract.py`.)

`--drop-theta` / `--keep-theta` (mutually exclusive; default: build both)
let a caller opt into building only one form when disk space or wall time
matters. `secondary_target` is recorded in the manifest either way, along
with every array actually written and its shape, so a future reader can tell
from the manifest alone which target(s) a given build produced -- and,
downstream, which target a checkpoint was trained against.

The continuous stage has only one target shape and is unaffected by any of
this: `continuous_X.npy` is `(N, 2)` (`[KineticEnergy, StepLength]`),
`continuous_Y.npy` is `(N,)` (`ContinuousLoss`), from `extract_continuous_xy`.

**The manifest is the provenance artifact.** Without a record of exactly
what truth file (identity + size), what code (git SHA), what filters, what
arrays, and how many rows survived, a claim that a checkpoint was trained
against a given dataset is unfalsifiable -- two checkpoints could silently
come from different data with no way to tell after the fact. `filters` is
imported from `data_handling.truth_extract.ROW_LOCAL_FILTER_DESCRIPTIONS` --
itself derived from `ROW_LOCAL_FILTERS`,
the executable `(description, callable)` list `_apply_row_local_filters`
actually runs -- rather than copied as a second hand-typed string list, so
the manifest cannot list a filter that isn't applied or omit one that is
(see that module's docstring).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from common.paths import DATASETS_DIR, PATH_TO_GEANT4_DATA
from data_handling.truth_extract import (
    ROW_LOCAL_FILTER_DESCRIPTIONS, SECONDARY_2COL_E_SEC_COL,
    extract_continuous_xy, extract_secondary_xy,
)

# Valid values for `secondary_target` / `build()`'s corresponding parameter.
SECONDARY_TARGET_BOTH = "both"
SECONDARY_TARGET_WITH_THETA = "with_theta"   # secondary_Y.npy, (N, 2)
SECONDARY_TARGET_DROP_THETA = "drop_theta"   # secondary_Y_drop_theta.npy, (N, 1)
_VALID_SECONDARY_TARGETS = (
    SECONDARY_TARGET_BOTH, SECONDARY_TARGET_WITH_THETA, SECONDARY_TARGET_DROP_THETA,
)


def _git_sha() -> str:
    """The git SHA of the code that produced this manifest. Falls back to
    "unknown" (never raises) so a build run outside a git checkout -- e.g.
    from an extracted tarball -- still produces a manifest, just a less
    complete one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def build(root_path: Path, out_dir: Path,
          secondary_target: str = SECONDARY_TARGET_BOTH) -> dict:
    """Extract both stages' `(X, Y)` pairs from `root_path`, save them as
    `.npy` under `out_dir`, write `manifest.json` alongside them, and return
    the manifest dict.

    `secondary_target` picks which secondary-generator Y array(s) get
    written -- see module docstring. Read-only on `root_path`:
    `truth_extract` streams the ROOT file via `uproot.iterate` and never
    opens it for writing. `out_dir` is created if missing; callers pass
    `DATASETS_DIR` in production and a tmp path in tests.
    """
    if secondary_target not in _VALID_SECONDARY_TARGETS:
        raise ValueError(
            f"secondary_target={secondary_target!r} must be one of "
            f"{_VALID_SECONDARY_TARGETS}")

    root_path = Path(root_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always extract the 2-column form: it is a strict superset of the
    # 1-column one (same X, same row order, its SecondaryLoss column is the
    # whole content of the 1-column form), so the 1-column array is derived
    # from it in memory rather than re-streaming the ROOT file a second time.
    # The column index comes from the EXTRACTOR's layout constant, never from
    # SecondaryFeats -- the enum describes the secondary generator's single
    # output column, not this truth array's [theta, e_sec] layout.
    x_sec, y_sec_2col = extract_secondary_xy(root_path, drop_theta=False)
    y_sec_1col = y_sec_2col[:, [SECONDARY_2COL_E_SEC_COL]]
    nz = y_sec_1col[y_sec_1col > 0]
    assert nz.size == 0 or nz.min() > 1.0, (
        "drop_theta target's nonzero minimum is below 1 eV -- a delta-ray "
        "energy is bounded below by the production threshold (keV scale), so "
        "this is the wrong column (theta?) or the wrong units")

    x_cnt, y_cnt = extract_continuous_xy(root_path)

    arrays: dict[str, list[int]] = {}

    np.save(out_dir / "secondary_X.npy", x_sec)
    arrays["secondary_X.npy"] = list(x_sec.shape)

    if secondary_target in (SECONDARY_TARGET_BOTH, SECONDARY_TARGET_WITH_THETA):
        np.save(out_dir / "secondary_Y.npy", y_sec_2col)
        arrays["secondary_Y.npy"] = list(y_sec_2col.shape)

    if secondary_target in (SECONDARY_TARGET_BOTH, SECONDARY_TARGET_DROP_THETA):
        np.save(out_dir / "secondary_Y_drop_theta.npy", y_sec_1col)
        arrays["secondary_Y_drop_theta.npy"] = list(y_sec_1col.shape)

    np.save(out_dir / "continuous_X.npy", x_cnt)
    arrays["continuous_X.npy"] = list(x_cnt.shape)
    np.save(out_dir / "continuous_Y.npy", y_cnt)
    arrays["continuous_Y.npy"] = list(y_cnt.shape)

    stat = root_path.stat()
    manifest = {
        "source_root": str(root_path.resolve()),
        "source_bytes": stat.st_size,
        "source_mtime": stat.st_mtime,
        "git_sha": _git_sha(),
        "filters": list(ROW_LOCAL_FILTER_DESCRIPTIONS),
        "secondary_target": secondary_target,
        "arrays": arrays,
        "n_rows_secondary": int(x_sec.shape[0]),
        "n_rows_continuous": int(x_cnt.shape[0]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build training datasets from a GEANT4 "
                     "truth ROOT file. Read-only on the source file.")
    p.add_argument("--root", type=Path, default=PATH_TO_GEANT4_DATA,
                   help="path to a GEANT4 truth ROOT file (default: the "
                        f"active beam's ionisation-only truth, "
                        f"{PATH_TO_GEANT4_DATA})")
    p.add_argument("--out", type=Path, default=DATASETS_DIR,
                   help=f"output directory, default {DATASETS_DIR}")
    target_group = p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--drop-theta", dest="secondary_target", action="store_const",
        const=SECONDARY_TARGET_DROP_THETA,
        help="write only the (N, 1) e_sec-only secondary target "
             "(secondary_Y_drop_theta.npy)")
    target_group.add_argument(
        "--keep-theta", dest="secondary_target", action="store_const",
        const=SECONDARY_TARGET_WITH_THETA,
        help="write only the (N, 2) [theta, e_sec] secondary target "
             "(secondary_Y.npy)")
    p.set_defaults(secondary_target=SECONDARY_TARGET_BOTH)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    manifest = build(args.root, args.out, secondary_target=args.secondary_target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
