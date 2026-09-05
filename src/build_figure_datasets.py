"""Build the input pickle files the figure scripts (system_vs_g4/: figs 8,
10, 11, 12) consume: six pickled numpy arrays per truth stem -- raw
step-length pairs plus filtered continuous/secondary pairs -- written under
DATASETS_DIR, with a provenance manifest alongside them (same shape as
build_training_datasets.py's).

`<stem>_secondary_Y_filtered.pkl` is (M, 1), e_sec-only, matching the
secondary generator's single output column; fig 10's `__main__` assumes the
1-column form and fig 12's secondary metric does not use theta.

Filter semantics:
- step_length X/Y: the raw (E, L) rows, unfiltered.
- continuous filtered: drop rows with e_cnt == 0, rows with e_cnt == E (all
  energy lost), and rows in the deterministic Bethe-Bloch region -- a NaN
  mu in the analytic continuous-loss lookup, the same predicate the
  continuous training datamodule filters by.
- secondary filtered: keep rows with post-continuous energy above
  MIN_ENERGY_CUTOFF and e_sec > 0.

Rows are shuffled per stage (`np.random.RandomState(0).permutation`): the
figure scripts batch sequentially, so event-ordered rows would make every
batch intra-track-correlated and bias fig 12's D_E distributions.
"""
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch

from common.config import MIN_ENERGY_CUTOFF
from common.paths import DATASETS_DIR, PATH_TO_GEANT4_DATA
from data_handling.truth_extract import (
    extract_continuous_xy, extract_secondary_xy,
)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousYScaler,
)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _deterministic_region_mask(x_el: np.ndarray) -> np.ndarray:
    """True where (E, L) sits in the deterministic Bethe-Bloch region,
    defined as a NaN mu in the analytic continuous-loss lookup -- mirror of
    the continuous training datamodule's NaN-row filter
    (train_networks/continuous_generator/datamodule.py:249-262).

    `_compute_lookup_indices` takes the two axes as SEPARATE 1-D tensors
    (`kinetic_energy`, `step_length`), not a combined (N, 2) tensor -- the
    datamodule calls it as
    `scaler._compute_lookup_indices(self.X_real[:, 0], self.X_real[:, 1])`,
    which this mirrors exactly.
    """
    scaler = ContinuousYScaler.load()
    x_t = torch.from_numpy(x_el).type(torch.float32)
    kin_idx, step_idx = scaler._compute_lookup_indices(x_t[:, 0], x_t[:, 1])
    mu = scaler.lookup_table[kin_idx, step_idx, 0]
    return torch.isnan(mu).numpy()


def _dump(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        pickle.dump(arr, f)


def build(root_path: Path, out_dir: Path = DATASETS_DIR) -> dict:
    root_path = Path(root_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = root_path.stem
    files: dict[str, list[int]] = {}

    # ---- continuous + step-length stages (one extraction pass) ----
    x_cnt, y_cnt = extract_continuous_xy(root_path)   # (N,2) [E,L] eV/m, (N,)
    n_raw = x_cnt.shape[0]
    perm = np.random.RandomState(0).permutation(n_raw)
    x_cnt, y_cnt = x_cnt[perm], y_cnt[perm]

    _dump(out_dir / f"{stem}_step_length_X.pkl",
          x_cnt[:, [0]].astype(np.float64))
    _dump(out_dir / f"{stem}_step_length_Y.pkl",
          x_cnt[:, [1]].astype(np.float64))
    files[f"{stem}_step_length_X.pkl"] = [n_raw, 1]
    files[f"{stem}_step_length_Y.pkl"] = [n_raw, 1]

    keep = (y_cnt != 0) & (y_cnt != x_cnt[:, 0])
    keep &= ~_deterministic_region_mask(x_cnt)
    xf = x_cnt[keep].astype(np.float32)
    yf = y_cnt[keep].reshape(-1, 1).astype(np.float32)
    _dump(out_dir / f"{stem}_continuous_X_filtered.pkl", xf)
    _dump(out_dir / f"{stem}_continuous_Y_filtered.pkl", yf)
    files[f"{stem}_continuous_X_filtered.pkl"] = list(xf.shape)
    files[f"{stem}_continuous_Y_filtered.pkl"] = list(yf.shape)
    n_cnt_filtered = int(xf.shape[0])
    del x_cnt, y_cnt, xf, yf

    # ---- secondary stage ----
    x_sec, y_sec = extract_secondary_xy(root_path, drop_theta=True)
    perm = np.random.RandomState(0).permutation(x_sec.shape[0])
    x_sec, y_sec = x_sec[perm], y_sec[perm]

    keep = (x_sec > MIN_ENERGY_CUTOFF) & (y_sec[:, 0] > 0)
    xs = x_sec[keep].astype(np.float32)
    ys = y_sec[keep].astype(np.float32)
    _dump(out_dir / f"{stem}_secondary_X_filtered.pkl", xs)
    _dump(out_dir / f"{stem}_secondary_Y_filtered.pkl", ys)
    files[f"{stem}_secondary_X_filtered.pkl"] = list(xs.shape)
    files[f"{stem}_secondary_Y_filtered.pkl"] = list(ys.shape)
    n_sec_filtered = int(xs.shape[0])

    stat = root_path.stat()
    manifest = {
        "source_root": str(root_path.resolve()),
        "source_bytes": stat.st_size,
        "source_mtime": stat.st_mtime,
        "git_sha": _git_sha(),
        "secondary_y_layout": "e_sec_only_1col",
        "n_rows_raw": int(n_raw),
        "n_rows_continuous_filtered": n_cnt_filtered,
        "n_rows_secondary_filtered": n_sec_filtered,
        "files": files,
    }
    (out_dir / f"{stem}_figure_datasets_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the figure scripts' pkl inputs from a GEANT4 "
                    "truth ROOT file. Read-only on the source file.")
    p.add_argument("--root", type=Path, default=PATH_TO_GEANT4_DATA,
                   help="truth ROOT file (default: the active beam's "
                        f"ionisation-only truth, {PATH_TO_GEANT4_DATA})")
    p.add_argument("--out", type=Path, default=DATASETS_DIR)
    args = p.parse_args()
    print(json.dumps(build(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
