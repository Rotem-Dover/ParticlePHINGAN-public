"""Shared G4 phase-space bin slicing.

`api.g4_data.hist1d` (the G4 truth histogram) and the bin-conditioned
sampling paths in `api.samplers` / `api.generators` must agree on which
rows fall inside a pinned (E, L) cell. This module owns that cut so the
G4 trace and the MC/NN conditioning labels always derive from the
identical row set.

Units follow the tracks dataframe: KineticEnergy and ContinuousLoss in
eV, StepLength in metres — the same units the samplers' label tensors
expect, so values pass through unconverted.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch

from common.enums import G4Columns
from common.paths import TRACKS_DIR
from data_handling.reader import read_geant4_simulation_output

from playground.cache import get_or_load
from playground.services.sampling import STAGE_CONFIG


# Full-frame loads (Table tab, G4 browser histograms) materialize every
# branch as a pandas DataFrame — tens of GB for the largest 100k-event
# tracks file, a guaranteed OOM on a 32 GB machine. Pkl-less ROOTs above
# this size are refused; the Pull tab is unaffected (it streams via
# data_handling.slim_reader).
_FULL_LOAD_MAX_ROOT_BYTES = 8 * 1024**3


class TracksFileTooLarge(RuntimeError):
    """A tracks ROOT too big for a full-frame load (no full pkl exists)."""


def normalize_awkward_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert awkward_pandas-backed columns to plain pandas categoricals.

    uproot's `.arrays(library="pd")` returns string branches (ProcessName)
    as awkward_pandas extension arrays — and the reader pickles the frame
    as-is, so cached .pkl files carry them too. Series-level ops tolerate
    them, but numpy-level ops raise a bare ValueError (e.g. stage_row_mask's
    `mask &= (df[col].values == proc)`). Paying the conversion once at load
    keeps every downstream consumer (row masks, hist1d, the Table tab's
    df.query) on fast plain-numpy comparisons.
    """
    for col in df.columns:
        if type(df[col].dtype).__module__.startswith("awkward"):
            df[col] = pd.Categorical(df[col].astype(str))
    return df


def load_tracks(name: str) -> pd.DataFrame:
    """Cache-backed loader for a tracks file by base name (e.g. 100MeV_10k)."""
    def loader():
        # Prefer pickle if present; fall back to ROOT.
        pkl = TRACKS_DIR / f"{name}.pkl"
        root = TRACKS_DIR / f"{name}.root"
        path = pkl if pkl.exists() else root
        if not path.exists():
            raise FileNotFoundError(f"No tracks file named {name}")
        if not pkl.exists() and root.exists() \
                and root.stat().st_size > _FULL_LOAD_MAX_ROOT_BYTES:
            size_gb = root.stat().st_size / 1024**3
            raise TracksFileTooLarge(
                f"{name}.root is {size_gb:.1f} GB — too large for a full "
                f"table/browser load; the Pull tab reads it via the slim "
                f"streaming loader")
        return normalize_awkward_columns(
            read_geant4_simulation_output(path, verbose=False))
    return get_or_load(f"g4:{name}", loader)


def _log_cut(values: np.ndarray, center: float, halfwidth_dex: float) -> np.ndarray:
    """Rows whose log10(value) lies within ±halfwidth_dex of log10(center)."""
    v = np.where(values > 0, values, np.nan)
    log_v = np.log10(v)
    cut = (log_v >= np.log10(center) - halfwidth_dex) & \
          (log_v <= np.log10(center) + halfwidth_dex)
    return np.nan_to_num(cut, nan=False).astype(bool)


def bin_mask(df: pd.DataFrame,
             e_center: Optional[float] = None, e_halfwidth: Optional[float] = None,
             l_center: Optional[float] = None, l_halfwidth: Optional[float] = None,
             ) -> np.ndarray:
    """Boolean mask of rows inside the pinned (E, L) cell.

    Each cut applies only when both its center and halfwidth are given and
    the column exists.
    """
    mask = np.ones(df.shape[0], dtype=bool)
    for center, hw, col in (
        (e_center, e_halfwidth, G4Columns.KineticEnergy.value),
        (l_center, l_halfwidth, G4Columns.StepLength.value),
    ):
        if center is None or hw is None or col not in df.columns:
            continue
        mask &= _log_cut(df[col].values, float(center), float(hw))
    return mask


# Which stages condition on L. length/secondary must NOT be L-cut even if
# the caller passes one (it would degenerate their histograms — see the
# needsL logic in static/js/tabs/generators.js).
_STAGE_NEEDS_L = {"length": False, "continuous": True, "secondary": False}


def stage_row_mask(df: pd.DataFrame, stage: str) -> np.ndarray:
    """Stage-specific row filter beyond the (E, L) cell cut.

    Driven by STAGE_CONFIG: `g4_process` restricts rows to a ProcessName,
    and `g4_min_value` drops rows whose `g4_col` value is at or below a
    floor. Stages without those keys pass every row.
    """
    cfg = STAGE_CONFIG[stage]
    mask = np.ones(df.shape[0], dtype=bool)
    proc = cfg.get("g4_process")
    if proc is not None and G4Columns.ProcessName.value in df.columns:
        mask &= (df[G4Columns.ProcessName.value].values == proc)
    min_val = cfg.get("g4_min_value")
    if min_val is not None and cfg["g4_col"] in df.columns:
        mask &= (df[cfg["g4_col"]].values > float(min_val))
    return mask


def in_bin_labels(file: str, stage: str,
                  e_center: float, e_halfwidth: float,
                  l_center: Optional[float] = None,
                  l_halfwidth: Optional[float] = None,
                  ) -> Optional[torch.Tensor]:
    """Per-event conditioning labels for `stage` from the in-bin G4 rows.

    One row per G4 event inside the (E, L) cell:
      length / secondary -> (n, 1): [E]
      continuous         -> (n, 2): [E, L]

    Returns None when the tracks file is missing or the bin is empty, so
    callers can fall back to point conditioning.
    """
    if stage not in _STAGE_NEEDS_L:
        raise KeyError(f"unknown stage {stage}")
    try:
        df = load_tracks(file)
    except FileNotFoundError:
        return None
    if _STAGE_NEEDS_L[stage]:
        mask = bin_mask(df, e_center, e_halfwidth, l_center, l_halfwidth)
    else:
        mask = bin_mask(df, e_center, e_halfwidth)
    mask &= stage_row_mask(df, stage)
    rows = df.loc[mask]
    if rows.empty:
        return None

    E = torch.tensor(rows[G4Columns.KineticEnergy.value].values, dtype=torch.float32)
    if stage in ("length", "secondary"):
        return E.reshape(-1, 1)
    L = torch.tensor(rows[G4Columns.StepLength.value].values, dtype=torch.float32)
    return torch.stack([E, L], dim=1)


def parse_bin_args(args: Mapping[str, str]) -> Optional[dict]:
    """Extract bin-conditioning params from a request query mapping.

    Returns kwargs for `in_bin_labels` (minus `stage`) when `file`,
    `e_center` and `e_halfwidth` are all present; None otherwise. The L
    pair is optional and included only when both halves are given.
    """
    file = args.get("file")
    e_center = args.get("e_center")
    e_halfwidth = args.get("e_halfwidth")
    if not (file and e_center and e_halfwidth):
        return None
    out = {"file": file, "e_center": float(e_center), "e_halfwidth": float(e_halfwidth)}
    l_center = args.get("l_center")
    l_halfwidth = args.get("l_halfwidth")
    if l_center is not None and l_halfwidth is not None:
        out["l_center"] = float(l_center)
        out["l_halfwidth"] = float(l_halfwidth)
    return out
