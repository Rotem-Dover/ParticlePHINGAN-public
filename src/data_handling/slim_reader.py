"""Slim chunked reader for the pull analysis.

Streams a GEANT4 tracks ROOT file (tree 'Data') with uproot.iterate and keeps
only the 4 columns the energy-deposition pull needs, applying
data_handling.reader's row-local semantics exactly per chunk:

1. StepNum > 0            (first step is in vacuum)
2. SecondaryLoss = 0 where NumOfSecondaries == 0   (numerical-noise fix)
3. KineticEnergy > MIN_ENERGY_CUTOFF
4. ProcessName != 'Transportation'
5. TOTAL_ENERGY_LOSS = ContinuousLoss + SecondaryLoss  (G4 truth files fold
   the brems photon energy into SecondaryLoss on eBrem rows)

All rules are row-local, so chunked application selects exactly the same rows
as reading the whole file at once and applying them there. The sort by
(EventNum, StepNum) that a full-frame read would apply is skipped —
histogram binning is order-independent. Output units are the file's raw
units (meters / eV); organize_df applies displace + unit conversion.

Peak memory is one decompressed chunk (~9 branches) plus the accumulating
4-column output, rather than a full 16-branch DataFrame for the whole file —
chunking keeps memory bounded regardless of file size.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import uproot

from common.config import MIN_ENERGY_CUTOFF
from common.enums import G4Columns, TOTAL_ENERGY_LOSS

log = logging.getLogger(__name__)

DEFAULT_STEP_ENTRIES = 15_000_000

# Bump this whenever the slim frame's schema changes (columns added/removed/
# renamed) or `read_pull_slim`'s filter semantics change (e.g. the StepNum/
# KineticEnergy/ProcessName cuts below). A version bump invalidates every
# on-disk `.pull_slim.pkl` cache on next load. Read by the playground pull
# API (`playground/api/pull.py`).
PULL_SLIM_CACHE_VERSION = 1

_READ_BRANCHES = [
    G4Columns.EventNum, G4Columns.StepNum,
    G4Columns.X, G4Columns.Y,
    G4Columns.SecondaryLoss, G4Columns.ContinuousLoss,
    G4Columns.NumOfSecondaries, G4Columns.KineticEnergy,
    G4Columns.ProcessName,
]

PULL_COLUMNS = [G4Columns.EventNum, G4Columns.X, G4Columns.Y, TOTAL_ENERGY_LOSS]


def read_pull_slim(root_path: Path,
                   step_entries: int = DEFAULT_STEP_ENTRIES,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   ) -> pd.DataFrame:
    """Stream `root_path` and return the slim pull frame (see module docstring)."""
    parts: list[pd.DataFrame] = []
    with uproot.open(root_path) as f:
        tree = f["Data"]
        n_entries = int(tree.num_entries)
        n_chunks = max(1, -(-n_entries // step_entries))  # ceil div
        for i, arrays in enumerate(tree.iterate(
                expressions=[str(b) for b in _READ_BRANCHES],
                step_size=step_entries, library="np")):
            parts.append(_slim_chunk(arrays))
            if progress_cb is not None:
                progress_cb(i + 1, n_chunks)
    return pd.concat(parts, ignore_index=True)


def _row_keep_and_sec(a: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The reader's row-local semantics on one chunk (see module docstring):
    the survivor mask (rules 1, 3, 4) and the noise-fixed SecondaryLoss
    (rule 2). Single source for every slim chunked reader."""
    proc = np.asarray(a[G4Columns.ProcessName], dtype=object)
    keep = ((a[G4Columns.StepNum] > 0)
            & (a[G4Columns.KineticEnergy] > MIN_ENERGY_CUTOFF)
            & (proc != "Transportation"))
    sec = np.where(a[G4Columns.NumOfSecondaries] == 0,
                   0.0, a[G4Columns.SecondaryLoss])
    return keep, sec


def _slim_chunk(a: dict[str, np.ndarray]) -> pd.DataFrame:
    keep, sec = _row_keep_and_sec(a)
    total = a[G4Columns.ContinuousLoss] + sec
    return pd.DataFrame({
        G4Columns.EventNum: a[G4Columns.EventNum][keep].astype(np.int32),
        G4Columns.X: a[G4Columns.X][keep].astype(np.float32),
        G4Columns.Y: a[G4Columns.Y][keep].astype(np.float32),
        TOTAL_ENERGY_LOSS: total[keep].astype(np.float64),
    })


# Row-local rule inputs every read needs even when not requested as output.
_FILTER_BRANCHES = [
    G4Columns.EventNum, G4Columns.StepNum, G4Columns.KineticEnergy,
    G4Columns.ProcessName, G4Columns.NumOfSecondaries, G4Columns.SecondaryLoss,
]


def read_steps_slim(root_path: Path,
                    columns: list,
                    max_events: Optional[int] = None,
                    step_entries: int = DEFAULT_STEP_ENTRIES,
                    progress_cb: Optional[Callable[[int, int], None]] = None,
                    ) -> pd.DataFrame:
    """Stream `root_path` and return the requested step-feature `columns`
    with the same row-local filter semantics as `read_pull_slim` (module
    docstring rules 1-4), without ever materializing the full 16-branch
    frame.

    `max_events` keeps only rows with EventNum < max_events and stops
    iterating at the first chunk whose last event reaches that bound, so
    reading a small event count off a large file touches only the leading
    chunks. Early stop assumes EventNum is non-decreasing through the file
    (GEANT4 writes events sequentially); the reader raises if a chunk
    boundary violates that, rather than silently dropping late rows.

    Output dtypes: EventNum int32, ProcessName category, all other columns
    as stored in the file (float64 for the GEANT4 dumps).
    """
    columns = [str(c) for c in columns]
    read_branches = sorted({*columns, *(str(b) for b in _FILTER_BRANCHES)})
    parts: list[pd.DataFrame] = []
    prev_last_event: Optional[int] = None
    with uproot.open(root_path) as f:
        tree = f["Data"]
        n_entries = int(tree.num_entries)
        n_chunks = max(1, -(-n_entries // step_entries))  # ceil div
        for i, a in enumerate(tree.iterate(
                expressions=read_branches,
                step_size=step_entries, library="np")):
            ev = a[G4Columns.EventNum]
            if (prev_last_event is not None and ev.size
                    and int(ev.min()) < prev_last_event):
                raise ValueError(
                    f"{root_path}: EventNum not monotone across chunks "
                    f"({int(ev.min())} after {prev_last_event}); "
                    "max_events early stop would drop rows")
            if ev.size:
                prev_last_event = int(ev.max())
            parts.append(_features_chunk(a, columns, max_events))
            if progress_cb is not None:
                progress_cb(i + 1, n_chunks)
            if (max_events is not None and ev.size
                    and int(ev.max()) >= max_events):
                break
    df = pd.concat(parts, ignore_index=True)
    if str(G4Columns.ProcessName) in df.columns:
        df[G4Columns.ProcessName] = df[G4Columns.ProcessName].astype("category")
    return df


def _features_chunk(a: dict[str, np.ndarray], columns: list[str],
                    max_events: Optional[int]) -> pd.DataFrame:
    keep, sec = _row_keep_and_sec(a)
    if max_events is not None:
        keep = keep & (a[G4Columns.EventNum] < max_events)
    out = {}
    for col in columns:
        if col == G4Columns.EventNum:
            out[col] = a[col][keep].astype(np.int32)
        elif col == G4Columns.SecondaryLoss:
            out[col] = sec[keep]
        elif col == G4Columns.ProcessName:
            out[col] = np.asarray(a[col], dtype=object)[keep]
        else:
            out[col] = a[col][keep]
    return pd.DataFrame(out)


def load_pull_slim_cached(root: Path,
                          progress_cb: Optional[Callable[[int, int], None]] = None,
                          ) -> pd.DataFrame:
    """Slim frame for a big pkl-less ROOT file, backed by a disk cache.

    The full 16-branch pkl is deliberately never created for these files —
    it would be much larger than needed. The slim cache holds only the 4
    pull columns, in raw file units.

    The cache is self-invalidating: it stores `PULL_SLIM_CACHE_VERSION`
    plus the source ROOT's mtime/size alongside the frame, and is discarded
    (regenerated from the ROOT) whenever any of those disagree with the
    current file on disk, or the cache is missing/unreadable/malformed.
    This guards against silently serving outdated G4 truth after the ROOT is
    regenerated under the same name, and against an outdated-schema cache
    surviving a `read_pull_slim`/schema change.
    """
    slim_pkl = root.with_suffix(".pull_slim.pkl")
    st = root.stat()
    cached = _read_valid_slim_cache(slim_pkl, st)
    if cached is not None:
        return cached
    df = read_pull_slim(root, progress_cb=progress_cb)
    pd.to_pickle(
        {
            "version": PULL_SLIM_CACHE_VERSION,
            "src_mtime": st.st_mtime,
            "src_size": st.st_size,
            "df": df,
        },
        slim_pkl,
    )
    return df


def _read_valid_slim_cache(slim_pkl: Path, src_stat) -> pd.DataFrame | None:
    """Return the cached slim frame if `slim_pkl` is fresh and well-formed,
    else `None` (any invalidity/error is logged and treated as a cache miss —
    never raised)."""
    if not slim_pkl.exists():
        return None
    try:
        payload = pd.read_pickle(slim_pkl)
    except Exception as exc:
        log.warning("discarding unreadable pull-slim cache %s: %s", slim_pkl, exc)
        return None
    if not isinstance(payload, dict) or "df" not in payload:
        log.warning("discarding malformed pull-slim cache %s (unexpected format)",
                    slim_pkl)
        return None
    if payload.get("version") != PULL_SLIM_CACHE_VERSION:
        log.info("discarding pull-slim cache %s: version %r != current %r",
                 slim_pkl, payload.get("version"), PULL_SLIM_CACHE_VERSION)
        return None
    if (payload.get("src_mtime") != src_stat.st_mtime
            or payload.get("src_size") != src_stat.st_size):
        log.info("discarding outdated pull-slim cache %s: source ROOT changed "
                 "(mtime/size mismatch)", slim_pkl)
        return None
    return payload["df"]
