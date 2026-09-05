"""Stream training targets straight out of a GEANT4 truth ROOT file.

`data_handling.reader.read_geant4_simulation_output` is not usable for large
truth files: it caches a `.pkl` beside its source and materialises the whole
frame in memory. This module instead streams the tree with `uproot.iterate`,
applies the row-local filters chunk by chunk, and never opens a file for
writing.

The row-local filters (ordered as in `data_handling/reader.py`) select
identically whether applied to the full frame at once or to one chunk at a
time, because none of them depends on any other row. `ROW_LOCAL_FILTERS`
below pairs each filter's description with the callable that actually
implements it, and `_apply_row_local_filters` just runs the list -- so the
tuple IS the implementation, not a second prose copy of it that could
silently drift out of sync (an outdated description would misdescribe a
filter that still runs; a genuinely missing filter, the direction that
matters for a provenance manifest -- see below -- is structurally
impossible, because a filter that isn't in the tuple never runs at all).

1. ``StepNum > 0`` -- the first step of a track is in vacuum.
2. ``SecondaryLoss = 0`` wherever ``NumOfSecondaries == 0`` -- clears
   numerical noise GEANT4 sometimes leaves in that column.
3. ``KineticEnergy > MIN_ENERGY_CUTOFF``.
4. ``ProcessName != 'Transportation'``.
"""
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import uproot

from common.config import MIN_ENERGY_CUTOFF
from common.enums import G4Columns

TREE_NAME = "Data"


def _zero_secondary_loss_where_no_secondaries(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filter 2: clears numerical noise GEANT4 sometimes leaves in
    `SecondaryLoss` on rows with no secondary. A mutation, not a row
    selection, so (unlike the other three filters) it needs its own `.copy()`
    before writing through `.loc[...]` -- pandas would otherwise raise
    `SettingWithCopyWarning` on a view produced by the previous filter's
    boolean indexing."""
    chunk = chunk.copy()
    chunk.loc[chunk[G4Columns.NumOfSecondaries] == 0, G4Columns.SecondaryLoss] = 0.0
    return chunk


# Each row-local filter as a `(description, callable)` pair, in the order
# `_apply_row_local_filters` runs them. This tuple is not just documentation
# of the filtering -- `_apply_row_local_filters` below runs exactly these
# callables and nothing else, so a filter added, removed, or reordered here
# changes both what the code does and what any caller's provenance record
# says was done, together, by construction. There is no second place that
# can drift out of sync.
ROW_LOCAL_FILTERS = (
    ("StepNum > 0",
     lambda chunk: chunk[chunk[G4Columns.StepNum] > 0]),
    ("SecondaryLoss = 0 wherever NumOfSecondaries == 0",
     _zero_secondary_loss_where_no_secondaries),
    ("KineticEnergy > MIN_ENERGY_CUTOFF",
     lambda chunk: chunk[chunk[G4Columns.KineticEnergy] > MIN_ENERGY_CUTOFF]),
    ("ProcessName != 'Transportation'",
     lambda chunk: chunk[chunk[G4Columns.ProcessName] != "Transportation"]),
)  # type: tuple[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]], ...]

# The description half alone, in order -- what a provenance manifest (e.g.
# `build_training_datasets.py`) actually wants. Derived from ROW_LOCAL_FILTERS
# by a plain comprehension, not retyped, so it cannot list a filter that
# isn't actually applied (under-reporting) or omit one that is.
ROW_LOCAL_FILTER_DESCRIPTIONS = tuple(desc for desc, _fn in ROW_LOCAL_FILTERS)


def _apply_row_local_filters(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply every filter in `ROW_LOCAL_FILTERS`, in order, to one streamed
    chunk. Never mutates the caller's chunk (each step either returns a
    fresh boolean-indexed frame or its own explicit `.copy()`)."""
    for _description, apply_filter in ROW_LOCAL_FILTERS:
        chunk = apply_filter(chunk)
    return chunk


def iter_truth_frames(
        root_path: Path,
        columns: list[str],
        step_size: str = "500 MB",
) -> Iterator[pd.DataFrame]:
    """Stream `root_path`'s `Data` tree in chunks, yielding one filtered
    `pandas.DataFrame` per chunk. Writes nothing to disk.

    `columns` must include every column the row-local filters need
    (`StepNum`, `NumOfSecondaries`, `SecondaryLoss`, `KineticEnergy`,
    `ProcessName`) in addition to whatever the caller wants back.
    """
    with uproot.open(root_path) as f:
        tree = f[TREE_NAME]
        for chunk in tree.iterate(columns, library="pd", step_size=step_size):
            yield _apply_row_local_filters(chunk)


# Column order of extract_secondary_xy's drop_theta=False output. This is the
# extractor's OWN fixed layout, deliberately not tied to common.enums.
# SecondaryFeats: that enum describes the NETWORK's single output column
# (`E_SEC_IDX = 0`), while this two-column truth array keeps theta at column
# 0. Indexing this array with the enum silently selects theta instead of the
# secondary loss -- use SECONDARY_2COL_E_SEC_COL, not SecondaryFeats.E_SEC_IDX,
# to read the secondary-loss column out of it.
SECONDARY_2COL_COLUMNS = (G4Columns.angleDiscrete, G4Columns.SecondaryLoss)
SECONDARY_2COL_E_SEC_COL = SECONDARY_2COL_COLUMNS.index(G4Columns.SecondaryLoss)


def extract_secondary_xy(
        root_path: Path,
        drop_theta: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the SecondaryGenerator training pair.

    X is the post-continuous kinetic energy (`KineticEnergy - ContinuousLoss`)
    -- the same quantity `G4hIonisation.post_step_do_it` conditions on at
    inference, since GEANT4 records the delta ray after applying the
    along-step continuous loss. Conditioning on the pre-step energy instead
    mismatches GEANT4's recorded `angleDiscrete` by ~1e-4 relative rather than
    ~1e-8.

    Y is `(N, 1)` of `SecondaryLoss` when `drop_theta`, else `(N, 2)` of
    `[angleDiscrete, SecondaryLoss]`.
    """
    columns = [
        G4Columns.StepNum, G4Columns.NumOfSecondaries, G4Columns.SecondaryLoss,
        G4Columns.KineticEnergy, G4Columns.ContinuousLoss, G4Columns.ProcessName,
        G4Columns.angleDiscrete,
    ]

    x_chunks = []
    y_chunks = []
    for chunk in iter_truth_frames(root_path, columns):
        x_chunks.append(
            (chunk[G4Columns.KineticEnergy] - chunk[G4Columns.ContinuousLoss]).to_numpy()
        )
        if drop_theta:
            y_chunks.append(chunk[[G4Columns.SecondaryLoss]].to_numpy())
        else:
            # The ROOT schema carries no momentum columns, so theta cannot be
            # derived from momentum components; the raw GEANT4 angleDiscrete
            # column is used directly instead. Training uses drop_theta=True
            # (the secondary generator emits only the secondary energy loss),
            # so this path only matters for callers that want the truth angle
            # alongside it.
            y_chunks.append(chunk[list(SECONDARY_2COL_COLUMNS)].to_numpy())

    x = np.concatenate(x_chunks) if x_chunks else np.empty((0,))
    y = np.concatenate(y_chunks) if y_chunks else np.empty((0, 1 if drop_theta else 2))
    return x, y


def extract_continuous_xy(root_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build the ContinuousGenerator training pair: X = (N, 2) of
    `[KineticEnergy, StepLength]`, Y = (N,) of `ContinuousLoss`."""
    columns = [
        G4Columns.StepNum, G4Columns.NumOfSecondaries, G4Columns.SecondaryLoss,
        G4Columns.KineticEnergy, G4Columns.StepLength, G4Columns.ContinuousLoss,
        G4Columns.ProcessName,
    ]

    x_chunks = []
    y_chunks = []
    for chunk in iter_truth_frames(root_path, columns):
        x_chunks.append(chunk[[G4Columns.KineticEnergy, G4Columns.StepLength]].to_numpy())
        y_chunks.append(chunk[G4Columns.ContinuousLoss].to_numpy())

    x = np.concatenate(x_chunks) if x_chunks else np.empty((0, 2))
    y = np.concatenate(y_chunks) if y_chunks else np.empty((0,))
    return x, y
