"""Slim chunked reader must reproduce the full-load path exactly on
the 4 pull columns — same surviving rows, same values (float32 tolerance on
the downcast X/Y)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.enums import G4Columns, TOTAL_ENERGY_LOSS
from common.paths import TRACKS_DIR
from data_handling.reader import read_geant4_simulation_output
from data_handling.slim_reader import read_pull_slim, read_steps_slim
from tests.helpers_synthetic_g4 import write_synthetic_root

_PULL_COLS = [G4Columns.EventNum, G4Columns.X, G4Columns.Y, TOTAL_ENERGY_LOSS]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    # X/Y must be cast to float32 on BOTH sides before sorting: the slim
    # frame's X/Y are float32(full-load X/Y) for the same surviving rows, so
    # sorting on the raw float64 full-load values vs. the float32-downcast slim
    # values can tie-break adjacent, close-together rows (dense trajectory
    # steps) in different orders on a real 22M-row file, desyncing the
    # positional comparison below. Casting both sides to float32 first makes
    # the sort keys identical, so any residual ties are genuinely identical
    # rows and their order is irrelevant.
    out = df[_PULL_COLS].copy()
    out[G4Columns.X] = out[G4Columns.X].astype("float32")
    out[G4Columns.Y] = out[G4Columns.Y].astype("float32")
    return out.sort_values(_PULL_COLS).reset_index(drop=True)


def _assert_equiv(slim: pd.DataFrame, full_load: pd.DataFrame) -> None:
    a, b = _sorted(slim), _sorted(full_load)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a[G4Columns.EventNum], b[G4Columns.EventNum])
    np.testing.assert_array_equal(a[G4Columns.X], b[G4Columns.X])
    np.testing.assert_array_equal(a[G4Columns.Y], b[G4Columns.Y])
    np.testing.assert_allclose(a[TOTAL_ENERGY_LOSS], b[TOTAL_ENERGY_LOSS], rtol=1e-12)


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic_tracks.root"
    write_synthetic_root(root)
    return root


def test_slim_matches_full_load_on_synthetic_file(synthetic_root: Path):
    slim = read_pull_slim(synthetic_root)
    full_load = read_geant4_simulation_output(synthetic_root, verbose=False)
    _assert_equiv(slim, full_load)


def test_slim_dtypes_and_columns(synthetic_root: Path):
    slim = read_pull_slim(synthetic_root)
    assert list(slim.columns) == _PULL_COLS
    assert slim[G4Columns.EventNum].dtype == np.int32
    assert slim[G4Columns.X].dtype == np.float32
    assert slim[G4Columns.Y].dtype == np.float32
    assert slim[TOTAL_ENERGY_LOSS].dtype == np.float64


def test_chunked_read_is_exact(synthetic_root: Path):
    whole = read_pull_slim(synthetic_root)
    chunked = read_pull_slim(synthetic_root, step_entries=97)  # forces ≥5 chunks
    pd.testing.assert_frame_equal(
        chunked.reset_index(drop=True), whole.reset_index(drop=True))


def test_progress_cb_reports_chunks(synthetic_root: Path):
    calls: list[tuple[int, int]] = []
    read_pull_slim(synthetic_root, step_entries=97,
                   progress_cb=lambda i, n: calls.append((i, n)))
    assert calls, "progress_cb never called"
    steps, totals = zip(*calls)
    assert steps == tuple(range(1, len(calls) + 1))
    assert set(totals) == {calls[0][1]}
    assert calls[-1][0] == calls[-1][1]


# ---------------------------------------------------------------------------
# read_steps_slim — generic chunked feature reader (pairwise reference arms)

_FEATURE_COLS = [
    G4Columns.EventNum, G4Columns.KineticEnergy, G4Columns.StepLength,
    G4Columns.ContinuousLoss, G4Columns.SecondaryLoss,
    G4Columns.angleDiscrete, G4Columns.angleMSC, G4Columns.ProcessName,
]


def test_steps_slim_matches_full_load_on_requested_columns(synthetic_root: Path):
    slim = read_steps_slim(synthetic_root, _FEATURE_COLS)
    full_load = read_geant4_simulation_output(synthetic_root, verbose=False)
    assert list(slim.columns) == [str(c) for c in _FEATURE_COLS]
    assert len(slim) == len(full_load)
    # The full-load reader sorts by (EventNum, StepNum), which is the
    # synthetic file's write order, so the surviving rows compare
    # positionally.
    for col in _FEATURE_COLS:
        np.testing.assert_array_equal(
            np.asarray(slim[col]), np.asarray(full_load[col]), err_msg=str(col))


def test_steps_slim_applies_secondary_loss_zero_fix(synthetic_root: Path):
    slim = read_steps_slim(
        synthetic_root, [G4Columns.EventNum, G4Columns.SecondaryLoss,
                         G4Columns.NumOfSecondaries])
    assert (slim.loc[slim[G4Columns.NumOfSecondaries] == 0,
                     G4Columns.SecondaryLoss] == 0.0).all()


def test_steps_slim_max_events_keeps_exactly_first_events(synthetic_root: Path):
    whole = read_steps_slim(synthetic_root, _FEATURE_COLS)
    head = read_steps_slim(synthetic_root, _FEATURE_COLS, max_events=10)
    expected = whole[whole[G4Columns.EventNum] < 10].reset_index(drop=True)
    # check_categorical=False: the head slice's ProcessName categories are
    # derived from fewer rows than the whole file's.
    pd.testing.assert_frame_equal(head.reset_index(drop=True), expected,
                                  check_categorical=False)


def test_steps_slim_max_events_stops_early(synthetic_root: Path):
    calls: list[tuple[int, int]] = []
    read_steps_slim(synthetic_root, _FEATURE_COLS, max_events=10,
                    step_entries=97,
                    progress_cb=lambda i, n: calls.append((i, n)))
    # 50 events x 10 steps = 500 rows -> 6 chunks of 97; events 0..9 end at
    # row 99, so the reader must stop after the 2nd chunk (first chunk whose
    # last event is >= max_events), not stream the whole file.
    assert calls[-1][1] == 6
    assert len(calls) == 2


def test_steps_slim_chunked_read_is_exact(synthetic_root: Path):
    whole = read_steps_slim(synthetic_root, _FEATURE_COLS)
    chunked = read_steps_slim(synthetic_root, _FEATURE_COLS, step_entries=97)
    pd.testing.assert_frame_equal(
        chunked.reset_index(drop=True), whole.reset_index(drop=True))


def test_steps_slim_event_dtype_and_process_category(synthetic_root: Path):
    slim = read_steps_slim(synthetic_root, _FEATURE_COLS)
    assert slim[G4Columns.EventNum].dtype == np.int32
    assert isinstance(slim[G4Columns.ProcessName].dtype, pd.CategoricalDtype)
    assert (slim[G4Columns.ProcessName] == "eBrem").any()


_BREMS_10K = TRACKS_DIR / "100MeV_brems_10k.root"


@pytest.mark.skipif(not _BREMS_10K.exists(), reason="storage-gated: 10k brems file absent")
def test_slim_matches_full_load_on_real_10k_file():
    slim = read_pull_slim(_BREMS_10K)
    full_load = read_geant4_simulation_output(_BREMS_10K, verbose=False)  # cached pkl path
    _assert_equiv(slim, full_load)


def test_load_pull_slim_cached_writes_and_reuses_disk_cache(
        synthetic_root: Path, monkeypatch):
    import data_handling.slim_reader as slim_reader
    df1 = slim_reader.load_pull_slim_cached(synthetic_root)
    slim_pkl = synthetic_root.with_suffix(".pull_slim.pkl")
    assert slim_pkl.exists()
    payload = pd.read_pickle(slim_pkl)
    assert payload["version"] == slim_reader.PULL_SLIM_CACHE_VERSION

    monkeypatch.setattr(slim_reader, "read_pull_slim",
                        lambda *a, **k: pytest.fail("must serve the disk cache"))
    df2 = slim_reader.load_pull_slim_cached(synthetic_root)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_pull_slim_cached_forwards_progress_cb(synthetic_root: Path):
    import data_handling.slim_reader as slim_reader
    calls: list[tuple[int, int]] = []
    slim_reader.load_pull_slim_cached(
        synthetic_root, progress_cb=lambda i, n: calls.append((i, n)))
    assert calls, "progress_cb not forwarded on a cold cache"
