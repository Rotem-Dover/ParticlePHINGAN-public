"""The fig-12 D_E cache (energy_distance_analysis --score / --plot): the
scored arrays round-trip through the npz, a cache scored under different
checkpoint pins is refused rather than silently plotted, and the plot renders
from the cache alone -- no simulator or network is ever built here."""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from benchmark.system_vs_g4.energy_distance import energy_distance_analysis as eda


def _fake_arrays(rng):
    return {f"{row}_{arm}": rng.random(300)
            for row in eda._ROWS for arm in ("null", "phys", "gan")}


def _write_cache(path, arrays, pins, seed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if seed is None else {"seed": np.array(seed)}
    np.savez(path, pins=np.array(pins), **extra, **arrays)


def test_cache_round_trips_and_plots_without_building_a_simulator(tmp_path):
    arrays = _fake_arrays(np.random.default_rng(0))
    cache = tmp_path / "energy_distance_wds.npz"
    _write_cache(cache, arrays, eda._current_pins())

    loaded = eda.load_cache(cache)
    assert set(loaded) == set(arrays)
    for k in arrays:
        np.testing.assert_array_equal(loaded[k], arrays[k])

    out = tmp_path / "energy_distance.pdf"
    fig = eda.plot_from_cache(cache, out_path=out)
    assert out.exists() and out.stat().st_size > 0
    # 3 rows x 3 arms all drawn
    assert len(fig.axes) == 3
    matplotlib.pyplot.close(fig)


def test_seed_stamp_is_metadata_not_an_array(tmp_path):
    """A seeded cache loads to exactly the nine D_E arrays -- the seed key is
    provenance, not a curve the plot could accidentally draw."""
    arrays = _fake_arrays(np.random.default_rng(2))
    cache = tmp_path / "energy_distance_wds.npz"
    _write_cache(cache, arrays, eda._current_pins(), seed=20260807)

    loaded = eda.load_cache(cache)
    assert set(loaded) == set(arrays)


def test_mismatched_pins_are_refused_with_both_pin_sets_named(tmp_path):
    arrays = _fake_arrays(np.random.default_rng(1))
    cache = tmp_path / "energy_distance_wds.npz"
    mismatched = list(eda._current_pins())
    mismatched[0] = "some_older_run/epoch=99.ckpt"
    _write_cache(cache, arrays, mismatched)

    with pytest.raises(RuntimeError, match="out of date") as excinfo:
        eda.load_cache(cache)
    assert "some_older_run/epoch=99.ckpt" in str(excinfo.value)
    assert eda._current_pins()[0] in str(excinfo.value)


def test_missing_cache_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--score"):
        eda.load_cache(tmp_path / "absent.npz")


def test_current_pins_track_the_loaded_checkpoint_constants():
    """The stamp must name all four preset checkpoints (run dir + epoch),
    so re-pinning any one of them invalidates the cache."""
    from common import paths

    pins = eda._current_pins()
    assert len(pins) == 4
    for pin, ckpt in zip(pins, (paths.CNT_CKPT, paths.SEC_CKPT,
                                paths.NOPHYS_CNT_CKPT,
                                paths.NOPHYS_SEC_CKPT)):
        assert pin == f"{ckpt.parent.parent.name}/{ckpt.name}"
