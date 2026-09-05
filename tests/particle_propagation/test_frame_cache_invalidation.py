"""Regression: TrackSimulator's cached_property DataFrames must be dropped on
every run(), otherwise a reused simulator (playground preset fast-path, pull
chunk loop) serves the previous run's frames."""
import pandas as pd

from particle_propagation.track_simulator import TrackSimulator


def _bare_simulator() -> TrackSimulator:
    # No __init__: we only exercise the cache-invalidation helper, which
    # touches nothing but the instance __dict__.
    return TrackSimulator.__new__(TrackSimulator)


def test_invalidate_frame_caches_drops_all_cached_frames():
    ts = _bare_simulator()
    previous_frame = pd.DataFrame({"a": [1]})
    for attr in TrackSimulator._FRAME_CACHE_ATTRS:
        ts.__dict__[attr] = previous_frame
    ts._invalidate_frame_caches()
    for attr in TrackSimulator._FRAME_CACHE_ATTRS:
        assert attr not in ts.__dict__


def test_frame_cache_attrs_cover_every_cached_property():
    from functools import cached_property
    cached = {name for name, val in vars(TrackSimulator).items()
              if isinstance(val, cached_property)}
    assert cached == set(TrackSimulator._FRAME_CACHE_ATTRS)


def test_invalidate_is_safe_on_cold_instance():
    _bare_simulator()._invalidate_frame_caches()  # must not raise
