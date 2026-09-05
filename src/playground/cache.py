"""Shared in-process caches for the playground.

The playground re-uses heavy objects across requests:
- GEANT4 track DataFrames (slow to read from ROOT)
- Built SteppingManager / TrackSimulator instances (loading checkpoints +
  scaler pickles takes seconds)
- Loaded NN generators keyed by class name + checkpoint path

These caches are intentionally simple in-process dicts. The Flask dev server
is single-process, so this is enough; if the playground is ever moved behind
a multi-worker WSGI server, the caches will need a Redis/disk backing.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

# RLock (reentrant) so a loader that itself calls get_or_load() with a
# different key from the same thread doesn't deadlock. The pull-analysis
# loader does exactly that — it calls run_simulation, which calls
# build_track_simulator, which is itself a get_or_load() call.
_lock = threading.RLock()
_store: dict[str, Any] = {}


def get_or_load(key: str, loader: Callable[[], Any]) -> Any:
    if key in _store:
        return _store[key]
    with _lock:
        if key in _store:
            return _store[key]
        value = loader()
        _store[key] = value
        return value


def peek(key: str) -> Any:
    """Return the cached value for `key` without invoking any loader.

    Raises KeyError if the key is absent. Used by cache-only paths (e.g. the
    pull re-bin endpoint) that must operate on an already-cached result and
    refuse rather than recompute when the cache is cold.

    No lock is taken: the single dict lookup is atomic under CPython's GIL,
    matching get_or_load's lock-free fast-path read.
    """
    return _store[key]


def invalidate(prefix: str = "") -> int:
    with _lock:
        if not prefix:
            n = len(_store)
            _store.clear()
            return n
        keys = [k for k in _store if k.startswith(prefix)]
        for k in keys:
            del _store[k]
        return len(keys)


def keys() -> list[str]:
    return sorted(_store.keys())
