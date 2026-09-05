"""Tiny thread-safe progress registry for the playground.

A single in-process dict tracks the current simulation's step/total counters
plus a free-text `phase` (e.g. "building manager", "simulating", "done").
The stepping API writes into it; the `/api/stepping/progress` endpoint reads
from it; the frontend polls that endpoint while a run is in flight.
"""
from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_state: dict[str, Any] = {
    "phase": "idle",
    "step": 0,
    "total": 0,
    "started_at": 0.0,
    "updated_at": 0.0,
}


def set_phase(phase: str, total: int = 0) -> None:
    with _lock:
        _state["phase"] = phase
        _state["step"] = 0
        _state["total"] = int(total)
        _state["started_at"] = time.time()
        _state["updated_at"] = _state["started_at"]


def update_step(step: int, total: int | None = None) -> None:
    with _lock:
        _state["step"] = int(step)
        if total is not None:
            _state["total"] = int(total)
        _state["updated_at"] = time.time()


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_state)
