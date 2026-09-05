"""Build a SteppingManager + TrackSimulator from a JSON config and run it.

The JSON config the frontend sends is a list of process dicts, each with
`name` and per-stage generator/scaler subconfigs — the same shape
`SteppingManager.from_config` expects. This module is a thin translator
from JSON to `SteppingManager.from_config`, plus a cache so repeated runs
reuse the (slow-to-build) manager. `build_track_simulator_for_manager` and
`run_simulation_with_manager` offer the same caching for a manager a caller
already built directly (e.g. via `benchmark.track_simulator_config`'s
`PRESETS`), bypassing the JSON translation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from common.config import DEVICE
from particle_propagation.stepping_manager import SteppingManager
from particle_propagation.track_simulator import TrackSimulator

from playground.cache import get_or_load
from playground.services import progress as progress_state


# Most-recently-run TrackSimulator, shared across requests (single-process
# Flask server). The Table tab reads this to show "the resulting table of the
# last simulation" regardless of which tab — trajectory / pull / pairwise —
# triggered it. Set on every actual `.run()` in `_run_with_simulator`; on a
# pull/pairwise cache hit the underlying `ts` is the same cached object, so the
# table stays correct.
_last_run: dict[str, Any] | None = None


def get_last_run() -> dict[str, Any] | None:
    """Return {ts, E0_eV, n_events, n_steps, source, g4_file} of the last sim, or None.

    `g4_file` is the base name of the GEANT4 tracks file the run was compared
    against (set by the Pull / Pairwise tabs), or None when the run carried no
    G4 reference (e.g. a plain Trajectory run). The Table tab uses it to lazily
    load the matching G4 truth table.

    For a chunked Pull run (n_events above `PULL_SIM_CHUNK_EVENTS`,
    `playground.api.pull`), the underlying `TrackSimulator` is `.run()`
    repeatedly across sequential event chunks, and this holder is overwritten
    on each call — so after a chunked run completes, the Table tab's "last
    run" reflects only the FINAL chunk, not the full chunked population.
    """
    return _last_run


def _config_key(cfg: dict[str, Any]) -> str:
    """Hash a config dict so identical configs share a built manager."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def build_stepping_manager(processes: list[dict], device: str = DEVICE) -> SteppingManager:
    key = f"sm:{_config_key({'processes': processes, 'device': device})}"
    def loader():
        progress_state.set_phase("building stepping manager")
        return SteppingManager.from_config(device=device, processes=processes)
    return get_or_load(key, loader)


def build_track_simulator(processes: list[dict], device: str = DEVICE) -> TrackSimulator:
    key = f"ts:{_config_key({'processes': processes, 'device': device})}"
    def loader():
        sm = build_stepping_manager(processes, device)
        return TrackSimulator(stepping_manager=sm, device=device)
    return get_or_load(key, loader)


def build_track_simulator_for_manager(sm: SteppingManager, device: str = DEVICE) -> TrackSimulator:
    """Wrap a pre-built SteppingManager in a TrackSimulator (cached by id).

    For a manager a caller already built directly, e.g. via
    `benchmark.track_simulator_config.build_stepping_manager(preset_name)`,
    so it does not need to be re-expressed as a JSON process list and built
    again through `build_track_simulator`.
    """
    key = f"ts_from_sm:{id(sm)}:{device}"
    return get_or_load(key, lambda: TrackSimulator(stepping_manager=sm, device=device))


def _run_with_simulator(ts: TrackSimulator, E0_eV: float, n_events: int,
                        n_steps: int, save_steps_states: bool,
                        source: str | None = None,
                        g4_file: str | None = None,
                        phase_label: str = "simulating") -> TrackSimulator:
    progress_state.set_phase(phase_label, total=int(n_steps))

    def _cb(step: int, total: int) -> None:
        progress_state.update_step(step, total)

    with torch.inference_mode():
        ts.run(E_0=float(E0_eV), n_events=int(n_events), n_steps=int(n_steps),
               save_steps_states=save_steps_states, progress_cb=_cb)
    progress_state.set_phase("done", total=int(n_steps))
    progress_state.update_step(int(n_steps), int(n_steps))

    global _last_run
    _last_run = {
        "ts": ts,
        "E0_eV": float(E0_eV),
        "n_events": int(n_events),
        "n_steps": int(n_steps),
        "source": source,
        "g4_file": g4_file,
    }
    return ts


def run_simulation(processes: list[dict], E0_eV: float, n_events: int,
                   n_steps: int, save_steps_states: bool = True,
                   device: str = DEVICE, source: str | None = None,
                   g4_file: str | None = None,
                   phase_label: str = "simulating") -> TrackSimulator:
    progress_state.set_phase("building stepping manager")
    ts = build_track_simulator(processes, device)
    return _run_with_simulator(ts, E0_eV, n_events, n_steps, save_steps_states,
                               source, g4_file, phase_label=phase_label)


def run_simulation_with_manager(sm: SteppingManager, E0_eV: float, n_events: int,
                                n_steps: int, save_steps_states: bool = True,
                                device: str = DEVICE,
                                source: str | None = None,
                                g4_file: str | None = None,
                                phase_label: str = "simulating") -> TrackSimulator:
    """Fast path: reuse an already-built SteppingManager (e.g. from a preset)."""
    progress_state.set_phase("wrapping pre-built manager")
    ts = build_track_simulator_for_manager(sm, device)
    return _run_with_simulator(ts, E0_eV, n_events, n_steps, save_steps_states,
                               source, g4_file, phase_label=phase_label)
