"""Track-simulator endpoints behind the Trajectory tab.

The frontend POSTs the ``processes`` list it builds from the top bar's
per-stage generator dropdowns (the same shape ``SteppingManager.from_config``
takes), the simulator runs through ``playground.services.sim_runner`` (which
caches the built manager), and the response carries only what the tab plots:
a subsample of 3D trajectories and a per-feature mean/std summary. The full
tensors stay in the cached ``TrackSimulator`` on the server, where the Table
tab reads them back as "the last run".

Like ``playground.api.pull``, no named presets are registered: an explicit
``processes`` list is required.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request

from common.config import BEAM_ENERGY_eV
from common.enums import G4Columns
from playground.services import progress as progress_state
from playground.services.sim_runner import run_simulation

log = logging.getLogger(__name__)

bp = Blueprint("stepping", __name__)


@bp.route("/progress", methods=["GET"])
def progress():
    """Snapshot of the in-flight run's phase/step/total; the tab polls it."""
    return jsonify(progress_state.snapshot())


@bp.route("/run", methods=["POST"])
def run():
    """Run the simulator and return trajectories for the 3D viewer.

    POST JSON shape (numbers may arrive as strings from the form inputs):
        {
          "processes": [...],            # required
          "E0_eV": 1e8,
          "n_events": 10,
          "n_steps": 10250,
          "trajectory_subsample": 25     # how many events to return
        }
    """
    payload: dict[str, Any] = request.get_json(force=True) or {}
    if "preset" in payload:
        return jsonify({"error": f"unknown preset {payload['preset']!r}: no "
                        "named presets are registered; send an explicit "
                        "'processes' list"}), 400
    processes = payload.get("processes")
    if not processes:
        return jsonify({"error": "'processes' is required"}), 400

    E0 = float(payload.get("E0_eV", BEAM_ENERGY_eV))
    n_events = int(payload.get("n_events", 100))
    n_steps = int(payload.get("n_steps", 1000))
    trajectory_subsample = int(payload.get("trajectory_subsample", 25))

    try:
        ts = run_simulation(processes, E0, n_events, n_steps,
                            save_steps_states=True, source="trajectory")
    except Exception as e:  # noqa: BLE001
        log.exception("trajectory simulation failed")
        return jsonify({"error": f"simulation failed: {type(e).__name__}: {e}"}), 500

    # Dead particles' rows are NaN-padded (see CLAUDE.md), so each
    # trajectory keeps only the states with finite coordinates.
    states = ts.states_along_propagation_df
    event_ids = sorted(states[G4Columns.EventNum].unique().tolist())
    selected = event_ids[:trajectory_subsample]
    sub = states[states[G4Columns.EventNum].isin(selected)]
    coords = [G4Columns.X, G4Columns.Y, G4Columns.Z, G4Columns.KineticEnergy]

    trajectories = []
    for ev_id, group in sub.groupby(G4Columns.EventNum):
        finite = group[group[coords].notna().all(axis=1)]
        trajectories.append({
            "event": int(ev_id),
            "x": finite[G4Columns.X].tolist(),
            "y": finite[G4Columns.Y].tolist(),
            "z": finite[G4Columns.Z].tolist(),
            "ke": finite[G4Columns.KineticEnergy].tolist(),
        })

    feats = ts.steps_features_df
    feature_summary = {
        col: {
            "mean": float(np.nanmean(feats[col].values)),
            "std":  float(np.nanstd(feats[col].values)),
        }
        for col in feats.columns
        if col not in (G4Columns.EventNum, G4Columns.StepNum)
    }

    return jsonify({
        "n_events": n_events,
        "n_steps":  n_steps,
        "E0_eV":    E0,
        "n_trajectories_returned": len(trajectories),
        "trajectories":   trajectories,
        "feature_summary": feature_summary,
    })
