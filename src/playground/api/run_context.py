"""Read-only run-context endpoint.

Reports the active (particle, material, beam energy) tuple and surfaces the
on-disk data inventory the playground will draw from: G4 tracks files,
analytical CDFs, portable scaler state (``<ClassName>.state.pt``), and
training datasets. Used by the frontend to populate dropdowns on initial
load.
"""
from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from common.run_context import (
    ACTIVE_NAME, PARTICLE, TARGET_MATERIAL, BEAM_ENERGY_eV, BEAM_ENERGY_TAG,
)
from physics.definitions.beam import PRESETS
from common.paths import (TRACKS_DIR, CDFS_DIR, SCALERS_DIR, TABLES_DIR, RUNS_DIR, DATASETS_DIR, ON_CLUSTER)

bp = Blueprint("run_context", __name__)


def _list_files(d, patterns):
    if not d.exists():
        return []
    out = []
    for pat in patterns:
        out.extend(sorted(p.name for p in d.glob(pat)))
    return out


@bp.route("", methods=["GET"])
@bp.route("/", methods=["GET"])
def get_run_context():
    return jsonify({
        "beam": ACTIVE_NAME,
        "available_beams": sorted(PRESETS),
        "particle": PARTICLE.name,
        "particle_mass_eV": PARTICLE.mass_eV,
        "material": TARGET_MATERIAL.name,
        "material_Tc_eV": TARGET_MATERIAL.Tc,
        "beam_energy_eV": BEAM_ENERGY_eV,
        "beam_energy_tag": BEAM_ENERGY_TAG,
        "on_cluster": ON_CLUSTER,
        "paths": {
            "tracks": str(TRACKS_DIR),
            "cdfs": str(CDFS_DIR),
            "scalers": str(SCALERS_DIR),
            "tables": str(TABLES_DIR),
            "datasets": str(DATASETS_DIR),
            "tensorboard": str(RUNS_DIR),
        },
        "inventory": {
            "tracks": _list_files(TRACKS_DIR, ["*.root", "*.pkl"]),
            "cdfs":   _list_files(CDFS_DIR,   ["*.pkl"]),
            # Scaler state is portable tensors, not pickled instances.
            "scalers": _list_files(SCALERS_DIR, ["*.state.pt"]),
            "datasets": _list_files(DATASETS_DIR, ["*.npy"]),
        },
    })


@bp.route("/set", methods=["POST"])
def set_run_context():
    """Switch the active beam preset and re-exec the playground server.

    The active beam is locked at import time by `common.run_context`
    (`PHINGAN_BEAM`, read once against `physics.definitions.beam.PRESETS`),
    and dozens of downstream modules cache scalers, tables, and checkpoints
    keyed off that snapshot. A clean swap therefore means restarting the
    Python process with a new `PHINGAN_BEAM` env var; this endpoint responds
    first, then re-execs after a short delay so the frontend can begin
    polling for the new server.
    """
    payload = request.get_json(force=True) or {}
    beam = str(payload.get("beam", "")).strip()
    if beam not in PRESETS:
        return jsonify({
            "error": f"beam must be one of {sorted(PRESETS)}, got {beam!r}",
        }), 400
    if beam == ACTIVE_NAME:
        return jsonify({"ok": True, "beam": beam, "restarting": False,
                        "note": "already active"})

    import playground.services.process_control as process_control
    process_control.spawn_replacement(env={"PHINGAN_BEAM": beam})
    port = os.environ.get("PORT", "5050")
    return jsonify({"ok": True, "beam": beam, "restarting": True,
                    "port": port})
