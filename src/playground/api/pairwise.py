"""Pairwise (feature x feature) comparison between the simulator and GEANT4.

Frontend picks two features (E, L, E_CNT, E_SEC, theta) and a G4
file. The backend:

  - runs the simulator with the user's globally-selected per-stage generators
    (the same `processes` shape as the Track / Pull tabs), reusing the cached
    `STEPPING_MANAGERS` fast-path when possible;
  - loads the G4 reference file via the same cache used by `g4_data`;
  - bins both into the canonical log10 bins from `PLOT_RANGES`;
  - returns four 2D histograms: raw G4, raw Sim, G4-vs-G4 split-half relative
    difference (noise floor), and Sim-vs-G4 relative difference.

The relative-difference computation mirrors
`benchmark/system_vs_g4/pairwise_comparison.py:plot_pairwise_relative_diff`:
density-normalised counts, |cmp - ref| / ref, with a low-stats mask
(threshold = 30 / N_ref) and NaN/Inf → 0.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from flask import Blueprint, jsonify, request

from common.config import BEAM_ENERGY_eV, MIN_ENERGY_CUTOFF
from common.enums import G4Columns
from common.run_context import PLOT_RANGES

from playground.api.g4_data import _load as _load_g4
from playground.cache import get_or_load
from playground.services.sim_runner import run_simulation


bp = Blueprint("pairwise", __name__)


# Canonical log-spaced bins, mirroring pairwise_comparison.py:30-36.
_N_BINS = 100
_BINS = {
    "E":         np.logspace(math.log10(MIN_ENERGY_CUTOFF), math.log10(BEAM_ENERGY_eV), _N_BINS),
    "L":         np.logspace(*PLOT_RANGES.step_length, _N_BINS),
    "E_CNT":     np.logspace(*PLOT_RANGES.cnt_loss,    _N_BINS),
    "E_SEC":     np.logspace(*PLOT_RANGES.sec_loss,    _N_BINS),
    "theta":     np.logspace(*PLOT_RANGES.theta,       _N_BINS),
}

# Plotly renders <sub>/<sup> natively but does not render LaTeX $...$ unless
# MathJax is explicitly loaded into the page. Use Unicode + HTML subscripts so
# labels read correctly without an extra dependency.
_LABELS = {
    "E":         "E [eV]",
    "L":         "L [m]",
    "E_CNT":     "ΔE<sub>cnt</sub> [eV]",
    "E_SEC":     "ΔE<sub>sec</sub> [eV]",
    "theta":     "θ<sub>discrete</sub> [rad]",
}

FEATURES = list(_BINS.keys())


def _g4_arrays(df) -> dict[str, np.ndarray]:
    return {
        "E":         df[G4Columns.KineticEnergy].values,
        "L":         df[G4Columns.StepLength].values,
        "E_CNT":     df[G4Columns.ContinuousLoss].values,
        "E_SEC":     df[G4Columns.SecondaryLoss].values,
        "theta":     df[G4Columns.angleDiscrete].values,
    }


def _sim_arrays(ts) -> dict[str, np.ndarray]:
    """Pair kinetic energy with step features by joining the cached DataFrames.

    `ts.states_and_steps_features_df` is an inner merge on (EventNum, StepNum)
    of the lab-frame state (carrying KineticEnergy) with steps_features_df, so
    every step row gets the pre-step KE used as the conditioning energy.
    """
    df = ts.states_and_steps_features_df
    return {
        "E":         df[G4Columns.KineticEnergy].values,
        "L":         df[G4Columns.StepLength].values,
        "E_CNT":     df[G4Columns.ContinuousLoss].values,
        "E_SEC":     df[G4Columns.SecondaryLoss].values,
        "theta":     df[G4Columns.angleDiscrete].values,
    }


def _rel_diff(
    x_ref: np.ndarray, y_ref: np.ndarray,
    x_cmp: np.ndarray, y_cmp: np.ndarray,
    x_bins: np.ndarray, y_bins: np.ndarray,
) -> np.ndarray:
    """Relative-difference 2D histogram, same masking as the matplotlib version."""
    ref_hist, _, _ = np.histogram2d(x_ref, np.abs(y_ref), bins=[x_bins, y_bins], density=False)
    cmp_hist, _, _ = np.histogram2d(x_cmp, np.abs(y_cmp), bins=[x_bins, y_bins], density=False)
    if x_ref.size > 0:
        ref_hist = ref_hist / x_ref.size
    if x_cmp.size > 0:
        cmp_hist = cmp_hist / x_cmp.size
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.abs(cmp_hist - ref_hist) / ref_hist
        threshold = 30.0 / x_ref.size if x_ref.size > 0 else 0.0
        diff[ref_hist < threshold] = 0.0
        diff = np.nan_to_num(diff)
    return diff


def _arrays_cache_key(processes: list[dict], g4_file: str,
                      E0_eV: float, n_events: int, n_steps: int) -> str:
    """Stable hash of the sim-defining inputs.

    Histogramming depends only on x_feature/y_feature, which are NOT part of
    the key — that's exactly what lets the same cached arrays serve every
    feature pair without re-running the simulator.
    """
    payload = {"p": processes, "g4": g4_file, "E0": E0_eV,
               "ne": n_events, "ns": n_steps}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    # The array-layout version is embedded in the key so a cache entry from
    # a different feature schema can never be served up under a new one.
    return f"pairwise:arrays:v3:{h}"


def _get_arrays(processes: list[dict], g4_file: str, E0_eV: float,
                n_events: int, n_steps: int) -> tuple[dict, dict]:
    """Load G4 + run sim, cache the extracted per-feature arrays."""
    key = _arrays_cache_key(processes, g4_file, E0_eV, n_events, n_steps)

    def loader():
        g4_df = _load_g4(g4_file)
        with torch.inference_mode():
            ts = run_simulation(processes, E0_eV, n_events, n_steps,
                                save_steps_states=True, source="pairwise",
                                g4_file=g4_file)
        return _g4_arrays(g4_df), _sim_arrays(ts)

    return get_or_load(key, loader)


def _raw_hist(x: np.ndarray, y: np.ndarray,
              x_bins: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(x, np.abs(y), bins=[x_bins, y_bins], density=False)
    if x.size > 0:
        h = h / x.size
    return h


@bp.route("/features", methods=["GET"])
def features():
    """Expose the feature catalogue + their LaTeX labels to the frontend."""
    return jsonify({
        "features": FEATURES,
        "labels":   _LABELS,
    })


@bp.route("/run", methods=["POST"])
def run():
    payload: dict[str, Any] = request.get_json(force=True) or {}

    processes = payload.get("processes")
    if not processes:
        return jsonify({"error": "'processes' is required"}), 400

    x_feature = payload.get("x_feature", "E")
    y_feature = payload.get("y_feature", "L")
    if x_feature not in FEATURES or y_feature not in FEATURES:
        return jsonify({"error": f"unknown feature(s): {x_feature}, {y_feature}"}), 400
    if x_feature == y_feature:
        return jsonify({"error": "x_feature and y_feature must differ"}), 400

    g4_file  = payload.get("g4_file", "100MeV_1k")
    E0       = float(payload.get("E0_eV", BEAM_ENERGY_eV))
    n_events = int(payload.get("n_events", 1000))
    n_steps  = int(payload.get("n_steps", 10250))

    # Sim + G4 arrays are cached jointly by (processes, g4_file, E0, n_events,
    # n_steps). Re-picking x_feature / y_feature reuses the cached arrays, so
    # subsequent calls only rerun the histogramming.
    try:
        g4_arr, sim_arr = _get_arrays(processes, g4_file, E0, n_events, n_steps)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"simulation failed: {type(e).__name__}: {e}"}), 500

    x_bins = _BINS[x_feature]
    y_bins = _BINS[y_feature]

    x_g4_full = g4_arr[x_feature]
    y_g4_full = g4_arr[y_feature]
    x_sim     = sim_arr[x_feature]
    y_sim     = sim_arr[y_feature]

    # Mask NaNs jointly per dataset.
    g4_mask  = np.isfinite(x_g4_full) & np.isfinite(y_g4_full)
    sim_mask = np.isfinite(x_sim)     & np.isfinite(y_sim)
    x_g4_full, y_g4_full = x_g4_full[g4_mask], y_g4_full[g4_mask]
    x_sim,     y_sim     = x_sim[sim_mask],   y_sim[sim_mask]

    # G4 vs G4 split-half (noise floor reference).
    n_half = len(x_g4_full) // 2
    x_g4_a, x_g4_b = x_g4_full[:n_half],     x_g4_full[n_half:2 * n_half]
    y_g4_a, y_g4_b = y_g4_full[:n_half],     y_g4_full[n_half:2 * n_half]

    g4_raw   = _raw_hist(x_g4_full, y_g4_full, x_bins, y_bins)
    sim_raw  = _raw_hist(x_sim,     y_sim,     x_bins, y_bins)
    g4_vs_g4 = _rel_diff(x_g4_a, y_g4_a, x_g4_b, y_g4_b, x_bins, y_bins)
    sim_vs_g4 = _rel_diff(x_g4_full, y_g4_full, x_sim, y_sim, x_bins, y_bins)

    return jsonify({
        "x_feature": x_feature,
        "y_feature": y_feature,
        "x_label":   _LABELS[x_feature],
        "y_label":   _LABELS[y_feature],
        "x_bins":    x_bins.tolist(),
        "y_bins":    y_bins.tolist(),
        # Histograms are (n_x, n_y); transpose at the client to match Plotly's
        # (rows = y, cols = x) heatmap convention.
        "g4_hist":     g4_raw.tolist(),
        "sim_hist":    sim_raw.tolist(),
        "g4_vs_g4":    g4_vs_g4.tolist(),
        "sim_vs_g4":   sim_vs_g4.tolist(),
        "n_g4":        int(x_g4_full.size),
        "n_sim":       int(x_sim.size),
        "g4_file":     g4_file,
    })
