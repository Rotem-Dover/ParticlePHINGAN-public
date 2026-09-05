"""Pull-analysis endpoint.

Runs a TrackSimulator with the user-chosen backend mix and compares the
energy-deposition map against a G4 reference using the helpers from
`benchmark/system_vs_g4/pull_analysis/energy_deposition_pull_analysis.py`.

The expensive parts (loading the G4 reference, running the simulator,
binning a 1500x1500 map) are cached so successive requests are responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from common.config import BEAM_ENERGY_eV
from common.paths import TRACKS_DIR
from data_handling.reader import read_geant4_simulation_output
from data_handling.slim_reader import (PULL_SLIM_CACHE_VERSION,
                                       load_pull_slim_cached)

from common.enums import G4Columns
from playground.services.g4_slices import _FULL_LOAD_MAX_ROOT_BYTES
from playground.cache import get_or_load, peek as _peek
from playground.services import progress as progress_state
from playground.services.sim_runner import run_simulation

import logging
log = logging.getLogger(__name__)


bp = Blueprint("pull", __name__)

# Chunk size for the pull's system simulation. 10k events x 10250 steps is
# the proven single-run workload (~5-6 GB transient); larger n_events values
# are simulated as sequential chunks of this size and their slim frames
# concatenated with per-chunk EventNum offsets.
PULL_SIM_CHUNK_EVENTS = 10_000

# PULL_SLIM_CACHE_VERSION is owned by data_handling.slim_reader and
# re-exported above for callers/tests.


def _g4_cache_key(file_stem: str, displace: float | None) -> str:
    return f"pull:g4:{file_stem}:{displace}"


def _sim_cache_key(E0_eV: float, n_events: int, n_steps: int,
                   processes: list[dict]) -> str:
    return f"pull:sim:{E0_eV}:{n_events}:{n_steps}:{hash(repr(processes))}"


def _load_pull_slim(root: Path) -> pd.DataFrame:
    """Slim frame for a big pkl-less ROOT file, backed by the shared
    self-invalidating disk cache (data_handling.slim_reader.
    load_pull_slim_cached — see its docstring for the invalidation rules)."""
    progress_state.set_phase("reading G4 truth (chunked)")
    return load_pull_slim_cached(root, progress_cb=progress_state.update_step)


def _organized_g4(file_stem: str, displace: float | None = None,
                  cache_only: bool = False):
    from benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis import organize_df
    from common.enums import G4Columns

    key = _g4_cache_key(file_stem, displace)
    if cache_only:
        return _peek(key)

    def loader():
        pkl = TRACKS_DIR / f"{file_stem}.pkl"
        root = TRACKS_DIR / f"{file_stem}.root"
        if (not pkl.exists() and root.exists()
                and root.stat().st_size > _FULL_LOAD_MAX_ROOT_BYTES):
            raw = _load_pull_slim(root)
        else:
            # Full-frame path: full pkl present, pkl-less root at or below
            # the full-load threshold (so the full .pkl side-effect still
            # gets written, benefiting the Table tab / G4 Browser), or
            # neither file on disk (the reader raises / a test monkeypatches
            # it).
            raw = read_geant4_simulation_output(pkl if pkl.exists() else root,
                                                verbose=False)
        d = displace
        if d is None:
            # Auto-align the G4 beam-entry face to x = 0, the system
            # simulator's frame. Each G4 production has its own world
            # geometry (proton file: target spans [-0.05, 0] m; electron
            # file: beam entry at -0.50 m), so a fixed offset cannot work
            # for all of them.
            d = -float(np.min(raw[G4Columns.X].values))
        return organize_df(raw, displace=d)
    return get_or_load(key, loader)


def _run_sim_once(processes: list[dict], preset_name: str | None,
                  E0_eV: float, n_events: int, n_steps: int,
                  g4_file: str | None,
                  phase_label: str = "simulating"):
    """One TrackSimulator run."""
    log.info("PULL SLOW PATH: rebuilding SteppingManager")
    return run_simulation(processes, E0_eV, n_events, n_steps,
                          save_steps_states=True, source="pull",
                          g4_file=g4_file, phase_label=phase_label)


def _organized_sim(processes: list[dict], E0_eV: float, n_events: int,
                   n_steps: int, displace: float = 0.0,
                   preset_name: str | None = None,
                   g4_file: str | None = None,
                   cache_only: bool = False):
    from benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis import organize_df

    key = _sim_cache_key(E0_eV, n_events, n_steps, processes)
    if cache_only:
        return _peek(key)

    def loader():
        n_chunks = -(-int(n_events) // PULL_SIM_CHUNK_EVENTS)  # ceil div
        parts: list[pd.DataFrame] = []
        for i in range(n_chunks):
            ev = min(PULL_SIM_CHUNK_EVENTS, int(n_events) - i * PULL_SIM_CHUNK_EVENTS)
            label = (f"pull sim chunk {i + 1}/{n_chunks}"
                     if n_chunks > 1 else "simulating")
            ts = _run_sim_once(processes, preset_name, E0_eV, ev, n_steps,
                               g4_file, phase_label=label)
            df = ts.states_and_steps_features_df.dropna()
            part = organize_df(df, displace=displace)
            if n_chunks > 1:
                # Distinct chunks' events must not merge in the per-event
                # variance estimator (compute_hist_stats groups by EventNum).
                part[G4Columns.EventNum] = part[G4Columns.EventNum] + i * PULL_SIM_CHUNK_EVENTS
            parts.append(part)
        if len(parts) == 1:
            return parts[0]
        return pd.concat(parts, ignore_index=True)
    return get_or_load(key, loader)


def _parse_common(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse the fields shared by /run and /rebin from the request payload."""
    _displace_raw = payload.get("displace_g4")
    return {
        "g4_file":    payload.get("g4_file", "100MeV_10k"),
        "E0":         float(payload.get("E0_eV", BEAM_ENERGY_eV)),
        "n_events":   int(payload.get("n_events", 10_000)),
        "n_steps":    int(payload.get("n_steps", 10_250)),
        "n_2d_bins":  tuple(payload.get("n_2d_bins", [1500, 1500])),
        "n_1d_bins":  int(payload.get("pull_1d_n_bins", 1000)),
        "pull_min":   int(payload.get("pull_min_counts_per_bin", 3)),
        "hist_range": tuple(payload.get("pull_1d_hist_range", [-5.0, 5.0])),
        "fit_range":  tuple(payload.get("pull_fit_range", [-30.0, 30.0])),
        # None → auto-align the G4 beam-entry face to x = 0 (see _organized_g4).
        "displace_g4": float(_displace_raw) if _displace_raw is not None else None,
    }


def _resolve_processes(payload: dict[str, Any]):
    """Return (processes, preset_name). Raises KeyError on unknown preset."""
    if "preset" in payload:
        raise KeyError(f"unknown preset {payload['preset']!r}: no named presets "
                       "are registered; send an explicit 'processes' list")
    return payload["processes"], None


def _compute_pull_response(df_g4, df_sim, *, n_2d_bins, n_1d_bins, pull_min,
                           hist_range, fit_range, preset_label):
    """Histogram + Gaussian-fit the pull map and build the JSON-ready response dict.

    The cheap stage shared by /run and /rebin: it consumes the (cached)
    organized G4 + sim DataFrames and returns the response payload as a plain
    dict (the caller wraps it in jsonify).
    """
    from benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis import (
        compute_hist_stats, compute_2d_pull, fit_pull_distribution,
    )
    from common.enums import G4Columns

    x_range = (df_g4[G4Columns.X].min(), df_g4[G4Columns.X].max())
    y_range = (df_g4[G4Columns.Y].min(), df_g4[G4Columns.Y].max())
    range_lims = [x_range, y_range]
    hist_g4,  var_g4,  cnt_g4  = compute_hist_stats(df_g4,  range_lims, n_2d_bins)
    hist_sim, var_sim, cnt_sim = compute_hist_stats(df_sim, range_lims, n_2d_bins)
    pull_2d = compute_2d_pull(hist_g4, var_g4, hist_sim, var_sim)

    avg_cnt_2d = (cnt_g4 + cnt_sim) / 2
    pull_2d_masked = pull_2d.astype(float).copy()
    pull_2d_masked[avg_cnt_2d <= pull_min] = np.nan

    pull_values = pull_2d_masked.ravel()
    avg_cnt = avg_cnt_2d.ravel()

    pull_histogram, popt, pcov, chi2 = fit_pull_distribution(
        pull_values=pull_values,
        n_bins_hist=n_1d_bins,
        hist_range=hist_range,
        fit_range=fit_range,
    )
    _amp, mu_fit, sigma_fit = popt
    mu_err, sigma_err = float(np.sqrt(pcov[1, 1])), float(np.sqrt(pcov[2, 2]))

    bin_centers, bin_counts = pull_histogram

    # Energy-deposition thumbnails (MeV) — for the PHIN-GAN / GEANT4 panels.
    eV_to_MeV = 1.0e-6
    hist_sim_mev = hist_sim * eV_to_MeV
    hist_g4_mev  = hist_g4  * eV_to_MeV

    # Dense Gaussian fit curve for the pull-distribution overlay.
    amp_fit = float(popt[0])
    fit_x = np.linspace(hist_range[0], hist_range[1], 400)
    fit_y = amp_fit * np.exp(-0.5 * ((fit_x - mu_fit) / sigma_fit) ** 2)

    # System backend label: prefer preset name, fall back to a short hash.
    label = preset_label if preset_label is not None else "custom"
    backend_label_map = {
        "mc":  "GEANT4-port physics MC",
        "net": "Neural ionisation",
    }
    system_title = backend_label_map.get(label, f"system ({label})")

    return {
        "mu":       float(mu_fit),
        "mu_err":   mu_err,
        "sigma":    float(sigma_fit),
        "sigma_err": sigma_err,
        "chi2_ndof": float(chi2),
        "pull_hist": {
            "centers": bin_centers.tolist(),
            "counts":  bin_counts.tolist(),
            "fit_x":   fit_x.tolist(),
            "fit_y":   fit_y.tolist(),
        },
        "pull_map": {
            # Downsample the full pull map (n_2d_bins shape) for transport. The frontend
            # gets a thumbnail; the raw map stays on the server. Cells with
            # too few counts are emitted as JSON null so Plotly renders them
            # transparent (panel background) instead of as a 0 / "green" cell.
            "thumbnail": _nan_to_none(_downsample(pull_2d_masked, 200)),
            "x_range":   [float(x_range[0]), float(x_range[1])],
            "y_range":   [float(y_range[0]), float(y_range[1])],
        },
        # Energy-deposition heatmaps (MeV), matching the top row of
        # artifacts/benchmark_outputs/energy_deposition_pull.pdf.
        "hist_sim": {
            "thumbnail": _downsample(hist_sim_mev, 200, agg="sum").tolist(),
            "x_range":   [float(x_range[0]), float(x_range[1])],
            "y_range":   [float(y_range[0]), float(y_range[1])],
            "title":     system_title,
        },
        "hist_g4": {
            "thumbnail": _downsample(hist_g4_mev, 200, agg="sum").tolist(),
            "x_range":   [float(x_range[0]), float(x_range[1])],
            "y_range":   [float(y_range[0]), float(y_range[1])],
            "title":     "GEANT4",
        },
        "n_bins_used": int(np.sum(avg_cnt > pull_min)),
        "n_2d_bins":   list(n_2d_bins),
    }


@bp.route("/run", methods=["POST"])
def run():
    """Compute a single pull (mu, sigma) between a simulated batch and a G4 file.

    POST JSON shape:
        {
          "g4_file": "100MeV_10k",
          "preset": "phin"  OR  "processes": [...],
          "E0_eV": 99.99e6,
          "n_events": 10000,
          "n_steps":  10250,
          "n_2d_bins": [1500, 1500],
          "pull_1d_n_bins": 1000,
          "pull_min_counts_per_bin": 3,
          "pull_1d_hist_range": [-5, 5],
          "pull_fit_range":     [-30, 30],
          "displace_g4": 0.05   # optional; default auto-aligns entry face to x=0
        }
    """
    payload: dict[str, Any] = request.get_json(force=True) or {}
    p = _parse_common(payload)

    try:
        processes, preset_name = _resolve_processes(payload)
    except KeyError as e:
        return jsonify({"error": f"unknown preset {e}"}), 400

    try:
        df_g4  = _organized_g4(p["g4_file"], displace=p["displace_g4"])
        df_sim = _organized_sim(processes, p["E0"], p["n_events"], p["n_steps"],
                                displace=0.0, preset_name=preset_name,
                                g4_file=p["g4_file"])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"prep failed: {type(e).__name__}: {e}"}), 500

    resp = _compute_pull_response(
        df_g4, df_sim,
        n_2d_bins=p["n_2d_bins"], n_1d_bins=p["n_1d_bins"], pull_min=p["pull_min"],
        hist_range=p["hist_range"], fit_range=p["fit_range"],
        preset_label=payload.get("preset"),
    )
    return jsonify(resp)


@bp.route("/rebin", methods=["POST"])
def rebin():
    """Recompute the pull from the *cached* dataframes at a new bin count.

    Cache-only: if no prior /run populated the g4/sim caches for these
    parameters, return 409 instead of re-simulating. This backs the Pull
    tab's fast "Re-bin" button.
    """
    payload: dict[str, Any] = request.get_json(force=True) or {}
    p = _parse_common(payload)

    try:
        processes, preset_name = _resolve_processes(payload)
    except KeyError as e:
        return jsonify({"error": f"unknown preset {e}"}), 400

    try:
        df_g4  = _organized_g4(p["g4_file"], displace=p["displace_g4"],
                               cache_only=True)
        df_sim = _organized_sim(processes, p["E0"], p["n_events"], p["n_steps"],
                                displace=0.0, preset_name=preset_name,
                                g4_file=p["g4_file"], cache_only=True)
    except KeyError:
        return jsonify({
            "error": "no cached run for these parameters — "
                     "run the full pull analysis first",
        }), 409

    resp = _compute_pull_response(
        df_g4, df_sim,
        n_2d_bins=p["n_2d_bins"], n_1d_bins=p["n_1d_bins"], pull_min=p["pull_min"],
        hist_range=p["hist_range"], fit_range=p["fit_range"],
        preset_label=payload.get("preset"),
    )
    return jsonify(resp)


def _nan_to_none(arr: np.ndarray) -> list:
    """Serialize a 2-D float array to a JSON-safe list-of-lists.

    NaN / inf are mapped to None so the browser receives valid JSON (standard
    JSON.parse rejects the NaN literal that Python's json emits by default).
    Plotly's heatmap renders None cells as transparent.
    """
    return [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in arr
    ]


def _downsample(arr: np.ndarray, target: int, agg: str = "mean") -> np.ndarray:
    """Block-aggregate downsample to ~target x target.

    agg='mean'  — average per block (for pull-map / signed quantities).
    agg='sum'   — sum per block, preserving total energy when reducing
                  an energy-deposition map; this matches the PDF where each
                  pixel reports total MeV deposited.
    NaNs are treated as zero for the sum, and skipped for the mean.
    """
    h, w = arr.shape
    if h <= target and w <= target:
        # Preserve NaN here: callers that need a JSON-safe payload run the
        # result through `_nan_to_none`; sums use `np.nan_to_num` themselves
        # if they want zero-fill, but the energy-deposition maps don't carry
        # NaNs upstream so this branch is effectively a no-op for them.
        return arr
    sh = max(1, h // target)
    sw = max(1, w // target)
    h2 = (h // sh) * sh
    w2 = (w // sw) * sw
    trimmed = arr[:h2, :w2]
    reshaped = trimmed.reshape(h2 // sh, sh, w2 // sw, sw)
    with np.errstate(invalid="ignore"):
        if agg == "sum":
            return np.nansum(reshaped, axis=(1, 3))
        return np.nanmean(reshaped, axis=(1, 3))
