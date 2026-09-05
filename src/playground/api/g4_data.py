"""G4 data browser endpoints.

Lists available GEANT4 track files and serves binned 1D/2D histograms of
step features at user-selectable cuts. Cached because the underlying
ROOT/pickle reads are slow (>1 s for 10k events).
"""
from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify, request

from common.enums import G4Columns, TOTAL_ENERGY_LOSS
from common.paths import TRACKS_DIR

from playground.services.g4_slices import (
    TracksFileTooLarge, bin_mask, load_tracks,
)
from physics.g4h_ionisation.explicit_physics.continuous.phase_space_boundaries import (
    compute_phase_space_boundaries, PhaseSpaceBoundaries,
)


bp = Blueprint("g4_data", __name__)

_regions_cache: PhaseSpaceBoundaries | None = None


# Columns the user is allowed to histogram. Some are derived (TotalEnergyLoss).
_ALLOWED_COLUMNS = [
    G4Columns.KineticEnergy.value,
    G4Columns.StepLength.value,
    G4Columns.GeomStepLength.value,
    G4Columns.LateralDisplacement.value,
    G4Columns.ContinuousLoss.value,
    G4Columns.SecondaryLoss.value,
    G4Columns.angleDiscrete.value,
    G4Columns.X.value, G4Columns.Y.value, G4Columns.Z.value,
    G4Columns.NumOfSecondaries.value,
    TOTAL_ENERGY_LOSS,
]


def _load(name: str):
    """Cache-backed loader for a tracks file by base name (e.g. 100MeV_10k)."""
    return load_tracks(name)


class _NoFile(Exception):
    """Raised when no track file is selected/available (empty `file` param)."""


def _require_name(default: str) -> str:
    """Return the requested file name, or raise `_NoFile` if it's blank.

    The frontend can send `file=` (empty) before any track file is known — e.g.
    a fresh checkout whose gitignored `storage/` has no GEANT4 tracks yet. That
    must surface as a clean 400, not a `FileNotFoundError` 500 from `load_tracks`.
    """
    name = (request.args.get("file", default) or "").strip()
    if not name:
        raise _NoFile
    return name


@bp.errorhandler(_NoFile)
def _handle_no_file(_):
    return jsonify({"error": "no track file selected (storage/.../tracks is empty?)"}), 400


@bp.errorhandler(TracksFileTooLarge)
def _handle_too_large(exc):
    # Root-only files above the full-load cap are still valid for the Pull
    # tab's streaming loader — tell the user that instead of a bare 500.
    return jsonify({"error": str(exc)}), 413


@bp.route("/files", methods=["GET"])
def list_files():
    if not TRACKS_DIR.exists():
        return jsonify({"files": []})
    seen: dict[str, dict] = {}
    for p in sorted(TRACKS_DIR.iterdir()):
        if p.suffix not in (".root", ".pkl"):
            continue
        # Exclude the Pull tab's internal `<stem>.pull_slim.pkl` disk cache
        # (see playground/api/pull.py::_load_pull_slim) — it is a derived
        # 4-column artifact, not a user-selectable track file.
        if p.name.endswith(".pull_slim.pkl"):
            continue
        seen.setdefault(p.stem, {"name": p.stem, "ext": []})
        seen[p.stem]["ext"].append(p.suffix.lstrip("."))
    return jsonify({"files": list(seen.values())})


@bp.route("/summary", methods=["GET"])
def summary():
    name = _require_name("100MeV_1k")
    df = _load(name)
    return jsonify({
        "name": name,
        "n_rows": int(df.shape[0]),
        "n_events": int(df[G4Columns.EventNum].nunique()),
        "columns": [c for c in _ALLOWED_COLUMNS if c in df.columns],
    })


def _safe_log_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    positive = values[values > 0]
    if positive.size == 0:
        return np.linspace(values.min() if values.size else 0.0,
                           values.max() if values.size else 1.0, n_bins + 1)
    return np.logspace(np.log10(positive.min()), np.log10(positive.max()), n_bins + 1)


@bp.route("/hist2d", methods=["GET"])
def hist2d():
    """2D histogram of (x_col vs y_col). Default: kinetic energy vs step length."""
    name  = _require_name("100MeV_1k")
    x_col = request.args.get("x", G4Columns.KineticEnergy.value)
    y_col = request.args.get("y", G4Columns.StepLength.value)
    nbins = int(request.args.get("bins", 120))
    log_x = request.args.get("log_x", "1") == "1"
    log_y = request.args.get("log_y", "1") == "1"

    if x_col not in _ALLOWED_COLUMNS or y_col not in _ALLOWED_COLUMNS:
        return jsonify({"error": f"column not allowed: {x_col} / {y_col}"}), 400

    df = _load(name)
    if x_col not in df.columns or y_col not in df.columns:
        return jsonify({"error": "column not in dataframe"}), 400

    x = np.abs(df[x_col].values) if log_x else df[x_col].values
    y = np.abs(df[y_col].values) if log_y else df[y_col].values

    x_bins = _safe_log_bins(x, nbins) if log_x else np.linspace(x.min(), x.max(), nbins + 1)
    y_bins = _safe_log_bins(y, nbins) if log_y else np.linspace(y.min(), y.max(), nbins + 1)
    h, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return jsonify({
        "x_edges": x_bins.tolist(),
        "y_edges": y_bins.tolist(),
        "z":       h.tolist(),     # shape (nx, ny)
        "x_col":   x_col,
        "y_col":   y_col,
        "log_x":   log_x,
        "log_y":   log_y,
    })


@bp.route("/hist1d", methods=["GET"])
def hist1d():
    """1D histogram of `col` from `file`, optionally filtered by an (E,L) cell.

    If `e_center` and `l_center` are provided (plus fractional half-widths
    `e_halfwidth` and `l_halfwidth`, in dex when log_e/log_l are 1), restrict
    to rows whose kinetic energy and step length fall inside that bin. This
    is how the G4 browser slices truth at a phase-space point.

    `transform=one_minus_cos` maps the histogrammed values theta -> 1-cos(theta)
    (after the phase-space cut), for columns that are stored as a raw angle in
    radians but compared against a 1-cos(theta) quantity elsewhere.
    """
    name = _require_name("100MeV_1k")
    col  = request.args.get("col", G4Columns.ContinuousLoss.value)
    nbins = int(request.args.get("bins", 120))
    log_x = request.args.get("log_x", "1") == "1"
    transform = request.args.get("transform")

    if col not in _ALLOWED_COLUMNS:
        return jsonify({"error": f"column not allowed: {col}"}), 400
    if transform not in (None, "one_minus_cos"):
        return jsonify({"error": f"unknown transform: {transform}"}), 400
    df = _load(name)
    if col not in df.columns:
        return jsonify({"error": f"column missing: {col}"}), 400

    # Shared with the bin-conditioned sampling paths (api.samplers /
    # api.generators) so the G4 histogram and the MC/NN labels always
    # derive from the identical row set.
    def _farg(name):
        v = request.args.get(name)
        return float(v) if v is not None else None

    mask = bin_mask(
        df,
        e_center=_farg("e_center"), e_halfwidth=_farg("e_halfwidth"),
        l_center=_farg("l_center"), l_halfwidth=_farg("l_halfwidth"),
    )
    # Optional row filters: `process` restricts to a ProcessName, `min_val`
    # drops values at or below a floor. Generic params — the endpoint stays a
    # stage-agnostic column browser.
    proc = request.args.get("process")
    if proc:
        pcol = G4Columns.ProcessName.value
        if pcol not in df.columns:
            return jsonify({"error": f"column missing: {pcol}"}), 400
        mask &= (df[pcol].values == proc)
    values = df[col].values[mask]
    min_val = request.args.get("min_val")
    if min_val is not None:
        values = values[values > float(min_val)]
    if transform == "one_minus_cos":
        values = 1.0 - np.cos(values)
    if values.size == 0:
        return jsonify({"edges": [], "counts": [], "n": 0})

    if log_x:
        vals = np.abs(values)
        vals = vals[vals > 0]
        if vals.size == 0:
            return jsonify({"edges": [], "counts": [], "n": 0})
        edges = np.logspace(np.log10(vals.min()), np.log10(vals.max()), nbins + 1)
        counts, _ = np.histogram(vals, bins=edges)
    else:
        edges = np.linspace(values.min(), values.max(), nbins + 1)
        counts, _ = np.histogram(values, bins=edges)

    return jsonify({
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "n": int(values.size),
        "col": col,
        "log_x": log_x,
    })


@bp.route("/phase_space_regions", methods=["GET"])
def phase_space_regions():
    """Straggling-regime boundary curves for the (E, L) phase space.

    Computed once per server process via root-finding (brentq along E-slices).
    Result is cached in _regions_cache; server restart clears it.
    """
    global _regions_cache
    if _regions_cache is None:
        _regions_cache = compute_phase_space_boundaries()

    def _nan_to_null(arr):
        return [None if not np.isfinite(v) else float(v) for v in arr]

    return jsonify({
        "energy_eV": _regions_cache.energy_eV.tolist(),
        "boundaries": {
            k: _nan_to_null(v)
            for k, v in _regions_cache.boundaries.items()
        },
        "regions": _regions_cache.regions,
    })
