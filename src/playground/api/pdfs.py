"""Analytical-PDF endpoint.

Returns the analytical straggling-function PDF curve for a stage at a given
phase-space point, on a domain chosen to match the MC sampler histogram so
the two can be overlaid directly on the frontend.
"""
from __future__ import annotations

import math

from flask import Blueprint, jsonify, request

from playground.services.sampling import analytical_pdf, available_stages


bp = Blueprint("pdfs", __name__)


@bp.route("/curve", methods=["GET"])
def curve():
    stage = request.args.get("stage", "continuous")
    if stage not in available_stages():
        return jsonify({"error": f"unknown or unavailable stage {stage}"}), 400
    try:
        E = float(request.args.get("E"))
    except (TypeError, ValueError):
        return jsonify({"error": "E (eV) required"}), 400
    L = request.args.get("L")
    L = float(L) if L is not None else None
    x_min = float(request.args.get("x_min", 1e-6))
    x_max = float(request.args.get("x_max", 1e6))
    n_points = int(request.args.get("n_points", 1024))
    log_x = request.args.get("log_x", "1") == "1"
    eloss_frac = float(request.args.get("eloss_frac", 0.0))

    try:
        xs, pdf, regime = analytical_pdf(stage, E, L, x_min, x_max, n_points,
                                         log_x, eloss_frac=eloss_frac)
    except Exception as e:  # noqa: BLE001 — surface as 500 with a useful message
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # NaN/Inf cleanup so the frontend can safely plot the curve.
    xs_clean  = [float(x) if math.isfinite(x) else None for x in xs]
    pdf_clean = [float(y) if math.isfinite(y) else None for y in pdf]
    return jsonify({
        "stage":  stage,
        "x":      xs_clean,
        "pdf":    pdf_clean,
        "regime": regime,
        "log_x":  log_x,
    })
