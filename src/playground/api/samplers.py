"""MC sampler endpoint for the verification tab.

Given a stage and a phase-space point, returns N samples from the physics-MC
sampler at that point, binned into a 1D histogram ready for Plotly.
"""
from __future__ import annotations

import numpy as np
from flask import Blueprint, jsonify, request

from playground.services.g4_slices import in_bin_labels, parse_bin_args
from playground.services.sampling import (
    sample_mc, STAGE_CONFIG, available_stages,
)


bp = Blueprint("samplers", __name__)


@bp.route("/stages", methods=["GET"])
def stages():
    avail = available_stages()
    return jsonify({"stages": list(avail),
                    "config": {s: STAGE_CONFIG[s] for s in avail}})


def _parse_query():
    stage = request.args.get("stage", "continuous")
    if stage not in available_stages():
        raise ValueError(f"unknown or unavailable stage {stage}")
    E = float(request.args.get("E"))
    L = request.args.get("L")
    L = float(L) if L is not None else None
    n = int(request.args.get("n", 50_000))
    nbins = int(request.args.get("bins", 120))
    log_x = request.args.get("log_x", "1") == "1"
    eloss_frac = float(request.args.get("eloss_frac", 0.0))
    eloss_mode = request.args.get("eloss_mode", "point")
    return stage, E, L, n, nbins, log_x, eloss_frac, eloss_mode


@bp.route("/sample", methods=["GET"])
def sample():
    try:
        stage, E, L, n, nbins, log_x, eloss_frac, eloss_mode = _parse_query()
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    # Bin conditioning: when the client pins a (E, L) cell, draw one MC
    # sample per in-bin G4 event, conditioned on that event's phase-space
    # values. Falls back to point conditioning at user n when the params
    # are absent, the file is missing, or the bin is empty.
    labels = None
    bin_q = parse_bin_args(request.args)
    if bin_q is not None:
        labels = in_bin_labels(stage=stage, **bin_q)
    conditioning = "bin" if labels is not None else "point"

    samples = sample_mc(stage, E, L, n, labels=labels, eloss_frac=eloss_frac,
                        eloss_mode=eloss_mode)
    samples = samples[np.isfinite(samples)]

    # Return raw samples so the client can bin MC and NN with shared edges.
    # Histograms are kept for backwards-compat callers that want a pre-binned
    # response (those still pass `bins` and may ignore `samples`).
    if log_x:
        positive = samples[samples > 0]
        if positive.size == 0:
            return jsonify({"edges": [], "counts": [], "samples_n": 0,
                            "samples": [],
                            "stage": stage, "log_x": log_x,
                            "conditioning": conditioning,
                            "bin_n": int(labels.shape[0]) if labels is not None else None,
                            "summary": {"mean": float("nan"), "std": float("nan")}})
        edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), nbins + 1)
        counts, _ = np.histogram(positive, bins=edges)
    else:
        edges = np.linspace(samples.min(), samples.max(), nbins + 1)
        counts, _ = np.histogram(samples, bins=edges)

    return jsonify({
        "edges":  edges.tolist(),
        "counts": counts.tolist(),
        "samples": samples.tolist(),
        "samples_n": int(samples.size),
        "summary": {
            "mean": float(np.mean(samples)) if samples.size else float("nan"),
            "std":  float(np.std(samples))  if samples.size else float("nan"),
            "min":  float(np.min(samples))  if samples.size else float("nan"),
            "max":  float(np.max(samples))  if samples.size else float("nan"),
        },
        "stage": stage,
        "log_x": log_x,
        "conditioning": conditioning,
        "bin_n": int(labels.shape[0]) if labels is not None else None,
    })
