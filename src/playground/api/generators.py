"""NN generator endpoint.

Loads a trained `GeneratorInterface` from a Lightning checkpoint, generates
N samples at the requested phase-space point, and returns a 1D histogram.
The loaded generator is cached by (stage, class, checkpoint) so repeated
calls with the same checkpoint don't re-read the file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flask import Blueprint, jsonify, request

from playground.cache import get_or_load
from playground.services.checkpoints import (
    all_checkpoints, STAGE_TO_CLASSES,
)
from playground.services.g4_slices import in_bin_labels, parse_bin_args


bp = Blueprint("generators", __name__)


@bp.route("/checkpoints", methods=["GET"])
def checkpoints():
    return jsonify({
        "checkpoints": all_checkpoints(),
        "classes": STAGE_TO_CLASSES,
    })


def _resolve_class(stage: str, class_name: str):
    """Resolve `class_name` to its actual Python class for the given stage."""
    if stage == "length":
        raise ValueError("the length stage has no trained network; "
                         "it is sampled analytically by PhysicsLengthGenerator")
    elif stage == "continuous":
        from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
            ContinuousGenerator,
        )
        opts = {ContinuousGenerator.__name__: ContinuousGenerator}
    elif stage == "secondary":
        from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
            SecondaryGenerator,
        )
        opts = {SecondaryGenerator.__name__: SecondaryGenerator}
    else:
        raise KeyError(stage)
    if class_name not in opts:
        raise KeyError(f"{class_name} not valid for stage {stage}")
    return opts[class_name]


def _resolve_scalers(stage: str, class_name: str):
    """Return the (x_scaler, y_scaler) instances for `stage`, loaded from SCALERS_DIR.

    Each `load()` classmethod reads the portable `<ClassName>.state.pt` from
    `SCALERS_DIR`. Bare constructors are NOT equivalent — e.g.
    `ContinuousXScaler()` carries `min_post_log=None` and crashes inside
    `predict`.
    """
    if stage == "length":
        raise ValueError("the length stage has no trained network; "
                         "it is sampled analytically by PhysicsLengthGenerator")
    if stage == "continuous":
        from physics.g4h_ionisation.generators.continuous_generator.scaler import (
            ContinuousXScaler, ContinuousYScaler,
        )
        return ContinuousXScaler.load(), ContinuousYScaler.load()
    if stage == "secondary":
        from physics.g4h_ionisation.generators.secondary_generator.scaler import (
            SecondaryXScaler, SecondaryYScaler,
        )
        return SecondaryXScaler.load(), SecondaryYScaler.load()
    raise KeyError(stage)


def _build_labels(stage: str, E: float, L: float, n: int,
                  eloss_frac: float = 0.0,
                  eloss_mode: str = "point") -> torch.Tensor:
    if stage == "length":
        return torch.full((n, 1), float(E), dtype=torch.float32)
    if stage == "continuous":
        return torch.tensor([[float(E), float(L)]] * n, dtype=torch.float32)
    if stage == "secondary":
        return torch.full((n, 1), float(E), dtype=torch.float32)
    raise KeyError(stage)


def _load_generator(stage: str, class_name: str, ckpt_path: str):
    def loader():
        cls = _resolve_class(stage, class_name)
        x_scaler, y_scaler = _resolve_scalers(stage, class_name)
        # infer_arch: a checkpoint's trunk width can differ from config.py's
        # current default (e.g. one fetched from a run trained at a
        # different width), so recover it from the state-dict shapes rather
        # than assuming the config constant applies.
        gen = cls.from_checkpoint(Path(ckpt_path), device="cpu",
                                  x_scaler=x_scaler, y_scaler=y_scaler,
                                  infer_arch=True)
        gen.freeze()
        return gen
    return get_or_load(f"nn:{stage}:{class_name}:{ckpt_path}", loader)


@bp.route("/generate", methods=["GET"])
def generate():
    stage = request.args.get("stage")
    class_name = request.args.get("class_name")
    ckpt = request.args.get("checkpoint")
    if not (stage and class_name and ckpt):
        return jsonify({"error": "stage, class_name and checkpoint required"}), 400

    from playground.services.sampling import available_stages
    if stage not in available_stages():
        return jsonify({"error": f"stage {stage!r} not available for the "
                                 "active particle"}), 400

    try:
        E = float(request.args.get("E"))
    except (TypeError, ValueError):
        return jsonify({"error": "E (eV) required"}), 400
    L = request.args.get("L")
    L = float(L) if L is not None else 0.0
    n = int(request.args.get("n", 20_000))
    nbins = int(request.args.get("bins", 120))
    log_x = request.args.get("log_x", "1") == "1"
    eloss_frac = float(request.args.get("eloss_frac", 0.0))
    eloss_mode = request.args.get("eloss_mode", "point")

    try:
        gen = _load_generator(stage, class_name, ckpt)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not load generator: {type(e).__name__}: {e}"}), 500

    # Bin conditioning: when the client pins a (E, L) cell, generate one NN
    # sample per in-bin G4 event, conditioned on that event's phase-space
    # values. Falls back to point conditioning at user n otherwise.
    labels = None
    bin_q = parse_bin_args(request.args)
    if bin_q is not None:
        labels = in_bin_labels(stage=stage, **bin_q)
    conditioning = "bin" if labels is not None else "point"
    bin_n = int(labels.shape[0]) if labels is not None else None
    if labels is None:
        labels = _build_labels(stage, E, L, n, eloss_frac=eloss_frac,
                               eloss_mode=eloss_mode)
    try:
        with torch.inference_mode():
            out = gen.predict(labels).detach().cpu().numpy().reshape(-1)
    except Exception as e:  # noqa: BLE001
        # Surface the real message: an uncaught exception here renders as
        # Flask's generic 500 page, which is undebuggable from the frontend.
        return jsonify({"error": f"predict failed: {type(e).__name__}: {e}"}), 500

    samples = out[np.isfinite(out)]
    # Return raw samples so the client can bin MC and NN with shared edges.
    if log_x:
        positive = samples[samples > 0]
        if positive.size == 0:
            return jsonify({"edges": [], "counts": [], "samples": [],
                            "samples_n": 0, "log_x": log_x,
                            "conditioning": conditioning,
                            "bin_n": bin_n})
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
        "stage":  stage,
        "log_x":  log_x,
        "class_name": class_name,
        "checkpoint": ckpt,
        "conditioning": conditioning,
        "bin_n": bin_n,
    })

