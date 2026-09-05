"""Per-stage MC sampler + analytical-PDF dispatch.

Each verification stage has up to four backends the playground can compare
against one another at a single phase-space point:

  * `g4`        – empirical histogram drawn from a GEANT4 truth file.
  * `mc`        – the physics-MC samplers exposed via `PhysicsXxxGenerator`
                  (drop-in `GeneratorInterface` subclasses that already
                  bypass `@scaled_forward`).
  * `analytical` – the closed-form straggling PDF / CDF for the stage.
  * `nn`        – a trained generator loaded from a Lightning checkpoint.

This module gives the Flask handlers a unified call surface for `mc`,
`analytical`, and `nn`; the `g4` slice is served by `api.g4_data.hist1d`.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
import torch

import common.units as U
from common.utils import PointInPhaseSpace
from common.enums import G4Columns

from playground.cache import get_or_load


# --- Stage metadata ---------------------------------------------------------

STAGES = ("length", "continuous", "secondary")

STAGE_CONFIG = {
    "length": {
        "needs": ("E",),
        "label": "Step length [m]",
        "g4_col": G4Columns.StepLength.value,
        "log_x": True,
    },
    "continuous": {
        "needs": ("E", "L"),
        "label": "Continuous energy loss [eV]",
        "g4_col": G4Columns.ContinuousLoss.value,
        "log_x": True,
    },
    "secondary": {
        "needs": ("E",),
        "label": "Secondary energy [eV]",
        "g4_col": G4Columns.SecondaryLoss.value,
        "log_x": True,
    },
}


def available_stages() -> tuple[str, ...]:
    """Stages valid for the active beam. Every registered beam is a proton
    beam, so every stage in STAGES is always available."""
    return STAGES


# --- MC samplers -----------------------------------------------------------

def _mc_sampler(stage: str):
    """Return a cached PhysicsXxxGenerator for `stage`."""
    def loader():
        if stage == "length":
            from physics.g4h_ionisation.generators.length_generator.physics_mc import PhysicsLengthGenerator
            return PhysicsLengthGenerator()
        if stage == "continuous":
            from physics.g4h_ionisation.generators.continuous_generator.physics_mc import PhysicsContinuousGenerator
            return PhysicsContinuousGenerator()
        if stage == "secondary":
            from physics.g4h_ionisation.generators.secondary_generator.physics_mc import PhysicsSecondaryGenerator
            return PhysicsSecondaryGenerator()
        raise KeyError(stage)
    return get_or_load(f"mc:{stage}", loader)


def sample_mc(stage: str, E_eV: float, L_m: Optional[float], n: int,
              labels: Optional[torch.Tensor] = None,
              eloss_frac: float = 0.0,
              eloss_mode: str = "point") -> np.ndarray:
    """Sample `stage` at a point (labels=None: repeat (E_eV, L_m) n times)
    or at per-event conditioning labels from g4_slices.in_bin_labels.

    `eloss_frac` / `eloss_mode` are accepted for API compatibility with the
    caller but no stage conditions on a post-step energy, so they are
    unused."""
    gen = _mc_sampler(stage)
    if labels is not None:
        # g4_slices returns 2-D (n, k); the length/secondary samplers take
        # 1-D E tensors (see the point-mode branches below).
        if stage in ("length", "secondary"):
            labels = labels.reshape(-1)
    elif stage == "length":
        labels = torch.full((n,), float(E_eV), dtype=torch.float32)
    elif stage == "continuous":
        labels = torch.tensor(
            [[float(E_eV), float(L_m)]] * n, dtype=torch.float32,
        )
    elif stage == "secondary":
        labels = torch.full((n,), float(E_eV), dtype=torch.float32)
    else:
        raise KeyError(stage)
    with torch.inference_mode():
        out = gen.predict(labels)
    return out.detach().cpu().numpy().reshape(-1)


# --- Analytical PDFs --------------------------------------------------------

def analytical_pdf(stage: str, E_eV: float, L_m: Optional[float],
                   x_min: float, x_max: float, n_points: int = 1024,
                   log_x: bool = True,
                   eloss_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (xs, pdf, regime_label) for the analytical straggling function.

    Domain conventions per stage:
      length     – xs in metres, no log floor: PDF is the truncated
                   exponential over [0, L_eloss].
      continuous – xs in eV, log-spaced.
      secondary  – xs in eV, log-spaced. Heavy primaries: Rutherford 1/E^2
                   tail; electrons: the Møller z(x)/x^2 spectrum.
    """
    if stage == "length":
        from physics.g4h_ionisation.explicit_physics.step_length.straggling_function import StepLengthModel
        model = get_or_load("ana:length", StepLengthModel)
        model.set_state(PointInPhaseSpace(primary_energy=float(E_eV)))
        # Model works in cm; convert from/to metres.
        xs_m = np.linspace(max(x_min, 1e-12), x_max, n_points) if not log_x else np.logspace(
            np.log10(max(x_min, 1e-12)), np.log10(max(x_max, x_min * 10)), n_points,
        )
        xs_cm = xs_m * U.m2cm
        pdf_per_cm = model.straggling_function(xs_cm)
        pdf_per_m = pdf_per_cm * U.m2cm  # Jacobian d(cm)/d(m) = 100
        return xs_m, pdf_per_m, model.step_length_type.name

    if stage == "continuous":
        from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import ContinuousStragglingModel
        model = get_or_load("ana:continuous", ContinuousStragglingModel)
        # set_state expects step_length in cm.
        point = PointInPhaseSpace(step_length=float(L_m) * U.m2cm,
                                  primary_energy=float(E_eV))
        model.set_state(point)
        # The continuous straggling PDF is assembled via fftconvolve and a
        # `np.diff(x)`-based normalization (see common.utils.convolve_2_pdfs /
        # __fix_convolution). Both require a UNIFORM x-grid; a log-spaced
        # grid produces a distorted shape and a wrong normalization integral.
        # So we always evaluate on a fine linear grid here, then interpolate
        # onto the requested (typically log-spaced) display grid.
        # The excitation/ionization PDF kernels assert x[0] == 0 (see
        # _excitation_straggling_function.py and _ionization_straggling_function.py)
        # and convolve_2_pdfs needs the grid anchored at zero loss.
        #
        # Resolution: scale the linear grid to the PHYSICS, not to the sample
        # band. The straggling PDF has a sharp ionization-onset step at
        # w3 ~ alfa*E0 ~ 15 eV; tying the grid to the wide sample band (the
        # Rutherford tail reaches ~1e4 eV) spreads a fixed point budget to
        # ~5 eV spacing, smearing that edge into a spurious 10-15 eV ramp.
        # Mimic physics.g4h_ionisation.generators.continuous_generator.generate_cdfs.process_one:
        # span [0, min(100*average_loss, 1e7)] with 100k points (~0.1 eV
        # spacing for a typical thin step), then interpolate onto the display
        # grid bounded by the model's support.
        lin_lo = 0.0
        avg_loss = float(getattr(model, "average_loss", 0.0) or 0.0)
        lin_hi = min(100.0 * avg_loss, 1e7) if avg_loss > 0 else max(x_max, 1.0)
        lin_hi = max(lin_hi, x_min + 1.0, 1.0)
        n_lin = 100_000
        xs_lin = np.linspace(lin_lo, lin_hi, n_lin)
        if log_x:
            out_hi = min(max(x_max, 1e3), lin_hi)
            xs_out = np.logspace(np.log10(max(x_min, 1.0)), np.log10(max(out_hi, 10.0)), n_points)
        else:
            xs_out = np.linspace(lin_lo, lin_hi, n_points)
        try:
            pdf_lin = model.straggling_function(xs_lin)
        except (RuntimeError, NotImplementedError, ValueError) as e:
            return xs_out, np.zeros_like(xs_out), f"unsupported: {e}"
        # The model packs exp(-lambda) of mass into pdf_lin[0] as a zero-loss
        # "delta" (see continuous/_utils.py::special_normalization). The
        # playground's MC histogram discards v<=0 and renormalizes by N_positive,
        # so it represents the density conditioned on positive loss. Drop the
        # delta and renormalize on the positive tail so the analytical curve
        # overlays the MC density correctly.
        pdf_lin = np.asarray(pdf_lin, dtype=float)
        if pdf_lin.size > 1:
            pdf_lin[0] = pdf_lin[1]
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        norm = trapz(pdf_lin, xs_lin)
        if norm > 0:
            pdf_lin = pdf_lin / norm
        pdf_out = np.interp(xs_out, xs_lin, pdf_lin, left=0.0, right=0.0)
        return xs_out, pdf_out, "+".join(t.name for t in model.straggling_type) or "Bethe-Bloch"

    if stage == "secondary":
        from physics.g4h_ionisation.explicit_physics.discrete.straggling_function import DiscreteStragglingModel
        model = get_or_load("ana:secondary", DiscreteStragglingModel)
        point = PointInPhaseSpace(primary_energy=float(E_eV))
        model.set_state(point)
        lo = max(model.min_ion_energy_transfer, 1.0)
        hi = model.max_ion_energy_transfer
        xs = np.logspace(np.log10(lo), np.log10(hi), n_points) if log_x \
            else np.linspace(lo, hi, n_points)
        pdf = model.straggling_function(xs)
        return xs, pdf, model.regime_label

    raise KeyError(stage)


def stage_g4_column(stage: str) -> str:
    return STAGE_CONFIG[stage]["g4_col"]
