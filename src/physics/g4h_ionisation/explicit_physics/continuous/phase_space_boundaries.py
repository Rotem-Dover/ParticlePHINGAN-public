"""
Computes the four phase-space boundary curves for the continuous-straggling
regime map, using root-finding instead of a brute-force grid.

Each curve is found by a coarse log-spaced scan (to locate the sign change)
followed by scipy.optimize.brentq (to pin the crossing). Total cost is
~(n_scan + ~10) * n_energy * 4 calls to ContinuousLossParams.set_state()
instead of n_energy^2.

Boundary ordering at fixed E (ascending in L):
  very_thin  <  ion_gauss  <  exc_freq  <  thick_limit
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq

import common.units as U
from common.utils import PointInPhaseSpace
from common.run_context import ACTIVE, PARTICLE, TARGET_MATERIAL
from physics.definitions.particle import Particle
from physics.definitions.material import Material
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import (
    ContinuousLossParams,
)

_L_MIN_M: float = 1e-14   # metres — lower bound of bisection search
_L_MAX_M: float = 1e-1    # metres — upper bound
_E_MIN_eV: float = ACTIVE.grid_bounds.energy_min_eV
_E_MAX_eV: float = 100e6  # 100 MeV
_N_SCAN: int = 40          # coarse-scan points per boundary per E

REGIONS: list[dict] = [
    {"id": 1, "label": "Very thin limit",                                "color": "#F7F7F7", "alpha": 0.4},
    {"id": 2, "label": "Thin limit Exc.\nThin limit Ion. w/o Gauss.",   "color": "#BC80BD", "alpha": 0.4},
    {"id": 3, "label": "Thin limit Exc.\nThin limit Ion. w/Gauss.",     "color": "#8DD3C7", "alpha": 0.4},
    {"id": 4, "label": "Gauss. Exc.\nThin limit Ion. w/Gauss.",         "color": "#FB8072", "alpha": 0.4},
    {"id": 5, "label": "Thick Limit",                                   "color": "#FDB462", "alpha": 0.4},
]


@dataclass
class PhaseSpaceBoundaries:
    energy_eV:  np.ndarray             # shape (n_energy,), log-spaced eV
    boundaries: dict[str, np.ndarray]  # L in metres; NaN where boundary not found
    regions:    list[dict]             # id, label, color, alpha


def _find_first_crossing(
    params: ContinuousLossParams,
    energy_eV: float,
    condition_fn,
    l_min: float = _L_MIN_M,
    l_max: float = _L_MAX_M,
) -> float:
    """
    Coarse log scan to find first sign change of condition_fn(params),
    then brentq to refine. Returns NaN if no sign change found.

    condition_fn receives the ContinuousLossParams *after* set_state()
    has been called with the given (energy_eV, L) point.
    """
    ls = np.logspace(np.log10(l_min), np.log10(l_max), _N_SCAN)

    def f(l: float) -> float:
        params.set_state(PointInPhaseSpace(l * U.m2cm, energy_eV))
        return condition_fn(params)

    fvals = [f(l) for l in ls]

    for j in range(len(fvals) - 1):
        fa, fb = fvals[j], fvals[j + 1]
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            try:
                return brentq(f, ls[j], ls[j + 1], xtol=1e-16, rtol=1e-8)
            except ValueError:
                continue
    return float("nan")


def _thick_limit_possible(energy_eV: float, particle: Particle, material: Material) -> bool:
    """True only when the Bohr Gaussian regime can be entered at this energy."""
    if particle.mass_eV <= electron_mass_eV:
        return False
    # Omits the mean_loss >= n_min_bohr*Tc guard from set_fluct_params by design:
    # at short steps where that guard would fail, getattr(p, "sn", -1.0) returns
    # -1.0, keeping f negative and producing no false sign change.
    return particle.t_max(energy_eV) <= 2.0 * material.Tc


def compute_phase_space_boundaries(
    particle: Particle | None = None,
    material: Material | None = None,
    n_energy: int = 150,
) -> PhaseSpaceBoundaries:
    """
    Compute the four straggling-regime boundary curves.

    Parameters
    ----------
    particle : Particle, optional
        Defaults to the active PARTICLE from common.run_context.
    material : Material, optional
        Defaults to the active TARGET_MATERIAL from common.run_context.
    n_energy : int
        Number of log-spaced energy points (default 150 for the API,
        use 500 for offline scripts where smoother curves are desired).

    Returns
    -------
    PhaseSpaceBoundaries
        energy_eV : np.ndarray, shape (n_energy,)
        boundaries : dict mapping key → L array (metres), NaN where not found
        regions : list of dicts with id, label, color, alpha
    """
    if particle is None:
        particle = PARTICLE
    if material is None:
        material = TARGET_MATERIAL

    energy_eV = np.logspace(np.log10(_E_MIN_eV), np.log10(_E_MAX_eV), n_energy)
    params = ContinuousLossParams(particle, material)

    very_thin   = np.full(n_energy, float("nan"))
    thick_limit = np.full(n_energy, float("nan"))
    exc_freq    = np.full(n_energy, float("nan"))
    ion_gauss   = np.full(n_energy, float("nan"))

    for i, e_eV in enumerate(energy_eV):
        # 1. very_thin: mean_loss crosses min_energy_loss (10 eV)
        very_thin[i] = _find_first_crossing(
            params, e_eV,
            lambda p: p.average_loss - p.min_energy_loss,
        )

        # 2. thick_limit: sn crosses 2.0 (Bohr Gaussian thick-target condition)
        if _thick_limit_possible(e_eV, particle, material):
            thick_limit[i] = _find_first_crossing(
                params, e_eV,
                lambda p: getattr(p, "sn", -1.0) - 2.0,
            )

        # For ion_gauss and exc_freq, restrict the upper search bound to just
        # below thick_limit (if it exists) to avoid the sign-reversal that
        # occurs when set_glandz_params is not reached (ThickLimit early return).
        l_upper = (thick_limit[i] * 0.999
                   if not np.isnan(thick_limit[i])
                   else _L_MAX_M)

        # 3. ion_gauss: a3 crosses nmaxCont (= 8)
        ion_gauss[i] = _find_first_crossing(
            params, e_eV,
            lambda p: getattr(p, "a3", 0.0) - p.nmaxCont,
            l_max=l_upper,
        )

        # 4. exc_freq: a1 crosses nmaxCont (= 8)
        l_lower_exc = (ion_gauss[i] * 1.001
                       if not np.isnan(ion_gauss[i])
                       else _L_MIN_M)
        exc_freq[i] = _find_first_crossing(
            params, e_eV,
            lambda p: getattr(p, "a1", 0.0) - p.nmaxCont,
            l_min=l_lower_exc,
            l_max=l_upper,
        )

    return PhaseSpaceBoundaries(
        energy_eV=energy_eV,
        boundaries={
            "very_thin":   very_thin,
            "thick_limit": thick_limit,
            "exc_freq":    exc_freq,
            "ion_gauss":   ion_gauss,
        },
        regions=REGIONS,
    )
