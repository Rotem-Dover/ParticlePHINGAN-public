"""
Runtime loader for the locally computed G4 initialisation tables.

`InitTableSet` is the single source of physics tables for the simulation
runtime (G4hIonisation, the physics-MC samplers) and the offline analytics.
It loads the four npz files produced by `physics.g4_init_tables.build_all`
for the active (particle, material) and encodes the G4 spline-target
subtlety exactly once:

  * ionisation: G4 stores and splines sigma(E) (G4EmModelManager::
    FillLambdaVector), with the sigma=0 node at the delta-ray threshold as
    a spline knot; lambda = 1/sigma is recovered at evaluation.

All node tensors are float64 and stay float64 — callers cast results to
their state dtype at the boundary.
"""
from __future__ import annotations

import numpy as np
import torch

from common.paths import (
    PATH_INIT_DEDX,
    PATH_INIT_LAMBDA_ION,
    PATH_INIT_RANGE,
    PATH_INIT_SCALED_KIN,
)
from physics.g4_init_tables.interpolator import G4PhysicsVector

_BUILD_HINT = (
    "Init table file missing: {path}\n"
    "Run `python -m physics.g4_init_tables.build_all` (from the repo root, "
    "with PYTHONPATH=src, or from src/ with PYTHONPATH=.) to generate it. "
    "It builds tables for the beam selected by common.run_context.ACTIVE "
    "(the PHINGAN_BEAM environment variable); the builder itself takes no "
    "--beam argument."
)

# G4EmParameters::LambdaFactor default (G4VEnergyLossProcess `lambdaFactor`).
# Used by the fEmOnePeak biased-denominator rule in sigma_ion_biased().
G4_LAMBDA_FACTOR: float = 0.8


def _load_npz(path, *keys: str):
    if not path.exists():
        raise FileNotFoundError(_BUILD_HINT.format(path=path))
    with np.load(path) as d:
        missing = [k for k in keys if k not in d.files]
        if missing:
            raise KeyError(
                f"{path} lacks keys {missing} — rebuild the init tables "
                f"with `python -m physics.g4_init_tables.build_all`."
            )
        return tuple(d[k].copy() for k in keys)


def _pv(x: np.ndarray, y: np.ndarray, log_grid: bool) -> G4PhysicsVector:
    return G4PhysicsVector(torch.from_numpy(x), torch.from_numpy(y),
                           log_grid=log_grid, use_spline=True)


class InitTableSet:
    """The four runtime physics vectors (units: eV, cm; float64 nodes)."""

    def __init__(self, dedx: G4PhysicsVector, rng: G4PhysicsVector,
                 inverse_range: G4PhysicsVector, sigma_ion: G4PhysicsVector):
        self.dedx = dedx                        # E [eV] -> dE/dx [eV/cm]
        self.range = rng                        # E [eV] -> CSDA range [cm]
        self.inverse_range = inverse_range      # range [cm] -> E [eV] (free vector)
        self.sigma_ion = sigma_ion              # E [eV] -> Sigma [1/cm]
        # Energy of the sigma_ion maximum (G4's theEnergyOfCrossSectionMax):
        # the biased-denominator rule pivots here (fEmOnePeak branch of
        # G4VEnergyLossProcess::ComputeLambdaForScaledEnergy). G4 scans its
        # spline for the true continuous maximum, which can sit between
        # nodes; this takes the arg-max grid node instead. For this table's
        # 100 MeV proton beam, the recorded epeak node sits well below the
        # beam energy, and every track slows down through that energy before
        # its kinetic energy reaches the kill threshold, so the node-resolution
        # approximation is exercised on every track's tail. Its numerical impact is
        # second-order: sigma(E) is flat at its own maximum by definition
        # (dSigma/dE = 0 there), so missing the true peak by up to half a
        # node spacing barely moves the recorded epeak value, and
        # `sigma_ion_biased` only consumes epeak as a clamp/threshold
        # argument (`torch.clamp(e64 * G4_LAMBDA_FACTOR, min=epeak)`), not
        # as sigma(E) itself — so an error in epeak's location does not
        # translate one-for-one into an error in the biased cross section.
        self.sigma_ion_epeak = float(
            sigma_ion.bin[int(torch.argmax(sigma_ion.data))]
        )
        self._shared = False

    @classmethod
    def load(cls, device: str = "cpu") -> "InitTableSet":
        e, d = _load_npz(PATH_INIT_DEDX, "energy_eV", "dedx_eV_per_cm")
        dedx = _pv(e, d, log_grid=True)
        e, r = _load_npz(PATH_INIT_RANGE, "energy_eV", "range_cm")
        rng = _pv(e, r, log_grid=True)
        # scaled_kin_energy: x-axis is range (NOT log-uniform) -> free vector.
        r, e = _load_npz(PATH_INIT_SCALED_KIN, "range_cm", "energy_eV")
        inv = _pv(r, e, log_grid=False)
        e, s = _load_npz(PATH_INIT_LAMBDA_ION, "energy_eV", "sigma_per_cm")
        sig = _pv(e, s, log_grid=True)
        return cls(dedx=dedx, rng=rng, inverse_range=inv,
                   sigma_ion=sig).to(device)

    def sigma_ion_biased(self, e: torch.Tensor) -> torch.Tensor:
        """G4's cached preStepLambda for the ionisation discrete process.

        Port of the fEmOnePeak branch of G4VEnergyLossProcess::
        ComputeLambdaForScaledEnergy (v11.2.0): fresh sigma(E) at or below
        the cross-section peak; sigma(max(epeak, lambdaFactor*E)) above it.
        mfpKinEnergy is reset in PostStepDoIt after every discrete
        interaction, so per-step this stateless form reproduces the
        preStepLambda logged in `common.paths.PATH_G4_DUMP_SEC_GEN_FRAC`.
        """
        e64 = e.to(torch.float64)
        epeak = self.sigma_ion_epeak
        e_den = torch.where(
            e64 > epeak,
            torch.clamp(e64 * G4_LAMBDA_FACTOR, min=epeak),
            e64,
        )
        return self.sigma_ion(e_den)

    def to(self, device=None, dtype=None) -> "InitTableSet":
        if self._shared and device not in (None, "cpu"):
            raise RuntimeError(
                "This InitTableSet is the shared CPU instance from "
                "get_init_table_set(); moving it would corrupt every other "
                "consumer. Own a fresh InitTableSet.load(device) instead."
            )
        # dtype intentionally accepted-and-ignored for downcasts: nodes stay
        # float64 (the spline's parity with G4 lives in fp64).
        for pv in (self.dedx, self.range, self.inverse_range,
                   self.sigma_ion):
            pv.to(device=device)
        return self


# Shared read-only CPU instance for offline/analytics consumers that are
# constructed in loops (Geant4Tables, cross-section table).
# Runtime processes that move tables across devices must own a fresh
# InitTableSet.load(device) instead of mutating this one.
_SHARED: dict[str, InitTableSet] = {}


def get_init_table_set() -> InitTableSet:
    ts = _SHARED.get("cpu")
    if ts is None:
        ts = InitTableSet.load("cpu")
        ts._shared = True
        _SHARED["cpu"] = ts
    return ts
