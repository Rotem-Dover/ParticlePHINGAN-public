"""
Physics-based Monte Carlo step-length generator that replicates GEANT4's G4hIonisation
deterministic and discrete stepping limits. Samples step lengths from the energy-loss
limit and macroscopic cross-section without neural networks, serving as a baseline
or drop-in replacement for the learned LengthGenerator.
"""
import torch
from torch import nn
from typing import Optional

import common.units as U

from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4_init_tables.runtime import InitTableSet

from physics.interfaces.generator_interface import GeneratorInterface


class PhysicsLengthGenerator(GeneratorInterface):
    """
    Step-length generator based on the G4hIonisation deterministic and discrete
    limits (explicit MC), used as a drop-in replacement for the neural-network
    LengthGenerator.
    """

    def __init__(self):
        super().__init__()
        # Register a dummy parameter so nn.Module.device / .to() work correctly
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        self.tables: Optional[InitTableSet] = None

        self._load_tables()
        self.freeze()

    def _load_tables(self) -> None:
        # Built G4 init tables (physics.g4_init_tables): kinetic energy is
        # eV and the range table's ordinate is centimetres, so no unit
        # conversion is needed after loading. Loaded on CPU; moved
        # independently via `to()`.
        #
        # `forward()` evaluates these two G4PhysicsVectors directly
        # (`InitTableSet.range` / `InitTableSet.sigma_ion_biased`), through
        # their branch-free scripted spline kernel (g4pv_eval,
        # use_spline=True) rather than resampling them onto a coarser table
        # and linearly interpolating that: linear interpolation at this
        # table's resolution carries a measurable one-signed bias against
        # G4PhysicsVector::Value (the model GEANT4 itself evaluates at
        # runtime), which evaluating the spline directly avoids.
        self.tables = InitTableSet.load("cpu")

    # ------------------------------------------------------------------
    # GeneratorInterface overrides
    # ------------------------------------------------------------------

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Sample step lengths from the physics MC algorithm.
        `labels` is kinetic energy in eV (physical units, unscaled).
        Returns step lengths in metres.
        """
        kinetic_energy = labels.flatten()
        dtype = kinetic_energy.dtype

        # 1. Deterministic energy-loss limit L_eloss (cm)
        # `InitTableSet.range.__call__` runs `g4pv_eval`
        # (physics/g4_init_tables/interpolator.py), a `@torch.jit.script`
        # kernel that is branch-free (no `.any()` / data-dependent control
        # flow -- out-of-range queries clamp to the edge values via
        # `torch.where`, matching G4PhysicsVector::Value's own semantics) and
        # whose boundary constants (`log_emin`, `inv_dbin_log`) are
        # precomputed Python floats at table-load time, not tensors read
        # inside the traced region. It therefore runs inside the compiled
        # `compute_step_limit` graph with no data-dependent breaks -- pinned
        # by tests/physics/test_length_generator_branchless.py -- while
        # evaluating the cubic spline G4PhysicsVector::Value uses.
        # The built table's ordinate is already centimetres
        # (InitTableSet.range), so no unit conversion is applied here.
        R_cm = self.tables.range(kinetic_energy).to(dtype)
        f = 0.01  # final_range = 100 um = 0.01 cm
        dRoverRange = 0.1

        L_eloss = torch.where(
            R_cm <= f,
            R_cm,
            R_cm * dRoverRange + f * (1 - dRoverRange) * (2 - f / R_cm),
        )

        # 2. Discrete macroscopic cross-section Sigma (1/cm)
        mass_ratio = electron_mass_eV / proton.mass_eV
        tau = kinetic_energy / proton.mass_eV
        gamma = tau + 1.0
        gamma_sq = gamma ** 2
        beta_sq = 1.0 - (1.0 / gamma_sq)
        t_max = (2.0 * electron_mass_eV * beta_sq * gamma_sq) / (
            1.0 + 2.0 * gamma * mass_ratio + mass_ratio ** 2
        )

        # G4's actual discrete-step sampler sums nILL/sigma with sigma = the
        # PROCESS'S CACHED preStepLambda (G4VEnergyLossProcess::
        # ComputeLambdaForScaledEnergy's fEmOnePeak branch), not the raw
        # ionisation cross-section.
        # `sigma_ion_biased` evaluates `self.sigma_ion(e_den)` internally, so
        # calling it directly on `kinetic_energy` here gives the spline
        # evaluation for free -- no separate resampled table and no second
        # interpolation path. forward() divides nILL directly by this value
        # (sigma, units 1/cm): it is already what forward() needs, no
        # sigma<->lambda inversion.
        sigma = self.tables.sigma_ion_biased(kinetic_energy).to(dtype)
        sigma = torch.where(t_max <= material.Tc, torch.zeros_like(sigma), sigma)
        sigma = torch.clamp(sigma, min=0.0)

        # 3. Sample
        nILL = -torch.log(torch.rand_like(kinetic_energy))
        L_discrete = torch.where(
            sigma > 0,
            nILL / sigma,
            torch.full_like(kinetic_energy, float('inf')),
        )

        step_length_cm = torch.minimum(L_discrete, L_eloss)
        step_length_m = step_length_cm * U.cm2m
        return step_length_m.to(dtype)

    def predict(
        self, labels: torch.Tensor, repeat_interleave: Optional[int] = None
    ) -> torch.Tensor:
        """Bypass scaling — the physics MC operates in physical units directly."""
        if repeat_interleave is not None:
            labels = labels.repeat_interleave(repeat_interleave, dim=0)
        return self.forward(labels)

    def embedding(self) -> nn.Sequential:
        return nn.Sequential()

    def to(self, device):
        super().to(device)
        if self.tables is not None:
            self.tables.to(device)
        return self
