"""
Physics-based Monte Carlo continuous-loss generator that replicates the
G4UniversalFluctuation algorithm vectorized in PyTorch. Drop-in replacement
for the trained `ContinuousGenerator` neural nets.

Mirrors the design of `physics.g4h_ionisation.generators.length_generator.physics_mc.PhysicsLengthGenerator`:
operates in physical units (eV, m), bypasses the `@scaled_forward` decorator,
and registers a dummy parameter so `nn.Module` device tracking works.
"""
import math
import torch
from torch import nn
from typing import Optional

import common.units as U

from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.physical_constants import electron_mass_eV, electron_radius_cm

from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import GeantConstants
from physics.g4h_ionisation.explicit_physics.continuous._kernels import _sample_continuous_loss_jit
from physics.g4h_ionisation.g4h_ionisation import _average_loss_fp64
from physics.g4_init_tables.runtime import InitTableSet

from physics.interfaces.generator_interface import GeneratorInterface


_G4CONSTS = GeantConstants()

# Module-level Python-float constants (closed over by JIT kernels)
_PRIMARY_MASS_EV: float = proton.mass_eV
# Møller-vs-heavy t_max branch inside the continuous kernel (mirrors
# secondary_generator/physics_mc.py).
_IS_ELECTRON: bool = proton.name == "electron"
_ELECTRON_MASS_EV: float = electron_mass_eV
_ELECTRON_RADIUS_CM: float = electron_radius_cm
_ELECTRON_DENSITY: float = float(material.electron_density)
_MATERIAL_TC: float = material.Tc
_MATERIAL_I: float = material.I
_E0: float = _G4CONSTS.e0
_LOG_W1: float = math.log(_MATERIAL_TC / _E0)
_IONIZATION_RATE: float = _G4CONSTS.ionization_rate
_FW: float = _G4CONSTS.FW
_A0: float = _G4CONSTS.A0
_NMAX_CONT: float = float(_G4CONSTS.nmaxCont)
_N_MIN_BOHR: float = float(_G4CONSTS.n_min_bohr)
_SCALING: float = min(1.0 + 500.0 / _MATERIAL_TC, 1.5)
_TWOPI_MC2_RCL2: float = 2.0 * math.pi * _ELECTRON_MASS_EV * _ELECTRON_RADIUS_CM ** 2

_M2CM: float = U.m2cm

_MAX_GAUSS_ITER: int = 25


class PhysicsContinuousGenerator(GeneratorInterface):
    """
    Continuous-energy-loss generator based on G4UniversalFluctuation::SampleFluctuations
    (vectorized PyTorch port). Drop-in for ContinuousGenerator.

    Input contract:  labels of shape [B, 2] = [kinetic_energy_eV, step_length_m]
    Output contract: e_cnt of shape [B] in eV
    """

    # The Glandz Poisson channels already produce exact zeros at the G4 rate;
    # G4hIonisation.along_step_do_it must not re-inject the zero-loss atom.
    samples_zero_loss: bool = True

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)

        self.tables = InitTableSet.load()
        self.freeze()

    # ------------------------------------------------------------------
    # GeneratorInterface overrides
    # ------------------------------------------------------------------
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        assert labels.dim() == 2 and labels.shape[1] == 2, (
            f"Expected [B, 2] input (KE_eV, L_m); got {tuple(labels.shape)}"
        )

        kinetic_energy = labels[:, 0].contiguous()
        step_length_m = labels[:, 1].contiguous()
        dtype = kinetic_energy.dtype

        average_loss = _average_loss_fp64(
            self.tables, kinetic_energy, step_length_m,
        ).to(kinetic_energy.dtype)

        step_length_cm = step_length_m * _M2CM

        e_cnt = _sample_continuous_loss_jit(
            kinetic_energy, average_loss, step_length_cm,
            _PRIMARY_MASS_EV, _ELECTRON_MASS_EV, _ELECTRON_RADIUS_CM,
            _ELECTRON_DENSITY, _MATERIAL_TC, _MATERIAL_I,
            _E0, _LOG_W1, _IONIZATION_RATE, _FW, _A0,
            _NMAX_CONT, _N_MIN_BOHR, _SCALING,
            _TWOPI_MC2_RCL2, _MAX_GAUSS_ITER, _IS_ELECTRON,
        )

        # Handle the deterministic and kill branches inline so this generator
        # behaves correctly even outside G4hIonisation's wrapping logic
        # (G4hIonisation.along_step_do_it re-applies these as a safety net).
        e_cnt = torch.where(
            average_loss < _G4CONSTS.min_energy_loss, average_loss, e_cnt,
        )
        e_cnt = torch.where(
            average_loss > kinetic_energy, kinetic_energy, e_cnt,
        )
        return e_cnt.to(dtype)

    def predict(
        self, labels: torch.Tensor, repeat_interleave: Optional[int] = None,
    ) -> torch.Tensor:
        """Bypass scaling — the physics MC operates in physical units."""
        if repeat_interleave is not None:
            labels = labels.repeat_interleave(repeat_interleave, dim=0)
        return self.forward(labels)

    def embedding(self) -> nn.Sequential:
        return nn.Sequential()

    def to(self, device, dtype: Optional[torch.dtype] = None):
        super().to(device)
        # dtype ignored for tables — nodes stay float64.
        self.tables.to(device)
        return self
