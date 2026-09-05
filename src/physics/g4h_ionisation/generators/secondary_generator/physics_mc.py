"""
Physics-based Monte Carlo secondary-loss generator, built on the vectorized
delta-ray rejection kernels in
`physics.g4h_ionisation.explicit_physics.discrete._kernels`. Drop-in
replacement for `SecondaryGenerator`.

Mirrors the design of `PhysicsLengthGenerator` and `PhysicsContinuousGenerator`:
operates in physical units, bypasses the `@scaled_forward` decorator, registers
a dummy parameter so `nn.Module` device tracking works.

Note: this generator unconditionally samples e_sec for every event. Whether a
secondary is actually emitted on a given step is decided in
`G4hIonisation.post_step_do_it` via the cross-section ratio
`sec_gen_prob(pre_E, post_E)`; events for which no secondary is emitted have
their e_sec zeroed (and theta is reconstructed from e_sec analytically) by
that wrapper. This matches how the trained `SecondaryGenerator` is consumed.
"""
import torch
from torch import nn
from typing import Optional

from common.enums import SecondaryFeats

from common.run_context import PARTICLE, TARGET_MATERIAL as material
from physics.definitions.physical_constants import electron_mass_eV

from physics.g4h_ionisation.explicit_physics.discrete._kernels import _sample_discrete_jit

from physics.interfaces.generator_interface import GeneratorInterface


_PRIMARY_MASS_EV: float = PARTICLE.mass_eV
_ELECTRON_MASS_EV: float = electron_mass_eV
_MATERIAL_TC: float = material.Tc
_MAX_REJECTION_ITER: int = 50
# Selects the Møller (e-e-) branch of the discrete sampler: T_max = T/2 and
# the Møller spectrum instead of the hadron T_max + Rutherford tail.
_IS_ELECTRON: bool = PARTICLE.name == "electron"


class PhysicsSecondaryGenerator(GeneratorInterface):
    """
    Secondary-loss generator based on the discrete delta-ray rejection
    sampler: hadron-on-electron Rutherford-tail kinematics for heavy
    primaries, the Møller (e-e-) spectrum with T_max = T/2 for electrons
    (selected via the active `common.run_context.PARTICLE`).

    Input contract:  labels of shape [B] OR [B, 1] = post-continuous KE in eV
                     (matches the existing trained generator)
    Output contract: shape [B, 1] = [e_sec_eV] indexed via
                     `SecondaryFeats.E_SEC_IDX`
    """

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.freeze()

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        kinetic_energy = labels.flatten()
        dtype = kinetic_energy.dtype

        _, e_sec = _sample_discrete_jit(
            kinetic_energy,
            _PRIMARY_MASS_EV, _ELECTRON_MASS_EV, _MATERIAL_TC,
            _MAX_REJECTION_ITER, _IS_ELECTRON,
        )

        out = torch.empty(kinetic_energy.shape[0], 1, device=kinetic_energy.device, dtype=dtype)
        # int(): torch 2.5 FX codegen cannot serialize IntEnum tensor indices
        # (this forward is inlined into the compiled post_step_do_it phase).
        out[:, int(SecondaryFeats.E_SEC_IDX)] = e_sec.to(dtype)
        return out

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
        return self
