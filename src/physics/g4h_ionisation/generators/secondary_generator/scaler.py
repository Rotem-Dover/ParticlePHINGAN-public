"""
Scalers for the secondary energy-loss generator stage. `SecondaryXScaler`
normalizes kinetic energy input via log10 + min-max. `SecondaryYScaler` uses
physics-derived t_max bounds for a log10 min-max normalization of the
secondary energy.

Every arithmetic expression, dtype and bound below is part of the trained
checkpoint's contract: a checkpoint's weights were fit against exactly this
scaling, so changing it invalidates the checkpoint.

The output column layout is `SecondaryFeats` (E_SEC_IDX = 0, one column) --
the checkpoint's output layout, not a convention.
"""
import torch
import logging

from pathlib import Path

from typing_extensions import Self

from common.enums import SecondaryFeats
import common.paths as paths
import common.units as U
from physics.interfaces.scaler_interface import ScalerInterface

import physics.definitions.physical_constants as C
from common.run_context import PARTICLE as proton, TARGET_MATERIAL as aluminum

log = logging.getLogger(__name__)


class SecondaryXScaler(ScalerInterface):
    """
    Input scaler for SecondaryGenerator.
    Input: kinetic_energy (1 feature)
    Applies log10 + min-max normalization.
    """
    def __init__(self):
        self.min_post_log = None
        self.max_post_log = None

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "SecondaryXScaler":
        """Rebuild from the portable tensor state a training run's
        datamodule wrote.

        `state_dir=None` defaults to `paths.SCALERS_DIR`. A caller passes
        `state_dir=` explicitly to read a different material's or a
        different arm's state instead.
        """
        state_dir = paths.SCALERS_DIR if state_dir is None else Path(state_dir)
        state = torch.load(state_dir / f"{cls.__name__}.state.pt",
                           map_location="cpu", weights_only=True)
        obj = cls()
        for k, v in state.items():
            setattr(obj, k, v)
        return obj

    def scale(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        tensor = torch.log10(tensor)
        tensor = (tensor - self.min_post_log) / (self.max_post_log - self.min_post_log)
        return tensor

    def rescale(self, tensor: torch.Tensor, differentiable_ops: bool = True) -> torch.Tensor:
        tensor = tensor * (self.max_post_log - self.min_post_log) + self.min_post_log
        tensor = 10 ** tensor

        return tensor

    def to(self, device: str) -> Self:
        self.min_post_log = self.min_post_log.to(device)
        self.max_post_log = self.max_post_log.to(device)

        return self

    @property
    def device(self) -> torch.device:
        return self.min_post_log.device


class SecondaryYScaler(ScalerInterface):
    """
    Output scaler for SecondaryGenerator.
    Output: [E_SEC] (1 feature), min-max in log10 using t_max-dependent bounds.

    **Carries no fitted state.** `_log10_Tc`, `_Tc` and the
    `min_max_e_secondary` constants are physics, defined at class level, so
    there is nothing to fit from data and two instances are interchangeable.
    Kept as a class (rather than folded into free functions) because
    `@scaled_forward` and `ScalerInterface` are the generator-side contract.
    """

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "SecondaryYScaler":
        """Rebuild from the portable tensor state a training run's
        datamodule wrote.

        `state_dir=None` defaults to `paths.SCALERS_DIR`. A caller passes
        `state_dir=` explicitly to read a different material's or a
        different arm's state instead.
        """
        state_dir = paths.SCALERS_DIR if state_dir is None else Path(state_dir)
        state = torch.load(state_dir / f"{cls.__name__}.state.pt",
                           map_location="cpu", weights_only=True)
        obj = cls()
        for k, v in state.items():
            setattr(obj, k, v)
        return obj

    def scale(self, tensor: torch.Tensor, phase_space: torch.Tensor = None) -> torch.Tensor:
        e_secondary = self.scale_e_secondary(tensor[:, int(SecondaryFeats.E_SEC_IDX)], phase_space)

        return torch.stack([e_secondary], dim=1)

    def scale_e_secondary(self, e_secondary: torch.Tensor, phase_space: torch.Tensor) -> torch.Tensor:
        kinetic_energy = phase_space.squeeze()

        _, max_ = self.min_max_e_secondary(kinetic_energy)
        log10_max = torch.log10(max_)
        log10_min = self._log10_Tc
        e_secondary = torch.log10(e_secondary)
        e_secondary = (e_secondary - log10_min) / (log10_max - log10_min)

        return e_secondary

    def rescale(self, tensor: torch.Tensor, phase_space: torch.Tensor = None) -> torch.Tensor:
        e_secondary = self.rescale_e_secondary(tensor[:, int(SecondaryFeats.E_SEC_IDX)], phase_space)

        return torch.stack([e_secondary], dim=1)

    def rescale_e_secondary(self, e_secondary: torch.Tensor, phase_space: torch.Tensor) -> torch.Tensor:
        kinetic_energy = phase_space.squeeze()

        _, max_ = self.min_max_e_secondary(kinetic_energy)
        log10_max = torch.log10(max_)
        log10_min = self._log10_Tc
        is_atom = e_secondary == 0
        e_secondary = e_secondary * (log10_max - log10_min) + log10_min
        e_secondary = 10 ** e_secondary
        # The z == 0 Dirac atom is routed AROUND the exponential. At z == 0 the
        # formula above collapses to `10 ** log10(Tc)` -- one repeated constant
        # on every atom lane, independent of KE. Producing it with an
        # exponential makes its exact bits depend on which kernel computes the
        # power (eager vs. a compiled one can round the last bit differently
        # at the same input), which would displace the whole atom depending on
        # which arm evaluates it. Routing it through the shared pinned
        # constant `_Tc` instead means every arm agrees bitwise, and `_Tc` is
        # float32(Tc) itself rather than the round-trip value, so it also sits
        # closer to the documented G4 production threshold than the
        # exponential's own result would.
        e_secondary = torch.where(is_atom, self._Tc, e_secondary)

        return e_secondary

    def to(self, device: str) -> Self:
        self._log10_Tc = self._log10_Tc.to(device)
        self._Tc = self._Tc.to(device)

        return self

    @property
    def device(self):
        # `_log10_Tc` is this scaler's only tensor, so there is nothing else
        # to compare it against for a device mismatch.
        return self._log10_Tc.device

    # Pre-compute constants used in min_max_e_secondary
    _inv_proton_mass = 1.0 / proton.mass_eV
    _2_electron_mass = 2.0 * C.electron_mass_eV
    _electron_mass_ratio_sq = proton.electron_mass_ratio ** 2
    _2_electron_mass_ratio = 2.0 * proton.electron_mass_ratio
    _log10_Tc = torch.log10(torch.tensor(aluminum.Tc))
    # The z == 0 atom's value, pinned as a shared constant rather than
    # recomputed by `10 ** _log10_Tc` -- see `rescale_e_secondary`. Taken
    # from `physics.definitions` via `common.run_context`, so it is float32
    # of the active material's documented G4 production threshold Tc
    # directly, not the exponential round-trip of its log10. Class-level
    # like `_log10_Tc`, and moved by `to()` the same way.
    _Tc = torch.tensor(aluminum.Tc, dtype=torch.float32)

    @staticmethod
    def min_max_e_secondary(kin_energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = kin_energy * SecondaryYScaler._inv_proton_mass + 1
        g_sq = g * g
        b_sq = 1 - 1 / g_sq
        t_max = (SecondaryYScaler._2_electron_mass * b_sq * g_sq) / (
            1 + SecondaryYScaler._2_electron_mass_ratio * g + SecondaryYScaler._electron_mass_ratio_sq
        )
        min_ = aluminum.Tc
        max_ = t_max
        return min_, max_


class NoPhysSecondaryYScaler(ScalerInterface):
    """Output scaler for the no-physics ablation's secondary stage, matching
    the single-column `SecondaryFeats` layout: eV to MeV, log10, then
    min-max on the one E_SEC column."""

    def __init__(self):
        super().__init__()
        self.min_post_log = None
        self.max_post_log = None

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "NoPhysSecondaryYScaler":
        state_dir = paths.SCALERS_DIR if state_dir is None else Path(state_dir)
        state = torch.load(state_dir / f"{cls.__name__}.state.pt",
                           map_location="cpu", weights_only=True)
        self = cls()
        self.min_post_log = state["min_post_log"]
        self.max_post_log = state["max_post_log"]
        return self

    def state_dict_tensors(self):
        return {"min_post_log": self.min_post_log,
                "max_post_log": self.max_post_log}

    def scale(self, tensor, **kwargs):
        tensor = tensor * U.eV2MeV
        tensor = torch.log10(tensor)
        return (tensor - self.min_post_log) / (self.max_post_log - self.min_post_log)

    def rescale(self, tensor, differentiable_ops=True, **kwargs):
        tensor = tensor * (self.max_post_log - self.min_post_log) + self.min_post_log
        if differentiable_ops:
            tensor = 10 ** tensor
        else:
            torch.pow(10, tensor, out=tensor)
        return tensor * U.MeV2eV

    def to(self, device):
        self.min_post_log = self.min_post_log.to(device)
        self.max_post_log = self.max_post_log.to(device)
        return self

    @property
    def device(self):
        return self.min_post_log.device
