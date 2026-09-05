"""
Scalers for the continuous energy-loss generator stage. `ContinuousXScaler`
normalizes the two-feature input (kinetic energy, step length) via log10 +
min-max. `ContinuousYScaler` uses a physics-informed lookup table of
mean/std (from the straggling model) for z-score normalization of the
continuous energy loss.

Every arithmetic expression, dtype, rounding order and clamp below is part
of the trained checkpoint's contract: a checkpoint's weights were fit
against exactly this scaling, so changing it invalidates the checkpoint.

Each scaler has a no-argument constructor plus a `load()` classmethod that
reads the portable tensor state a training run's datamodule writes, rather
than an unpicklable trained instance.
"""
import pickle
import logging

from pathlib import Path

import torch
from typing_extensions import Self

import common.paths as paths
import common.units as U

from physics.interfaces.scaler_interface import ScalerInterface

log = logging.getLogger(__name__)


class ContinuousXScaler(ScalerInterface):
    """
    Input scaler for ContinuousGenerator.
    Input: [kinetic_energy, step_length] (2 features)
    Applies log10 + min-max normalization.
    """
    def __init__(self):
        self.min_post_log = None
        self.max_post_log = None

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "ContinuousXScaler":
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
        tensor = torch.log10(tensor)          # fresh tensor; caller's is untouched
        tensor[:, 1] = torch.abs(tensor[:, 1])

        tensor = (tensor - self.min_post_log) / (self.max_post_log - self.min_post_log)
        return tensor

    def rescale(self, tensor: torch.Tensor, differentiable_ops: bool = True) -> torch.Tensor:
        tensor = tensor * (self.max_post_log - self.min_post_log)
        tensor = tensor + self.min_post_log
        if differentiable_ops:
            mask = torch.ones_like(tensor)
            mask[:, 1] = -1
            tensor = tensor * mask
            tensor = 10 ** tensor
        else:
            tensor[:, 1] *= -1
            torch.pow(10, tensor, out=tensor)

        return tensor

    def to(self, device: str) -> Self:
        self.min_post_log = self.min_post_log.to(device)
        self.max_post_log = self.max_post_log.to(device)

        return self

    @property
    def device(self):
        return self.min_post_log.device


class ContinuousYScaler(ScalerInterface):
    """
    Output scaler for ContinuousGenerator.
    Output: E_CNT (1 feature)
    Uses z-score normalization with physics-informed mean/std lookup table.
    """
    def __init__(self):
        self.kin_table = None
        self.step_table = None
        self.lookup_table = None

        self._kin_delta = None
        self._steps_delta = None
        self._kin_table_start = None
        self._step_table_start = None
        self._kin_max_idx = None
        self._step_max_idx = None

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "ContinuousYScaler":
        """Rebuild from the portable tensor state a training run's
        datamodule wrote, then recompute the integer lookup bounds
        `__init__` would otherwise derive directly (the saved state holds
        only the raw tensors, not these derived scalars).

        `state_dir=None` defaults to `paths.SCALERS_DIR`. The lookup table
        embedded in this scaler's state is built by
        `build_analytic_mean_std_table.py` from the explicit-physics
        straggling model and written into portable state by
        `scripts/build_continuous_scaler_state.py`.
        """
        state_dir = paths.SCALERS_DIR if state_dir is None else Path(state_dir)
        state = torch.load(state_dir / f"{cls.__name__}.state.pt",
                           map_location="cpu", weights_only=True)
        obj = cls()
        for k, v in state.items():
            setattr(obj, k, v)

        obj._kin_delta = (obj.kin_table[-1] - obj.kin_table[0]) / (obj.kin_table.shape[0] - 1)
        obj._steps_delta = (obj.step_table[-1] - obj.step_table[0]) / (obj.step_table.shape[0] - 1)
        obj._kin_table_start = obj.kin_table[0]
        obj._step_table_start = obj.step_table[0]
        obj._kin_max_idx = obj.kin_table.shape[0] - 1
        obj._step_max_idx = obj.step_table.shape[0] - 1
        return obj

    _LOOKUP_NAN_MSG = (
        "NaN mu/std from the mean/std lookup: an (E, L) pair landed in the "
        "analytic table's deterministic region (or was clamped into it by "
        "_compute_lookup_indices, which clips both axes with no diagnostic). "
        "The analytic table stores NaN there rather than a fabricated "
        "standard deviation. In the FORWARD (training) direction there is "
        "no downstream mask to absorb that, so it would become a NaN "
        "target and a NaN gradient; filter deterministic rows out of the "
        "training set."
    )

    @staticmethod
    def _assert_lookup_finite(mu: torch.Tensor, std: torch.Tensor) -> None:
        """Refuse a forward-direction gather that landed in the NaN region.

        Guarded on `scale` only, not on `rescale`, because the two
        directions treat a NaN gather differently. The table's NaN region
        is interior and physically reachable -- roughly 29% of cells, a
        contiguous short-step prefix at every kinetic energy -- not merely
        an out-of-grid clamp artifact: the builder writes NaN there because
        `G4hIonisation` treats those cells deterministically, and
        `along_step_do_it` handles them the same way, except that it
        computes the network first and then overwrites those lanes with
        `average_loss` -- GEANT4's own deterministic Bethe-Bloch mean for
        that regime, not a placeholder -- via
        `torch.where(is_deterministic, average_loss, e_cnt)`. So on the
        inference path a NaN gather is expected and absorbed downstream,
        and asserting here would refuse every ordinary run.

        The escape hatch that actually matters is guarded at its real site,
        `G4hIonisation.along_step_do_it`, on the post-mask `e_cnt`.

        One `isnan().any()` on the already-gathered pair, not a scan of the
        whole table.

        Deliberately unconditional -- no `torch.compiler.is_compiling()`
        skip, unlike the sibling guard in `G4hIonisation.along_step_do_it`.
        That one is skipped while tracing because it sits in the per-step
        inference hot loop, where a data-dependent assert is a graph break
        and a host synchronisation. This one is on the training path, which
        runs eager, is not per-step-latency-bound, and has no downstream
        mask to absorb a NaN: a NaN here is a NaN target and a NaN
        gradient, i.e. a silently ruined run. If the training path is later
        compiled, the correct response is to filter deterministic rows out
        of the dataset (see `_LOOKUP_NAN_MSG`), not to drop this check.
        """
        assert not torch.isnan(mu).any(), ContinuousYScaler._LOOKUP_NAN_MSG
        assert not torch.isnan(std).any(), ContinuousYScaler._LOOKUP_NAN_MSG

    def scale(self, tensor: torch.Tensor, phase_space: torch.Tensor = None) -> torch.Tensor:
        return self.scale_e_continuous(tensor, phase_space)

    def _compute_lookup_indices(self, kinetic_energy: torch.Tensor, step_length: torch.Tensor):
        kin_indexes = torch.round((torch.log10(kinetic_energy) - self._kin_table_start) / self._kin_delta).to(int)
        kin_indexes = torch.clip(kin_indexes, 0, self._kin_max_idx)

        steps_indexes = torch.round((torch.log10(step_length) - self._step_table_start) / self._steps_delta).to(int)
        steps_indexes = torch.clip(steps_indexes, 0, self._step_max_idx)

        return kin_indexes, steps_indexes

    def lookup_is_nan(self, phase_space: torch.Tensor) -> torch.Tensor:
        """Per-lane: does this (E, L) gather the table's deterministic NaN?

        `G4hIonisation._along_step_core` ORs this into its own
        `is_deterministic` predicate rather than re-deriving the
        deterministic regime independently: the table's own NaN mask is the
        authority on which cells it could not normalize, so asking it
        directly makes the two agree by construction.

        Both channels carry the same NaN pattern (set jointly per cell by
        the builder), so one channel is read, not two.

        Pointwise + gather only: no host synchronisation, no data-dependent
        branch, and it traces into the caller's single graph.
        """
        kin_indexes, steps_indexes = self._compute_lookup_indices(
            phase_space[:, 0], phase_space[:, 1])
        return torch.isnan(self.lookup_table[kin_indexes, steps_indexes, 0])

    def scale_e_continuous(self, e_continuous: torch.Tensor, phase_space: torch.Tensor) -> torch.Tensor:
        kin_indexes, steps_indexes = self._compute_lookup_indices(phase_space[:, 0], phase_space[:, 1])

        mu, std = self.lookup_table[kin_indexes, steps_indexes, 0], self.lookup_table[kin_indexes, steps_indexes, 1]
        self._assert_lookup_finite(mu, std)
        mu = mu.reshape_as(e_continuous)
        std = std.reshape_as(e_continuous)
        e_continuous = (torch.log10(e_continuous) - mu) / std

        return e_continuous

    def rescale(self, tensor: torch.Tensor, phase_space: torch.Tensor = None) -> torch.Tensor:
        return self.rescale_e_continuous(tensor, phase_space)

    def rescale_e_continuous(self, e_continuous: torch.Tensor, phase_space: torch.Tensor) -> torch.Tensor:
        kin_indexes, steps_indexes = self._compute_lookup_indices(phase_space[:, 0], phase_space[:, 1])
        mu, std = self.lookup_table[kin_indexes, steps_indexes, 0], self.lookup_table[kin_indexes, steps_indexes, 1]
        # Deliberately not guarded -- see `_assert_lookup_finite`. NaN here is
        # expected on deterministic-regime lanes and is replaced by
        # `average_loss` in `G4hIonisation.along_step_do_it`'s where-chain,
        # which guards the escape itself: it ORs `isnan(e_cnt).any()` into a
        # device-resident latch on every device (no per-step host read), and
        # `TrackSimulator.run()` reads that latch once per run via
        # `check_nan_guard()`.
        mu = mu.reshape_as(e_continuous)
        std = std.reshape_as(e_continuous)

        e_continuous = e_continuous * std + mu
        e_continuous = 10 ** e_continuous

        return e_continuous

    def to(self, device: str) -> Self:
        self.kin_table = self.kin_table.to(device)
        self.step_table = self.step_table.to(device)
        self.lookup_table = self.lookup_table.to(device)
        self._kin_table_start = self._kin_table_start.to(device) if isinstance(self._kin_table_start, torch.Tensor) else self._kin_table_start
        self._step_table_start = self._step_table_start.to(device) if isinstance(self._step_table_start, torch.Tensor) else self._step_table_start
        self._kin_delta = self._kin_delta.to(device) if isinstance(self._kin_delta, torch.Tensor) else self._kin_delta
        self._steps_delta = self._steps_delta.to(device) if isinstance(self._steps_delta, torch.Tensor) else self._steps_delta

        return self

    @property
    def device(self):
        devices = {
            self.kin_table.device,
            self.step_table.device,
            self.lookup_table.device
        }
        assert len(devices) == 1, f"Components are on different devices: {devices}"
        return devices.pop()

    @staticmethod
    def continuous_mean_std_lookup_table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the analytic mean/std lookup. Never build on miss.

        The lookup is expensive to build (see
        `build_analytic_mean_std_table.py`), so an ordinary import must not
        trigger a build; it raises instead, with the command to run.
        """
        analytic_path = paths.PATH_ANALYTIC_MEAN_STD_TABLE
        if not analytic_path.exists():
            raise FileNotFoundError(
                f"Analytic mean/std lookup table not found at {analytic_path}. "
                f"Build it first with: PYTHONPATH=src python -m "
                f"physics.g4h_ionisation.generators.continuous_generator.build_analytic_mean_std_table"
            )
        log.info(f"Loading analytic lookup table from {analytic_path}")
        with open(analytic_path, 'rb') as f:
            return pickle.load(f)


class NoPhysContinuousYScaler(ScalerInterface):
    """Output scaler for the no-physics ablation's continuous stage: eV to
    MeV, an abs(log10) skew fold, then min-max -- no physics lookup table.
    No-argument constructor plus `load()`; the fit arithmetic lives in the
    training datamodule, which writes the same `.state.pt` this `load()`
    reads."""

    def __init__(self):
        super().__init__()
        self.min_post_log = None
        self.max_post_log = None

    @classmethod
    def load(cls, state_dir: "Path | None" = None) -> "NoPhysContinuousYScaler":
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
        tensor = torch.abs(torch.log10(tensor))
        return (tensor - self.min_post_log) / (self.max_post_log - self.min_post_log)

    def rescale(self, tensor, differentiable_ops=True, **kwargs):
        tensor = tensor * (self.max_post_log - self.min_post_log) + self.min_post_log
        tensor = -tensor          # invert the abs fold (values were negative-log)
        tensor = 10 ** tensor
        return tensor * U.MeV2eV

    def to(self, device):
        self.min_post_log = self.min_post_log.to(device)
        self.max_post_log = self.max_post_log.to(device)
        return self

    @property
    def device(self):
        return self.min_post_log.device
