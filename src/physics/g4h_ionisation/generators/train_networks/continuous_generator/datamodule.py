"""
PyTorch Lightning data module for the continuous generator's WGAN-GP training.

**Data source.** Reads `continuous_X.npy` / `continuous_Y.npy` as built by
`build_training_datasets.py` (via `data_handling.truth_extract.
extract_continuous_xy`) under `DATASETS_DIR`. There is no ROOT-extraction
entrypoint in this module (unlike `secondary_generator/datamodule.py`'s
`build_arrays_from_root`), because the build script already covers both
stages in one pass over the truth file.

**On-disk cache format is `.npy`**: `X`/`Y` are plain arrays with no Python
object structure to preserve, so `.npy` is a self-describing,
code-execution-free fit.

**`_filter_bethe_bloch_steps` filters out deterministic (non-stochastic)
Bethe-Bloch steps** by instantiating
`physics.g4h_ionisation.explicit_physics.continuous.
straggling_function.ContinuousStragglingModel` per row and checking
`StragglingType.DeterministicBetheBloch in straggling_model.straggling_type`
-- a Python loop over every training row, wrapped in `tqdm` because of the
cost. There is deliberately no on-disk caching of the post-filter arrays: a
cache keyed only by a file's existence (not a hash of the inputs) is a
correctness footgun, since a cached result from an earlier, different
dataset would be silently reused. A real training run against the full
truth-derived
`continuous_X.npy` pays this filter's full per-row cost on every `setup()`
call; if that becomes a bottleneck, the fix is a content-hashed cache.

**Y-scaler: `.load()`, not a fit.** The continuous stage's Y-scaler
(`physics.g4h_ionisation.generators.continuous_generator.scaler.
ContinuousYScaler`) has no dataset-dependent fit: its state is a
physics-derived kinetic-energy/step-length -> (mean, std) lookup table from
the straggling model, a function of the (grid, physics parameters) pair
only, never of the specific training dataset. This datamodule calls
`ContinuousYScaler.load()` (reading `SCALERS_DIR/ContinuousYScaler.state.pt`,
the analytic lookup -- see that classmethod). Rebuilding the lookup
table from scratch would cost a 5000x5000-point sweep
(`build_analytic_mean_std_table.py`), which is exactly the kind of expensive
physics precomputation this datamodule should not own. The X-scaler
(`ContinuousXScaler`, log10 + min-max over the two input columns) genuinely
is fit to the dataset, so its fit is implemented here (`_fit_x_scaler`), the
same way `secondary_generator/datamodule.py` fits the secondary X-scaler.

Writes only under `common.paths.DATASETS_DIR` / `SCALERS_DIR`.
"""

import logging
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

import common.units as U
from common.config import DEVICE, N_BATCHES, N_CONTINUOUS_FEATURES
from common.paths import DATASETS_DIR, SCALERS_DIR
from common.utils import PointInPhaseSpace
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import ContinuousStragglingModel, StragglingType
from physics.g4h_ionisation.generators.continuous_generator.scaler import NoPhysContinuousYScaler
from physics.interfaces.scaler_interface import ScalerInterface

log = logging.getLogger(__name__)


def fit_nophys_y_scaler(y: torch.Tensor, state_dir: Union[str, Path] = SCALERS_DIR) -> None:
    """Fit-and-write the NoPhys Y-scaler state (min/max in the class's own
    log space). Same write-what-load-reads contract as the inline fits."""
    t = y * U.eV2MeV
    t = torch.abs(torch.log10(t))                # continuous: abs fold
    state = {"min_post_log": t.min(), "max_post_log": t.max()}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(state_dir) / "NoPhysContinuousYScaler.state.pt")


class CustomDataLoader:
    """
    Custom data loader for the continuous generator.
    X: [kinetic_energy, step_length] (2 features)
    Y: E_CNT (N_CONTINUOUS_FEATURES features)
    """
    def __init__(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            batch_size: int = 100_000,
    ):
        dataset_size = len(X) - len(X) % batch_size
        self.X, self.Y = X, Y
        self.X = self.X[:dataset_size, :].reshape(-1, batch_size, 2).to(DEVICE)
        self.Y = self.Y[:dataset_size].reshape(-1, batch_size, N_CONTINUOUS_FEATURES).to(DEVICE)

    def __iter__(self):
        for X, Y in zip(self.X, self.Y):
            yield X, Y.squeeze(-1)

    def __len__(self):
        return self.Y.shape[0]


class DataModule(LightningDataModule):
    """
    Continuous-generator data module: loads the `([kinetic_energy,
    step_length], e_cnt)` training pair, fits/loads the X/Y scalers, and
    yields scaled batches.
    """
    def __init__(
            self,
            x_scaler_type:       Type[ScalerInterface],
            y_scaler_type:       Type[ScalerInterface],
            data_dir:            Union[str, Path] = DATASETS_DIR,
            scaler_dir:          Union[str, Path] = SCALERS_DIR,
            file_name_stem:      str              = "continuous",
            n_batches:           int              = N_BATCHES,
    ):
        super().__init__()
        self.x_scaler_type  = x_scaler_type
        self.y_scaler_type  = y_scaler_type
        self.data_dir       = Path(data_dir)
        self.scaler_dir     = Path(scaler_dir)
        self.file_name_stem = file_name_stem
        self.n_batches      = n_batches

        self.X_real: Optional[torch.Tensor] = None
        self.Y_real: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None

        self.scaler_x: Optional[ScalerInterface] = None
        self.scaler_y: Optional[ScalerInterface] = None

        self.has_been_setup = False

    # ------------------------------- paths -------------------------------- #
    @property
    def x_path(self) -> Path:
        return self.data_dir / f"{self.file_name_stem}_X.npy"

    @property
    def y_path(self) -> Path:
        return self.data_dir / f"{self.file_name_stem}_Y.npy"

    # -------------------------------- setup --------------------------------- #
    def setup(self, stage: Optional[str] = None) -> None:
        if self.has_been_setup:
            return None

        log.info("Setting up continuous DataModule...")
        if self.X_real is None or self.Y_real is None:
            self.X_real, self.Y_real = self._load_data()

        self._filter_samples()

        if self.X is None or self.Y is None:
            self.X, self.Y = self._prepare_data_for_training(self.X_real, self.Y_real)

        self.has_been_setup = True

    def train_dataloader(self) -> CustomDataLoader:
        return CustomDataLoader(self.X, self.Y, self.batch_size)

    @property
    def batch_size(self) -> int:
        return round(len(self.Y) / self.n_batches)

    # ------------------------------ internals -------------------------------- #
    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        for path in (self.x_path, self.y_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} doesn't exist -- run build_training_datasets.py "
                    f"first to extract it from a GEANT4 truth ROOT file")

        X = torch.from_numpy(np.load(self.x_path)).type(torch.float32)
        Y = torch.from_numpy(np.load(self.y_path)).type(torch.float32)
        assert X.shape[0] == Y.shape[0], "Number of steps should be equal both in X and Y"
        return X, Y

    def _filter_samples(self) -> None:
        # The first two filters this datamodule applies: drop
        # zero-continuous-loss steps and steps that lost all of the
        # particle's kinetic energy in one continuous step (both edge cases
        # the WGAN-GP generator is not trained to reproduce).
        zero_mask = self.Y_real != 0
        all_energy_lost_mask = self.Y_real != self.X_real[:, 0]
        mask = zero_mask & all_energy_lost_mask

        n_zero = int((~zero_mask).sum())
        n_all_lost = int((~all_energy_lost_mask & zero_mask).sum())
        log.info(f"Filtered out {n_zero} zero energy loss steps...")
        log.info(f"Filtered out {n_all_lost} all energy lost steps...")

        self.X_real = self.X_real[mask]
        self.Y_real = self.Y_real[mask]

        self._filter_bethe_bloch_steps()
        self._filter_nan_lookup_rows()

    def _filter_nan_lookup_rows(self) -> None:
        """Drop rows whose (E, L) gathers a NaN mu/sigma from the Y-scaler's
        lookup -- the rows that would abort training on
        `ContinuousYScaler._assert_lookup_finite`.

        `_filter_bethe_bloch_steps` above tests ONE condition,
        `StragglingType.DeterministicBetheBloch`. The analytic table writes
        NaN on a THREE-way OR (`build_analytic_mean_std_table.py`: the
        straggling type, an `average_loss > primary_energy` capped early
        return, and a NaN mu out of the CDF moments), so the Bethe-Bloch
        filter alone does not bound the NaN set. Empirically the residue is
        empty, but "empirically empty" is not a guarantee: an unfiltered row
        that gathers a NaN mu/sigma here becomes a RuntimeError partway into
        a training run.

        This filters on the LOOKUP ITSELF rather than re-deriving the builder's
        three conditions, so it is exhaustive with respect to the assert it
        protects BY CONSTRUCTION: the gather it performs is the same gather
        `scale_e_continuous` performs, through the same
        `_compute_lookup_indices` (clamps included). It is not a claim about
        the physics -- a row may be dropped here for a reason the straggling
        model would not call deterministic -- only that no surviving row can
        trip the guard.

        No-op for Y-scalers with no lookup table (the secondary-style scalers
        and the test fakes), guarded by `hasattr` for those -- EXCEPT
        `NoPhysContinuousYScaler`, which short-circuits by class, above the
        `hasattr` check, rather than through it: that scaler's state is
        fit-and-written by `_setup_scalers`' NoPhys branch, which has not run
        yet at this point in `setup()` (`_filter_samples` -> here ->
        `_prepare_data_for_training` -> `_setup_scalers`), so eagerly calling
        `self.y_scaler_type.load()` for it would raise `FileNotFoundError`
        before a single row of state exists -- loading it before it is fit
        is exactly the crash this guard exists to avoid, so the `hasattr` path
        can never be reached for this class. Keeping every row for the
        no-physics ablation is correct: that arm has no physics lookup table
        to filter against.
        """
        if self.y_scaler_type is NoPhysContinuousYScaler:
            return

        if self.scaler_y is None:
            self.scaler_y = self.y_scaler_type.load()

        scaler = self.scaler_y
        if getattr(scaler, "lookup_table", None) is None or not hasattr(
                scaler, "_compute_lookup_indices"):
            return

        kin_idx, step_idx = scaler._compute_lookup_indices(
            self.X_real[:, 0], self.X_real[:, 1])
        mu = scaler.lookup_table[kin_idx, step_idx, 0]
        std = scaler.lookup_table[kin_idx, step_idx, 1]
        keep = ~(torch.isnan(mu) | torch.isnan(std))

        n_dropped = int((~keep).sum())
        if n_dropped:
            log.info(f"Filtered out {n_dropped} NaN mean/std lookup steps...")
        self.X_real = self.X_real[keep]
        self.Y_real = self.Y_real[keep]

    def _filter_bethe_bloch_steps(self) -> None:
        """Drop deterministic (non-stochastic) Bethe-Bloch steps -- the
        WGAN-GP generator is trained only on the stochastic straggling
        regime. No result caching: see the module docstring for why."""
        straggling_model = ContinuousStragglingModel()
        deterministic_indexes = []

        for i in tqdm(range(self.Y_real.shape[0]), desc="Filtering deterministic steps"):
            L = float(self.X_real[i, 1])
            kin_energy = float(self.X_real[i, 0])
            point = PointInPhaseSpace(L * U.m2cm, kin_energy)
            straggling_model.set_state(point)
            if StragglingType.DeterministicBetheBloch in straggling_model.straggling_type:
                deterministic_indexes.append(i)

        remaining_indexes = np.setdiff1d(np.arange(self.Y_real.shape[0]), deterministic_indexes)
        self.X_real = self.X_real[remaining_indexes]
        self.Y_real = self.Y_real[remaining_indexes]

        log.info(f"Filtered out {len(deterministic_indexes)} deterministic bethe-bloch steps...")

    def _prepare_data_for_training(
            self, X_real: torch.Tensor, Y_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._setup_scalers(X_real, Y_real)
        X = self.scaler_x.scale(X_real)
        Y = self.scaler_y.scale(Y_real, phase_space=X_real)
        return X, Y

    def _setup_scalers(self, X_real: torch.Tensor, Y_real: torch.Tensor) -> None:
        if self.scaler_x is None:
            log.info(f"Fitting {self.x_scaler_type.__name__} on {X_real.shape[0]} samples...")
            self.scaler_x = self._fit_x_scaler(X_real)

        if self.scaler_y is None:
            if self.y_scaler_type is NoPhysContinuousYScaler:
                log.info(f"Fitting {self.y_scaler_type.__name__} on {Y_real.shape[0]} samples...")
                fit_nophys_y_scaler(Y_real, state_dir=self.scaler_dir)
                self.scaler_y = self.y_scaler_type.load(state_dir=self.scaler_dir)
            else:
                # ContinuousYScaler has no fit path -- its state is a
                # physics-derived lookup table, not fit to this dataset. See
                # the module docstring.
                log.info(f"Loading {self.y_scaler_type.__name__} (physics lookup table, not dataset-fit)...")
                self.scaler_y = self.y_scaler_type.load()

    def _fit_x_scaler(self, X_real: torch.Tensor) -> ScalerInterface:
        """Fits `ContinuousXScaler` (log10 min/max per column, with the
        step-length column's sign flipped to correct its skew); the scaler
        class itself carries no fit path (see module docstring). Writes the
        fitted state to `SCALERS_DIR`."""
        log_x = torch.log10(X_real).clone()
        log_x[:, 1] = torch.abs(log_x[:, 1])  # changes L skew to the right
        scaler = self.x_scaler_type()
        scaler.min_post_log = log_x.min(dim=0).values
        scaler.max_post_log = log_x.max(dim=0).values
        self._save_scaler_state(scaler, {
            "min_post_log": scaler.min_post_log,
            "max_post_log": scaler.max_post_log,
        })
        return scaler

    def _save_scaler_state(self, scaler: ScalerInterface, state: dict) -> None:
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.scaler_dir / f"{type(scaler).__name__}.state.pt"
        torch.save(state, out_path)
        log.info(f"Saved {type(scaler).__name__} state to {out_path}")
