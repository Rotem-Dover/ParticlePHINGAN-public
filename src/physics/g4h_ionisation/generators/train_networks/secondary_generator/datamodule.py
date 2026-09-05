"""
PyTorch Lightning data module for the secondary generator's WGAN-GP training.

**Data source.** Builds the `(X, Y)` training pair from
`data_handling.truth_extract.extract_secondary_xy` -- streamed straight out
of a GEANT4 truth ROOT file, no caching beside the source. `drop_theta`
defaults to **True**: `N_SECONDARY_FEATURES` is 1, so the target is the
one-column `[e_sec]` form and `y_path` resolves to
`secondary_Y_drop_theta.npy`. The two-column path is kept reachable with
`drop_theta=False` for comparison; training uses one column.

**On-disk cache format is `.npy`.** `X`/`Y` here are always plain
`float32`/`float64` `ndarray`s with no Python object structure to preserve
(unlike e.g. a dict of per-key CDFs), so `.npy` is a self-describing (dtype
+ shape in the header) fit that does not run arbitrary code on load the way
`pickle.load` can.

**Scaler fitting lives here, not in the scaler module.** `SecondaryXScaler`
/ `SecondaryYScaler`
(`physics/g4h_ionisation/generators/secondary_generator/scaler.py`) have a
no-argument constructor plus a `load()` classmethod reading a portable
`.state.pt` -- an inference-only shape with no fit path of their own. This
datamodule fits both scalers' minimal min/max arithmetic itself, scoped to
this module, and writes the result out through the exact same `.state.pt`
tensor-dict format `load()` reads, so a freshly fit scaler and one loaded
back from disk are interchangeable from the scaler class's point of view.

Writes only under `common.paths.DATASETS_DIR` / `SCALERS_DIR`.
"""

import logging
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule

import common.units as U
from common.config import DEVICE, N_BATCHES, N_SECONDARY_FEATURES
from common.enums import SecondaryFeats
from common.paths import DATASETS_DIR, SCALERS_DIR
from data_handling.truth_extract import extract_secondary_xy
from physics.g4h_ionisation.generators.secondary_generator.scaler import NoPhysSecondaryYScaler
from physics.interfaces.scaler_interface import ScalerInterface

log = logging.getLogger(__name__)


def fit_nophys_y_scaler(y: torch.Tensor, state_dir: Union[str, Path] = SCALERS_DIR) -> None:
    """Fit-and-write the NoPhys Y-scaler state (min/max in the class's own
    log space, per column over the (N, 1) tensor). Same write-what-load-reads
    contract as the inline fits."""
    t = y * U.eV2MeV
    t = torch.log10(t)
    state = {"min_post_log": t.min(dim=0).values, "max_post_log": t.max(dim=0).values}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(state_dir) / "NoPhysSecondaryYScaler.state.pt")


class CustomDataLoader:
    """
    Custom data loader for the secondary generator.
    X: post-continuous kinetic energy (1 feature)
    Y: [E_SEC] (N_SECONDARY_FEATURES features)
    """
    def __init__(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            batch_size: int = 100_000,
    ):
        dataset_size = len(X) - len(X) % batch_size
        self.X, self.Y = X, Y
        self.X = self.X[:dataset_size].reshape(-1, batch_size).to(DEVICE)
        self.Y = self.Y[:dataset_size, :].reshape(-1, batch_size, N_SECONDARY_FEATURES).to(DEVICE)

    def __iter__(self):
        for X, Y in zip(self.X, self.Y):
            yield X, Y

    def __len__(self):
        return self.Y.shape[0]


class DataModule(LightningDataModule):
    """
    Secondary-generator data module: loads the `(kinetic_energy, [e_sec])`
    training pair, fits/loads the X/Y scalers, and yields scaled batches.
    """
    def __init__(
            self,
            x_scaler_type:       Type[ScalerInterface],
            y_scaler_type:       Type[ScalerInterface],
            data_dir:            Union[str, Path] = DATASETS_DIR,
            scaler_dir:          Union[str, Path] = SCALERS_DIR,
            file_name_stem:      str              = "secondary",
            n_batches:           int              = N_BATCHES,
            drop_theta:          bool             = True,
    ):
        super().__init__()
        self.x_scaler_type = x_scaler_type
        self.y_scaler_type = y_scaler_type
        self.data_dir       = Path(data_dir)
        self.scaler_dir      = Path(scaler_dir)
        self.file_name_stem = file_name_stem
        self.n_batches       = n_batches
        self.drop_theta      = drop_theta

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
        """The one-column target lives under its own name, not as a narrower
        `secondary_Y.npy` -- `build_training_datasets.py` writes BOTH widths
        from one pass over the truth file, so selecting the width is a
        filename choice, not a rebuild."""
        suffix = "_Y_drop_theta.npy" if self.drop_theta else "_Y.npy"
        return self.data_dir / f"{self.file_name_stem}{suffix}"

    # ------------------------------ building ------------------------------- #
    def build_arrays_from_root(self, root_path: Path) -> None:
        """Extract `(X, Y)` from a GEANT4 truth ROOT file via
        `data_handling.truth_extract.extract_secondary_xy` and cache them as
        `.npy` under `DATASETS_DIR`. Never writes into the ROOT file's own
        directory.
        """
        log.info(f"Extracting secondary (X, Y) from {root_path}...")
        X, Y = extract_secondary_xy(root_path, drop_theta=self.drop_theta)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.x_path, X)
        np.save(self.y_path, Y)
        log.info(f"Saved {X.shape[0]} rows to {self.x_path} / {self.y_path}")

    # -------------------------------- setup --------------------------------- #
    def setup(self, stage: Optional[str] = None) -> None:
        if self.has_been_setup:
            return None

        log.info("Setting up secondary DataModule...")
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
                    f"{path} doesn't exist -- call build_arrays_from_root() "
                    f"first to extract it from a GEANT4 truth ROOT file")

        X = torch.from_numpy(np.load(self.x_path)).type(torch.float32)
        Y = torch.from_numpy(np.load(self.y_path)).type(torch.float32)
        assert X.shape[0] == Y.shape[0], "Number of steps should be equal both in X and Y"
        return X, Y

    def _filter_samples(self) -> None:
        # Remove rows where no secondary is generated (E_SEC == 0): the
        # generator is trained only on the nonzero continuum; the zero atom
        # is handled separately by G4hIonisation.along_step_do_it at
        # inference.
        mask = self.Y_real[:, SecondaryFeats.E_SEC_IDX] > 0
        self.X_real = self.X_real[mask]
        self.Y_real = self.Y_real[mask]

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
            log.info(f"Fitting {self.y_scaler_type.__name__} on {Y_real.shape[0]} samples...")
            self.scaler_y = self._fit_y_scaler(Y_real)

    def _fit_x_scaler(self, X_real: torch.Tensor) -> ScalerInterface:
        """Fits `SecondaryXScaler` (log10 min/max); the scaler class itself
        carries no fit path (see module docstring). Writes the fitted state
        to `SCALERS_DIR`."""
        log_x = torch.log10(X_real)
        scaler = self.x_scaler_type()
        scaler.min_post_log = log_x.min()
        scaler.max_post_log = log_x.max()
        self._save_scaler_state(scaler, {
            "min_post_log": scaler.min_post_log,
            "max_post_log": scaler.max_post_log,
        })
        return scaler

    def _fit_y_scaler(self, Y_real: torch.Tensor) -> ScalerInterface:
        """`SecondaryYScaler` has nothing to fit: its E_SEC bound is
        physics-derived, recomputed at scale time from
        `SecondaryYScaler.min_max_e_secondary`.

        An empty state file is still written, deliberately, for that scaler:
        `load()` reads one unconditionally, and a present-but-empty file
        records that this run fitted nothing, where an absent file is
        indistinguishable from a run that crashed before saving.

        `NoPhysSecondaryYScaler` is different in kind -- its min/max state IS
        the dataset-fit ablation arm (see `fit_nophys_y_scaler` above) -- so
        it gets the real fit-and-write instead of the empty-state no-op.
        """
        if self.y_scaler_type is NoPhysSecondaryYScaler:
            fit_nophys_y_scaler(Y_real, state_dir=self.scaler_dir)
            return self.y_scaler_type.load(state_dir=self.scaler_dir)

        scaler = self.y_scaler_type()
        self._save_scaler_state(scaler, {})
        return scaler

    def _save_scaler_state(self, scaler: ScalerInterface, state: dict) -> None:
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.scaler_dir / f"{type(scaler).__name__}.state.pt"
        torch.save(state, out_path)
        log.info(f"Saved {type(scaler).__name__} state to {out_path}")
