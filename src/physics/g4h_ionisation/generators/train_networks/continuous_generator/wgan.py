"""
WGAN-GP implementation for continuous energy loss generation.
Extends WGANInterface with physics-informed KS-test regularization against theoretical
straggling CDFs, plus validation logging and histogram comparison plots.

Training-time diagnostic plots land under `common.paths.FIGURES_DIR`.

**Physics-penalty / CDF path is required, not optional.**
`common.paths.PATH_TO_CONTINUOUS_CDFS` holds the theoretical continuous-loss
(straggling) CDFs this penalty scores against. The constructor raises
`FileNotFoundError` if that file is absent, so training refuses loudly
instead of silently running without physics regularization.

**Both the physics and no-physics ablation arms are supported, and
`setup()` is arm-aware.** `PHINGAN_NO_PHYS` (read by
`continuous_generator/train.py::select_arm()`) picks between
`ContinuousGenerator` and `NoPhysContinuousGenerator`. The constructor takes
`generator` as a plain argument, so the caller's choice of architecture
needs no change here; it also takes a `y_scaler_type` parameter (default
`ContinuousYScaler`), and `setup()` calls
`self.y_scaler_type.load(paths.SCALERS_DIR)` -- pairing each arm with its
own Y-scaler's `rescale()` semantics rather than always loading the physics
scaler regardless of which generator is in use. The **X-scaler load stays
hardcoded** to `ContinuousXScaler`: the input scaler is shared between arms,
since both condition on the same `(kinetic energy, step length)` input
space and only the *output* scaling and architecture differ. The
`use_physics_regularization` flag (`config.USE_PHYSICS` by default, though
`train.py` passes the selected arm's `use_physics` explicitly) gates only
the *loss term* (`add_generator_loss_terms`) and is independent of which
generator/scaler pair `train.py` selected, so a no-physics run and a
`USE_PHYSICS=False`-but-physics-architecture run are two different,
orthogonal knobs.
"""

import os
import numpy as np
import torch
import logging
import matplotlib as mpl

from pathlib import Path
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

import common.config as config
import common.units as U
from common import paths
from common.enums import ContinuousFeats
from common.paths import FIGURES_DIR, PATH_TO_CONTINUOUS_CDFS

from physics.penalty_models import PhysicsPenaltyModel, KSTestPenaltyModel
from physics.interfaces.wgan_interface import WGANInterface
from physics.g4h_ionisation.generators.continuous_generator.scaler import ContinuousXScaler, ContinuousYScaler
from physics.g4h_ionisation.generators.continuous_generator.neural_net import ContinuousGenerator
from physics.interfaces.generator_interface import GeneratorInterface
from physics.interfaces.scaler_interface import ScalerInterface
from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import Critic
from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import DataModule

log = logging.getLogger(__name__)


class ContinuousGAN(WGANInterface):
    """
    WGAN-GP for continuous energy loss generation.
    Physics regularization via KS distance to continuous loss CDF.
    """
    def __init__(
            self,
            generator:                      GeneratorInterface,
            critic:                         Critic,
            datamodule:                     DataModule = None,
            lr:                             float                     = config.LEARNING_RATE,
            lambda_gp:                      float                     = config.LAMBDA_GP,
            opt_g_frequency:                int                       = 1,
            opt_d_frequency:                int                       = 3,
            device:                         str                       = config.DEVICE,
            lambda_physics:                 float                     = config.LAMBDA_PHYSICS_CONTINUOUS,
            penalty_model:                  type[PhysicsPenaltyModel] = KSTestPenaltyModel,
            use_physics_regularization:     bool                      = config.USE_PHYSICS,
            y_scaler_type:                  type[ScalerInterface]     = ContinuousYScaler,
    ):
        super().__init__(
            generator=generator,
            critic=critic,
            lr=lr,
            lambda_gp=lambda_gp,
            opt_g_frequency=opt_g_frequency,
            opt_d_frequency=opt_d_frequency
        )
        # Loaded in setup() (below), not here: Lightning runs
        # DataModule.setup() before LightningModule.setup(), so by then the
        # datamodule has fitted and written the X-scaler state to
        # SCALERS_DIR. Loading here in __init__ would run before that
        # state exists.
        self._scaler_device = device
        self.x_scaler = None
        self.y_scaler = None
        self.y_scaler_type = y_scaler_type

        self.lambda_physics = lambda_physics
        self.use_physics_regularization = use_physics_regularization

        self.physics_continuous_penalizer = None

        if PATH_TO_CONTINUOUS_CDFS.exists():
            self.physics_continuous_penalizer = penalty_model(
                path_to_cdf=PATH_TO_CONTINUOUS_CDFS,
                columns_idx=ContinuousFeats.E_CNT_IDX
            )
        else:
            raise FileNotFoundError(
                f"Physics regularization enabled but continuous CDFs not found at "
                f"{str(PATH_TO_CONTINUOUS_CDFS)}.")

        self.save_hyperparameters(ignore=["datamodule", "generator", "critic", "physics_continuous_penalizer"])

        self.datamodule = datamodule

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.x_scaler is None:
            # X-scaler load stays hardcoded to ContinuousXScaler -- the input
            # scaler is shared between the phys and noPhys arms (spec 2.4).
            # The Y-scaler is arm-specific, hence self.y_scaler_type.
            self.x_scaler = ContinuousXScaler.load(paths.SCALERS_DIR).to(self._scaler_device)
            self.y_scaler = self.y_scaler_type.load(paths.SCALERS_DIR).to(self._scaler_device)
            self.generator.set_scalers(self.x_scaler, self.y_scaler)

    def add_generator_loss_terms(self, g_loss: torch.Tensor, labels: torch.Tensor, generated_steps: torch.Tensor) -> torch.Tensor:
        if not self.use_physics_regularization:
            return g_loss

        self.generator.eval()

        continuous_regularization = \
            self.physics_continuous_penalizer.physics_penalty_loss(generator=self.generator)
        g_loss = g_loss + self.lambda_physics * continuous_regularization

        self.generator.train()

        self.log("Physics Regularization Continuous", continuous_regularization, on_epoch=True, on_step=False)

        return g_loss

    def on_train_epoch_end(self):
        if not config.SIMULATE_ON_EPOCH_END:
            return None
        if self.current_epoch % config.EVERY_N_EPOCHS != 0:
            return None
        if self.datamodule is None:
            return None

        log.info(f"\nEvaluating continuous generator at epoch {self.current_epoch}")
        self.freeze()

        logger: TensorBoardLogger = self.logger
        base_path = FIGURES_DIR / "ContinuousGAN" / f"version_{logger.version}"
        n_samples_ = 10_000_000

        # Generate predictions (unscaled, real-space)
        X_real = self.datamodule.X_real[:n_samples_]  # [E, L]
        Y_real = self.datamodule.Y_real[:n_samples_]  # [E_CNT]
        E_cnt_real = Y_real.cpu().numpy().ravel()
        E_cnt_fake = self.generator.predict(X_real.to(self.device)).cpu().detach().numpy().ravel()

        E = X_real[:, 0].cpu().numpy()
        L = X_real[:, 1].cpu().numpy()

        # --- 1. Output distribution comparison ---
        log.info("\tLogging continuous energy loss distribution...")
        e_cnt_bins = np.logspace(
            np.log10(max(1e-20, min(E_cnt_real.min(), E_cnt_fake.min()))),
            np.log10(max(E_cnt_real.max(), E_cnt_fake.max())),
            100
        )
        fig, ax = plt.subplots()
        ax.hist(E_cnt_real + 1e-20, bins=e_cnt_bins, color='blue', alpha=0.5, label='Real', log=True, histtype='stepfilled')
        ax.hist(E_cnt_fake + 1e-20, bins=e_cnt_bins, color='orange', alpha=0.5, label='Fake', log=True, histtype='stepfilled')
        ax.set_xscale('log')
        ax.set_xlabel('Continuous Energy Loss [eV]')
        ax.set_ylabel('Frequency')
        ax.legend()
        logger.experiment.add_figure("Continuous Loss Distribution", fig, self.current_epoch)
        self.savefig(fig, base_path / "Continuous Loss Distribution")
        plt.close(fig)

        # --- 2. 2D histogram: E vs E_CNT (fake) ---
        log.info("\tLogging 2D histogram E vs E_CNT...")
        e_bins = np.logspace(np.log10(max(1e-1, E.min())), np.log10(E.max()), 100)
        fig, ax = plt.subplots()
        ax.hist2d(E, E_cnt_fake, bins=(e_bins, e_cnt_bins), cmap='jet', norm=mpl.colors.LogNorm())
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Kinetic Energy [eV]')
        ax.set_ylabel('Continuous Energy Loss [eV]')
        logger.experiment.add_figure("Fake E vs E_CNT", fig, self.current_epoch)
        self.savefig(fig, base_path / "Fake E vs E_CNT")
        plt.close(fig)

        # --- 3. Relative difference ---
        log.info("\tLogging relative difference E vs E_CNT...")
        hist2d_fake, _, _ = np.histogram2d(E, E_cnt_fake, bins=(e_bins, e_cnt_bins))
        hist2d_real, _, _ = np.histogram2d(E, E_cnt_real, bins=(e_bins, e_cnt_bins))
        mask = hist2d_real > 0
        relative_diff = np.zeros_like(hist2d_real)
        relative_diff[mask] = np.abs(hist2d_fake[mask] - hist2d_real[mask]) / hist2d_real[mask]
        fig, ax = plt.subplots()
        X_mesh, Y_mesh = np.meshgrid(e_bins, e_cnt_bins)
        cax = ax.pcolormesh(X_mesh, Y_mesh, relative_diff.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1e0))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Kinetic Energy [eV]')
        ax.set_ylabel('Continuous Energy Loss [eV]')
        fig.colorbar(cax, ax=ax, label='Relative Difference')
        logger.experiment.add_figure("Continuous Relative Difference E vs E_CNT", fig, self.current_epoch)
        self.savefig(fig, base_path / "Continuous Relative Difference E vs E_CNT")
        plt.close(fig)

        # --- 3b. Relative difference (L vs E_CNT) ---
        log.info("\tLogging relative difference L vs E_CNT...")
        l_bins = np.logspace(np.log10(max(1e-12, L.min())), np.log10(L.max()), 100)
        hist2d_fake_L, _, _ = np.histogram2d(L, E_cnt_fake, bins=(l_bins, e_cnt_bins))
        hist2d_real_L, _, _ = np.histogram2d(L, E_cnt_real, bins=(l_bins, e_cnt_bins))
        mask_L = hist2d_real_L > 0
        relative_diff_L = np.zeros_like(hist2d_real_L)
        relative_diff_L[mask_L] = np.abs(hist2d_fake_L[mask_L] - hist2d_real_L[mask_L]) / hist2d_real_L[mask_L]
        fig, ax = plt.subplots()
        X_mesh_L, Y_mesh_L = np.meshgrid(l_bins, e_cnt_bins)
        cax = ax.pcolormesh(X_mesh_L, Y_mesh_L, relative_diff_L.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1e0))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Step Length [m]')
        ax.set_ylabel('Continuous Energy Loss [eV]')
        fig.colorbar(cax, ax=ax, label='Relative Difference')
        logger.experiment.add_figure("Continuous Relative Difference L vs E_CNT", fig, self.current_epoch)
        self.savefig(fig, base_path / "Continuous Relative Difference L vs E_CNT")
        plt.close(fig)

        # --- 4. Physics loss grid ---
        log.info("\tLogging continuous physics loss grid...")
        fig = self._plot_physics_loss_grid(self.physics_continuous_penalizer)
        logger.experiment.add_figure("Continuous Physics Loss at phasespace points", fig, self.current_epoch)
        self.savefig(fig, base_path / "Continuous Physics Loss at phasespace points")
        plt.close(fig)

        self.unfreeze()

    def _plot_physics_loss_grid(self, penalizer: KSTestPenaltyModel) -> plt.Figure:
        Ls, kins, loss_vals = [], [], []
        for (L, kin_energy), cdf in penalizer.physics_cdfs.items():
            phase_space = torch.tensor([kin_energy, L], dtype=torch.float32, device=self.device).unsqueeze(0)
            prediction = self.generator.predict(phase_space, repeat_interleave=1_000).detach().cpu().numpy().ravel()
            pred_cdf_x = np.sort(prediction)
            pred_cdf_y = np.cumsum(np.ones_like(pred_cdf_x)) / len(pred_cdf_x)
            ys_for_comp = cdf(torch.from_numpy(pred_cdf_x)).numpy()
            loss_vals.append(float(np.max(np.abs(ys_for_comp - pred_cdf_y))))
            Ls.append(L)
            kins.append(kin_energy)

        fig = plt.figure(figsize=(6, 5))
        sc = plt.scatter(np.array(kins) * U.eV2MeV, np.array(Ls) * U.m2um,
                         c=loss_vals, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1))
        plt.colorbar(sc)
        plt.xlabel('Kinetic Energy [MeV]')
        plt.ylabel('Step Length [um]')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        return fig

    def savefig(self, fig: plt.Figure, path: Path):
        # Off by default: every figure this writes is also add_figure()'d
        # into the TensorBoard events at the same epoch, so a second on-disk
        # PNG copy per epoch only multiplies the /storage inode count for no
        # additional information. PHINGAN_SAVE_FIGS=1 re-enables the on-disk
        # copies.
        if os.environ.get("PHINGAN_SAVE_FIGS", "").strip() != "1":
            return
        if not path.exists():
            path.mkdir(parents=True)
        fig.savefig(path / f"{self.current_epoch}.png", dpi=300)
