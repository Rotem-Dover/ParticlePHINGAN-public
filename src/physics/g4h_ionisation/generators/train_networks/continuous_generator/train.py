"""
Training entrypoint for the continuous energy loss WGAN-GP generator.
Builds the PyTorch Lightning trainer, instantiates the ContinuousGAN model with its
critic and data module, and runs the training loop with optional physics regularization.

Checkpoints and TensorBoard logs write under `common.paths.RUNS_DIR` only.
`RESUME_FROM_CHECKPOINT` resumes from a `last.ckpt` under this run's own
logger directory (`RUNS_DIR / "logger_cnt" / <version> /
checkpoints/last.ckpt`).

The `USE_PHYSICS`-gated non-physics ablation arm (`NoPhysContinuousGenerator`
/ `NoPhysContinuousYScaler`) is selected per-run by the `PHINGAN_NO_PHYS`
env var rather than a config constant, since the arm is a per-run property
like the beam and the seed. See `select_arm()`: `PHINGAN_NO_PHYS=1` swaps
the generator class, the `y_scaler_type` passed to BOTH the datamodule (fit
target) and `ContinuousGAN` itself (the `y_scaler_type=` constructor
argument `setup()` loads through -- see `continuous_generator/wgan.py`'s
docstring), `use_physics_regularization=False`, and the TensorBoard logger
name (`logger_cnt` -> `logger_cnt_nophys`) so a no-physics run's checkpoints
and logs can never be confused with a physics run's.

**Known cost:** the continuous datamodule's `_filter_bethe_bloch_steps` runs
a Python loop over every training row through `ContinuousStragglingModel`
and has no result cache; see that module's docstring for the reasoning. A
real training run against the full dataset should expect this filter's full
per-row cost on every `setup()` call.

This module is a runnable entrypoint (`main()` / `__main__`), not something
imported for its side effects -- importing it must not touch storage at all.
"""

import os
import torch
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from common.config import MAX_EPOCHS, DEVICE
from common.paths import RUNS_DIR, ON_CLUSTER
from common.run_context import ACTIVE_NAME

from physics.g4h_ionisation.generators.train_networks.continuous_generator.wgan import ContinuousGAN
from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import DataModule
from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import Critic
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator, NoPhysContinuousGenerator)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousXScaler, ContinuousYScaler, NoPhysContinuousYScaler)
from physics.g4h_ionisation.generators.train_networks.run_version import (
    get_run_version, resume_requested, write_run_meta)

torch.set_float32_matmul_precision('medium')


def select_arm():
    """Per-run arm selection (PHINGAN_NO_PHYS=1 -> the no-physics ablation).

    Env-var, not config: config.py's constants are value-pinned by the
    branch-scope test, and the arm is a per-run property like the beam."""
    if os.environ.get("PHINGAN_NO_PHYS", "").strip() == "1":
        return (NoPhysContinuousGenerator(), NoPhysContinuousYScaler,
                False, "logger_cnt_nophys")
    return ContinuousGenerator(), ContinuousYScaler, True, "logger_cnt"


def select_seed() -> int:
    """Per-run RNG seed (PHINGAN_SEED, default 42). Env-var, not config, for
    the same reason as the arm and the beam: it is a per-run property. A
    non-integer value raises rather than silently seeding something else."""
    raw = os.environ.get("PHINGAN_SEED", "").strip()
    return int(raw) if raw else 42


def select_ckpt_every() -> int:
    """Checkpoint cadence in epochs (PHINGAN_CKPT_EVERY, default 100).
    Env-var, not config, because it is a per-run property and the cluster
    src/ is shared with queued jobs. Note save_top_k=-1 keeps every file, so
    at MAX_EPOCHS = 100_000 a low cadence retains many thousands of
    checkpoints per stage against the /storage inode quota."""
    raw = os.environ.get("PHINGAN_CKPT_EVERY", "").strip()
    n = int(raw) if raw else 100
    if n < 1:
        raise ValueError(f"PHINGAN_CKPT_EVERY={raw!r}: must be a positive integer")
    return n


def build_trainer(logger_name: str = "logger_cnt") -> pl.Trainer:
    logger = TensorBoardLogger(save_dir=str(RUNS_DIR), name=logger_name,
                               version=get_run_version(), log_graph=False)
    # Provenance for whoever finds this run dir later: which code trained it.
    # PHINGAN_GIT_SHA is the snapshot commit deploy_and_train.sh recorded on
    # the runs/<stage>/<ts> branch; empty for local runs.
    write_run_meta(logger.log_dir, {
        "run_id": get_run_version(),
        "git_sha": os.environ.get("PHINGAN_GIT_SHA", "").strip(),
        "stage": "continuous",
        "beam": ACTIVE_NAME,
        "seed": select_seed(),
        "ckpt_every": select_ckpt_every(),
    })
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}",
        save_last=True,
        save_top_k=-1,
        # Default 100 (see select_ckpt_every for the inode-quota reasoning);
        # deploy_and_train.sh --ckpt-every=N overrides per run.
        every_n_epochs=select_ckpt_every(),
    )

    devices = 1 if ON_CLUSTER else 'auto'

    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            model_checkpoint,
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=20),
        ],
        logger=logger,
        log_every_n_steps=1
    )


def _resume_ckpt_path(logger_name: str = "logger_cnt") -> str:
    """Which logger version to resume. Defaults to `version_0`, overridable
    per job so a wall-clock-limited run (against `MAX_EPOCHS = 100_000`) can
    be walked forward across several submissions.

    Takes the selected arm's logger name so a resume of a no-physics run
    does not silently resume a physics one."""
    version = os.environ.get("PHINGAN_RESUME_VERSION", "").strip() or "version_0"
    return str(RUNS_DIR / logger_name / version / "checkpoints" / "last.ckpt")


def train_gan() -> ContinuousGAN:
    from physics.penalty_models import KSTestPenaltyModel
    torch.serialization.add_safe_globals([KSTestPenaltyModel])
    seed_everything(select_seed())

    generator, y_scaler_type, use_physics, logger_name = select_arm()

    datamodule = DataModule(
        x_scaler_type=ContinuousXScaler,
        y_scaler_type=y_scaler_type,
    )

    critic = Critic()
    model = ContinuousGAN(generator=generator, critic=critic, datamodule=datamodule,
                          use_physics_regularization=use_physics,
                          y_scaler_type=y_scaler_type)

    kwargs = {
        "model": model,
        "datamodule": datamodule,
    }

    if resume_requested():
        kwargs["ckpt_path"] = _resume_ckpt_path(logger_name)

    trainer = build_trainer(logger_name)
    trainer.fit(**kwargs)

    return model


def main() -> ContinuousGAN:
    return train_gan()


if __name__ == "__main__":
    main()
