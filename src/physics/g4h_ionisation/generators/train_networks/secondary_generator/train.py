"""
Training entrypoint for the secondary energy loss WGAN-GP generator.
Builds the PyTorch Lightning trainer, instantiates the SecondaryGAN model with its
critic and data module, and runs the training loop with optional physics regularization.

Checkpoints and TensorBoard logs write under `common.paths.RUNS_DIR` only.
`RESUME_FROM_CHECKPOINT` resumes from a `last.ckpt` under this run's own
logger directory (`RUNS_DIR / "logger_sec" / <version> /
checkpoints/last.ckpt`).

The `USE_PHYSICS`-gated non-physics ablation arm (`NoPhysSecondaryGenerator`
/ `NoPhysSecondaryYScaler`) is selected per-run by the `PHINGAN_NO_PHYS` env
var rather than a config constant, since the arm is a per-run property like
the beam and the seed. See `select_arm()`: `PHINGAN_NO_PHYS=1` swaps the
generator class, the `y_scaler_type` passed to BOTH the datamodule (fit
target) and `SecondaryGAN` itself (the `y_scaler_type=` constructor argument
`setup()` loads through -- see `secondary_generator/wgan.py`'s docstring),
`use_physics_regularization=False`, and the TensorBoard logger name
(`logger_sec` -> `logger_sec_nophys`) so a no-physics run's checkpoints and
logs can never be confused with a physics run's.

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

from physics.g4h_ionisation.generators.train_networks.secondary_generator.wgan import SecondaryGAN
from physics.g4h_ionisation.generators.train_networks.secondary_generator.datamodule import DataModule
from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import Critic
from common.config import N_SECONDARY_FEATURES
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator, NoPhysSecondaryGenerator, OUTPUT_ACTIVATION_IDS)
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    SecondaryXScaler, SecondaryYScaler, NoPhysSecondaryYScaler)
from physics.g4h_ionisation.generators.train_networks.run_version import (
    get_run_version, resume_requested, write_run_meta)

torch.set_float32_matmul_precision('medium')


def select_arm():
    """Per-run arm selection (PHINGAN_NO_PHYS=1 -> the no-physics ablation).

    Env-var, not config: config.py's constants are value-pinned by the
    branch-scope test, and the arm is a per-run property like the beam."""
    if os.environ.get("PHINGAN_NO_PHYS", "").strip() == "1":
        return (NoPhysSecondaryGenerator(), NoPhysSecondaryYScaler,
                False, "logger_sec_nophys")
    width = select_width()
    return (SecondaryGenerator(n_trunk=width * N_SECONDARY_FEATURES,
                               n_half=(width // 2) * N_SECONDARY_FEATURES,
                               output_activation=select_output_activation(),
                               output_stretch_eps=select_stretch_eps()),
            SecondaryYScaler, True, "logger_sec")


def select_width() -> int:
    """Per-run trunk width multiplier of the physics SecondaryGenerator
    (PHINGAN_SEC_WIDTH, default common.config.N_NEURON_MULTIPLIER_SEC).
    Env-var, not config, so a width sweep can vary this per run without
    editing config.py per arm; the env var records the width in meta.json
    and the RUN_ID. Loading a non-default-width checkpoint into the runtime
    still requires setting the config constant to match
    (`from_checkpoint` builds a bare `cls()`); the state-dict shapes catch a
    mismatch. Must be a positive even integer (n_half = width // 2)."""
    from common.config import N_NEURON_MULTIPLIER_SEC
    raw = os.environ.get("PHINGAN_SEC_WIDTH", "").strip()
    if not raw:
        return N_NEURON_MULTIPLIER_SEC
    if not raw.isdigit() or int(raw) < 2 or int(raw) % 2:
        raise ValueError(f"PHINGAN_SEC_WIDTH={raw!r}: must be a positive even integer")
    return int(raw)


def select_output_activation() -> str:
    """Per-run terminal activation of the physics SecondaryGenerator
    (PHINGAN_SEC_OUTPUT_ACT, default common.config.SECONDARY_OUTPUT_ACTIVATION).
    Env-var, not config, so the Sigmoid retrain can run while the local
    runtime still loads the Hardtanh checkpoint the config constant names;
    the trained checkpoint records the choice in its own state dict and
    from_checkpoint() refuses a mismatch, so the constant is flipped only
    when the new checkpoint is pinned. Unknown values raise."""
    from common.config import SECONDARY_OUTPUT_ACTIVATION
    raw = os.environ.get("PHINGAN_SEC_OUTPUT_ACT", "").strip()
    name = raw or SECONDARY_OUTPUT_ACTIVATION
    if name not in OUTPUT_ACTIVATION_IDS:
        raise ValueError(f"PHINGAN_SEC_OUTPUT_ACT={raw!r}: must be one of "
                         f"{sorted(OUTPUT_ACTIVATION_IDS)}")
    return name


def select_stretch_eps() -> float:
    """Per-run stretch of the "stretched_sigmoid" terminal
    (PHINGAN_SEC_STRETCH_EPS, default common.config.SECONDARY_STRETCH_EPS).
    Env-var for the same reason as the activation: the sweep runs several
    stretches while the runtime keeps loading the pinned checkpoint. Inert
    unless the activation is "stretched_sigmoid" (the net records 0.0 then).
    Must be a positive float."""
    from common.config import SECONDARY_STRETCH_EPS
    raw = os.environ.get("PHINGAN_SEC_STRETCH_EPS", "").strip()
    if not raw:
        return SECONDARY_STRETCH_EPS
    try:
        eps = float(raw)
    except ValueError:
        raise ValueError(f"PHINGAN_SEC_STRETCH_EPS={raw!r}: must be a positive float")
    if not eps > 0:
        raise ValueError(f"PHINGAN_SEC_STRETCH_EPS={raw!r}: must be a positive float")
    return eps


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


def build_trainer(logger_name: str = "logger_sec") -> pl.Trainer:
    logger = TensorBoardLogger(save_dir=str(RUNS_DIR), name=logger_name,
                               version=get_run_version(), log_graph=False)
    # Provenance for whoever finds this run dir later: which code trained it.
    # PHINGAN_GIT_SHA is the snapshot commit deploy_and_train.sh recorded on
    # the runs/<stage>/<ts> branch; empty for local runs.
    write_run_meta(logger.log_dir, {
        "run_id": get_run_version(),
        "git_sha": os.environ.get("PHINGAN_GIT_SHA", "").strip(),
        "stage": "secondary",
        "beam": ACTIVE_NAME,
        "seed": select_seed(),
        "ckpt_every": select_ckpt_every(),
        "secondary_output_activation": select_output_activation(),
        "secondary_stretch_eps": (select_stretch_eps()
                                  if select_output_activation() == "stretched_sigmoid" else 0.0),
        "secondary_width": select_width(),
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


def _resume_ckpt_path(logger_name: str = "logger_sec") -> str:
    """Which logger version to resume. Defaults to `version_0`, overridable
    per job so a wall-clock-limited run (against `MAX_EPOCHS = 100_000`) can
    be walked forward across several submissions.

    Takes the selected arm's logger name so a resume of a no-physics run
    does not silently resume a physics one."""
    version = os.environ.get("PHINGAN_RESUME_VERSION", "").strip() or "version_0"
    return str(RUNS_DIR / logger_name / version / "checkpoints" / "last.ckpt")


def train_gan() -> SecondaryGAN:
    from physics.penalty_models import KSTestPenaltyModel
    torch.serialization.add_safe_globals([KSTestPenaltyModel])
    seed_everything(select_seed())

    generator, y_scaler_type, use_physics, logger_name = select_arm()

    datamodule = DataModule(
        x_scaler_type=SecondaryXScaler,
        y_scaler_type=y_scaler_type,
    )

    critic = Critic()
    model = SecondaryGAN(generator=generator, critic=critic, datamodule=datamodule,
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


def main() -> SecondaryGAN:
    return train_gan()


if __name__ == "__main__":
    main()
