"""Simulator presets.

  phin_gan - the physics-informed configuration: analytic MC length + WGAN
             continuous + WGAN secondary (CNT_CKPT / SEC_CKPT), with every
             scaler loaded from SCALERS_DIR.
  gan      - the no-physics ablation arm: the same three-phase pipeline and
             analytic-MC length stage, but the continuous/secondary nets and
             their Y-scalers are the NoPhys (BatchNorm1d in place of
             LayerNorm) variants (NOPHYS_CNT_CKPT / NOPHYS_SEC_CKPT). The
             continuous generator's readout runs through
             `1.2 * sigmoid(x) - 0.1`; the secondary generator ends in a
             bare Linear and rectifies the e_sec column with ReLU in
             `forward`. The continuous stage's X- and Y-scalers are loaded
             from NOPHYS_CNT_SCALERS_DIR, its own fitted state, because its
             bounds differ materially from the physics stage's; the
             secondary stage's scalers come from SCALERS_DIR.

Both presets share the length stage. Each preset entry carries its own
scaler state_dir per stage.
"""
import logging

from common.config import DEVICE
from common.paths import (NOPHYS_CNT_CKPT, CNT_CKPT, NOPHYS_SEC_CKPT, SEC_CKPT, NOPHYS_CNT_SCALERS_DIR, SCALERS_DIR)
from particle_propagation.stepping_manager import SteppingManager
from particle_propagation.track_simulator import TrackSimulator
from physics.g4h_ionisation.g4h_ionisation import G4hIonisation
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator, NoPhysContinuousGenerator)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousXScaler, ContinuousYScaler, NoPhysContinuousYScaler)
from physics.g4h_ionisation.generators.length_generator.physics_mc import (
    PhysicsLengthGenerator)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    NoPhysSecondaryGenerator, SecondaryGenerator)
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    NoPhysSecondaryYScaler, SecondaryXScaler, SecondaryYScaler)

log = logging.getLogger(__name__)

# Checkpoints are pinned, not resolved by recency: _newest_checkpoint expects
# timestamp-prefixed RUN_IDs and would not find these version_N directories.
# Each stage entry is (generator class, checkpoint, Y-scaler class,
# scaler state_dir) -- the state_dir feeds BOTH that stage's X- and Y-scaler
# load(), because a checkpoint is only meaningful with the scaler state it
# trained against. The gan/cnt stage reads its own fitted state
# (NOPHYS_CNT_SCALERS_DIR); everything else reads SCALERS_DIR.
PRESETS = {
    "phin_gan": {"cnt": (ContinuousGenerator, CNT_CKPT, ContinuousYScaler, SCALERS_DIR),
                 "sec": (SecondaryGenerator, SEC_CKPT, SecondaryYScaler, SCALERS_DIR)},
    "gan":      {"cnt": (NoPhysContinuousGenerator, NOPHYS_CNT_CKPT, NoPhysContinuousYScaler, NOPHYS_CNT_SCALERS_DIR),
                 "sec": (NoPhysSecondaryGenerator, NOPHYS_SEC_CKPT, NoPhysSecondaryYScaler, SCALERS_DIR)},
}


# The no-physics ablation was trained for the paper's beam only; every other
# material has `phin_gan` alone, and the figure scripts drop their GAN arm
# when `has_gan_preset()` is False rather than failing on a missing pin.
GAN_PRESET_MATERIALS = ("aluminum",)


def has_gan_preset() -> bool:
    """Whether the `gan` preset exists for the active beam's material."""
    from common.paths import _MATERIAL_DIR
    return _MATERIAL_DIR in GAN_PRESET_MATERIALS


def build_stepping_manager(name: str, device: str = DEVICE) -> SteppingManager:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]
    cnt_cls, cnt_ckpt, cnt_y_cls, cnt_scaler_dir = cfg["cnt"]
    sec_cls, sec_ckpt, sec_y_cls, sec_scaler_dir = cfg["sec"]
    process = G4hIonisation(
        length_generator=PhysicsLengthGenerator(),
        continuous_generator=cnt_cls.from_checkpoint(
            cnt_ckpt, device=device,
            x_scaler=ContinuousXScaler.load(state_dir=cnt_scaler_dir),
            y_scaler=cnt_y_cls.load(state_dir=cnt_scaler_dir)),
        secondary_generator=sec_cls.from_checkpoint(
            sec_ckpt, device=device,
            x_scaler=SecondaryXScaler.load(state_dir=sec_scaler_dir),
            y_scaler=sec_y_cls.load(state_dir=sec_scaler_dir)),
        device=device)
    if name == "phin_gan":
        log.info(
            "phin_gan: physics-informed generators, scalers from "
            "SCALERS_DIR")
    elif name == "gan":
        log.info(
            "gan: no-physics ablation (NoPhys generators); continuous "
            "scalers from NOPHYS_CNT_SCALERS_DIR")
    return SteppingManager(processes=[process], device=device)


def build_track_simulator(name: str, device: str = DEVICE) -> TrackSimulator:
    return TrackSimulator(
        stepping_manager=build_stepping_manager(name, device), device=device)
