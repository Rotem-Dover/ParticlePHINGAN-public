"""The gan preset wires the NoPhys continuous/secondary generators and their
NoPhys Y-scalers into the same three-phase pipeline the phin_gan preset
uses. The two presets differ in generator classes, checkpoints, and the
continuous stage's scaler directory (NOPHYS_CNT_SCALERS_DIR vs
SCALERS_DIR); they share the secondary stage's scaler directory
(SCALERS_DIR) and the length stage."""
import pytest

from benchmark.track_simulator_config import PRESETS, build_stepping_manager
from common.paths import (NOPHYS_CNT_CKPT, NOPHYS_SEC_CKPT, SEC_CKPT, NOPHYS_CNT_SCALERS_DIR, SCALERS_DIR)
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator, NoPhysContinuousGenerator,
)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousYScaler, NoPhysContinuousYScaler,
)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    NoPhysSecondaryGenerator, SecondaryGenerator,
)
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    NoPhysSecondaryYScaler, SecondaryYScaler,
)


def test_phin_gan_composition_pinned():
    cnt_cls, _, cnt_y, cnt_dir = PRESETS["phin_gan"]["cnt"]
    assert cnt_cls is ContinuousGenerator and cnt_y is ContinuousYScaler
    sec_cls, sec_ckpt, sec_y, sec_dir = PRESETS["phin_gan"]["sec"]
    assert sec_cls is SecondaryGenerator and sec_y is SecondaryYScaler
    assert sec_ckpt is SEC_CKPT
    assert cnt_dir is SCALERS_DIR and sec_dir is SCALERS_DIR


def test_gan_wires_the_nophys_classes():
    # The gan preset's continuous stage pairs its checkpoint with
    # NOPHYS_CNT_SCALERS_DIR, a scaler tree distinct from the one the
    # secondary stage uses (SCALERS_DIR) -- each generator's checkpoint is
    # only numerically valid against the scaler state it was fit against.
    cnt_cls, cnt_ckpt, cnt_y, cnt_dir = PRESETS["gan"]["cnt"]
    assert cnt_cls is NoPhysContinuousGenerator and cnt_y is NoPhysContinuousYScaler
    assert cnt_ckpt is NOPHYS_CNT_CKPT
    assert cnt_dir is NOPHYS_CNT_SCALERS_DIR
    sec_cls, sec_ckpt, sec_y, sec_dir = PRESETS["gan"]["sec"]
    assert sec_cls is NoPhysSecondaryGenerator and sec_y is NoPhysSecondaryYScaler
    assert sec_ckpt is NOPHYS_SEC_CKPT
    assert sec_ckpt.name == "epoch=3199.ckpt"
    assert sec_dir is SCALERS_DIR


def test_gan_builds():
    # The gan preset's checkpoint and scaler paths are pinned in
    # common.paths, so building it on CPU must succeed and wire the NoPhys
    # nets and NoPhys Y-scalers, with the continuous stage's scaler state
    # holding the values its own scaler tree was fit to.
    import torch
    sm = build_stepping_manager("gan", device="cpu")
    process = sm.processes[0]
    assert isinstance(process.continuous_generator, NoPhysContinuousGenerator)
    assert isinstance(process.secondary_generator, NoPhysSecondaryGenerator)
    assert isinstance(process.continuous_generator.y_scaler, NoPhysContinuousYScaler)
    assert isinstance(process.secondary_generator.y_scaler, NoPhysSecondaryYScaler)
    # this scaler tree's Y-state max_post_log sits above 9.0
    assert float(process.continuous_generator.y_scaler.max_post_log) > 9.0
    # and its X-scaler L min_post_log is 3.9571
    assert torch.isclose(process.continuous_generator.x_scaler.min_post_log[1],
                         torch.tensor(3.9571), atol=1e-3)


def test_unknown_preset_still_keyerrors():
    with pytest.raises(KeyError):
        build_stepping_manager("nope", device="cpu")
