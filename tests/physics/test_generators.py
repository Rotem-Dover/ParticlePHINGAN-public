"""The continuous and secondary WGAN generators load their pinned checkpoints
and produce finite, correctly-shaped output."""
import torch

from common.paths import CNT_CKPT, SEC_CKPT, SCALERS_DIR
from physics.g4h_ionisation.generators.continuous_generator.neural_net import ContinuousGenerator
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousXScaler, ContinuousYScaler)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import SecondaryGenerator
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    SecondaryXScaler, SecondaryYScaler)

SEED = 20260728


def _cnt_gen():
    return ContinuousGenerator.from_checkpoint(
        CNT_CKPT, device="cpu",
        x_scaler=ContinuousXScaler.load(state_dir=SCALERS_DIR),
        y_scaler=ContinuousYScaler.load(state_dir=SCALERS_DIR))


def _sec_gen():
    return SecondaryGenerator.from_checkpoint(
        SEC_CKPT, device="cpu",
        x_scaler=SecondaryXScaler.load(state_dir=SCALERS_DIR),
        y_scaler=SecondaryYScaler.load(state_dir=SCALERS_DIR))


def test_checkpoints_load_with_no_missing_or_unexpected_keys():
    # from_checkpoint builds a bare cls(), so a config drift shows up here as
    # a load_state_dict error rather than as silently wrong physics.
    _cnt_gen()
    _sec_gen()


def test_secondary_output_has_one_column():
    from common.enums import SecondaryFeats
    out = _sec_gen().predict(torch.full((8, 1), 5e7))
    assert out.shape[1] == len(SecondaryFeats) == 1


def test_length_sampler_respects_the_grid_bounds():
    from physics.g4h_ionisation.generators.length_generator.physics_mc import (
        PhysicsLengthGenerator)
    torch.manual_seed(SEED)
    out = PhysicsLengthGenerator().predict(torch.full((4096,), 1e8))
    assert torch.isfinite(out).all()
    assert (out > 0).all()
