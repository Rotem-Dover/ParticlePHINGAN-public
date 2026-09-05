"""Single-step output shape and geometry invariants."""
import torch

from benchmark.track_simulator_config import build_stepping_manager
from common.enums import StepFeaturesIdxes


def test_step_output_has_five_columns():
    sm = build_stepping_manager("phin_gan", device="cpu")
    out = sm(torch.full((256,), 1e8))
    assert out.shape == (256, len(StepFeaturesIdxes))


def test_geom_step_equals_true_step():
    sm = build_stepping_manager("phin_gan", device="cpu")
    out = sm(torch.full((256,), 1e8))
    torch.testing.assert_close(out[:, int(StepFeaturesIdxes.GEOM_STEP)],
                               out[:, int(StepFeaturesIdxes.L)], rtol=0, atol=0)


def test_unknown_preset_raises_rather_than_guessing():
    import pytest
    build_stepping_manager("phin_gan", device="cpu")
    with pytest.raises(KeyError):
        build_stepping_manager("nope", device="cpu")
