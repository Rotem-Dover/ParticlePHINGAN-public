"""The phin_gan preset builds from exactly one continuous and one secondary
checkpoint, both pinned by explicit path, with every scaler read from
SCALERS_DIR."""
import pytest

from common.paths import (CNT_CKPT, SEC_CKPT, SCALERS_DIR)

_ARTIFACTS = [
    CNT_CKPT, SEC_CKPT,
    SCALERS_DIR / "ContinuousXScaler.state.pt",
    SCALERS_DIR / "ContinuousYScaler.state.pt",
    SCALERS_DIR / "SecondaryXScaler.state.pt",
    SCALERS_DIR / "SecondaryYScaler.state.pt",
]
needs_artifacts = pytest.mark.skipif(
    not all(p.exists() for p in _ARTIFACTS),
    reason="checkpoints/scaler states not staged locally")


def test_ckpt_pins_point_at_the_configured_runs():
    # The default beam is proton in aluminium; the pins must follow it.
    assert "2026-08-12-164426_continuous_aluminum_60281d7c67_e29e061b" in str(CNT_CKPT)
    assert CNT_CKPT.name == "epoch=23499.ckpt"
    # SECONDARY_OUTPUT_ACTIVATION / N_NEURON_MULTIPLIER_SEC in
    # common/config.py fix this checkpoint's architecture and eval rule.
    assert "2026-08-19-192831_secondary_s2_hardtanh_open_top_w16_aluminum_1eabc3aea3_ef3bec7d" in str(SEC_CKPT)
    assert SEC_CKPT.name == "epoch=17999.ckpt"


def test_every_material_has_a_pin_pair_in_its_own_run_tree():
    """One (run, epoch) pair per stage per material, each run named for the
    material it was trained on, so PHINGAN_BEAM retargets the pins with the
    beam and never loads another material's weights."""
    from common.paths import _PHIN_GAN_PINS
    assert set(_PHIN_GAN_PINS) == {"aluminum", "iron", "beryllium"}
    for material, ((cnt_run, cnt_epoch), (sec_run, sec_epoch)) in _PHIN_GAN_PINS.items():
        assert f"continuous_{material}" in cnt_run
        assert "hardtanh_open_top_w16_" + material in sec_run
        assert isinstance(cnt_epoch, int) and isinstance(sec_epoch, int)


@needs_artifacts
def test_phin_gan_builds_the_configured_architecture():
    from benchmark.track_simulator_config import build_stepping_manager
    sm = build_stepping_manager("phin_gan", device="cpu")
    proc = sm.processes[0]
    assert proc.secondary_generator.n_embedding == 5
    assert proc.secondary_generator.n_noise == 5
    assert proc.continuous_generator.n_embedding == 25
    assert proc.continuous_generator.n_noise == 25
