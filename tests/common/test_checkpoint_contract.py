"""The architecture constants in `common.config` are part of the checkpoint
contract: `from_checkpoint()` builds a bare generator from them, so a changed
value silently invalidates the pinned checkpoints. Pin them by value."""
import pytest

import common.config as cfg

CONTRACT = {
    "N_CONTINUOUS_FEATURES": 1,
    "N_SECONDARY_FEATURES": 1,
    "N_EMBEDDING_CNT": 25,
    "N_NOISE_CNT": 25,
    "N_EMBEDDING_SEC": 5,
    "N_NOISE_SEC": 5,
    "N_NEURON_MULTIPLIER_CNT": 50,
    "N_NEURON_MULTIPLIER_SEC": 16,
    "N_NEURON_MULTIPLIER_SEC_NOPHYS": 32,
    "N_EMBEDDING_DISC_CNT": 25,
    "N_EMBEDDING_DISC_SEC": 5,
    "SECONDARY_OUTPUT_ACTIVATION": "hardtanh_reflect_top",
    "SECONDARY_STRETCH_EPS": 0.01,
}


@pytest.mark.parametrize("name,value", sorted(CONTRACT.items()))
def test_contract_constant_is_pinned(name, value):
    assert getattr(cfg, name) == value, (
        f"{name} moved to {getattr(cfg, name)!r}; the pinned checkpoints were "
        f"trained at {value!r} — retrain or re-pin, do not just change the constant")
