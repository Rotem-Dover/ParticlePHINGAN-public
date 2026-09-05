import os

import pytest

from benchmark.runtime_profiling.simulator_builders import (
    build_eager_simulator, build_optimized_simulator, env_latches)


def test_env_latches_sets_and_restores():
    os.environ.pop("PHINGAN_FUSED_MLP", None)
    with env_latches(fused=True):
        assert os.environ["PHINGAN_FUSED_MLP"] == "1"
    assert "PHINGAN_FUSED_MLP" not in os.environ


def test_env_latches_restores_previous_value_on_exception():
    os.environ["PHINGAN_FUSED_MLP"] = "0"
    try:
        with env_latches(fused=True):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert os.environ["PHINGAN_FUSED_MLP"] == "0"
    os.environ.pop("PHINGAN_FUSED_MLP", None)


def test_env_latches_does_not_set_a_multiple_scattering_latch():
    """The simulator has no multiple-scattering stage, so the builders
    must not manage an env latch for one."""
    import benchmark.runtime_profiling.simulator_builders as sb
    assert "PHINGAN_MSC_FP32_RESCALE" not in sb._LATCH_KEYS


def test_eager_builder_presets_the_compile_flags():
    simulator, sm = build_eager_simulator("phin_gan", device="cpu")
    assert simulator._geometry_compiled is True
    for proc in sm.processes:
        assert proc._compiled is True


def test_optimized_builder_returns_a_runnable_simulator():
    simulator, sm = build_optimized_simulator("phin_gan", device="cpu")
    assert len(sm.processes) == 1
    assert type(sm.processes[0]).__name__ == "G4hIonisation"


def test_unknown_preset_raises_keyerror():
    with pytest.raises(KeyError):
        build_eager_simulator("no_such_preset", device="cpu")
