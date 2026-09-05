"""The playground must load cluster-fetched checkpoints of any sweep size.

Two distinct failure modes behind one generic 500:

1. `_resolve_scalers` constructed BARE scalers (``ContinuousXScaler()`` has
   ``min_post_log=None``) instead of ``load()``-ing their portable state, so
   the continuous generator loaded and then crashed inside ``predict``.
2. The fetched secondary run is a net-size-sweep arm (embed 5, trunk
   32->32->16, 1 output). ``from_checkpoint`` builds the net from config.py
   constants (embed 25, trunk 50), so ``load_state_dict`` cannot fit it, and
   the width is recorded nowhere in the checkpoint — only in the run's
   ``runs/`` snapshot branch. The playground therefore infers the
   architecture from the state-dict shapes, via an opt-in
   ``infer_arch=True`` on ``from_checkpoint`` that the runtime never passes.
"""
from __future__ import annotations

import torch
import pytest

from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator,
)


def test_bare_constructor_shapes_are_the_checkpoint_contract():
    """`cls()` with no args must keep building the config-shaped net."""
    from common.config import (
        N_EMBEDDING_SEC, N_NOISE_SEC, N_SECONDARY_FEATURES,
        N_NEURON_MULTIPLIER_SEC,
    )
    gen = SecondaryGenerator()
    sd = gen.state_dict()
    trunk = N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES
    assert tuple(sd["embed.0.weight"].shape) == (N_EMBEDDING_SEC, 1)
    assert tuple(sd["model.0.0.weight"].shape) == (trunk, N_NOISE_SEC + N_EMBEDDING_SEC)
    assert tuple(sd["model.2.0.weight"].shape) == (trunk // 2, trunk)
    assert tuple(sd["model.2.3.weight"].shape) == (N_SECONDARY_FEATURES, trunk // 2)


@pytest.fixture
def sweep_ckpt(tmp_path):
    """A Lightning-style checkpoint of a W=32 sweep arm (n_emb=n_noise=5)."""
    small = SecondaryGenerator(n_embedding=5, n_noise=5,
                               n_trunk=32, n_half=16, n_features=1)
    state_dict = {f"generator.{k}": v for k, v in small.state_dict().items()}
    path = tmp_path / "epoch=46999.ckpt"
    torch.save({"state_dict": state_dict}, path)
    return path


def test_from_checkpoint_infer_arch_loads_a_sweep_arm(sweep_ckpt):
    gen = SecondaryGenerator.from_checkpoint(sweep_ckpt, device="cpu",
                                             infer_arch=True)
    assert gen.n_embedding == 5 and gen.n_noise == 5
    with torch.inference_mode():
        out = gen.predict(torch.full((16, 1), 0.5))
    assert out.shape == (16, 1)


@pytest.fixture
def mismatched_ckpt(tmp_path):
    """A 50-wide shape, different from the configured architecture, so a
    strict from_checkpoint must fail loud on it."""
    big = SecondaryGenerator(n_embedding=25, n_noise=25,
                             n_trunk=50, n_half=25, n_features=1)
    state_dict = {f"generator.{k}": v for k, v in big.state_dict().items()}
    path = tmp_path / "epoch=0.ckpt"
    torch.save({"state_dict": state_dict}, path)
    return path


def test_from_checkpoint_stays_strict_by_default(mismatched_ckpt):
    with pytest.raises(RuntimeError, match="size mismatch"):
        SecondaryGenerator.from_checkpoint(mismatched_ckpt, device="cpu")


def test_strict_from_checkpoint_loads_the_configured_shape(tmp_path):
    """A bare net IS the configured shape (config constants are the
    architecture), so the runtime's strict path must load a checkpoint
    written at exactly that shape."""
    configured = SecondaryGenerator()
    state_dict = {f"generator.{k}": v for k, v in configured.state_dict().items()}
    path = tmp_path / "epoch=32799.ckpt"
    torch.save({"state_dict": state_dict}, path)
    gen = SecondaryGenerator.from_checkpoint(path, device="cpu")
    assert gen.n_embedding == 5 and gen.n_noise == 5


def test_stepping_manager_from_config_loads_a_sweep_arm(sweep_ckpt):
    """The pull/trajectory tabs build via `SteppingManager.from_config` →
    `_build_generator`, whose only caller is the playground's sim_runner —
    it must infer sweep widths the same way `/api/generators/generate` does.
    (The preset builder `track_simulator_config.py` calls `from_checkpoint`
    directly and deliberately stays strict.)"""
    from particle_propagation.stepping_manager import SteppingManager

    gen = SteppingManager._build_generator(
        {"name": "SecondaryGenerator", "checkpoint": str(sweep_ckpt)},
        {"SecondaryGenerator": SecondaryGenerator},
    )
    assert gen.n_embedding == 5 and gen.n_noise == 5
    with torch.inference_mode():
        assert gen.predict(torch.full((8, 1), 0.5)).shape == (8, 1)


def test_resolve_scalers_returns_loaded_state(monkeypatch):
    """The endpoint must `load()` scaler state, never construct bare ones."""
    import playground.api.generators as gen_mod
    import physics.g4h_ionisation.generators.continuous_generator.scaler as c
    import physics.g4h_ionisation.generators.secondary_generator.scaler as s

    loaded = []
    for klass in (c.ContinuousXScaler, c.ContinuousYScaler,
                  s.SecondaryXScaler, s.SecondaryYScaler):
        def fake_load(cls, state_dir=None, _k=klass):
            loaded.append(_k.__name__)
            return object.__new__(_k)
        monkeypatch.setattr(klass, "load", classmethod(fake_load))

    x, y = gen_mod._resolve_scalers("continuous", "ContinuousGenerator")
    assert isinstance(x, c.ContinuousXScaler) and isinstance(y, c.ContinuousYScaler)
    x, y = gen_mod._resolve_scalers("secondary", "SecondaryGenerator")
    assert isinstance(x, s.SecondaryXScaler) and isinstance(y, s.SecondaryYScaler)
    assert loaded == ["ContinuousXScaler", "ContinuousYScaler",
                      "SecondaryXScaler", "SecondaryYScaler"]


def test_generate_endpoint_reports_predict_failures_as_json(monkeypatch):
    """A predict-time crash must surface its message, not Flask's default
    500 page, which carries no information about which scaler or generator
    failed or why."""
    import json
    from flask import Flask
    import playground.api.generators as gen_mod
    import playground.services.sampling as sampling_mod

    class Boom:
        def predict(self, labels):
            raise RuntimeError("boom: scaler state missing")

    monkeypatch.setattr(gen_mod, "_load_generator",
                        lambda stage, cls, ckpt: Boom())
    monkeypatch.setattr(sampling_mod, "available_stages",
                        lambda: ("continuous", "secondary"))

    app = Flask(__name__)
    app.register_blueprint(gen_mod.bp, url_prefix="/api/generators")
    resp = app.test_client().get(
        "/api/generators/generate?stage=secondary&class_name=SecondaryGenerator"
        "&checkpoint=x.ckpt&E=1e7")
    assert resp.status_code == 500
    body = json.loads(resp.data)
    assert "boom: scaler state missing" in body["error"]


def test_stepping_manager_from_config_accepts_the_mc_baseline():
    """The top bar's default for every stage is "MC baseline (Physics
    sampler)", which `processes.js` sends as the three Physics* class names.
    `from_config` must resolve all three, or every Trajectory / Pull /
    Pairwise run fails out of the box."""
    from particle_propagation.stepping_manager import SteppingManager
    from physics.g4h_ionisation.generators.continuous_generator.physics_mc import PhysicsContinuousGenerator
    from physics.g4h_ionisation.generators.secondary_generator.physics_mc import PhysicsSecondaryGenerator

    sm = SteppingManager.from_config(device="cpu", processes=[{
        "name": "g4h_ionisation",
        "length_generator":     {"name": "PhysicsLengthGenerator"},
        "continuous_generator": {"name": "PhysicsContinuousGenerator"},
        "secondary_generator":  {"name": "PhysicsSecondaryGenerator"},
    }])
    ion = sm.processes[0]
    assert isinstance(ion.continuous_generator, PhysicsContinuousGenerator)
    assert isinstance(ion.secondary_generator, PhysicsSecondaryGenerator)
