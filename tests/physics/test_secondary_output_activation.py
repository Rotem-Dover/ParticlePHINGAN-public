"""The secondary generator's terminal activation is part of the checkpoint
contract: `SecondaryGenerator(output_activation=...)` selects among several
terminal kinds (hardtanh, sigmoid, stretched_sigmoid, hardtanh_open_top,
hardtanh_reflect_top) that all produce identical parameter shapes, so an
`output_activation_id` buffer in the state dict is what tells them apart.
Loading a checkpoint of one kind into a net of a different kind MUST fail
loud -- the state-dict shapes alone would not catch it.

Sigmoid and its variants exist because Hardtanh(0, 1) clamps its output
during training, which gives the generator a dead gradient at the clamp and
lets it accumulate probability mass in Dirac atoms at z==0 and z==1 that
GEANT4 truth does not have.
"""
import os
import pytest
import torch
from torch import nn

from common import config
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    SecondaryGenerator, OUTPUT_ACTIVATION_IDS)


def _terminal(gen):
    return list(gen.model.modules())[-1]


def test_hardtanh_variant_is_still_buildable_explicitly():
    gen = SecondaryGenerator(output_activation="hardtanh").eval()
    t = _terminal(gen)
    assert isinstance(t, nn.Hardtanh) and (t.min_val, t.max_val) == (0, 1)


def test_sigmoid_variant_builds_with_identical_parameter_shapes():
    a = SecondaryGenerator(output_activation="hardtanh")
    b = SecondaryGenerator(output_activation="sigmoid")
    assert isinstance(_terminal(b), nn.Sigmoid)
    sa = {k: v.shape for k, v in a.state_dict().items()}
    sb = {k: v.shape for k, v in b.state_dict().items()}
    assert sa == sb
    # and the id buffer is what tells them apart
    assert int(a.state_dict()["output_activation_id"]) == OUTPUT_ACTIVATION_IDS["hardtanh"]
    assert int(b.state_dict()["output_activation_id"]) == OUTPUT_ACTIVATION_IDS["sigmoid"]


def test_unknown_activation_is_refused():
    with pytest.raises(ValueError, match="output_activation"):
        SecondaryGenerator(output_activation="tanh")


def test_sigmoid_output_never_hits_the_bounds_exactly():
    torch.manual_seed(0)
    gen = SecondaryGenerator(output_activation="sigmoid").eval()
    with torch.no_grad():
        z = gen(torch.rand(20_000))
    assert torch.all(z > 0) and torch.all(z < 1)


def test_loading_a_mismatched_checkpoint_kind_raises():
    src = SecondaryGenerator(output_activation="sigmoid")
    dst = SecondaryGenerator(output_activation="hardtanh")
    with pytest.raises(RuntimeError, match="output_activation"):
        dst.load_state_dict(src.state_dict())
    # and the other direction
    with pytest.raises(RuntimeError, match="output_activation"):
        SecondaryGenerator(output_activation="sigmoid").load_state_dict(
            SecondaryGenerator(output_activation="hardtanh").state_dict())


def test_state_dict_without_the_id_buffer_defaults_to_hardtanh():
    """A state dict with no `output_activation_id` buffer is treated as
    Hardtanh: it must load into the hardtanh net, and must be refused by
    a sigmoid net."""
    old = SecondaryGenerator(output_activation="hardtanh").state_dict()
    del old["output_activation_id"]
    SecondaryGenerator(output_activation="hardtanh").load_state_dict(old)  # ok
    with pytest.raises(RuntimeError, match="output_activation"):
        SecondaryGenerator(output_activation="sigmoid").load_state_dict(old)


def test_ctor_kwargs_from_state_dict_recovers_the_activation():
    sd = SecondaryGenerator(output_activation="sigmoid").state_dict()
    assert SecondaryGenerator.ctor_kwargs_from_state_dict(sd)["output_activation"] == "sigmoid"
    del sd["output_activation_id"]
    assert SecondaryGenerator.ctor_kwargs_from_state_dict(sd)["output_activation"] == "hardtanh"


def test_trainer_selects_the_activation_from_the_env(monkeypatch):
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import train
    monkeypatch.delenv("PHINGAN_SEC_OUTPUT_ACT", raising=False)
    assert train.select_output_activation() == config.SECONDARY_OUTPUT_ACTIVATION
    monkeypatch.setenv("PHINGAN_SEC_OUTPUT_ACT", "sigmoid")
    assert train.select_output_activation() == "sigmoid"
    monkeypatch.setenv("PHINGAN_SEC_OUTPUT_ACT", "relu6")
    with pytest.raises(ValueError, match="PHINGAN_SEC_OUTPUT_ACT"):
        train.select_output_activation()


def test_fused_path_refuses_a_sigmoid_terminal():
    from physics.interfaces.fused_mlp import build_spec
    # old fusable widths so the refusal is the activation, not the LN widths
    gen = SecondaryGenerator(n_embedding=25, n_noise=25, n_trunk=50, n_half=25,
                             n_features=1, output_activation="sigmoid").eval()
    with pytest.raises(ValueError, match="Sigmoid"):
        build_spec(gen)


def test_trainer_selects_the_width_from_the_env(monkeypatch):
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import train
    monkeypatch.delenv("PHINGAN_SEC_WIDTH", raising=False)
    assert train.select_width() == config.N_NEURON_MULTIPLIER_SEC
    monkeypatch.setenv("PHINGAN_SEC_WIDTH", "16")
    assert train.select_width() == 16
    monkeypatch.delenv("PHINGAN_NO_PHYS", raising=False)
    monkeypatch.setenv("PHINGAN_SEC_OUTPUT_ACT", "sigmoid")
    gen, _, _, _ = train.select_arm()
    assert SecondaryGenerator.ctor_kwargs_from_state_dict(gen.state_dict()) == {
        "n_embedding": config.N_EMBEDDING_SEC, "n_noise": config.N_NOISE_SEC,
        "n_trunk": 16, "n_half": 8, "n_features": config.N_SECONDARY_FEATURES,
        "output_activation": "sigmoid", "output_stretch_eps": 0.0}
    for bad in ("15", "0", "x"):
        monkeypatch.setenv("PHINGAN_SEC_WIDTH", bad)
        with pytest.raises(ValueError, match="PHINGAN_SEC_WIDTH"):
            train.select_width()


# ---------------------------------------------------------------------------
# "stretched_sigmoid": z = (1 + 2 eps) * sigmoid(x) - eps, unclamped while
# training (the critic sees the overflow and pushes it back), clamped to
# [0, 1] in eval. Motivation: a plain Sigmoid terminal's WGAN gradient
# through sigma'(x) = z(1-z) dies in the tail, so the generator's z can get
# stuck away from 1 no matter how long training continues; a small stretch
# maps that reachable ceiling onto z = 1.
# ---------------------------------------------------------------------------

def test_stretched_sigmoid_is_a_registered_kind_with_its_own_id():
    assert OUTPUT_ACTIVATION_IDS["stretched_sigmoid"] == 2
    assert isinstance(config.SECONDARY_STRETCH_EPS, float) and config.SECONDARY_STRETCH_EPS > 0


def test_stretched_sigmoid_overflows_in_train_and_clamps_in_eval():
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import StretchedSigmoid
    m = StretchedSigmoid(0.01)
    x = torch.tensor([-50.0, 0.0, 50.0])
    y_train = m.train()(x)
    assert torch.allclose(y_train, torch.tensor([-0.01, 0.5, 1.01]), atol=1e-6)
    y_eval = m.eval()(x)
    assert torch.equal(y_eval, torch.tensor([0.0, 0.5, 1.0]))   # exact endpoints reachable
    with pytest.raises(ValueError):
        StretchedSigmoid(0.0)


def test_stretched_generator_records_eps_and_refuses_a_mismatch():
    a = SecondaryGenerator(output_activation="stretched_sigmoid", output_stretch_eps=0.02)
    sd = a.state_dict()
    assert int(sd["output_activation_id"]) == OUTPUT_ACTIVATION_IDS["stretched_sigmoid"]
    assert float(sd["output_stretch_eps"]) == 0.02
    assert set(sd) - set(SecondaryGenerator(output_activation="sigmoid").state_dict()) == set()
    # same eps loads
    SecondaryGenerator(output_activation="stretched_sigmoid", output_stretch_eps=0.02).load_state_dict(sd)
    # different eps refuses
    with pytest.raises(RuntimeError, match="output_stretch_eps"):
        SecondaryGenerator(output_activation="stretched_sigmoid", output_stretch_eps=0.01).load_state_dict(sd)
    # a plain sigmoid net refuses it too
    with pytest.raises(RuntimeError, match="output_activation"):
        SecondaryGenerator(output_activation="sigmoid").load_state_dict(sd)
    # eps buffer is inert for the other kinds: a sigmoid ckpt written before
    # the buffer existed (no key) still loads into a sigmoid net
    old = SecondaryGenerator(output_activation="sigmoid").state_dict()
    del old["output_stretch_eps"]
    SecondaryGenerator(output_activation="sigmoid").load_state_dict(old)


def test_ctor_kwargs_from_state_dict_recovers_the_stretch():
    sd = SecondaryGenerator(output_activation="stretched_sigmoid", output_stretch_eps=0.005).state_dict()
    kw = SecondaryGenerator.ctor_kwargs_from_state_dict(sd)
    assert kw["output_activation"] == "stretched_sigmoid" and kw["output_stretch_eps"] == 0.005
    g = SecondaryGenerator(**kw); g.load_state_dict(sd)


def test_stretched_generator_eval_output_stays_in_unit_interval():
    torch.manual_seed(0)
    gen = SecondaryGenerator(output_activation="stretched_sigmoid", output_stretch_eps=0.02).eval()
    with torch.no_grad():
        z = gen.forward(torch.rand(4096) * 2 - 1)
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0


def test_stretched_generator_is_refused_by_the_fused_kernel_and_falls_back():
    from physics.interfaces.fused_mlp import build_spec, try_enable_fused_forward
    gen = SecondaryGenerator(output_activation="stretched_sigmoid").eval()
    with pytest.raises(ValueError):
        build_spec(gen)
    try_enable_fused_forward(gen)   # must not raise
    with torch.inference_mode():
        assert gen.forward(torch.full((8,), 0.5)).shape == (8, 1)


def test_trainer_selects_the_stretch_from_the_env(monkeypatch):
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import train
    monkeypatch.delenv("PHINGAN_SEC_STRETCH_EPS", raising=False)
    assert train.select_stretch_eps() == config.SECONDARY_STRETCH_EPS
    monkeypatch.setenv("PHINGAN_SEC_STRETCH_EPS", "0.02")
    assert train.select_stretch_eps() == 0.02
    monkeypatch.setenv("PHINGAN_SEC_STRETCH_EPS", "0")
    with pytest.raises(ValueError):
        train.select_stretch_eps()
    monkeypatch.setenv("PHINGAN_SEC_OUTPUT_ACT", "stretched_sigmoid")
    assert train.select_output_activation() == "stretched_sigmoid"


# ---------------------------------------------------------------------------
# "hardtanh_open_top": Hardtanh's lower clip at 0 in BOTH training and eval,
# the upper clip at 1 in eval ONLY. Motivation: a bounded terminal (sigmoid,
# stretched sigmoid) puts a ceiling somewhere below 1 because nothing in the
# objective rewards the last fraction of the delta-ray spectrum; an open
# top lets the density run to (and past) z = 1 during training, and the
# inference clip collects the spill.
# ---------------------------------------------------------------------------

def test_hardtanh_open_top_is_a_registered_kind_with_its_own_id():
    assert OUTPUT_ACTIVATION_IDS["hardtanh_open_top"] == 3


def test_hardtanh_open_top_clips_bottom_always_and_top_only_in_eval():
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import HardtanhOpenTop
    m = HardtanhOpenTop()
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.7])
    assert torch.equal(m.train()(x), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.7]))
    assert torch.equal(m.eval()(x),  torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    # gradient: dead below 0 (as Hardtanh), live above 1 in train mode
    xg = torch.tensor([-0.5, 0.5, 1.7], requires_grad=True)
    m.train()(xg).sum().backward()
    assert xg.grad.tolist() == [0.0, 1.0, 1.0]


def test_hardtanh_open_top_generator_records_kind_and_refuses_the_others():
    a = SecondaryGenerator(output_activation="hardtanh_open_top")
    sd = a.state_dict()
    assert int(sd["output_activation_id"]) == OUTPUT_ACTIVATION_IDS["hardtanh_open_top"]
    assert float(sd["output_stretch_eps"]) == 0.0
    SecondaryGenerator(output_activation="hardtanh_open_top").load_state_dict(sd)
    for other in ("hardtanh", "sigmoid", "stretched_sigmoid"):
        with pytest.raises(RuntimeError, match="output_activation"):
            SecondaryGenerator(output_activation=other).load_state_dict(sd)
    assert SecondaryGenerator.ctor_kwargs_from_state_dict(sd)["output_activation"] == "hardtanh_open_top"


def test_hardtanh_open_top_generator_eval_output_stays_in_unit_interval():
    torch.manual_seed(0)
    gen = SecondaryGenerator(output_activation="hardtanh_open_top").eval()
    with torch.no_grad():
        z = gen.forward(torch.rand(4096) * 2 - 1)
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0


def test_hardtanh_open_top_generator_is_refused_by_the_fused_kernel_and_falls_back():
    from physics.interfaces.fused_mlp import build_spec, try_enable_fused_forward
    gen = SecondaryGenerator(output_activation="hardtanh_open_top").eval()
    with pytest.raises(ValueError):
        build_spec(gen)
    try_enable_fused_forward(gen)
    with torch.inference_mode():
        assert gen.forward(torch.full((8,), 0.5)).shape == (8, 1)


def test_trainer_selects_hardtanh_open_top_from_the_env(monkeypatch):
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import train
    monkeypatch.setenv("PHINGAN_SEC_OUTPUT_ACT", "hardtanh_open_top")
    assert train.select_output_activation() == "hardtanh_open_top"


# ---------------------------------------------------------------------------
# "hardtanh_reflect_top": identical to hardtanh_open_top in TRAINING
# (clamp(min=0), top open); in EVAL the spill past 1 is REFLECTED
# (z > 1 -> 2 - z) before the [0, 1] clamp instead of piled onto z == 1.
# Reflecting the spill rather than clamping it spreads that probability
# mass into the tail below 1, rather than stacking it into a spurious
# z == 1 Dirac atom. An open_top checkpoint is the SAME net at training
# time, so it loads into a reflect net (one-directional compatibility).
# ---------------------------------------------------------------------------

def test_hardtanh_reflect_top_is_a_registered_kind_with_its_own_id():
    assert OUTPUT_ACTIVATION_IDS["hardtanh_reflect_top"] == 4


def test_hardtanh_reflect_top_reflects_in_eval_and_is_open_in_train():
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import HardtanhReflectTop
    m = HardtanhReflectTop()
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.008, 2.5])
    assert torch.equal(m.train()(x), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.008, 2.5]))
    y = m.eval()(x)
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 0.5, 1.0, 0.992, 0.0]))
    assert float(y.max()) <= 1.0


def test_open_top_checkpoint_loads_into_a_reflect_net():
    src = SecondaryGenerator(output_activation="hardtanh_open_top")
    dst = SecondaryGenerator(output_activation="hardtanh_reflect_top")
    dst.load_state_dict(src.state_dict())   # compat: same net at training time
    # and the adopted buffer records the built net's kind, not the source's
    assert int(dst.state_dict()["output_activation_id"]) == OUTPUT_ACTIVATION_IDS["hardtanh_reflect_top"]


def test_reflect_net_refuses_every_other_kind_and_is_not_loadable_back():
    for other in ("hardtanh", "sigmoid", "stretched_sigmoid"):
        with pytest.raises(RuntimeError, match="output_activation"):
            SecondaryGenerator(output_activation="hardtanh_reflect_top").load_state_dict(
                SecondaryGenerator(output_activation=other).state_dict())
    # the compat is one-directional: a reflect checkpoint does NOT load into
    # an open_top net (the eval semantics the checkpoint was picked under
    # would silently change back to the atom-producing clamp)
    with pytest.raises(RuntimeError, match="output_activation"):
        SecondaryGenerator(output_activation="hardtanh_open_top").load_state_dict(
            SecondaryGenerator(output_activation="hardtanh_reflect_top").state_dict())


def test_config_default_is_hardtanh_reflect_top_and_predict_stays_in_unit_interval():
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import HardtanhReflectTop
    assert config.SECONDARY_OUTPUT_ACTIVATION == "hardtanh_reflect_top"
    gen = SecondaryGenerator().eval()
    assert isinstance(_terminal(gen), HardtanhReflectTop)
    torch.manual_seed(0)
    with torch.no_grad():
        z = gen.forward(torch.rand(4096) * 2 - 1)
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0


def test_hardtanh_reflect_top_packs_in_the_fused_kernel():
    """HardtanhReflectTop is accepted (clamp_kind=2), unlike
    Sigmoid/StretchedSigmoid/HardtanhOpenTop, which still refuse (see
    test_fused_path_refuses_a_sigmoid_terminal,
    test_stretched_generator_is_refused_by_the_fused_kernel_and_falls_back,
    test_hardtanh_open_top_generator_is_refused_by_the_fused_kernel_and_falls_back).
    On CPU try_enable_fused_forward still refuses on the device check
    (fusion is CUDA-only), so the eager forward keeps working regardless."""
    from physics.interfaces.fused_mlp import build_spec, try_enable_fused_forward
    gen = SecondaryGenerator(output_activation="hardtanh_reflect_top").eval()
    spec = build_spec(gen)
    assert spec.clamp_kind == 2
    try_enable_fused_forward(gen)
    with torch.inference_mode():
        assert gen.forward(torch.full((8,), 0.5)).shape == (8, 1)
