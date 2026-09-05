"""The process's stage-by-stage behavior: energy budget, theta reconstruction,
and post-step masking."""
import torch

from common.run_context import PARTICLE
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4h_ionisation.kinematics import compute_primary_theta
from benchmark.track_simulator_config import build_stepping_manager


def _process():
    return build_stepping_manager("phin_gan", device="cpu").processes[0]


def test_process_competes_post_step():
    assert _process().competes_post_step is True


def test_along_step_returns_only_e_cnt():
    p = _process()
    ke = torch.full((512,), 1e8)
    out = p.along_step_do_it(ke, p.compute_step_limit(ke))
    assert set(out) == {"e_cnt"}
    assert out["e_cnt"].shape == ke.shape


def test_along_step_never_removes_more_than_the_available_energy():
    p = _process()
    ke = torch.full((4096,), 1e6)
    e_cnt = p.along_step_do_it(ke, p.compute_step_limit(ke))["e_cnt"]
    assert (e_cnt <= ke + 1e-6).all()
    assert (e_cnt >= 0).all()


def test_post_step_theta_is_the_energy_momentum_reconstruction():
    # theta is computed from energy-momentum conservation
    # (compute_primary_theta), not read off a network output column. Since
    # post_step_do_it and this test call the same kernel on the same
    # inputs, the deviation is exactly zero by construction, so this
    # asserts bit-level agreement rather than a tolerance.
    p = _process()
    ke = torch.full((8192,), 1e8)
    sl = p.compute_step_limit(ke)
    e_cnt = p.along_step_do_it(ke, sl)["e_cnt"]
    out = p.post_step_do_it(ke, sl, e_cnt=e_cnt)
    nz = out["e_sec"] > 0
    assert nz.sum() > 10, "no secondaries produced; test is vacuous"

    expected = compute_primary_theta(
        (ke - e_cnt)[nz], out["e_sec"][nz], PARTICLE.mass_eV, electron_mass_eV)
    assert torch.equal(out["theta"][nz], expected)


def test_post_step_zero_e_sec_implies_zero_theta():
    p = _process()
    ke = torch.full((8192,), 1e8)
    sl = p.compute_step_limit(ke)
    e_cnt = p.along_step_do_it(ke, sl)["e_cnt"]
    out = p.post_step_do_it(ke, sl, e_cnt=e_cnt)
    zero = out["e_sec"] == 0
    assert (out["theta"][zero] == 0).all()


def test_won_mask_suppresses_the_vertex():
    p = _process()
    ke = torch.full((1024,), 1e8)
    sl = p.compute_step_limit(ke)
    e_cnt = p.along_step_do_it(ke, sl)["e_cnt"]
    won = torch.zeros_like(ke, dtype=torch.bool)
    out = p.post_step_do_it(ke, sl, e_cnt=e_cnt, won=won)
    assert (out["e_sec"] == 0).all() and (out["theta"] == 0).all()
