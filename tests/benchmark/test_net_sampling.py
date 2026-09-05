"""net_sampling exposes generate_continuous / generate_secondary as
convenience samplers over the phin_gan preset's G4hIonisation process, for
the figure scripts. The secondary sampler's output is (N, 1): one column,
e_sec."""
import pytest
import torch

from benchmark.system_vs_g4.net_sampling import (
    generate_continuous, generate_secondary, phin_gan_ionisation,
)


@pytest.fixture(scope="module")
def proc():
    return phin_gan_ionisation(device="cpu")


def test_generate_continuous_uncorrected_is_bare_predict(proc):
    # With both corrections off the function must be exactly the generator's
    # own predict (plus the never-triggering slow-kill clamp) -- it adds no
    # hidden transformation of its own.
    ps = torch.tensor([[5e6, 1e-6]])
    torch.manual_seed(0)
    a = generate_continuous(proc, ps.clone(), correct_zero=False,
                            correct_deterministic=False, repeat_interleave=64)
    torch.manual_seed(0)
    b = proc.continuous_generator.predict(ps.clone(), repeat_interleave=64)
    assert torch.equal(a, b)


def test_generate_continuous_refuses_zero_without_deterministic(proc):
    with pytest.raises(ValueError):
        generate_continuous(proc, torch.tensor([[5e6, 1e-6]]),
                            correct_zero=True, correct_deterministic=False)


def test_deterministic_point_returns_bethe_bloch_mean(proc):
    # 5 MeV / 1e-10 m sits in the deterministic (NaN-lookup) region: the raw
    # network output gathers NaN there by design, and the deterministic
    # correction must overwrite every draw with the analytic mean -- the same
    # torch.where semantics as G4hIonisation.along_step_do_it. The average
    # loss at this (KE, L) point is below G4CONSTS.min_energy_loss (10 eV),
    # which is what puts it in the deterministic region and makes the raw
    # predict() output NaN there.
    ps = torch.tensor([[5e6, 1e-10]])
    out = generate_continuous(proc, ps.clone(), correct_zero=False,
                              repeat_interleave=8)
    avg = proc.compute_average_loss(ps[:, 0], ps[:, 1])
    assert torch.isfinite(out).all()
    assert torch.allclose(out, avg.expand(8), rtol=1e-5)


def test_generate_secondary_shape_and_masking(proc):
    ke = torch.tensor([5e6, 2e6])
    out = generate_secondary(proc, ke, e_cnt=torch.zeros(2),
                             correct_zero=True, repeat_interleave=500)
    assert out.shape == (1000, 1)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    nz = out[out > 0]
    # a delta ray is bounded below by the production threshold (~1 keV in Al)
    assert nz.numel() == 0 or nz.min() > 1e3
