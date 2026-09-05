"""
Tests for the differentiable Kolmogorov-Smirnov distance implementation, verifying
correctness for Gaussian and uniform distributions and checking that gradients
propagate correctly through the KS statistic via PyTorch autograd.
"""
import torch

from torch.autograd import gradcheck

from common.utils import Interp1dLinear
from benchmark.ks_stats import kolmogorov_smirnov_distance, _get_empirical_cdf_values


def __samples_to_callable_cdf(samples):
    sorted_samples, _ = torch.sort(samples)
    y = _get_empirical_cdf_values(sorted_samples)
    return Interp1dLinear(sorted_samples, y, clamp_values=(0.0, 1.0), do_clamp=True)


def test_kstest_pytorch_gauss():
    samples_for_theoretical_cdf = torch.randn(10_000_000)
    theoretical_cdf             = __samples_to_callable_cdf(samples_for_theoretical_cdf)
    data = torch.randn(100_000)

    distance = kolmogorov_smirnov_distance(data, theoretical_cdf)

    assert torch.isclose(distance, torch.tensor(0.0, dtype=distance.dtype), atol=1e-2)


def test_kstest_pytorch_uniform():
    samples_for_theoretical_cdf = torch.rand(10_000_000, dtype=torch.float32)
    theoretical_cdf             = __samples_to_callable_cdf(samples_for_theoretical_cdf)
    data = torch.randn(100_000, dtype=torch.float32)

    distance = kolmogorov_smirnov_distance(data, theoretical_cdf)

    assert torch.isclose(distance, torch.tensor(0.5, dtype=distance.dtype), atol=1e-2)


def test_kstest_pytorch_backprop():
    samples_for_theoretical_cdf = torch.randn(100_000, dtype=torch.float64, requires_grad=False)
    theoretical_cdf             = __samples_to_callable_cdf(samples_for_theoretical_cdf)

    data = torch.randn(1_000, dtype=torch.float64, requires_grad=True)
    rand_locs = torch.randint(2, size=(data.shape[0], ), dtype=torch.bool)

    distance = kolmogorov_smirnov_distance(data[rand_locs], theoretical_cdf)
    distance.backward()

    assert data.grad is not None


def test_kstest_pytorch_gradcheck():
    # Seeded: the KS statistic is a max over a piecewise-linear ECDF, so it
    # is non-differentiable at argmax ties and interpolation knots. With
    # unseeded data ~10% of RNG states land a sample near such a point and
    # gradcheck legitimately fails (numeric vs analytic Jacobian mismatch),
    # which made this test order-dependent in full-suite runs.
    torch.manual_seed(0)
    samples_for_theoretical_cdf = torch.randn(100_000, dtype=torch.float64, requires_grad=False)
    theoretical_cdf             = __samples_to_callable_cdf(samples_for_theoretical_cdf)

    data = torch.randn(1_000, dtype=torch.float64, requires_grad=True)

    assert gradcheck(kolmogorov_smirnov_distance, (data, theoretical_cdf))
