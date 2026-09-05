"""
Tests for Interp1dLinear, a differentiable 1-D linear interpolation utility.
Covers bounds errors, extrapolation, clamping, fill values, numerical agreement
with scipy.interpolate.interp1d, and gradient correctness via torch.autograd.gradcheck.
"""
import pytest
import torch
import scipy
from torch.autograd import gradcheck

from common.utils import Interp1dLinear


def test_Interp1dLinear_bounds_error():
    x     = torch.tensor([0., 1., 2., 3., 4.])
    y     = torch.tensor([0., 1., 4., 9., 16.])
    x_new = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    interp = Interp1dLinear(x, y, bounds_error=True)
    with pytest.raises(ValueError):
        interp(x_new)


def test_Interp1dLinear_extrapolate():
    x     = torch.tensor([0., 1., 2., 3., 4.])
    y     = torch.tensor([0., 1., 4., 9., 16.])
    x_new = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    interp = Interp1dLinear(x, y)
    y_new  = interp(x_new)
    assert torch.allclose(y_new, torch.tensor([-0.5, 0.5, 2.5, 6.5, 12.5, 19.5]))


def test_Interp1dLinear_clamp_values():
    x     = torch.tensor([0., 1., 2., 3., 4.])
    y     = torch.tensor([0., 1., 4., 9., 16.])
    x_new = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    interp = Interp1dLinear(x, y, clamp_values=(0., 16.), do_clamp=True)
    y_new  = interp(x_new)
    assert torch.allclose(y_new, torch.tensor([0, 0.5, 2.5, 6.5, 12.5, 16.]))


def test_Interp1dLinear_fill_values():
    x = torch.tensor([0., 1., 2., 3., 4.])
    y = torch.tensor([0., 1., 4., 9., 16.])
    x_new = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    interp = Interp1dLinear(x, y, fill_values=(0., 16.), do_fill=True)
    y_new = interp(x_new)
    assert torch.allclose(y_new, torch.tensor([0, 0.5, 2.5, 6.5, 12.5, 16.]))


def test_Interp1dLinear_to_numpy():
    x     = torch.arange(0, 10, 1,     dtype=torch.float64)
    x_new = torch.arange(-10, 10, 0.1, dtype=torch.float64)

    for _ in range(1_000):
        y        = torch.randn(len(x))
        interp   = Interp1dLinear(x, y)
        output   = interp(x_new)
        expected = scipy.interpolate.interp1d(
            x.numpy(), y.numpy(), bounds_error=False, fill_value='extrapolate')(x_new.numpy())
        expected = torch.from_numpy(expected)
        assert torch.allclose(output, expected, atol=1e-7)


def test_Interp1dLinear_grad_in_bounds():
    # Define some small sample data for testing
    x = torch.linspace(-10, 10, steps=5, dtype=torch.double, requires_grad=False)
    y = torch.sigmoid(x)  # A simple function to interpolate

    # Instantiate the Interp1dLinear object
    interp_fn = Interp1dLinear(x, y, fill_values=(0., 1.))

    for _ in range(1_000):
        # Generate some random x_new values
        x_new = torch.randn(10, dtype=torch.double, requires_grad=True)

        # Verify that there is no value in x_new that is equal to any value in x
        for x_val in x_new:
            if torch.any(torch.isclose(x_val, x, atol=1e-5)):
                continue

        # Perform the gradient check
        is_compatible = gradcheck(interp_fn, (x_new,))

        assert is_compatible


def test_Interp1dLinear_grad_out_of_bounds():
    # Define some small sample data for testing
    x = torch.linspace(-10, 10, steps=5, dtype=torch.double, requires_grad=False)
    y = torch.sigmoid(x)  # A simple function to interpolate

    # Instantiate the Interp1dLinear object
    interp_fn = Interp1dLinear(x, y, fill_values=(0., 1.))

    for i in range(1_000):
        shift = -10 if i < 5_000 else 10

        # Generate some random x_new values
        x_new = shift + torch.randn(1, dtype=torch.double, requires_grad=True)

        # Verify that there is no value in x_new that is equal to any value in x
        for x_val in x_new:
            if torch.any(torch.isclose(x_val, x, atol=1e-5)):
                continue

        # Perform the gradient check
        is_compatible = gradcheck(interp_fn, (x_new,))#, atol=1e-4, eps=1e-10)
        assert is_compatible


# TODO: This is not passing at this moment. Slope is not well defined when x_new is equal to x
@pytest.mark.xfail(reason="Gradient undefined when x_new coincides with knot points")
def test_Interp1dLinear_grad_x_new_equals_x():
    # Define some small sample data for testing
    x = torch.linspace(-10, 10, steps=5, dtype=torch.double, requires_grad=False)
    y = torch.sigmoid(x)  # A simple function to interpolate
    x_new = x.clone().detach().requires_grad_(True)

    # Instantiate the Interp1dLinear object
    interp_fn = Interp1dLinear(x, y)

    # Perform the gradient check
    is_compatible = gradcheck(interp_fn, (x_new,))

    assert is_compatible

