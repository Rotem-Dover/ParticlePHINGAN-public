"""
Spline interpolator tests. Verifies the cubic-spline port reproduces values
at node points exactly and interpolates smoothly between them.
"""
import math

import torch

from physics.g4_init_tables.grid import log_energy_grid
from physics.g4_init_tables.interpolator import G4PhysicsVector


def test_exact_at_nodes_no_spline():
    e = log_energy_grid(1.0, 1000.0, n=21)
    v = torch.log(e)
    pv = G4PhysicsVector(e, v, log_grid=True, use_spline=False)
    out = pv(e)
    assert torch.allclose(out, v, atol=1e-12)


def test_exact_at_nodes_with_spline():
    e = log_energy_grid(1.0, 1000.0, n=21)
    v = torch.log(e)
    pv = G4PhysicsVector(e, v, log_grid=True, use_spline=True)
    out = pv(e)
    # at nodes b∈{0,1}, the spline correction term is b(b-1)*(...)=0 — exact
    assert torch.allclose(out, v, atol=1e-12)


def test_spline_better_than_linear_on_curved():
    """On a curved function the spline beats linear at off-node points."""
    e = log_energy_grid(1.0, 100.0, n=21)
    v = torch.sqrt(e)  # curved in linear space
    pv_lin = G4PhysicsVector(e, v, log_grid=True, use_spline=False)
    pv_spl = G4PhysicsVector(e, v, log_grid=True, use_spline=True)
    probe = torch.tensor([1.5, 7.3, 12.1, 55.5, 91.0], dtype=torch.float64)
    truth = torch.sqrt(probe)
    err_lin = (pv_lin(probe) - truth).abs().max()
    err_spl = (pv_spl(probe) - truth).abs().max()
    assert err_spl < err_lin


def test_clamp_outside():
    e = log_energy_grid(1.0, 100.0, n=11)
    v = torch.full_like(e, 3.0)
    pv = G4PhysicsVector(e, v)
    out = pv(torch.tensor([0.1, 200.0], dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([3.0, 3.0], dtype=torch.float64))
