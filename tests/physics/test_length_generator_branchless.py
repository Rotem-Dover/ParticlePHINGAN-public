"""The step-length stage must interpolate branch-free, and it must evaluate
the built init tables' cubic spline rather than a linear resample of them.

A `.any()`-guarded masked write is a data-dependent branch: `torch.compile`
cannot trace through it, and each such write forces a blocking GPU-to-CPU
sync (`aten::nonzero`). `PhysicsLengthGenerator.forward` calls
`InitTableSet.range` / `InitTableSet.sigma_ion_biased` directly: `g4pv_eval`
(`physics/g4_init_tables/interpolator.py`, a port of GEANT4's
`G4PhysicsVector::Value`) is branch-free -- out-of-range queries clamp via
`torch.where`, never `if mask.any():` -- and its boundary constants
(`log_emin`, `inv_dbin_log`) are Python floats precomputed once at
table-load time rather than tensors read inside the traced region, since
`float(tensor)` inside a traced region is itself a scalar-output sync.

Evaluating the tables' spline rather than a linear resample of the same
121 sparse nodes matters because GEANT4 itself evaluates them with that
same cubic spline; a linear interpolant over such sparse nodes disagrees
with it systematically.

These tests pin:

* the generator evaluates the tables' spline (`use_spline=True`), not a
  linear resample of them -- and that this is not a no-op, i.e. the spline
  actually disagrees with a naive linear interpolant between the same nodes;
* `forward` traces into a single graph with zero breaks;
* the table vectors' boundary constants are Python floats, not tensors.
  Leaving them as tensors reintroduces a graph break of its own, because
  `float(tensor)` inside a traced region is a scalar-output sync.
"""
import torch
import torch._dynamo as dynamo

from physics.g4h_ionisation.generators.length_generator.physics_mc import (
    PhysicsLengthGenerator)

BEAM_ENERGY_eV = 1e8


def _gen():
    return PhysicsLengthGenerator()


def test_generator_evaluates_the_tables_spline_not_a_linear_resample():
    gen = _gen()
    assert gen.tables.range.use_spline is True
    assert gen.tables.sigma_ion.use_spline is True

    # Midpoint between two knots: a real spline correction term is nonzero
    # there (the `use_spline` branch in g4pv_eval adds a cubic term keyed on
    # `sec_deriv`), so the spline value must disagree with the naive linear
    # interpolant between the same two knots. A regression back to linear
    # interpolation would silently collapse this gap to ~0.
    bin_x = gen.tables.range.bin
    i = bin_x.numel() // 2
    x_lo, x_hi = bin_x[i], bin_x[i + 1]
    midpoint = (x_lo + x_hi) / 2

    spline_val = gen.tables.range(midpoint.unsqueeze(0))
    y_lo, y_hi = gen.tables.range.data[i], gen.tables.range.data[i + 1]
    linear_val = (y_lo + y_hi) / 2  # midpoint of a linear interpolant

    rel_gap = ((spline_val - linear_val).abs() / linear_val.abs()).item()
    assert rel_gap > 1e-4, (
        f"spline and linear interpolation agree to {rel_gap:.2e} at a table "
        f"midpoint -- suspicious for a cubic-spline table; the generator may "
        f"have regressed to linear interpolation")


def test_forward_traces_into_one_graph_with_no_breaks():
    gen = _gen()
    dynamo.reset()
    explanation = dynamo.explain(gen.forward)(torch.full((4096,), BEAM_ENERGY_eV))
    reasons = "\n".join(
        f"  {r.reason} @ {r.user_stack[-1].filename}:{r.user_stack[-1].lineno}"
        for r in explanation.break_reasons)
    assert explanation.graph_break_count == 0, (
        f"the step-length stage graph-broke {explanation.graph_break_count} "
        f"time(s):\n{reasons}")
    assert explanation.graph_count == 1, (
        f"expected a single fused graph, got {explanation.graph_count}")


def test_table_boundary_constants_are_python_floats():
    gen = _gen()
    constants = {
        "range.log_emin": gen.tables.range.log_emin,
        "range.inv_dbin_log": gen.tables.range.inv_dbin_log,
        "sigma_ion.log_emin": gen.tables.sigma_ion.log_emin,
        "sigma_ion.inv_dbin_log": gen.tables.sigma_ion.inv_dbin_log,
    }
    for name, value in constants.items():
        assert type(value) is float, (
            f"{name} is {type(value).__name__}, not float -- a tensor here "
            f"forces a `float(tensor)` scalar-output sync inside the traced "
            f"region, reintroducing the graph break this stage exists to avoid")


def test_forward_still_respects_the_two_stepping_limits():
    """The sampled step is min(energy-loss limit, discrete limit) and positive.

    A branch-free rewrite that dropped one of the two `torch.minimum` arms
    would still trace cleanly and still evaluate through the spline tables,
    so the physics contract is pinned separately.
    """
    gen = _gen()
    torch.manual_seed(20260729)
    ke = torch.full((8192,), BEAM_ENERGY_eV)
    step_m = gen.forward(ke)

    assert step_m.shape == ke.shape
    assert torch.isfinite(step_m).all(), "step length must be finite"
    assert (step_m > 0).all(), "step length must be strictly positive"

    # The energy-loss limit is deterministic, so it is a hard per-lane ceiling
    # regardless of the discrete draw. Built table's ordinate is already
    # centimetres -- no unit conversion here (see physics_mc.py).
    import common.units as U
    r_cm = gen.tables.range(ke)
    f, d_r_over_range = 0.01, 0.1
    l_eloss_cm = torch.where(
        r_cm <= f, r_cm,
        r_cm * d_r_over_range + f * (1 - d_r_over_range) * (2 - f / r_cm))
    assert (step_m <= l_eloss_cm * U.cm2m).all(), (
        "a sampled step exceeded the deterministic energy-loss limit")
