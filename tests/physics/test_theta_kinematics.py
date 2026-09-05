"""The closed-form primary deflection for delta-ray production.

Reference values are fp64 evaluations of the same kinematics with the
`physics/definitions` constants (M = 938272013.0 eV, m_e = 510998.91 eV),
computed independently of the implementation.
"""
import math

import pytest
import torch

from physics.g4h_ionisation.kinematics import compute_primary_theta

M_PROTON_eV = 938272013.0
M_ELECTRON_eV = 510998.91

# (post-continuous KE [eV], delta-ray KE [eV], theta [rad])
# t_max at T0 = 1e8 eV is 229179.44 eV, so every T below is physical.
REFERENCE = [
    (1.0e8, 0.0,      0.0),
    (1.0e8, 5.0e3,    1.590296900691e-04),
    (1.0e8, 5.0e4,    4.497043726863e-04),
    (1.0e8, 1.0e5,    5.401431534669e-04),
    (1.0e8, 2.0e5,    3.632403738239e-04),
]


def _theta(t0, t, dtype):
    return compute_primary_theta(
        torch.tensor([t0], dtype=dtype),
        torch.tensor([t], dtype=dtype),
        M_PROTON_eV, M_ELECTRON_eV,
    ).item()


@pytest.mark.parametrize("t0,t,expected", REFERENCE)
def test_matches_the_fp64_reference_in_float64(t0, t, expected):
    got = _theta(t0, t, torch.float64)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-18)


@pytest.mark.parametrize("t0,t,expected", REFERENCE)
def test_matches_the_fp64_reference_in_float32(t0, t, expected):
    # The whole point of atan2 over acos: no fp64 is needed at runtime.
    # float32 carries ~7 significant digits, so rel=1e-6 is a real bound,
    # not a loosened one.
    got = _theta(t0, t, torch.float32)
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_zero_delta_energy_gives_exactly_zero_theta():
    # No secondary produced => no deflection, as an emergent property of the
    # formula rather than an imposed mask. G4hIonisation.post_step_do_it
    # relies on this to justify deleting its theta masks (see Task 3).
    for dtype in (torch.float32, torch.float64):
        out = compute_primary_theta(
            torch.full((1024,), 1.0e8, dtype=dtype),
            torch.zeros(1024, dtype=dtype),
            M_PROTON_eV, M_ELECTRON_eV,
        )
        assert (out == 0).all(), f"nonzero theta at e_sec=0 in {dtype}"


def test_float32_resolves_small_angles_that_acos_would_destroy():
    # THE ANTI-acos REGRESSION. acos(p1_z/|p1|) in float32 returns exactly 0
    # here, because cos(1.59e-4) differs from 1.0 by 1.3e-8 while float32's
    # spacing below 1.0 is 6e-8. atan2 never forms that near-1 number.
    # If this test fails with got == 0.0, someone reintroduced acos.
    t0, t, expected = 1.0e8, 5.0e3, 1.590296900691e-04
    got = _theta(t0, t, torch.float32)
    assert got > 0.0, "small-angle theta collapsed to zero -- acos regression"
    assert got == pytest.approx(expected, rel=1e-6)


def test_acos_formulation_really_does_fail_here():
    # Negative control for the test above: proves the anti-acos assertion is
    # discriminating rather than vacuous. If this ever passes, float32
    # acos has stopped being lossy at this angle, which would make the
    # regression test above vacuous.
    t0, t = 1.0e8, 5.0e3
    M, me = M_PROTON_eV, M_ELECTRON_eV
    p0 = math.sqrt(t0 * (t0 + 2 * M))
    pe = math.sqrt(t * (t + 2 * me))
    cos_e = min(1.0, t * (t0 + M + me) / (pe * p0))
    sin_e = math.sqrt(max(0.0, 1 - cos_e * cos_e))
    z = torch.tensor(p0 - pe * cos_e, dtype=torch.float32)
    p = torch.tensor(pe * sin_e, dtype=torch.float32)
    mag = torch.sqrt(z * z + p * p)
    acos_theta = torch.acos(torch.clamp(z / mag, max=1.0)).item()
    assert acos_theta == 0.0, (
        "float32 acos does not destroy this angle; the anti-acos regression "
        "test has stopped discriminating")


def test_preserves_input_dtype_and_shape():
    for dtype in (torch.float32, torch.float64):
        out = compute_primary_theta(
            torch.full((7, 3), 1.0e8, dtype=dtype),
            torch.full((7, 3), 5.0e4, dtype=dtype),
            M_PROTON_eV, M_ELECTRON_eV,
        )
        assert out.dtype == dtype
        assert out.shape == (7, 3)


def test_is_scriptable_so_dynamo_can_inline_it():
    # torch 2.5 dynamo traces THROUGH ScriptFunction callees, which is how
    # this kernel fuses into the compiled post_step_do_it graph.
    assert isinstance(compute_primary_theta, torch.jit.ScriptFunction)


def test_the_pinned_constants_are_the_definitions_pair():
    # physics/definitions is the repo's ONLY constant set. Pinning these
    # values here means a wrong or duplicated constant set would surface
    # as a mismatch against PARTICLE.mass_eV / electron_mass_eV rather
    # than as a silent numerical drift downstream. If a G4 version bump
    # changes CLHEP's values, update physics/definitions AND these
    # numbers together.
    from common.run_context import PARTICLE
    from physics.definitions.physical_constants import electron_mass_eV

    assert PARTICLE.mass_eV == M_PROTON_eV
    assert electron_mass_eV == M_ELECTRON_eV
