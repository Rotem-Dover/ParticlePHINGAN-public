"""The (E [MeV], L [um]) phase-space points figs 8 and 9 mark and sample.

The paper's four aluminium points span the straggling regimes: very thin
(1 MeV, 1e-3 um), thin with Gaussian excitation (10 MeV, 0.5 um), thick
(30 MeV, 10 um) and the low-energy end (0.7 MeV, 1 um). The same regimes
are marked for the other materials, moved where aluminium's coordinates
leave the explicit-physics models' domain: in iron the delta-ray threshold
Tc = 5.6 keV exceeds t_max below about 2.5 MeV (no delta rays, and GEANT4
stops the proton in one step there), so the low-energy points move up to
3-5 MeV; in beryllium the thin-limit excitation PDF has no support at
(10 MeV, 0.5 um), so that point moves to a thinner step.
"""
from common.run_context import TARGET_MATERIAL

MARKED_POINTS = {
    "aluminum": [
        (1.0, 1e-3),   # 1 MeV, 1e-3 um  -- very thin
        (10.0, 0.5),   # 10 MeV, 0.5 um  -- thin, Gaussian excitation
        (30.0, 10.0),  # 30 MeV, 10 um   -- thick
        (0.7, 1.0),    # 0.7 MeV, 1 um   -- low energy
    ],
    "iron": [
        (5.0, 1e-3),
        (10.0, 0.5),
        (30.0, 10.0),
        (3.0, 1.0),
    ],
    "beryllium": [
        (1.0, 1e-3),
        (10.0, 0.05),
        (30.0, 10.0),
        (0.7, 1.0),
    ],
}


def marked_points() -> list[tuple[float, float]]:
    """The active beam's four (E [MeV], L [um]) points."""
    return MARKED_POINTS[TARGET_MATERIAL.name.lower()]
