"""Closed-form kinematics for delta-ray (secondary electron) production.

The primary's deflection is NOT an independent random variable: once the
delta-ray's kinetic energy is drawn, energy-momentum conservation determines
the scattering angle exactly. This mirrors GEANT4, where
`G4BetheBlochModel::SampleSecondaries` samples only the secondary energy and
the primary's `angleDiscrete` follows from the two-body kinematics.

    p0      = sqrt(T0 (T0 + 2M))         primary momentum before the vertex
    pe      = sqrt(T  (T  + 2 m_e))      delta-ray momentum
    cos t_e = T (T0 + M + m_e) / (pe p0) delta-ray angle to the primary axis
    p1_z    = p0 - pe cos t_e            primary's longitudinal momentum after
    p1_perp = pe sin t_e                 primary's transverse momentum after
    theta   = atan2(p1_perp, p1_z)

`T0` is the kinetic energy entering the DISCRETE vertex, i.e. AFTER the
along-step continuous loss -- GEANT4 produces the delta-ray after the
continuous loss. Passing the pre-step energy instead mismatches G4's recorded
`angleDiscrete` by ~1e-4 relative rather than ~1e-8.

WHY atan2 AND NOT acos
----------------------
The obvious form, `acos(p1_z / |p1|)`, is catastrophically ill-conditioned at
the deflections this beam produces (~1e-4 rad). cos(1e-4) = 0.999999995 differs
from 1.0 by 5e-9, while consecutive float32 values just below 1.0 are
2**-24 ~ 6e-8 apart -- eight times that gap -- so the value rounds to exactly
1.0 and acos returns 0. Systematically, the smallest angle expressible that way
is sqrt(2 * 2**-24) ~ 3.5e-4 rad, LARGER than a typical deflection: the whole
distribution would collapse to a spike at zero with the survivors snapped onto
a coarse ladder, reading as "the model predicts no scattering".

The angle atan2 RETURNS never forms a number near 1.0 -- `cos_e` itself does
sit near 1.0 at the kinematic endpoint (see "A second, benign path to
theta == 0" below), and `sin_e = sqrt(1 - cos_e**2)` inherits the same
cancellation acos would suffer there. But atan2 is not fed that ratio: its
two arguments are `p1_perp` and `p1_z`, which are free of cancellation
(`p1_z` subtracts a small momentum from a large one; `p1_perp` is a product),
their ratio ~1e-4 carries full float32 relative precision, and
atan2(y, x) ~ y/x is well-conditioned there. Consequences:

  * no float64 is needed at runtime -- which also keeps fp64 transcendentals,
    markedly slower per lane than float32 ones on GPU hardware, out of the
    hot loop;
  * it stays quadrant-correct for ANY primary mass. `asin(p1_perp/|p1|)` also
    works for protons -- their deflection is bounded by asin(m_e/M) = 5.45e-4
    rad, never near pi/2 -- but that bound is a proton fact, and atan2 survives
    an electron beam.

Pinned by tests/physics/test_theta_kinematics.py, which includes a negative
control asserting that the acos formulation really does fail here.

`e_sec = 0` propagates to `theta = 0` on its own: pe = 0 makes p1_perp = 0 and
p1_z = p0, so atan2(0, p0) = 0. Callers rely on this instead of masking.

A SECOND, BENIGN PATH TO theta == 0
------------------------------------
`e_sec = 0` is not the only way to land on `theta == 0` exactly: at the
kinematic endpoint of the delta-ray spectrum (`e_sec` near the `t_max` bound
enforced by `SecondaryYScaler.min_max_e_secondary`, which the network's
terminal activation's eval-time upper clamp to 1 saturates against), `cos_e`
clamps to `1.0`, so
`sin_e = 0`, `p1_perp = 0`, and `atan2(0, p1_z>0)` returns exactly `0` even
though `e_sec > 0`. This is reachable in production simply because the
scaler's `t_max` bound and this module's endpoint are computed by different
expressions from the same constants: floating-point rounding alone lets a
network output the scaler judged in-bounds land exactly on this module's
endpoint. `tests/physics/test_g4h_ionisation.py` exercises this on the real
process and shows it occurring with `e_sec > 0`, more often at lower beam
energy (the delta-ray spectrum's kinematic endpoint sits relatively closer
to the bulk of the distribution there). This is correct physics, not an
artefact: the primary's deflection genuinely tends to 0 as the delta-ray
takes the maximum allowed momentum along the beam axis, so the
reconstruction returning exactly 0 there is the right limit, not a
cancellation failure. It does mean `theta`'s exact-zero population mixes
this case together with "no secondary was generated" -- any statistic that
treats theta's zero atom as a proxy for one or the other should account for
both sources.
"""
import torch


@torch.jit.script
def compute_primary_theta(
    post_cnt_energy: torch.Tensor,
    delta_kinetic_energy: torch.Tensor,
    primary_mass_eV: float,
    electron_mass_eV: float,
) -> torch.Tensor:
    """Primary scattering angle [rad] from the delta-ray kinetic energy.

    Args:
        post_cnt_energy: primary KE entering the discrete vertex [eV], i.e.
            AFTER the along-step continuous loss.
        delta_kinetic_energy: sampled secondary KE [eV]; 0 means no secondary.
        primary_mass_eV: primary rest mass [eV].
        electron_mass_eV: electron rest mass [eV].

    Returns:
        theta [rad], same shape and dtype as `post_cnt_energy`.
    """
    t0 = post_cnt_energy
    t = delta_kinetic_energy

    p0 = torch.sqrt(torch.clamp(t0 * (t0 + 2.0 * primary_mass_eV), min=0.0))
    pe = torch.sqrt(torch.clamp(t * (t + 2.0 * electron_mass_eV), min=0.0))

    # cos(theta_e) of the delta ray relative to the primary's incoming
    # direction (G4BetheBlochModel::SampleSecondaries), with E0 = T0 + M.
    # The clamp on the denominator keeps t = 0 finite; it yields cos = 0,
    # sin = 1, and p1_perp = pe * 1 = 0, so theta = 0 as required.
    cos_e = t * (t0 + primary_mass_eV + electron_mass_eV) / torch.clamp(pe * p0, min=1e-30)
    cos_e = torch.clamp(cos_e, min=-1.0, max=1.0)
    sin_e = torch.sqrt(torch.clamp(1.0 - cos_e * cos_e, min=0.0))

    p1_z = p0 - pe * cos_e
    p1_perp = pe * sin_e

    # atan2, NOT acos(p1_z / |p1|) -- see the module docstring.
    return torch.atan2(p1_perp, p1_z)
