"""
@torch.jit.script kernels that implement the discrete energy-loss sampler for
delta-ray (secondary electron) production, vectorized for PyTorch. These
kernels are the single source of truth for discrete sampling.

Two operations:

    1. Secondary kinetic energy `e_sec` — sampled from the Rutherford-like
       1/E² differential cross-section via rejection sampling of the
       truncated f-variable (a scalar version loops until acceptance; this
       vectorized version uses a fixed-iteration resample-where-rejected
       scheme so it can run lock-step on the GPU).
       Acceptance rate is high (≈ 0.97 for 100 MeV protons in Al), so the
       fixed cap is generous.

    2. Primary scattering angle `theta` (the `angleDiscrete` recorded by
       GEANT4) — derived from energy-momentum conservation given the sampled
       delta-ray kinetic energy, mirroring the kinematics used inside
       G4BetheBlochModel::SampleSecondaries followed by
       G4VEnergyLossProcess::PostStepDoIt:

           p0  = sqrt(T0 * (T0 + 2 M))                primary momentum
           pe  = sqrt(T  * (T  + 2 m_e))              delta-ray momentum
           cos(theta_e) = T * (T0 + M + m_e) / (pe * p0)
           p1_vec = p0 * z_hat - pe * delta_dir       (delta_dir at theta_e, phi)
           cos(theta_primary) = (p0 - pe * cos(theta_e)) / |p1_vec|

       The azimuthal angle is sampled uniformly in [0, 2π) but cancels for
       theta_primary itself.
"""
import math
import torch
from typing import Tuple

from physics.g4_init_tables.models.kinematics import _kinematics_jit


# ---------------------------------------------------------------------------
# Vectorized rejection sampler for delta-ray kinetic energy
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_secondary_kinetic_energy_jit(
    t_max: torch.Tensor,
    w: torch.Tensor,
    beta_sq: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    """
    Rejection-sample the delta-ray kinetic energy `T` from the Rutherford-like
    cross-section `dσ/dT ∝ 1/T² * (1 - β² T/T_max)` over the interval [Tc, T_max].

    Rejection scheme (the truncated f-variable construction):

        proposal:     T_prop = T_max / (1 + w * U)        with U ~ Uniform(0,1)
        accept rule:  Q < f(T_prop), where f = 1 - β² / (1 + w*U)
                      Q ~ Uniform(0,1) independent of U
        return:       T_prop  (== T_max / β² * (1 - f) by algebra)

    Resamples rejected entries each iteration; with a typical acceptance rate
    near 1 - β² log(1+w)/w (≈ 0.97 for 100 MeV p in Al), `max_iter ≈ 50` is
    far more than enough.
    """
    n = t_max.shape[0]
    device = t_max.device
    dtype = t_max.dtype

    # Initial proposal across all events
    u = torch.rand(n, device=device, dtype=dtype)
    q = torch.rand(n, device=device, dtype=dtype)
    denom = torch.clamp(1.0 + w * u, min=1e-30)
    f = 1.0 - beta_sq / denom
    t = t_max / denom
    accepted = q < f

    for _ in range(max_iter):
        if torch.all(accepted):
            break
        u_new = torch.rand(n, device=device, dtype=dtype)
        q_new = torch.rand(n, device=device, dtype=dtype)
        denom_new = torch.clamp(1.0 + w * u_new, min=1e-30)
        f_new = 1.0 - beta_sq / denom_new
        t_new = t_max / denom_new
        new_accept = q_new < f_new

        replace = (~accepted) & new_accept
        t = torch.where(replace, t_new, t)
        accepted = accepted | new_accept

    # Fallback for any (extremely rare) leftover: clamp to the proposal bounds
    t_min = t_max / torch.clamp(1.0 + w, min=1e-30)
    t = torch.clamp(t, min=t_min, max=t_max)
    return t


# ---------------------------------------------------------------------------
# Vectorized rejection sampler for the Møller (e-e-) delta-ray fraction
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_moller_fraction_jit(
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    gam: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    """
    Rejection-sample the Møller delta-ray energy fraction `x = T/T0` from
    `dσ/dx ∝ z(x)/x²` over [xmin, xmax] (xmax = 1/2 unless capped further).

    Mirrors the e-e- branch of G4MollerBhabhaModel::SampleSecondaries
    (geant4-11.2.0):

        proposal:    x = xmin*xmax / (xmin*(1-U) + xmax*U)    (∝ 1/x²)
        accept rule: grej * Q <= z(x), with
                     gg   = (2γ - 1)/γ²,  y = 1 - x
                     z(x) = 1 - gg*x + x²*(1 - gg + (1 - gg*y)/y²)
                     grej = z(xmax)   (G4's majorant over the interval)
    """
    n = xmin.shape[0]
    device = xmin.device
    dtype = xmin.dtype

    gamma_sq = gam * gam
    gg = (2.0 * gam - 1.0) / gamma_sq

    y_max = torch.clamp(1.0 - xmax, min=1e-30)
    grej = 1.0 - gg * xmax + xmax * xmax * (
        1.0 - gg + (1.0 - gg * y_max) / (y_max * y_max)
    )
    grej = torch.clamp(grej, min=1e-30)

    # Initial proposal across all events
    u = torch.rand(n, device=device, dtype=dtype)
    q = torch.rand(n, device=device, dtype=dtype)
    denom = torch.clamp(xmin * (1.0 - u) + xmax * u, min=1e-30)
    x = xmin * xmax / denom
    y = torch.clamp(1.0 - x, min=1e-30)
    z = 1.0 - gg * x + x * x * (1.0 - gg + (1.0 - gg * y) / (y * y))
    accepted = grej * q <= z

    for _ in range(max_iter):
        if torch.all(accepted):
            break
        u_new = torch.rand(n, device=device, dtype=dtype)
        q_new = torch.rand(n, device=device, dtype=dtype)
        denom_new = torch.clamp(xmin * (1.0 - u_new) + xmax * u_new, min=1e-30)
        x_new = xmin * xmax / denom_new
        y_new = torch.clamp(1.0 - x_new, min=1e-30)
        z_new = 1.0 - gg * x_new + x_new * x_new * (
            1.0 - gg + (1.0 - gg * y_new) / (y_new * y_new)
        )
        new_accept = grej * q_new <= z_new

        replace = (~accepted) & new_accept
        x = torch.where(replace, x_new, x)
        accepted = accepted | new_accept

    # Fallback for any (extremely rare) leftover: clamp to the proposal bounds
    x = torch.clamp(x, min=torch.minimum(xmin, xmax), max=xmax)
    return x


# ---------------------------------------------------------------------------
# Primary deflection angle from delta-ray kinematics
# ---------------------------------------------------------------------------
@torch.jit.script
def _compute_primary_theta_jit(
    primary_kinetic_energy: torch.Tensor,
    delta_kinetic_energy: torch.Tensor,
    primary_mass: float,
    electron_mass: float,
) -> torch.Tensor:
    """
    Compute the primary's scattering angle (`angleDiscrete` in the GEANT4
    output) from energy-momentum conservation.

    Inputs in eV:
        primary_kinetic_energy : T0  – primary KE entering the discrete event,
                                       i.e. *after* the along-step continuous
                                       loss (G4 produces the delta-ray after
                                       the continuous loss, so the call site
                                       passes post_cnt_energy; using the
                                       pre-step energy mismatches G4's recorded
                                       angleDiscrete by ~1e-4)
        delta_kinetic_energy   : T   – sampled secondary KE (T <= T_max)
    Returns:
        theta_primary in radians, broadcast to the input shape.

    Evaluated internally in fp64 regardless of input dtype: heavy-primary
    deflections are ~1e-4 rad, i.e. cos(theta) ~ 1 - 1e-8, which is below
    float32 resolution at 1.0 (2^-24) — acos in fp32 would quantize theta
    onto multiples of sqrt(2*2^-24) ≈ 3.5e-4 and zero out the bulk. The
    result is cast back to the input dtype at the boundary (theta itself
    is perfectly representable in fp32).
    """
    input_dtype = primary_kinetic_energy.dtype
    t0 = primary_kinetic_energy.to(torch.float64)
    t = delta_kinetic_energy.to(torch.float64)

    # Momenta
    p0 = torch.sqrt(torch.clamp(t0 * (t0 + 2.0 * primary_mass), min=0.0))
    pe = torch.sqrt(torch.clamp(t * (t + 2.0 * electron_mass), min=0.0))

    # cos(theta_e) of the delta ray relative to the primary direction.
    # Standard hadron-on-electron elastic kinematics
    # (G4BetheBlochModel::SampleSecondaries):
    #     cos(theta_e) = T * (E0 + m_e) / (pe * p0)
    # where E0 = T0 + M is the primary's total energy.
    e0_total = t0 + primary_mass
    cos_theta_e = t * (e0_total + electron_mass) / torch.clamp(pe * p0, min=1e-30)
    cos_theta_e = torch.clamp(cos_theta_e, min=-1.0, max=1.0)
    sin_theta_e = torch.sqrt(torch.clamp(1.0 - cos_theta_e * cos_theta_e, min=0.0))

    # Primary's final momentum vector: p1 = p0 z_hat - pe * (sin θ_e cos φ, sin θ_e sin φ, cos θ_e)
    # |p1|² = p0² - 2 p0 pe cos θ_e + pe²
    # cos(theta_primary) = (p0 - pe cos θ_e) / |p1|
    p1_z = p0 - pe * cos_theta_e
    p1_perp = pe * sin_theta_e
    p1_mag = torch.sqrt(torch.clamp(p1_z * p1_z + p1_perp * p1_perp, min=1e-60))
    cos_theta_p = torch.clamp(p1_z / p1_mag, min=-1.0, max=1.0)

    return torch.acos(cos_theta_p).to(input_dtype)


# ---------------------------------------------------------------------------
# Top-level discrete MC sampler
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_discrete_jit(
    primary_kinetic_energy: torch.Tensor,
    primary_mass: float,
    electron_mass: float,
    material_Tc: float,
    max_iter: int,
    is_electron: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (theta_primary, e_sec) for a batch of primaries that just produced a
    delta-ray.  Returns zero-filled outputs where the kinematics forbid a
    secondary (T_max <= Tc).

    `is_electron` selects the e- Møller branch (G4MollerBhabhaModel):
    T_max = T/2 (identical outgoing particles) and the Møller spectrum.
    Otherwise the heavy-particle branch (G4BetheBlochModel) applies:
    the hadron T_max and the Rutherford-tail spectrum.

    The caller (G4hIonisation.post_step_do_it) is responsible for deciding
    *whether* a secondary is produced on this step (based on the cross-section
    ratio) — this kernel only samples the secondary's energy and the primary's
    angle assuming the event happened.
    """
    _, gamma, _, beta_sq, heavy_t_max = _kinematics_jit(
        primary_kinetic_energy, primary_mass, electron_mass
    )
    beta_sq = torch.clamp(beta_sq, min=1e-30, max=1.0 - 1e-30)

    if is_electron:
        # Møller kinematic limit (G4MollerBhabhaModel::MaxSecondaryEnergy)
        t_max = 0.5 * primary_kinetic_energy
    else:
        t_max = heavy_t_max

    # Kinematic bounds of the secondary energy: [Tc, T_max]
    has_secondary = t_max > material_Tc

    if is_electron:
        ke_safe = torch.clamp(primary_kinetic_energy, min=1e-30)
        xmin = material_Tc / ke_safe
        xmax = t_max / ke_safe  # = 1/2
        x = _sample_moller_fraction_jit(xmin, xmax, gamma, max_iter)
        e_sec = x * primary_kinetic_energy
    else:
        w = torch.clamp(t_max / material_Tc - 1.0, min=0.0)
        e_sec = _sample_secondary_kinetic_energy_jit(t_max, w, beta_sq, max_iter)
    e_sec = torch.where(has_secondary, e_sec, torch.zeros_like(e_sec))

    theta = _compute_primary_theta_jit(
        primary_kinetic_energy, e_sec, primary_mass, electron_mass,
    )
    theta = torch.where(has_secondary, theta, torch.zeros_like(theta))
    return theta, e_sec
