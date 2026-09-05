"""
@torch.jit.script kernels that implement the G4UniversalFluctuation continuous
energy-loss sampler (G4UniversalFluctuation::SampleFluctuations / SampleGlandz)
in vectorized form. The parameter math mirrors the scalar reference
`physics.g4h_ionisation.explicit_physics.continuous.straggling_function`
(`ContinuousLossParams`), and the two are pinned together by
`tests/physics/g4h_ionisation/explicit_physics/test_glandz_params_parity.py`.
These primitives accept the per-event mean energy loss and
batched kinetic energies, and return a per-event continuous energy-loss sample
that mirrors GEANT4's behavior across the four regimes:

    - thick truncated Gaussian          (heavy particle, mean_loss >= n_min_bohr*Tc, t_max <= 2*Tc, sn >= 2)
    - low-frequency excitation          (Poisson over excitation collisions, a1 <= nmaxCont)
    - high-frequency excitation         (Gaussian addition, a1 > nmaxCont)
    - low-frequency ionisation          (compound Poisson over single collisions, Tc > w3)
    - high-frequency ionisation         (Gaussian addition, a3 > nmaxCont)

The deterministic-loss branch (mean_loss < min_energy_loss) and the kill branch
(mean_loss > kinetic_energy) are *not* applied here: they are re-applied by the
caller (G4hIonisation.along_step_do_it) which already overrides those events.
"""
import torch
from typing import Tuple

from physics.g4_init_tables.models.kinematics import _kinematics_jit


# ---------------------------------------------------------------------------
# Vectorized truncated Gaussian sampler (G4UniversalFluctuation::SampleGauss)
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_gauss_jit(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    active: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    """
    Truncated-Gaussian rejection sampler that matches G4UniversalFluctuation::SampleGauss:

        if mean < 0.25*sigma:
            x = mean + (2*rand-1) * mean    # uniform fallback for narrow mean
        else:
            do x = randn()*sigma + mean while x < 0 or x > 2*mean

    `active` selects which entries to actually sample for; inactive entries are
    returned untouched (zero) so callers can sum over the full batch.
    """
    n = mean.shape[0]
    device = mean.device
    dtype = mean.dtype

    use_uniform = mean < 0.25 * sigma

    r_uniform = torch.rand(n, device=device, dtype=dtype)
    x_uniform = mean + (2.0 * r_uniform - 1.0) * mean

    x_normal = torch.randn(n, device=device, dtype=dtype) * sigma + mean
    for _ in range(max_iter):
        bad = (x_normal < 0.0) | (x_normal > 2.0 * mean)
        if not torch.any(bad):
            break
        x_resample = torch.randn(n, device=device, dtype=dtype) * sigma + mean
        x_normal = torch.where(bad, x_resample, x_normal)
    # fallback clamp in case max_iter exhausted
    x_normal = torch.clamp(x_normal, min=torch.zeros_like(mean), max=2.0 * mean)

    x = torch.where(use_uniform, x_uniform, x_normal)
    return torch.where(active, x, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Excitation: low-frequency Poisson contribution
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_low_freq_excitation_jit(
    a1: torch.Tensor,
    e1: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """
    Low-collision-frequency excitation (a1 in (0, nmaxCont]):

        p = Poisson(a1)
        loss = ((p+1) - 2*rand) * e1   if p > 0 else 0
    """
    n = a1.shape[0]
    device = a1.device
    dtype = a1.dtype

    rate = torch.where(active, torch.clamp(a1, min=0.0), torch.zeros_like(a1))
    p = torch.poisson(rate).to(dtype)

    has_events = p > 0
    rand = torch.rand(n, device=device, dtype=dtype)
    loss = ((p + 1.0) - 2.0 * rand) * e1
    return torch.where(active & has_events, loss, torch.zeros_like(loss))


# ---------------------------------------------------------------------------
# Ionisation: low-frequency compound Poisson contribution
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_low_freq_ionisation_jit(
    p3: torch.Tensor,
    w3: torch.Tensor,
    w: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """
    Low-collision-frequency ionisation (Tc > w3): compound Poisson sampler

        nnb = Poisson(p3)
        loss = sum_{k=1..nnb} w3 / (1 - w*rand_k)

    Vectorized via a (B, M) padding buffer where M = max(nnb).  In practice p3
    stays bounded (~ nmaxCont) because the high-mean tail is absorbed by the
    Gaussian addition term, so M is small.
    """
    n = p3.shape[0]
    device = p3.device
    dtype = p3.dtype

    rate = torch.where(active, torch.clamp(p3, min=0.0), torch.zeros_like(p3))
    nnb = torch.poisson(rate).to(torch.int64)

    if nnb.numel() == 0:
        return torch.zeros_like(p3)

    m_int = int(torch.max(nnb).item())
    if m_int == 0:
        return torch.zeros_like(p3)

    # (B, M) buffer of single-collision energy losses; mask out k >= nnb[i]
    rand = torch.rand(n, m_int, device=device, dtype=dtype)
    w_b = w.unsqueeze(1)
    w3_b = w3.unsqueeze(1)
    per_collision = w3_b / torch.clamp(1.0 - w_b * rand, min=1e-30)

    arange = torch.arange(m_int, device=device).unsqueeze(0)
    keep = arange < nnb.unsqueeze(1)
    per_collision = torch.where(keep, per_collision, torch.zeros_like(per_collision))

    return torch.where(active, per_collision.sum(dim=1), torch.zeros_like(p3))


# ---------------------------------------------------------------------------
# Glandz parameter computation (vectorized)
# ---------------------------------------------------------------------------
@torch.jit.script
def _compute_glandz_params_jit(
    mean_loss_scaled: torch.Tensor,
    material_Tc: float,
    material_I: float,
    e0: float,
    log_w1: float,
    ionization_rate: float,
    fw: float,
    a0: float,
    nmax_cont: float,
) -> Tuple[
    torch.Tensor, torch.Tensor,    # a1, e1_eff
    torch.Tensor, torch.Tensor,    # emean_exc, sig2e_exc
    torch.Tensor,                   # a3
    torch.Tensor, torch.Tensor,    # alfa, alfa1
    torch.Tensor, torch.Tensor,    # namean, p3
    torch.Tensor, torch.Tensor,    # emean_ion, sig2e_ion
    torch.Tensor, torch.Tensor,    # w3, w
]:
    """
    Compute Glandz fluctuation-model parameters per event.

    Mirrors `ContinuousLossParams.set_glandz_params` but in pure tensor ops.
    `mean_loss_scaled` is the average loss already divided by the global Glandz
    `scaling` factor (min(1+500/Tc, 1.5)); this matches the GEANT4 source.
    """
    zero = torch.zeros_like(mean_loss_scaled)

    # --- excitation parameters ---
    if material_Tc > material_I:
        a1_pre = mean_loss_scaled * (1.0 - ionization_rate) / material_I
    else:
        a1_pre = zero
    a1_pre = torch.clamp(a1_pre, min=0.0)

    fwnow = torch.where(
        a1_pre < a0,
        0.1 + (fw - 0.1) * torch.sqrt(torch.clamp(a1_pre / a0, min=0.0)),
        torch.full_like(a1_pre, fw),
    )
    a1_safe = torch.where(a1_pre > 0.0, a1_pre, torch.ones_like(a1_pre))
    a1 = torch.where(a1_pre > 0.0, a1_safe / fwnow, zero)
    e1_eff = torch.where(a1_pre > 0.0, material_I * fwnow, torch.full_like(a1_pre, material_I))

    is_high_freq_exc = a1 > nmax_cont
    emean_exc = torch.where(is_high_freq_exc, a1 * e1_eff, zero)
    sig2e_exc = torch.where(is_high_freq_exc, a1 * e1_eff * e1_eff, zero)

    # --- ionisation parameters ---
    a3_raw = ionization_rate * mean_loss_scaled * (material_Tc - e0) / (e0 * material_Tc * log_w1)
    a3 = torch.where(a1_pre <= 0.0, a3_raw / ionization_rate, a3_raw)
    a3 = torch.clamp(a3, min=0.0)

    is_high_freq_ion = a3 > nmax_cont

    # alfa branch (only valid when a3 > nmax_cont)
    denom_alfa = torch.clamp(nmax_cont * material_Tc / e0 + a3, min=1e-30)
    alfa_high = (material_Tc / e0) * (nmax_cont + a3) / denom_alfa
    alfa = torch.where(is_high_freq_ion, alfa_high, torch.ones_like(a3))

    alfa_minus_one = torch.clamp(alfa - 1.0, min=1e-30)
    alfa1 = alfa * torch.log(torch.clamp(alfa, min=1e-30)) / alfa_minus_one
    alfa1 = torch.where(is_high_freq_ion, alfa1, torch.ones_like(alfa))

    w1_minus_one = (material_Tc / e0) - 1.0
    namean = a3 * (material_Tc / e0) * alfa_minus_one / (
        torch.clamp(w1_minus_one * alfa, min=1e-30)
    )
    namean = torch.where(is_high_freq_ion, namean, zero)

    emean_ion = torch.where(is_high_freq_ion, namean * e0 * alfa1, zero)
    sig2e_ion = torch.where(
        is_high_freq_ion,
        e0 * e0 * namean * (alfa - alfa1 * alfa1),
        zero,
    )
    sig2e_ion = torch.clamp(sig2e_ion, min=0.0)

    p3 = torch.where(is_high_freq_ion, a3 - namean, a3)
    p3 = torch.clamp(p3, min=0.0)

    w3 = alfa * e0
    # w = (Tc - w3) / Tc, only valid when Tc > w3
    w = torch.clamp((material_Tc - w3) / material_Tc, min=0.0, max=1.0 - 1e-12)

    return (
        a1, e1_eff,
        emean_exc, sig2e_exc,
        a3,
        alfa, alfa1,
        namean, p3,
        emean_ion, sig2e_ion,
        w3, w,
    )


# ---------------------------------------------------------------------------
# Top-level continuous-loss MC sampler
# ---------------------------------------------------------------------------
@torch.jit.script
def _sample_continuous_loss_jit(
    kinetic_energy: torch.Tensor,
    average_loss: torch.Tensor,
    step_length_cm: torch.Tensor,
    primary_mass: float,
    electron_mass: float,
    electron_radius_cm: float,
    electron_density: float,
    material_Tc: float,
    material_I: float,
    e0: float,
    log_w1: float,
    ionization_rate: float,
    fw: float,
    a0: float,
    nmax_cont: float,
    n_min_bohr: float,
    scaling: float,
    twopi_mc2_rcl2: float,
    max_gauss_iter: int,
    is_electron: bool,
) -> torch.Tensor:
    """
    Vectorized G4UniversalFluctuation::SampleFluctuations kernel.

    Returns the per-event continuous energy loss `e_cnt` in eV.

    Inputs:
        kinetic_energy   – pre-step KE in eV, shape [B]
        average_loss     – Bethe-Bloch * step_length (with range-correction for
                           long steps), shape [B], in eV
        step_length_cm   – step length in cm (used for thick-Gaussian sigma)
        primary_mass     – primary mass in eV; the Gaussian regime is gated on
                           primary_mass > electron_mass (verbatim G4)
        electron_mass    – electron mass in eV
        electron_radius_cm – classical electron radius in cm
        electron_density – material electron density in 1/cm³
        material_Tc      – delta-ray production threshold in eV
        material_I       – mean excitation energy in eV
        e0               – outer-shell excitation energy (10 eV)
        log_w1           – log(Tc / e0)
        ionization_rate  – Glandz constant 0.56
        fw, a0, nmax_cont – Glandz constants
        n_min_bohr       – Bohr-regime threshold (10)
        scaling          – min(1 + 500/Tc, 1.5)
        twopi_mc2_rcl2   – 2π m_e r_e²
        max_gauss_iter   – truncated-Gaussian rejection iteration cap
        is_electron      – Møller primary: t_max = T/2 instead of the
                           heavy-particle kinematic limit
    """
    # --- thick-Gaussian regime predicates ---
    _, _, _, beta_sq, heavy_t_max = _kinematics_jit(
        kinetic_energy, primary_mass, electron_mass
    )

    if is_electron:
        # Møller kinematics: identical outgoing electrons => T_max = T/2
        # (geant4-11.2.0 G4MollerBhabhaModel::MaxSecondaryEnergy). This is
        # the tmax G4VEnergyLossProcess passes into SampleFluctuations for
        # an e- primary. It only feeds the heavy-gated Gaussian regime, so
        # it is output-inert for electrons — kept faithful regardless.
        t_max = 0.5 * kinetic_energy
    else:
        t_max = heavy_t_max

    is_heavy = primary_mass > electron_mass
    is_above_n_min_bohr = average_loss >= n_min_bohr * material_Tc
    is_below_2tc = t_max <= 2.0 * material_Tc
    in_gauss_regime = is_above_n_min_bohr & is_below_2tc
    if not is_heavy:
        in_gauss_regime = torch.zeros_like(in_gauss_regime)

    # thick-Gaussian sigma
    siga = torch.sqrt(torch.clamp(
        (t_max / torch.clamp(beta_sq, min=1e-30) - 0.5 * material_Tc)
        * twopi_mc2_rcl2 * step_length_cm * electron_density,
        min=0.0,
    ))
    sn = average_loss / torch.clamp(siga, min=1e-30)
    use_thick_gauss = in_gauss_regime & (sn >= 2.0)

    # --- Glandz path: scale the mean loss ---
    mean_loss_scaled = average_loss / scaling

    (
        a1, e1_eff,
        emean_exc, sig2e_exc,
        a3,
        alfa, alfa1,
        namean, p3,
        emean_ion, sig2e_ion,
        w3, w,
    ) = _compute_glandz_params_jit(
        mean_loss_scaled, material_Tc, material_I,
        e0, log_w1, ionization_rate, fw, a0, nmax_cont,
    )

    # All Glandz contributions are accumulated into `glandz_loss` and rescaled
    glandz_loss = torch.zeros_like(average_loss)

    # excitation
    is_low_freq_exc = (a1 > 0.0) & (a1 <= nmax_cont)
    glandz_loss = glandz_loss + _sample_low_freq_excitation_jit(a1, e1_eff, is_low_freq_exc)

    is_high_freq_exc = sig2e_exc > 0.0
    glandz_loss = glandz_loss + _sample_gauss_jit(
        emean_exc, torch.sqrt(sig2e_exc), is_high_freq_exc, max_gauss_iter,
    )

    # ionisation
    is_low_freq_ion = (a3 > 0.0) & (material_Tc > w3) & (p3 > 0.0)
    glandz_loss = glandz_loss + _sample_low_freq_ionisation_jit(p3, w3, w, is_low_freq_ion)

    is_high_freq_ion = sig2e_ion > 0.0
    glandz_loss = glandz_loss + _sample_gauss_jit(
        emean_ion, torch.sqrt(sig2e_ion), is_high_freq_ion, max_gauss_iter,
    )

    glandz_loss = glandz_loss * scaling

    # --- thick truncated Gaussian regime ---
    thick_gauss_loss = _sample_gauss_jit(
        average_loss, siga, use_thick_gauss, max_gauss_iter,
    )

    # --- Combine ---
    # Default: Glandz path; thick-Gaussian regime overrides it.
    total = torch.where(use_thick_gauss, thick_gauss_loss, glandz_loss)

    # Tc <= e0 guard (very small Tc): fall back to mean loss with no fluctuation
    if material_Tc <= e0:
        total = average_loss

    # Numerical safety
    total = torch.clamp(total, min=0.0)
    return total
