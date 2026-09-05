"""
Branch-free JIT-scripted interpolation and zero-loss-probability kernels,
plus the module-level physics constants (Python floats) the latter needs.

* `_interp1d_jit` is a branch-free linear interpolation kernel. It has no
  callers in this codebase: `PhysicsLengthGenerator`
  (`generators/length_generator/physics_mc.py`) interpolates its two
  G4-init-table lookups through `InitTableSet`'s `g4pv_eval`
  (`physics/g4_init_tables/interpolator.py`) instead, which is also
  branch-free -- so `compute_step_limit` compiles into a single graph with
  no data-dependent breaks -- see
  `tests/physics/test_length_generator_branchless.py`.
* `_compute_zero_prob_jit` is a closed-form `G4UniversalFluctuation`
  expression over material constants, not a table lookup.
  `G4hIonisation.post_step_do_it` calls it directly (module-qualified, not
  re-derived) via `physics.g4h_ionisation.g4h_ionisation`.
"""
import math

from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import GeantConstants
from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.physical_constants import electron_mass_eV

import torch

G4CONSTS = GeantConstants()

# ---------------------------------------------------------------------------
# Module-level physics constants (Python floats) for the JIT-scripted kernels
# ---------------------------------------------------------------------------
_PROTON_MASS_EV: float = proton.mass_eV
_ELECTRON_MASS_EV: float = electron_mass_eV
_MATERIAL_TC: float = material.Tc
_MATERIAL_I: float = material.I
_IONIZATION_RATE: float = G4CONSTS.ionization_rate
_FW: float = G4CONSTS.FW
_A0: float = G4CONSTS.A0
_NMAX_CONT: float = float(G4CONSTS.nmaxCont)
_E0: float = G4CONSTS.e0
_N_MIN_BOHR: float = float(G4CONSTS.n_min_bohr)
_LOG_W1: float = math.log(_MATERIAL_TC / _E0)
_SCALING: float = min(1.0 + 500.0 / _MATERIAL_TC, 1.5)


@torch.jit.script
def _interp1d_jit(
    x_table: torch.Tensor, y_table: torch.Tensor, x_new: torch.Tensor,
    left_slope: float, right_slope: float,
    fill_left: float, fill_right: float, do_fill: bool,
) -> torch.Tensor:
    """JIT-fused linear interpolation replacing Interp1dLinear for GPU kernel fusion."""
    n = x_table.shape[0]
    x_min = x_table[0]
    x_max = x_table[n - 1]

    left_idx = torch.searchsorted(x_table, x_new, right=True) - 1
    right_idx = left_idx + 1
    left_idx = torch.clamp(left_idx, 0, n - 1)
    right_idx = torch.clamp(right_idx, 0, n - 1)

    x_left = x_table[left_idx]
    x_right = x_table[right_idx]
    y_left = y_table[left_idx]
    y_right = y_table[right_idx]

    denom = x_right - x_left
    same_idx = denom == 0.0
    safe_denom = torch.where(same_idx, torch.ones_like(denom), denom)
    slope = (y_right - y_left) / safe_denom

    # Branchless boundary-slope assignment (no .any() GPU sync)
    slope = torch.where(same_idx & (x_new <= x_min), left_slope, slope)
    slope = torch.where(same_idx & (x_new >= x_max), right_slope, slope)

    y_new = y_left + slope * (x_new - x_left)

    if do_fill:
        y_new = torch.where(x_new < x_min, fill_left, y_new)
        y_new = torch.where(x_new > x_max, fill_right, y_new)

    return y_new


@torch.jit.script
def _compute_zero_prob_jit(
    mean_loss: torch.Tensor,
    kinetic_energy: torch.Tensor,
    proton_mass: float, electron_mass: float,
    material_Tc: float, material_I: float,
    ionization_rate: float, fw: float, a0: float,
    nmax_cont: float, e0: float, n_min_bohr: float,
    log_w1: float, scaling: float,
) -> torch.Tensor:
    """JIT-fused zero-probability computation — replaces ~20 kernel launches with 1-2."""
    zero_prob = torch.zeros_like(mean_loss)

    mass_ratio = electron_mass / proton_mass
    tau = kinetic_energy / proton_mass
    gamma = tau + 1.0
    gamma_sq = gamma * gamma
    beta_sq = 1.0 - 1.0 / gamma_sq

    t_max_tensor = (2.0 * electron_mass * beta_sq * gamma_sq) / (
        1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio
    )

    # is_heavy_particle is always True for protons, folded out
    is_above_loss_threshold = mean_loss >= n_min_bohr * material_Tc
    is_below_tmax_threshold = t_max_tensor <= 2.0 * material_Tc
    is_gaussian = is_above_loss_threshold & is_below_tmax_threshold
    calc_mask = ~is_gaussian

    # Compute for ALL elements (branchless — no .any() GPU-CPU sync), mask at the end
    scaled_mean_loss = mean_loss / scaling

    # --- excitation component ---
    a1 = torch.zeros_like(mean_loss)
    if material_Tc > material_I:  # compile-time constant branch
        a1_temp = scaled_mean_loss * (1.0 - ionization_rate) / material_I
        fwnow = torch.where(
            a1_temp < a0,
            0.1 + (fw - 0.1) * torch.sqrt(torch.clamp(a1_temp / a0, min=0.0)),
            torch.full_like(a1_temp, fw),
        )
        a1 = a1_temp / fwnow

    frac_zero_exc = torch.where(
        (a1 > 0.0) & (a1 <= nmax_cont), torch.exp(-a1),
        torch.where(a1 > nmax_cont, torch.zeros_like(a1), torch.ones_like(a1)),
    )

    # --- ionization component ---
    a3 = ionization_rate * scaled_mean_loss * (material_Tc - e0) / (e0 * material_Tc * log_w1)
    a3 = torch.where(a1 <= 0.0, a3 / ionization_rate, a3)

    frac_zero_ion = torch.where(
        (a3 > 0.0) & (a3 <= nmax_cont), torch.exp(-a3),
        torch.where(a3 > nmax_cont, torch.zeros_like(a3), torch.ones_like(a3)),
    )

    zero_prob = torch.where(calc_mask, frac_zero_exc * frac_zero_ion, zero_prob)
    return zero_prob
