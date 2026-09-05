"""
Generates and caches secondary (discrete ionization) energy-loss fluctuation CDFs
as a function of kinetic energy. Uses the discrete straggling model to compute
theoretical CDFs and stores them as interpolated lookup tables (Interp1dLinear).
"""
import numpy as np
import torch
import pickle
import logging

from tqdm import tqdm
from joblib import Parallel, delayed

from common.paths import PATH_SECONDARIES_CDFS_XS, PATH_SECONDARIES_CDFS_YS, PATH_TO_SECONDARIES_CDFS
from common.config import BEAM_ENERGY_eV
from common.run_context import PARTICLE, TARGET_MATERIAL
from common.utils import Interp1dLinear, PointInPhaseSpace
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4h_ionisation.explicit_physics.discrete.straggling_function import DiscreteStragglingModel

log = logging.getLogger(__name__)


def _heavy_min_ke(t_min: float) -> float:
    """Lowest heavy-primary kinetic energy whose max single-collision transfer
    reaches `t_min` — the closed-form inverse of `Particle.t_max`.

    `t_max(T) = 2*m_e*(beta*gamma)^2 / (1 + 2*gamma*r + r^2)` with `r = m_e/M`
    and `gamma = 1 + T/M` (G4BetheBlochModel::MaxSecondaryEnergy). Setting
    `t_max = t_min` and using `(beta*gamma)^2 = gamma^2 - 1` gives a quadratic
    in gamma,

        2*m_e*gamma^2 - 2*t_min*r*gamma - (2*m_e + t_min*(1 + r^2)) = 0,

    whose positive root inverts to `T = (gamma - 1)*M`. Same constants the
    builder already uses; no new physics.
    """
    r = PARTICLE.electron_mass_ratio
    a = 2.0 * electron_mass_eV
    b = -2.0 * t_min * r
    c = -(2.0 * electron_mass_eV + t_min * (1.0 + r * r))
    gamma = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return (gamma - 1.0) * PARTICLE.mass_eV


def _energy_grid() -> np.ndarray:
    """Conditioning-energy grid (the CDF/oracle keys), spanning the range where a
    delta-ray above the production cut Tc can be emitted, up to the beam energy.

    Heavy primaries use a grid over 0.5-100 MeV, unless the target's
    production cut Tc makes 0.5 MeV kinematically empty:
    `DiscreteLossParams.set_state` asserts `t_max(KE) >= Tc`, so if `t_max`
    at 0.5 MeV would fall below the material's Tc, the floor is raised
    instead to the kinetic energy at which `t_max` first reaches Tc, with
    the same 1% margin the electron branch uses (so the [Tc, t_max] band is
    non-degenerate at the first grid point). Electrons follow the Møller
    kinematics: a secondary requires t_max = E/2 > Tc, i.e. E > 2*Tc. The
    grid starts just above that threshold and spans to the beam energy,
    with a denser point budget to keep per-decade resolution over the wider
    electron range.
    """
    if PARTICLE.name == "electron":
        min_ke = 2.0 * TARGET_MATERIAL.Tc * 1.01
        max_ke = BEAM_ENERGY_eV
        N = 400
    else:
        min_ke = max(0.5e6, _heavy_min_ke(TARGET_MATERIAL.Tc * 1.01))
        max_ke = 100e6
        N = 200
    return np.logspace(np.log10(min_ke), np.log10(max_ke), num=N)


def _process_energy(i, kinetic_energy):
    straggling_model = DiscreteStragglingModel()
    point = PointInPhaseSpace(step_length=None, primary_energy=kinetic_energy)
    straggling_model.set_state(point)

    # Log-spaced grid + trapezoidal CDF. The secondary spectrum has a sharp
    # peak just above Tc (1/E^2 Rutherford for heavy primaries, z(x)/x^2
    # Møller for electrons). A linear grid over the full [Tc, t_max] band --
    # which for electrons spans to t_max = E/2 (tens of MeV) -- under-resolves
    # that peak and biases the low-energy CDF under a left-Riemann sum. Log
    # spacing holds fixed per-decade resolution at the peak regardless of how
    # wide the band is.
    xs = np.logspace(
        np.log10(straggling_model.min_ion_energy_transfer),
        np.log10(straggling_model.max_ion_energy_transfer),
        10_000
    )
    pdf = straggling_model.straggling_function(xs)
    ys = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xs))])

    return i, xs, ys


def _fix_y_values(ys: np.ndarray) -> np.ndarray:
    # Keep CDF monotonic even if model numerics introduce tiny local decreases.
    if (np.diff(ys) < 0).any():
        ys = np.maximum.accumulate(ys)

    # Remove the zero-collision mass and renormalize to [0, 1].
    if ys.min() != 1.:
        ys = (ys - ys.min()) / (ys - ys.min()).max()

    return ys


def _gen_sec_fluc_cdfs_raw(
    force_recalculation: bool = False,
) -> tuple[dict[tuple[float], np.ndarray], dict[tuple[float], np.ndarray]]:
    if PATH_SECONDARIES_CDFS_XS.exists() and PATH_SECONDARIES_CDFS_YS.exists() and not force_recalculation:
        with open(PATH_SECONDARIES_CDFS_XS, 'rb') as file:
            cdfs_xs = pickle.load(file)
        with open(PATH_SECONDARIES_CDFS_YS, 'rb') as file:
            cdfs_ys = pickle.load(file)
    else:
        kinetic_energies = _energy_grid()
        N = len(kinetic_energies)

        log.info(f"Generating secondary CDFs for {N} energy points "
                 f"({PARTICLE.name}: {kinetic_energies[0]:.1f} - {kinetic_energies[-1]:.1f} eV)")

        tasks = [(i, ke) for i, ke in enumerate(kinetic_energies)]

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_process_energy)(i, ke) for i, ke in tqdm(tasks)
        )

        cdfs_xs = {}
        cdfs_ys = {}
        for i, xs, ys in results:
            if xs is not None:
                E = float(kinetic_energies[i])
                cdfs_xs[(E,)] = xs
                cdfs_ys[(E,)] = ys

        # Save cdf_xs and cdfs_ys to files
        with open(PATH_SECONDARIES_CDFS_XS, "wb") as f:
            pickle.dump(cdfs_xs, f)

        with open(PATH_SECONDARIES_CDFS_YS, "wb") as f:
            pickle.dump(cdfs_ys, f)

    return cdfs_xs, cdfs_ys


def gen_sec_fluc_cdfs(
    load_cdfs: bool = True,
    force_recalculation: bool = False,
) -> dict[tuple[float], Interp1dLinear]:
    message = "Instead of simulating the physics model, loading the CDFs from:"

    if load_cdfs and PATH_TO_SECONDARIES_CDFS.exists() and not force_recalculation:
        log.info(f"{message} {str(PATH_TO_SECONDARIES_CDFS)}")
        with open(PATH_TO_SECONDARIES_CDFS, 'rb') as file:
            return pickle.load(file)

    if load_cdfs and PATH_SECONDARIES_CDFS_XS.exists() and PATH_SECONDARIES_CDFS_YS.exists() and not force_recalculation:
        log.info(f"{message} {str(PATH_SECONDARIES_CDFS_YS)}")
        with open(PATH_SECONDARIES_CDFS_XS, 'rb') as file:
            sec_cdfs_xs: dict[tuple[float], np.ndarray] = pickle.load(file)
        with open(PATH_SECONDARIES_CDFS_YS, 'rb') as file:
            sec_cdfs_ys: dict[tuple[float], np.ndarray] = pickle.load(file)
    else:
        log.info("Simulating the secondary fluctuation CDFs... This can take ~30 minutes or more.")
        sec_cdfs_xs, sec_cdfs_ys = _gen_sec_fluc_cdfs_raw(force_recalculation=force_recalculation)

    for key in sec_cdfs_ys:
        sec_cdfs_ys[key] = _fix_y_values(sec_cdfs_ys[key])

    cdfs_secondaries = {
        key: Interp1dLinear(
            x=torch.from_numpy(sec_cdfs_xs[key]),
            y=torch.from_numpy(sec_cdfs_ys[key]),
            fill_values=(0., 1.)
        )
        for key in sec_cdfs_ys
    }

    with open(PATH_TO_SECONDARIES_CDFS, "wb") as file:
        log.info(f"Saving the secondaries CDFs to {str(PATH_TO_SECONDARIES_CDFS)}")
        pickle.dump(cdfs_secondaries, file)

    return cdfs_secondaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cdfs = gen_sec_fluc_cdfs(load_cdfs=True, force_recalculation=False)
    print(len(cdfs))
