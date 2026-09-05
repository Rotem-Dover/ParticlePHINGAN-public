"""
Generates and caches continuous energy-loss fluctuation CDFs over a 2D phase space
of kinetic energy and step length. Uses the continuous straggling model to compute
theoretical CDFs, stores them as interpolated lookup tables (Interp1dLinear), and
filters to the convex hull of the GEANT4 training data.
"""
import numpy as np
import torch
import pickle
import logging

from shapely.geometry import Point, MultiPoint

from common.config import (
    BEAM_ENERGY_eV,
    GRID_ENERGY_MIN_eV, GRID_STEP_LEN_MIN_m, GRID_STEP_LEN_MAX_m,
)
from common.paths import PATH_CONTINUOUS_CDFS_XS, PATH_CONTINUOUS_CDFS_YS, PATH_TO_CONTINUOUS_CDFS, PATH_TO_GEANT4_DATA
from tqdm import tqdm
from joblib import Parallel, delayed

import common.units as U
from common.enums import G4Columns
from common.utils import PointInPhaseSpace, Interp1dLinear
from data_handling.reader import read_geant4_simulation_output
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import ContinuousStragglingModel
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import StragglingType


log = logging.getLogger(__name__)


def define_phase_space_points() -> tuple[np.ndarray, np.ndarray]:
    """
    Define the energy and step length points in phase space for the straggling model.
    Points are log-spaced with ~20% spacing, matching the resolution needed for
    ~5% stopping power variation per step.
    """
    min_length, max_length = GRID_STEP_LEN_MIN_m, GRID_STEP_LEN_MAX_m
    step_lengths = [min_length]
    val = min_length
    while True:
        val *= 1.2
        if val < max_length:
            step_lengths.append(val)
        else:
            step_lengths.append(max_length)
            break
    step_lengths = np.array(step_lengths)

    min_ke = GRID_ENERGY_MIN_eV
    max_ke = BEAM_ENERGY_eV
    kinetic_energies = [max_ke]
    val = max_ke
    while True:
        val = val * 1.2 ** (-1 / 0.8)
        if val > min_ke:
            kinetic_energies.append(val)
        else:
            kinetic_energies.append(min_ke)
            break
    kinetic_energies = np.sort(kinetic_energies)

    return kinetic_energies, step_lengths


def process_one(i, j, E, L):

    straggling_model = ContinuousStragglingModel()
    point_in_phase_space = PointInPhaseSpace(step_length=L * U.m2cm, primary_energy=E)
    straggling_model.set_state(point_in_phase_space)

    if StragglingType.DeterministicBetheBloch in straggling_model.straggling_type:
        return i, j, None, None

    xs = np.linspace(0, min(100 * straggling_model.average_loss, 1e7), 100_000)
    theoretical_cdf = straggling_model.straggling_cdf(xs)

    return i, j, xs, theoretical_cdf


def _fix_y_values(ys: np.ndarray) -> np.ndarray:
    # Keep CDF monotonic even if model numerics introduce tiny local decreases.
    if (np.diff(ys) < 0).any():
        ys = np.maximum.accumulate(ys)

    # Remove the zero-collision mass and renormalize to [0, 1].
    if ys.min() != 1.:
        ys = (ys - ys.min()) / (ys - ys.min()).max()

    return ys


def _define_target_phase_space_shape() -> MultiPoint:
    geant4_csv_output = read_geant4_simulation_output(PATH_TO_GEANT4_DATA, verbose=True)
    points = geant4_csv_output[[G4Columns.KineticEnergy, G4Columns.StepLength]].to_numpy()
    return MultiPoint(points).convex_hull


def _gen_cnt_fluc_cdfs(force_recalculation: bool = False) -> tuple[dict[tuple[float, float], np.ndarray], dict[tuple[float, float], np.ndarray]]:
    if PATH_CONTINUOUS_CDFS_XS.exists() and PATH_CONTINUOUS_CDFS_YS.exists() and not force_recalculation:
        with open(PATH_CONTINUOUS_CDFS_XS, 'rb') as file:
            cdfs_xs = pickle.load(file)
        with open(PATH_CONTINUOUS_CDFS_YS, 'rb') as file:
            cdfs_ys = pickle.load(file)

    else:
        kinetic_energies, step_lengths = define_phase_space_points()

        # Prepare all tasks
        tasks = [(i, j, float(ke), float(L))
                 for i, ke in enumerate(kinetic_energies)
                 for j, L in enumerate(step_lengths)]

        # Parallel processing
        results = Parallel(n_jobs=-1, backend='loky')(delayed(process_one)(i, j, ke, L) for i, j, ke, L in tqdm(tasks))

        # Post-processing
        cdfs_xs = {}
        cdfs_ys = {}

        for i, j, xs, ys in results:
            if xs is not None:
                L = float(step_lengths[j])
                E = float(kinetic_energies[i])
                cdfs_xs[(L, E)] = xs
                cdfs_ys[(L, E)] = ys

        # Save cdf_xs and cdfs_ys to files
        with open(PATH_CONTINUOUS_CDFS_XS, "wb") as f:
            pickle.dump(cdfs_xs, f)

        with open(PATH_CONTINUOUS_CDFS_YS, "wb") as f:
            pickle.dump(cdfs_ys, f)

    return cdfs_xs, cdfs_ys


def gen_cnt_fluc_cdfs(
    load_cdfs: bool = True,
    force_recalculation: bool = False,
) -> dict[tuple[float, float], Interp1dLinear]:
    message = "Instead of simulating the physics model, loading the CDFs from:"

    if load_cdfs and PATH_TO_CONTINUOUS_CDFS.exists() and not force_recalculation:
        log.info(f"{message} {str(PATH_TO_CONTINUOUS_CDFS)}")
        with open(PATH_TO_CONTINUOUS_CDFS, 'rb') as file:
            return pickle.load(file)

    if load_cdfs and PATH_CONTINUOUS_CDFS_XS.exists() and PATH_CONTINUOUS_CDFS_YS.exists() and not force_recalculation:
        log.info(f"{message} {str(PATH_CONTINUOUS_CDFS_YS)}")
        with open(PATH_CONTINUOUS_CDFS_XS, 'rb') as file:
            cnt_cdfs_xs: dict[tuple[float, float], np.ndarray] = pickle.load(file)
        with open(PATH_CONTINUOUS_CDFS_YS, 'rb') as file:
            cnt_cdfs_ys: dict[tuple[float, float], np.ndarray] = pickle.load(file)
    else:
        log.info("Simulating the continuous fluctuation CDFs... This can take few minutes.")
        cnt_cdfs_xs, cnt_cdfs_ys = _gen_cnt_fluc_cdfs(force_recalculation=force_recalculation)

    log.info("Removing 0 energy loss samples from continuous CDFs... (These samples should be removed also from the dataset)")
    for key in cnt_cdfs_ys:
        cnt_cdfs_ys[key] = _fix_y_values(cnt_cdfs_ys[key])

    target_phase_space_shape = _define_target_phase_space_shape()
    cdfs_continuous = {
        key: Interp1dLinear(
            x=torch.from_numpy(cnt_cdfs_xs[key]),
            y=torch.from_numpy(cnt_cdfs_ys[key]),
            fill_values=(0., 1.)
        )
        for key in cnt_cdfs_ys
        if target_phase_space_shape.covers(Point(key[1], key[0]))
    }

    with open(PATH_TO_CONTINUOUS_CDFS, "wb") as file:
        log.info(f"Saving the continuous CDFs to {str(PATH_TO_CONTINUOUS_CDFS)}")
        pickle.dump(cdfs_continuous, file)

    return cdfs_continuous


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cdfs = gen_cnt_fluc_cdfs(load_cdfs=True, force_recalculation=False)
    print(len(cdfs))
