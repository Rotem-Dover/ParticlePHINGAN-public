"""
Analytical builder for the continuous-stage mean/std lookup table.

Computes exact log10 moments of the analytical straggling CDF on a coarse
log-log grid and bilinearly upsamples them to a dense 5000x5000 table. The
table gives `ContinuousYScaler` a per-(E, L) mean/std it can gather in O(1)
at inference to z-score `E_CNT` before it reaches the network.

Per coarse cell the CDF is evaluated with the same construction as the CDF
builder (linspace over [0, min(100*avg, 1e7)] + `_fix_y_values` zero-mass
renormalization), so mu/sigma are the moments of the *same* distribution the
z-target is defined against -- it becomes a properly standardized variable
per key. Deterministic cells (average_loss below the model's min loss, or
above the primary energy) are filtered out of the coarse grid (stored as
NaN) rather than assigned a placeholder sigma. This keeps the
stochastic/deterministic boundary from bleeding an arbitrary value into
neighboring stochastic cells via bilinear interpolation. `G4hIonisation`
bypasses the network for deterministic (E, L) pairs (see
`g4h_ionisation.py`'s `is_deterministic`), so the NaN fine-table cells are
never queried at runtime for a normalized value.

Bilinear (not bicubic) upsampling is deliberate: on the smooth stochastic
surfaces its interpolation error at this grid density is well below the
scale of what would need fitting, and it cannot ring near the
stochastic/deterministic boundary where the surface transitions to NaN.

Run (idempotent -- skips if the cache exists):
    python src/physics/g4h_ionisation/generators/continuous_generator/build_analytic_mean_std_table.py
    # options: --n-coarse 500 --force
"""
import argparse
import logging
import math
import pickle

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import common.paths as paths
import common.units as U
from common.config import (
    BEAM_ENERGY_eV,
    GRID_ENERGY_MIN_eV, GRID_STEP_LEN_MIN_m, GRID_STEP_LEN_MAX_m,
)
from common.utils import PointInPhaseSpace
from physics.g4h_ionisation.generators.continuous_generator.generate_cdfs import _fix_y_values
from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import (
    ContinuousStragglingModel, StragglingType,
)

log = logging.getLogger(__name__)

FINE_STEPS = 5_000          # must match ContinuousYScaler.continuous_mean_std_lookup_table
DEFAULT_COARSE_STEPS = 500
_N_CDF_POINTS = 100_000     # same as gen_continuous_fluc_cdfs.process_one
_STD_FLOOR = 1e-5           # mirror the MC path's degenerate-cell fallback

# Wall-band exact patch: the mean-log-loss surface mu(E, L) kinks along the
# stopping wall L = range(E) (steep rise -> flat saturation at avg_loss = E).
# Bilinear interpolation of the coarse grid cuts that corner, undershooting
# mu at fine cells hugging the wall from below -- a constant offset baked
# into the z-target and into rescale at end-of-track cells. The patch
# recomputes exact moments for fine cells within a few coarse-cell widths of
# the wall. Widths are in units of the coarse grid spacing, since one
# straddling coarse cell is what loses the corner.
_WALL_BAND_BELOW_COARSE = 2.0
_WALL_BAND_ABOVE_COARSE = 1.0


def analytic_cache_path(fine_steps: int = FINE_STEPS):
    return paths.TABLES_DIR / f"continuous_mean_std_lookup_table_steps_{fine_steps}_analytic.pkl"


def log10_moments_from_cdf(xs: np.ndarray, Fs: np.ndarray) -> tuple[float, float]:
    """Mean/std of log10(x) under the (zero-mass-removed, renormalized) CDF.

    Probability mass per bin comes from CDF differences, with log10
    evaluated at bin midpoints. Returns (nan, nan) for a degenerate CDF with
    no usable mass.
    """
    Fs = _fix_y_values(np.asarray(Fs, dtype=np.float64))
    w = np.diff(Fs)
    x_mid = 0.5 * (xs[1:] + xs[:-1])
    valid = (w > 0) & (x_mid > 0)
    w, x_mid = w[valid], x_mid[valid]
    w_sum = w.sum()
    if w_sum <= 0:
        return float("nan"), float("nan")
    w = w / w_sum
    log_x = np.log10(x_mid)
    mean = float((w * log_x).sum())
    var = float((w * (log_x - mean) ** 2).sum())
    return mean, math.sqrt(max(var, 0.0))


def _compute_row(i: int, ke_eV: float, step_lengths_m: np.ndarray) -> tuple[int, np.ndarray]:
    """mu/sigma for one kinetic energy across all step lengths. Returns (i, [n_steps, 2])."""
    model = ContinuousStragglingModel()
    out = np.empty((step_lengths_m.shape[0], 2), dtype=np.float64)

    for j, sl_m in enumerate(step_lengths_m):
        model.set_state(PointInPhaseSpace(step_length=sl_m * U.m2cm, primary_energy=ke_eV))

        deterministic = (
            StragglingType.DeterministicBetheBloch in model.straggling_type
            or model.average_loss > model.primary_energy  # capped-at-E early return
        )
        if not deterministic:
            xs = np.linspace(0, min(100 * model.average_loss, 1e7), _N_CDF_POINTS)
            mu, sd = log10_moments_from_cdf(xs, model.straggling_cdf(xs))
            deterministic = math.isnan(mu)

        if deterministic:
            out[j] = (np.nan, np.nan)
        elif sd < _STD_FLOOR:
            out[j] = (mu, 1.0)
        else:
            out[j] = (mu, sd)

    return i, out


def _log_grids(n: int) -> tuple[np.ndarray, np.ndarray]:
    """log10 (E [eV], L [m]) grids over the active beam's GridBounds."""
    kin_log = np.log10(
        np.logspace(math.log10(GRID_ENERGY_MIN_eV), math.log10(BEAM_ENERGY_eV), n)
    )
    step_log = np.log10(
        np.logspace(math.log10(GRID_STEP_LEN_MIN_m), math.log10(GRID_STEP_LEN_MAX_m), n)
    )
    return kin_log, step_log


def compute_coarse_table(n_coarse: int, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kin_log, step_log = _log_grids(n_coarse)
    kin_eV = 10.0 ** kin_log
    step_m = 10.0 ** step_log

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_row)(i, float(ke), step_m)
        for i, ke in enumerate(tqdm(kin_eV, desc=f"Analytic mu/sigma {n_coarse}x{n_coarse}"))
    )

    table = np.empty((n_coarse, n_coarse, 2), dtype=np.float64)
    for i, row in results:
        table[i] = row
    return kin_log, step_log, table


def _masked_bilinear_upsample(surface: np.ndarray, kin_log: np.ndarray, step_log: np.ndarray,
                              kin_fine: np.ndarray, step_fine: np.ndarray) -> np.ndarray:
    """NaN-aware bilinear upsample of one coarse surface onto the fine grid.

    Ordinary bilinear interpolation wherever the 2x2 stencil is fully finite.
    Where the stencil straddles the stochastic/deterministic boundary, the NaN
    (deterministic) corners are dropped and the surviving weights renormalized,
    so the interpolation degrades bilinear -> linear -> nearest as corners vanish
    and is always evaluated from the non-NaN (stochastic) side. A fine cell is
    NaN only when all four stencil corners are NaN (the deterministic interior).

    Consequence: the finite region is one stencil-width wider than the coarse
    stochastic region, so every stochastic oracle key -- which rounds to a fine
    cell on or just inside the boundary -- lands on a finite value, while the
    deterministic interior (bypassed by G4hIonisation at runtime) stays NaN.
    """
    n_coarse = surface.shape[0]
    dk = kin_log[1] - kin_log[0]
    ds = step_log[1] - step_log[0]

    gx = np.clip((kin_fine - kin_log[0]) / dk, 0.0, n_coarse - 1.0)
    gy = np.clip((step_fine - step_log[0]) / ds, 0.0, n_coarse - 1.0)
    i0 = np.minimum(np.floor(gx).astype(int), n_coarse - 2)
    j0 = np.minimum(np.floor(gy).astype(int), n_coarse - 2)
    fx = gx - i0          # (n_fine_kin,)  fractional offset in [0, 1]
    fy = gy - j0          # (n_fine_step,)

    wx = (1.0 - fx, fx)
    wy = (1.0 - fy, fy)

    num = np.zeros((kin_fine.size, step_fine.size), dtype=np.float64)
    wsum = np.zeros_like(num)
    for oi in (0, 1):
        for oj in (0, 1):
            v = surface[np.ix_(i0 + oi, j0 + oj)]
            w = wx[oi][:, None] * wy[oj][None, :]
            finite = np.isfinite(v)
            num += np.where(finite, v, 0.0) * w
            wsum += np.where(finite, w, 0.0)

    out = np.full_like(num, np.nan)
    np.divide(num, wsum, out=out, where=wsum > 0)
    return out


def upsample_bilinear(kin_log: np.ndarray, step_log: np.ndarray, table: np.ndarray,
                      fine_steps: int = FINE_STEPS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NaN-aware bilinear upsample of the mu and sigma surfaces in log-log space.

    Deterministic coarse cells are NaN. Each fine cell renormalizes its bilinear
    stencil over the finite (stochastic) corners only (see
    ``_masked_bilinear_upsample``), so the boundary interpolates purely from the
    stochastic side and a fine cell stays NaN only where the whole stencil is
    deterministic. Both channels share the same coarse NaN pattern (set jointly
    in ``_compute_row``), so the two fine surfaces are NaN on the same cells.
    """
    kin_fine, step_fine = _log_grids(fine_steps)
    fine = torch.empty((fine_steps, fine_steps, 2), dtype=torch.float32)
    for c in range(2):
        up = _masked_bilinear_upsample(table[..., c], kin_log, step_log, kin_fine, step_fine)
        fine[..., c] = torch.from_numpy(up).to(torch.float32)

    return (torch.from_numpy(kin_fine).to(torch.float32),
            torch.from_numpy(step_fine).to(torch.float32),
            fine)


def _exact_cell(model: ContinuousStragglingModel, ke_eV: float, sl_m: float):
    """Exact (mu, sigma) at one cell with the coarse builder's recipe.

    Returns None where the builder's deterministic rules apply (the existing
    fine value is kept there: the finite one-stencil-wide skirt around the
    stochastic region must survive the patch, see _masked_bilinear_upsample).
    """
    model.set_state(PointInPhaseSpace(step_length=sl_m * U.m2cm, primary_energy=ke_eV))
    if (StragglingType.DeterministicBetheBloch in model.straggling_type
            or model.average_loss > model.primary_energy):
        return None
    xs = np.linspace(0, min(100 * model.average_loss, 1e7), _N_CDF_POINTS)
    mu, sd = log10_moments_from_cdf(xs, model.straggling_cdf(xs))
    if math.isnan(mu):
        return None
    return (mu, 1.0) if sd < _STD_FLOOR else (mu, sd)


def _patch_wall_row(i: int, ke_eV: float, step_m: np.ndarray, js: np.ndarray):
    """Exact moments for one energy row's wall-band cells.

    Beyond the wall the loss saturates at the full kinetic energy, so every
    saturated cell in the row shares one distribution — computed once and
    reused. Returns (i, js_written, values [n, 2]).
    """
    model = ContinuousStragglingModel()
    out_js, out_vals = [], []
    saturated = None
    for j in js:
        sl_m = float(step_m[j])
        model.set_state(PointInPhaseSpace(step_length=sl_m * U.m2cm, primary_energy=ke_eV))
        is_saturated = model.average_loss >= ke_eV * (1.0 - 1e-12)
        if is_saturated and saturated is not None:
            out_js.append(j)
            out_vals.append(saturated)
            continue
        cell = _exact_cell(model, ke_eV, sl_m)
        if cell is None:
            continue
        if is_saturated:
            saturated = cell
        out_js.append(j)
        out_vals.append(cell)
    return i, np.asarray(out_js, dtype=int), np.asarray(out_vals, dtype=np.float64)


def patch_wall_band(kin_fine: torch.Tensor, step_fine: torch.Tensor, fine: torch.Tensor,
                    n_coarse: int = DEFAULT_COARSE_STEPS, n_jobs: int = -1) -> torch.Tensor:
    """Recompute exact (mu, sigma) for fine cells hugging the stopping wall.

    The wall L = range(E) is located from the init-tables CSDA range (stored
    in cm). Only rows whose wall lies inside the step grid carry the kink;
    each such row patches the fine cells within _WALL_BAND_BELOW/ABOVE_COARSE
    coarse-cell widths of the wall. Cells the builder's rules call
    deterministic keep their existing (skirt) values.
    """
    from physics.g4_init_tables.runtime import InitTableSet

    kin_log = kin_fine.numpy()
    step_log = step_fine.numpy()
    step_m = 10.0 ** step_log
    coarse_width = (step_log[-1] - step_log[0]) / (n_coarse - 1)

    ts = InitTableSet.load()
    range_cm = ts.range(torch.tensor(10.0 ** kin_log, dtype=torch.float64))
    wall_log = torch.log10(range_cm * U.cm2m).numpy()

    lo = wall_log - _WALL_BAND_BELOW_COARSE * coarse_width
    hi = wall_log + _WALL_BAND_ABOVE_COARSE * coarse_width
    rows = np.where((wall_log >= step_log[0]) & (wall_log <= step_log[-1]))[0]
    tasks = []
    for i in rows:
        js = np.where((step_log >= lo[i]) & (step_log <= hi[i]))[0]
        if js.size:
            tasks.append((int(i), float(10.0 ** kin_log[i]), js))
    n_cells = sum(js.size for _, _, js in tasks)
    log.info(f"Wall-band patch: {len(tasks)} energy rows, {n_cells} exact cells...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_patch_wall_row)(i, ke, step_m, js)
        for i, ke, js in tqdm(tasks, desc="Wall-band exact moments")
    )
    n_written = 0
    for i, js, vals in results:
        if js.size:
            fine[i, js] = torch.from_numpy(vals).to(torch.float32)
            n_written += js.size
    log.info(f"Wall-band patch wrote {n_written} cells.")
    return fine


def _save_with_backup(result, cache_path) -> None:
    if cache_path.exists():
        backup = cache_path.with_suffix(cache_path.suffix + ".bak")
        if not backup.exists():
            cache_path.rename(backup)
            log.info(f"Backed up original table to {backup}")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)


def patch_existing_table(n_jobs: int = -1):
    """Apply the wall-band exact patch to the cached fine table in place
    (backing up the prior file once). No coarse rebuild."""
    cache_path = analytic_cache_path()
    if not cache_path.exists():
        raise FileNotFoundError(f"No analytic table to patch at {cache_path}; run a full build first.")
    with open(cache_path, "rb") as f:
        kin_fine, step_fine, fine = pickle.load(f)
    fine = patch_wall_band(kin_fine, step_fine, fine, n_jobs=n_jobs)
    result = (kin_fine, step_fine, fine)
    _save_with_backup(result, cache_path)
    log.info(f"Saved wall-patched analytic lookup table to {cache_path}")
    return result


def build_analytic_mean_std_table(n_coarse: int = DEFAULT_COARSE_STEPS,
                                  force: bool = False, n_jobs: int = -1):
    cache_path = analytic_cache_path()
    if cache_path.exists() and not force:
        log.info(f"Analytic lookup table already exists at {cache_path}; skipping (use --force to rebuild).")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    kin_log, step_log, coarse = compute_coarse_table(n_coarse, n_jobs=n_jobs)
    kin_fine, step_fine, fine = upsample_bilinear(kin_log, step_log, coarse)
    fine = patch_wall_band(kin_fine, step_fine, fine, n_coarse=n_coarse, n_jobs=n_jobs)
    result = (kin_fine, step_fine, fine)

    _save_with_backup(result, cache_path)
    log.info(f"Saved analytic lookup table ({n_coarse}x{n_coarse} -> {FINE_STEPS}x{FINE_STEPS}) to {cache_path}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-coarse", type=int, default=DEFAULT_COARSE_STEPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--patch-wall", action="store_true",
                        help="apply the wall-band exact patch to the existing "
                             "cached table (no coarse rebuild)")
    args = parser.parse_args()
    if args.patch_wall:
        patch_existing_table(n_jobs=args.n_jobs)
    else:
        build_analytic_mean_std_table(n_coarse=args.n_coarse, force=args.force, n_jobs=args.n_jobs)
