import logging
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import curve_fit

from common import units as U
from common.enums import G4Columns, TOTAL_ENERGY_LOSS


logging.basicConfig(level=logging.INFO)


plt.rcParams.update({
        'font.size': 28,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 24,
        'legend.frameon': False,
        'figure.dpi': 300
    })


def organize_df(df: pd.DataFrame, displace: float = 0.) -> pd.DataFrame:
    if TOTAL_ENERGY_LOSS not in df.columns:
        total = df[G4Columns.SecondaryLoss] + df[G4Columns.ContinuousLoss]
        df[TOTAL_ENERGY_LOSS] = total
    df = df[
        [G4Columns.EventNum, G4Columns.X, G4Columns.Y, TOTAL_ENERGY_LOSS]].copy()
    df[G4Columns.X] += displace
    df[[G4Columns.X, G4Columns.Y]] *= U.m2mm
    df[TOTAL_ENERGY_LOSS] *= U.eV2MeV

    return df


def compute_hist_stats(
        df: pd.DataFrame,
        range_lims: tuple[tuple[float, float], tuple[float, float]],
        bins: tuple[int, int] = (600, 600),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes histogram stats with correct event-level variance calculation.

    Uses sparse matrix operations instead of a per-event Python loop:
    digitize all points into bins, build a sparse (event × bin) matrix
    where duplicate (event, bin) entries are auto-summed, then derive
    h_sum, h_sum_sq, and h_counts from the sparse matrix.

    Args:
        df: DataFrame containing the data
        range_lims: List of ranges [[min_x, max_x], [min_y, max_y]]
        bins: List of bin counts [nx, ny]

    Returns:
        h_sum.T: The image (Sum of energy).
        h_sum_sq.T: The variance-related quantity (sum of squares of per-event energy depositions).
        h_counts.T: The number of events contributing to each bin.
    """
    x = df[G4Columns.X].values
    y = df[G4Columns.Y].values
    w = df[TOTAL_ENERGY_LOSS].values
    events = df[G4Columns.EventNum].values

    # Drop NaN weights
    finite_mask = np.isfinite(w)
    if not finite_mask.all():
        x, y, w, events = x[finite_mask], y[finite_mask], w[finite_mask], events[finite_mask]

    (x_min, x_max), (y_min, y_max) = range_lims
    nx, ny = bins

    # Digitize into bin indices (0-based)
    x_bin = np.floor((x - x_min) / (x_max - x_min) * nx).astype(np.int64)
    y_bin = np.floor((y - y_min) / (y_max - y_min) * ny).astype(np.int64)

    # Keep only points inside the range (clip right edge into last bin)
    x_bin = np.clip(x_bin, 0, nx - 1)
    y_bin = np.clip(y_bin, 0, ny - 1)
    valid = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x_bin, y_bin, w, events = x_bin[valid], y_bin[valid], w[valid], events[valid]

    # Flat bin index
    flat_bin = x_bin * ny + y_bin
    total_bins = nx * ny

    # Remap event numbers to contiguous 0..N-1 for sparse matrix row indices
    unique_events, event_idx = np.unique(events, return_inverse=True)
    n_events = len(unique_events)

    # Sparse matrix (events × flat_bins): duplicate entries are summed,
    # giving per-event-per-bin total energy deposition
    event_bin = sparse.csr_matrix(
        (w, (event_idx, flat_bin)),
        shape=(n_events, total_bins),
    )

    # h_sum: total energy per bin (sum over events)
    h_sum = np.asarray(event_bin.sum(axis=0)).reshape(nx, ny)

    # h_sum_sq: sum of squared per-event energies per bin
    h_sum_sq = np.asarray(event_bin.power(2).sum(axis=0)).reshape(nx, ny)

    # h_counts: number of events contributing to each bin
    h_counts = np.asarray((event_bin != 0).sum(axis=0)).reshape(nx, ny)

    # Transpose to match the (y, x) convention typically used in imshow
    return h_sum.T, h_sum_sq.T, h_counts.T


def compute_2d_pull(
        hist1: NDArray[np.float64], var_hist_1: NDArray[np.float64],
        hist2: NDArray[np.float64], var_hist_2: NDArray[np.float64],
        # counts_hist_1: NDArray[np.float64],
        # min_counts: int = 30
) -> NDArray[np.float64]:
    # Compute Pull Map
    with np.errstate(divide='ignore', invalid='ignore'):
        # Pull = (Obs - Exp) / sqrt(sigma_obs^2 + sigma_exp^2)
        numerator = hist1 - hist2
        denominator = np.sqrt(var_hist_1 + var_hist_2)
        # pull = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=counts_hist_1 > min_counts)
        pull = numerator / denominator
        # Mask out 0/0 cases clearly
        pull[denominator == 0] = 0

    return pull


def fit_pull_distribution(
        pull_values: NDArray[np.float64],
        n_bins_hist: int = 1_000, hist_range: tuple[float, float] = (-10.0, 10.0),
        fit_range: tuple[float, float] = (-5.0, 5.0)
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64], float]:
    from iminuit import Minuit
    from scipy.stats import norm
    # 1. Clean and filter data to the unbinned fit window
    pull_values = pull_values[np.isfinite(pull_values) & (~np.isnan(pull_values))]
    data_fit = pull_values[(pull_values >= fit_range[0]) & (pull_values <= fit_range[1])]
    print(f"Mean: {np.mean(data_fit)}, std: {np.std(data_fit)}")
    n_obs = len(data_fit)
    bin_width = (hist_range[1] - hist_range[0]) / n_bins_hist

    # 2. Define the Extended Negative Log-Likelihood (NLL)
    def nll(amp, mu, sigma):
        # Prevent non-physical parameters during gradient descent
        if sigma <= 0 or amp <= 0:
            return 1e30

        # Relate the histogram amplitude parameter back to total expected events
        n_exp = amp * sigma * np.sqrt(2 * np.pi) / bin_width

        # Calculate the log-probability of the Gaussian PDF for all data points
        logpdf = norm.logpdf(data_fit, loc=mu, scale=sigma)

        # Extended Maximum Likelihood equation
        return n_exp - n_obs * np.log(n_exp) - np.sum(logpdf)

    # 3. Initial Guesses
    mu_init = np.mean(data_fit)
    sigma_init = np.std(data_fit)
    amp_init = n_obs * bin_width / (sigma_init * np.sqrt(2 * np.pi))

    # 4. Initialize Minuit
    m = Minuit(nll, amp=amp_init, mu=mu_init, sigma=sigma_init)

    # errordef=0.5 is mathematically required for a negative log-likelihood fit
    m.errordef = Minuit.LIKELIHOOD

    # Set physical boundaries to help the MIGRAD algorithm converge safely
    m.limits["sigma"] = (0, None)
    m.limits["amp"] = (0, None)

    # 5. Run the Optimization (MIGRAD) and Error Analysis (HESSE)
    m.migrad()
    m.hesse()

    # 6. Extract Results
    amp_fit = m.values["amp"]
    mean_fit = m.values["mu"]
    sigma_fit = m.values["sigma"]

    popt = np.array([amp_fit, mean_fit, sigma_fit])
    pcov = np.array(m.covariance)  # This is now a true, calculated 3x3 matrix

    # --- Post-fit Goodness-of-Fit (Binned Chi2) ---
    counts, bin_edges = np.histogram(pull_values, bins=n_bins_hist, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
    x_fit_binned = bin_centers[mask]
    y_fit_binned = counts[mask]

    y_err_binned = np.sqrt(y_fit_binned)
    y_err_binned[y_err_binned == 0] = 1

    # Project the unbinned fit back onto the bins using our fitted Amplitude
    y_expected = amp_fit * np.exp(-0.5 * ((x_fit_binned - mean_fit) / sigma_fit) ** 2)

    residuals = y_fit_binned - y_expected
    chi2 = np.sum((residuals / y_err_binned) ** 2)

    # We now actively fit 3 parameters (amp, mu, sigma)
    ndof = len(x_fit_binned) - 3
    chi2_ndof = chi2 / ndof if ndof > 0 else 0

    print(
        f"iminuit EML Fit: Amplitude={amp_fit:.2f}+-{m.errors['amp']:.2f}, "
        f"Mean={mean_fit:.4f}+-{m.errors['mu']:.4f}, "
        f"Sigma={sigma_fit:.4f}+-{m.errors['sigma']:.4f}, "
        f"Chi2/NDOF={chi2_ndof:.2f}"
    )

    return (bin_centers, counts), popt, pcov, chi2_ndof



def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def plot_tracks(
        ax: plt.Axes,
        hist_2d: np.ndarray,
        extent: tuple[float, float, float, float],
        norm: matplotlib.colors.Normalize,
        title: str,
) -> AxesImage:

    # Create 2D histogram plot
    im = ax.imshow(
        hist_2d,
        origin='lower',
        extent=extent,
        cmap='jet',
        norm=norm,
        aspect='auto'
    )
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('y (transverse axis) [mm]')
    ax.set_xlim(extent[0], extent[1])

    ax.set_ylim(extent[2], extent[3])

    return im


def pull_analysis(
        df_1: pd.DataFrame, df_2: pd.DataFrame,
        n_2d_bins: tuple[int, int] = (400, 400),
        pull_min_counts_per_bin: int = 30,
        pull_1d_hist_range: tuple[float, float] = (-10.0, 10.0),
        pull_fit_range: tuple[float, float] = (-5.0, 5.0),
        pull_1d_bins: int = 1_000,
        label: str = "PHIN-GAN",
        tolerance_text: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """`label` names the simulated arm (df_2) in every panel title -- the
    paper's Fig 13 is the PHIN-GAN arm, its appendix Fig E.2 the no-physics
    GAN ablation. `tolerance_text` (optional) is drawn verbatim below the
    fitted mu_0/sigma_0 in the 1-D pull panel, separated by a dashed rule --
    the paper's figure carries its G4-vs-G4 tolerance band there, and the
    caller owns that text (this library never computes it). Nothing is
    written to disk here: the caller saves the returned figure (an earlier
    revision also `plt.savefig`'d it to `track_comparison.pdf`, which
    produced a byte-for-byte duplicate of the driver's
    `energy_deposition_pull*.pdf`)."""
    # Compute Track Histograms
    # Define common range and bins
    x_range = (df_1[G4Columns.X].min(), df_1[G4Columns.X].max())
    y_range = (df_1[G4Columns.Y].min(), df_1[G4Columns.Y].max())
    range_lims = [x_range, y_range]
    hist_1, var_hist_1, counts_hist_1 = compute_hist_stats(df_1, range_lims, n_2d_bins)
    hist_2, var_hist_2, counts_hist_2 = compute_hist_stats(df_2, range_lims, n_2d_bins)

    pull_2d = compute_2d_pull(hist_1, var_hist_1, hist_2, var_hist_2)#, counts_hist_1, min_counts=pull_min_counts_per_bin)

    # zero for bins with less than pull_min_counts_per_bin in the reference histogram
    pull_values = pull_2d.copy().ravel()
    avg_count = (counts_hist_1.ravel() + counts_hist_2.ravel())/2
    pull_values[avg_count <= pull_min_counts_per_bin] = np.nan
    print(f"Number of values above threshold: {sum(avg_count > pull_min_counts_per_bin)}")

    pull_histogram, popt, pcov, chi2_ndof = fit_pull_distribution(
        pull_values=pull_values,
        n_bins_hist=pull_1d_bins,
        hist_range=pull_1d_hist_range,
        fit_range=pull_fit_range,
    )
    amp_fit, mean_fit, sigma_fit = popt
    mean_err, sigma_err = np.sqrt(pcov[1, 1]), np.sqrt(pcov[2, 2])
    bin_centers, bin_counts = pull_histogram

    # --- Plotting Setup ---
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.05)

    axs_top = subfigs[0].subplots(1, 2)
    ax_nn_tracks, ax_g4_tracks = axs_top
    axs_bot = subfigs[1].subplots(1, 2)
    ax_pull_map, ax_pull_hist = axs_bot

    # --- Shared Configuration ---
    x_label = 'z (primary beam axis) [mm]'

    # Link Y-axes so zooming one affects the other
    ax_g4_tracks.sharey(ax_nn_tracks)

    # --- 1. PHIN-GAN Tracks (Top Left) ---
    im_nn = plot_tracks(
        ax=ax_nn_tracks,
        hist_2d=hist_2,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        norm=LogNorm(vmin=1, vmax=100),
        title=label
    )

    # --- SHARED X-LABEL (Top Row) ---
    subfigs[0].supxlabel(x_label)

    # --- 2. GEANT4 Tracks (Top Right) ---
    im_g4 = plot_tracks(
        ax=ax_g4_tracks,
        hist_2d=hist_1,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        norm=LogNorm(vmin=1, vmax=100),
        title="GEANT4"
    )
    # Remove Y ticks and labels since they are shared with the left plot
    ax_g4_tracks.tick_params(axis='y', left=False, labelleft=False)

    # --- SHARED COLORBAR (Top Row) ---
    # ax=ax_g4_tracks: Anchors the bar to the right plot
    # pad=0.01: Minimizes the gap between the plot and the colorbar
    # cbar_shared = fig.colorbar(im_nn, ax=ax_g4_tracks, location='right', aspect=30, pad=0.01)
    # cbar_shared.set_label('Energy Deposition [MeV]')
    cbar_shared = fig.colorbar(
        im_nn,
        ax=[ax_nn_tracks, ax_g4_tracks],  # <--- The Critical Change
        location='right',
        aspect=30,
        pad=0.01
    )
    cbar_shared.set_label('Energy Deposition [MeV]')

    # --- 3. Pull Map (Bottom Left) ---
    # Mask low stats
    pull_masked = pull_2d.copy()
    pull_masked[counts_hist_1 == 0] = np.nan
    # pull_masked[counts_hist_1 <= pull_min_counts_per_bin] = np.nan

    im_pull = plot_tracks(
        ax=ax_pull_map,
        hist_2d=pull_masked,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        norm=plt.Normalize(vmin=-4, vmax=4),
        title=f"{label} vs GEANT4 pull map"
    )
    ax_pull_map.set_xlabel(x_label)

    # Pull Map Colorbar
    cbar_pull = fig.colorbar(im_pull, ax=ax_pull_map, location='right', aspect=30, pad=0.02)
    cbar_pull.set_label(r"Pull ($\sigma$)")

    # --- 4. Pull Histogram (Bottom Right) ---
    # Move the Y-axis to the right side
    ax_pull_hist.yaxis.tick_right()
    ax_pull_hist.yaxis.set_label_position("right")

    x_fine = np.linspace(pull_1d_hist_range[0], pull_1d_hist_range[1], 200)
    ax_pull_hist.bar(
        bin_centers, bin_counts,
        width=bin_centers[1] - bin_centers[0],
        color='black', label='Data', zorder=1, alpha=0.7,
    )
    ax_pull_hist.plot(x_fine, gaussian(x_fine, *popt), color='#D55E00', linewidth=4, label='Fit', zorder=2)

    text_str = (
        f"$\\mu_0$: {mean_fit:.3f} $\\pm$ {mean_err:.3f}\n"
        f"$\\sigma_0$: {sigma_fit:.3f} $\\pm$ {sigma_err:.3f}\n"
    )
    if tolerance_text is not None:
        text_str += "-" * 22 + "\n" + tolerance_text
    ax_pull_hist.text(0.03, 0.95, text_str, transform=ax_pull_hist.transAxes, zorder=0,
                      fontsize=24, verticalalignment='top',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_pull_hist.set_title(f"{label} vs GEANT4 pull", fontweight='bold')
    ax_pull_hist.set_xlabel(r"Pull Values (N$\sigma$)")
    ax_pull_hist.set_ylabel("Bins in y-z plane")
    ax_pull_hist.legend(loc='upper right')
    ax_pull_hist.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_pull_hist.set_xlim(pull_1d_hist_range[0], pull_1d_hist_range[1])

    return fig, (ax_nn_tracks, ax_g4_tracks, ax_pull_map, ax_pull_hist)


# ---------------------------------------------------------------------------
# Jackknife baseline & Z-score evaluation
# ---------------------------------------------------------------------------

def compute_pull_mu_sigma(
        df_1: pd.DataFrame, df_2: pd.DataFrame,
        n_2d_bins: tuple[int, int] = (1500, 1500),
        pull_min_counts_per_bin: int = 20,
        pull_1d_hist_range: tuple[float, float] = (-5.0, 5.0),
        pull_fit_range: tuple[float, float] = (-30.0, 30.0),
) -> tuple[float, float]:
    """Compute pull distribution mu and sigma for a pair of track DataFrames (no plotting)."""
    x_range = (df_1[G4Columns.X].min(), df_1[G4Columns.X].max())
    y_range = (df_1[G4Columns.Y].min(), df_1[G4Columns.Y].max())
    range_lims = [x_range, y_range]

    hist_1, var_hist_1, counts_hist_1 = compute_hist_stats(df_1, range_lims, n_2d_bins)
    hist_2, var_hist_2, counts_hist_2 = compute_hist_stats(df_2, range_lims, n_2d_bins)

    pull_2d = compute_2d_pull(hist_1, var_hist_1, hist_2, var_hist_2)

    pull_values = pull_2d.ravel()
    avg_count = (counts_hist_1.ravel() + counts_hist_2.ravel()) / 2
    pull_values[avg_count <= pull_min_counts_per_bin] = np.nan

    _, popt, _, _ = fit_pull_distribution(
        pull_values=pull_values,
        hist_range=pull_1d_hist_range,
        fit_range=pull_fit_range,
    )
    mu, sigma = popt[1], popt[2]
    return mu, sigma


def compute_all_pairwise_pulls(
        dfs: list[pd.DataFrame],
        **pull_kwargs,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[tuple[int, int]]]:
    """
    Compute pull (mu, sigma) for all C(N,2) pairs of DataFrames.

    Returns:
        mus: array of shape (n_pairs,)
        sigmas: array of shape (n_pairs,)
        pair_indices: list of (i, j) tuples identifying each pair
    """
    n = len(dfs)
    pair_indices = list(combinations(range(n), 2))
    mus = np.empty(len(pair_indices))
    sigmas = np.empty(len(pair_indices))

    for k, (i, j) in enumerate(pair_indices):
        logging.info(f"Computing pull for pair ({i}, {j})  [{k + 1}/{len(pair_indices)}]")
        mu, sigma = compute_pull_mu_sigma(dfs[i], dfs[j], **pull_kwargs)
        mus[k] = mu
        sigmas[k] = sigma

    return mus, sigmas, pair_indices


def jackknife_pull_estimate(
        mus: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        pair_indices: list[tuple[int, int]],
        n_batches: int,
) -> dict[str, float]:
    """
    Jackknife estimate of mean and standard error for pull mu and sigma.

    Leave-one-batch-out: when batch k is removed, all pairs involving k are dropped.
    This gives n_batches jackknife replicates.

    Returns dict with keys: mu_mean, mu_se, sigma_mean, sigma_se
    """
    pair_indices_arr = np.array(pair_indices)

    # Full-sample estimates
    mu_full = np.mean(mus)
    sigma_full = np.mean(sigmas)

    # Jackknife replicates
    mu_jk = np.empty(n_batches)
    sigma_jk = np.empty(n_batches)

    for k in range(n_batches):
        # Keep pairs that don't involve batch k
        mask = (pair_indices_arr[:, 0] != k) & (pair_indices_arr[:, 1] != k)
        mu_jk[k] = np.mean(mus[mask])
        sigma_jk[k] = np.mean(sigmas[mask])

    # Jackknife bias-adjusted estimate: n * theta_full - (n-1) * mean(theta_jk)
    n = n_batches
    mu_mean = n * mu_full - (n - 1) * np.mean(mu_jk)
    sigma_mean = n * sigma_full - (n - 1) * np.mean(sigma_jk)

    # Jackknife standard error
    mu_se = np.sqrt((n - 1) / n * np.sum((mu_jk - np.mean(mu_jk)) ** 2))
    sigma_se = np.sqrt((n - 1) / n * np.sum((sigma_jk - np.mean(sigma_jk)) ** 2))

    return {
        'mu_mean': mu_mean, 'mu_se': mu_se,
        'sigma_mean': sigma_mean, 'sigma_se': sigma_se,
        'mu_jk_replicates': mu_jk, 'sigma_jk_replicates': sigma_jk,
    }


