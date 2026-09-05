"""Figure 11 -- pairwise relative-difference comparison of PHIN-GAN vs GEANT4."""

import numpy as np
import math
import matplotlib as mpl

from matplotlib import pyplot as plt

import common.units as U
from common.config import MIN_ENERGY_CUTOFF, BEAM_ENERGY_eV
from common.run_context import PLOT_RANGES

# --- Nature Communications Styling (High Visibility) ---
plt.rcParams.update({
    'font.size': 30, #20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.5,  # Thicker spines for visibility
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize': 28, #18,
    'ytick.labelsize': 28, #18,
    'axes.labelsize': 30,#20,
    'legend.fontsize': 26,#16,
    'legend.frameon': False,
    'figure.dpi': 300
})


# --- Bins ---
ke_bins        = np.logspace(math.log10(MIN_ENERGY_CUTOFF), math.log10(BEAM_ENERGY_eV), 100)
l_bins         = np.logspace(*PLOT_RANGES.step_length, 100)
theta_bins     = np.logspace(*PLOT_RANGES.theta,       100)
cnt_bins       = np.logspace(*PLOT_RANGES.cnt_loss,    100)
sec_bins       = np.logspace(*PLOT_RANGES.sec_loss,    100)


def plot_pairwise_relative_diff(
        ax: plt.Axes,
        x_ref: np.ndarray, y_ref: np.ndarray,
        x_cmp: np.ndarray, y_cmp: np.ndarray,
        x_bins: np.ndarray, y_bins: np.ndarray,
        ylabel: str = '',
        xlabel: str = '',
        show_ylabel: bool = True,
        show_xlabel: bool = True,
        bold_labels: bool = True,
):
    ref_hist, _, _ = np.histogram2d(x_ref, np.abs(y_ref), bins=[x_bins, y_bins], density=False)
    cmp_hist, _, _ = np.histogram2d(x_cmp, np.abs(y_cmp), bins=[x_bins, y_bins], density=False)

    if x_ref.size > 0:
        ref_hist = ref_hist / x_ref.size
    if x_cmp.size > 0:
        cmp_hist = cmp_hist / x_cmp.size

    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(cmp_hist - ref_hist) / ref_hist
        threshold = 30 / x_ref.size if x_ref.size > 0 else 0
        diff[ref_hist < threshold] = 0
        diff = np.nan_to_num(diff)

    X, Y = np.meshgrid(x_bins, y_bins)
    mesh = ax.pcolormesh(X, Y, diff.T, cmap='jet',
                         norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1e0),
                         shading='auto', rasterized=True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    weight = 'bold' if bold_labels else 'normal'
    if show_ylabel:
        ax.set_ylabel(ylabel, fontweight=weight)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel(xlabel, fontweight=weight)

    return mesh


# Within-generator pairwise comparisons (same filtered dataset per row)
COMPARISONS = [
    # (x_key, y_key, x_bins, y_bins, xlabel, ylabel)
    ('E_l',   'L',         ke_bins, l_bins,         '$E$ [eV]',    '$L$ [m]'),
    ('E_cnt', 'E_CNT',     ke_bins, cnt_bins,       '$E$ [eV]',    r'$\Delta E^{\rm cnt}$ [eV]'),
    ('L_cnt', 'E_CNT',     l_bins,  cnt_bins,       '$L$ [m]',     r'$\Delta E^{\rm cnt}$ [eV]'),
    ('E_sec', 'theta',     ke_bins, theta_bins,     '$E$ [eV]',    r'$\theta_{\rm discrete}$ [rad]'),
    ('E_sec', 'E_SEC',     ke_bins, sec_bins,       '$E$ [eV]',    r'$\Delta E^{\rm sec}$ [eV]'),
    ('E_SEC', 'theta',     sec_bins, theta_bins,    r'$\Delta E^{\rm sec}$ [eV]', r'$\theta_{\rm discrete}$ [rad]'),
]


if __name__ == '__main__':
    import torch

    from common.paths import BENCHMARKS_OUTPUTS
    from common.enums import G4Columns, StepFeaturesIdxes as SFI
    from data_handling.reader import read_geant4_simulation_output
    from benchmark.track_simulator_config import build_stepping_manager

    # --- Load G4 reference data ---
    df = read_geant4_simulation_output()
    kinetic_energy = torch.tensor(df[G4Columns.KineticEnergy].values, dtype=torch.float64)

    # --- Run end-to-end stepping manager (float64 to preserve small MSC angles) ---
    net_sm = build_stepping_manager("phin_gan", device='cpu')
    net_sm.to('cpu', dtype=torch.float64)

    with torch.inference_mode():
        phys_out = net_sm(kinetic_energy).numpy()

    # --- No-physics GAN ablation arm, run the same way. The ablation exists
    # for aluminium only; other beams draw two columns. ---
    from benchmark.track_simulator_config import has_gan_preset
    gan_out = None
    if has_gan_preset():
        gan_sm = build_stepping_manager("gan", device='cpu')
        gan_sm.to('cpu', dtype=torch.float64)

        with torch.inference_mode():
            gan_out = gan_sm(kinetic_energy).numpy()

    # --- Extract G4 truth arrays ---
    E        = df[G4Columns.KineticEnergy].values
    L_g4     = df[G4Columns.StepLength].values
    E_CNT_g4 = df[G4Columns.ContinuousLoss].values
    theta_g4 = df[G4Columns.angleDiscrete].values
    E_SEC_g4 = df[G4Columns.SecondaryLoss].values

    # --- Extract prediction arrays ---
    L_pred         = phys_out[:, SFI.L]
    E_CNT_pred     = phys_out[:, SFI.E_CNT]
    theta_pred     = phys_out[:, SFI.THETA]
    E_SEC_pred     = phys_out[:, SFI.E_SEC]

    # Map keys to arrays for lookup by the comparison table
    g4_arrays = {
        'E_l': E, 'L': L_g4,
        'E_cnt': E, 'L_cnt': L_g4, 'E_CNT': E_CNT_g4,
        'E_sec': E, 'theta': theta_g4, 'E_SEC': E_SEC_g4,
    }
    gen_arrays = {
        'E_l': E, 'L': L_pred,
        'E_cnt': E, 'L_cnt': L_pred, 'E_CNT': E_CNT_pred,
        'E_sec': E, 'theta': theta_pred, 'E_SEC': E_SEC_pred,
    }

    # (title, cmp_arrays) -- the paper's fig 11 is three columns:
    # GAN vs G4 | PHIN-GAN vs G4 | G4 vs G4. G4-vs-G4 is the split-halves
    # self-comparison. Beams without the ablation drop the first column.
    ARMS = [
        ("PHIN-GAN vs GEANT4", gen_arrays),
        ("GEANT4 vs GEANT4", None),   # None => split-halves self-comparison
    ]
    if gan_out is not None:
        gan_arrays = {
            'E_l': E, 'L': gan_out[:, SFI.L],
            'E_cnt': E, 'L_cnt': gan_out[:, SFI.L], 'E_CNT': gan_out[:, SFI.E_CNT],
            'E_sec': E, 'theta': gan_out[:, SFI.THETA], 'E_SEC': gan_out[:, SFI.E_SEC],
        }
        ARMS.insert(0, ("GAN vs GEANT4", gan_arrays))
    n_rows, n_cols = len(COMPARISONS), len(ARMS)

    fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols + 1,
                          width_ratios=[1] * n_cols + [0.05])

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    for col, (title, _) in enumerate(ARMS):
        axes[0, col].set_title(title, fontsize=28, fontweight='bold', pad=20)

    for row, (x_key, y_key, xb, yb, xlabel, ylabel) in enumerate(COMPARISONS):
        x_g4, y_g4 = g4_arrays[x_key], g4_arrays[y_key]
        N_half = len(x_g4) // 2
        x_g4_1, y_g4_1 = x_g4[:N_half], y_g4[:N_half]
        for col, (title, cmp_arrays) in enumerate(ARMS):
            if cmp_arrays is None:
                x_cmp = x_g4[N_half:2 * N_half]
                y_cmp = y_g4[N_half:2 * N_half]
            else:
                x_cmp = cmp_arrays[x_key][:N_half]
                y_cmp = cmp_arrays[y_key][:N_half]
            plot_pairwise_relative_diff(
                axes[row, col], x_g4_1, y_g4_1, x_cmp, y_cmp, xb, yb,
                ylabel=ylabel, xlabel=xlabel,
                show_ylabel=(col == 0), show_xlabel=(col == n_cols // 2),
            )

    # Shared colorbar
    norm = mpl.colors.LogNorm(vmin=1e-2, vmax=1e0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])

    cax = fig.add_subplot(gs[:, n_cols])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Relative difference", fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=28)

    fig.savefig(BENCHMARKS_OUTPUTS / "pairwise_comparison.pdf", format="pdf", bbox_inches='tight')
    plt.show()
