"""Figure 10: 1D histogram comparisons of step-level features.

Produces 1D histogram comparisons of step-level features (step length,
continuous loss, secondary loss, scattering angle) for GEANT4, PHIN-GAN,
and no-physics GAN, styled for publication.
"""

import numpy as np
from matplotlib import pyplot as plt

# --- Style Configuration ---
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.5,  # Slightly thicker for publication quality
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'legend.frameon': False,
    'figure.dpi': 150
})

# Colors: High contrast, Colorblind friendly
# Deep Blue (GAN), Vermilion (PHIN-GAN), Orange/Amber (GEANT4) - Adjusted slightly for visibility
colors = ['#E69F00', '#D55E00', '#0072B2']
labels = ['GEANT4', 'PHIN-GAN', 'GAN']


def plot_stylish_hist(ax, data_list, bins, xlabel, panel_label):
    """
    Plots histograms with specific styles:
    0 (GEANT4): Filled Area
    1 (PHIN-GAN): Solid Line
    2 (GAN): Dashed Line
    """

    for i, data in enumerate(data_list):
        label = labels[i]
        color = colors[i]

        match i:
            case 0:  # GEANT4 -> Solid Fill Area
                # Main Fill
                ax.hist(data, bins=bins, density=True,
                        histtype='stepfilled', alpha=0.4,  # Alpha makes it look "Solid" but allows grid visibility
                        color=color, label=label, zorder=1)
                # Add a thin edge line to the fill for definition
                ax.hist(data, bins=bins, density=True,
                        histtype='step', linewidth=1,
                        color=color, zorder=2)
            case 1:  # PHIN-GAN -> Solid Line
                ax.hist(data, bins=bins, density=True,
                        histtype='step', linewidth=2, linestyle='-',
                        color=color, label=label, zorder=4)  # Highest zorder to be on top
            case 2:  # GAN -> Dashed Line
                ax.hist(data, bins=bins, density=True,
                        histtype='step', linewidth=2, linestyle='--',
                        color=color, label=label, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontweight='bold')

    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    # Legend only on the first plot to avoid clutter
    if panel_label == 'd':
        ax.legend(loc='upper left')  # 'best' usually avoids covering data, or use 'upper right'


if __name__ == "__main__":
    import pickle as pkl
    import torch

    from common.enums import SecondaryFeats
    from common.paths import BENCHMARKS_OUTPUTS, DATASETS_DIR, PATH_TO_GEANT4_DATA
    from physics.g4h_ionisation.explicit_physics.discrete._kernels import _compute_primary_theta_jit
    from common.run_context import PARTICLE as proton
    from physics.definitions.physical_constants import electron_mass_eV

    def _theta_from_esec(E_primary: torch.Tensor, e_sec: torch.Tensor) -> torch.Tensor:
        return _compute_primary_theta_jit(
            E_primary.flatten(), e_sec.flatten(),
            float(proton.mass_eV), float(electron_mass_eV),
        )

    from benchmark.system_vs_g4.net_sampling import (
        phin_gan_ionisation, preset_ionisation)
    from benchmark.track_simulator_config import has_gan_preset
    net_stepping_manager = phin_gan_ionisation()
    # The no-physics ablation exists for aluminium only; other beams draw
    # two arms (GEANT4, PHIN-GAN).
    gan_proc = preset_ionisation("gan") if has_gan_preset() else None

    stem = PATH_TO_GEANT4_DATA.stem

    # --- Load filtered data for each generator ---
    # Length (no filtering applied by datamodule)
    with open(DATASETS_DIR / f'{stem}_step_length_X.pkl', 'rb') as f:
        length_X = torch.from_numpy(pkl.load(f)).float()
    with open(DATASETS_DIR / f'{stem}_step_length_Y.pkl', 'rb') as f:
        length_Y = torch.from_numpy(pkl.load(f)).float()

    # Continuous (filtered: no zero/deterministic/all-energy-lost)
    with open(DATASETS_DIR / f'{stem}_continuous_X_filtered.pkl', 'rb') as f:
        continuous_X = torch.from_numpy(pkl.load(f)).float()
    with open(DATASETS_DIR / f'{stem}_continuous_Y_filtered.pkl', 'rb') as f:
        continuous_Y = torch.from_numpy(pkl.load(f)).float()

    # Secondary (filtered: no zero-secondary rows)
    with open(DATASETS_DIR / f'{stem}_secondary_X_filtered.pkl', 'rb') as f:
        secondary_X = torch.from_numpy(pkl.load(f)).float()
    with open(DATASETS_DIR / f'{stem}_secondary_Y_filtered.pkl', 'rb') as f:
        secondary_Y = torch.from_numpy(pkl.load(f)).float()

    # --- Generate predictions using individual generators ---
    # Neural system
    L_gen     = net_stepping_manager.length_generator.predict(length_X).detach().numpy()
    cnt_gen   = net_stepping_manager.continuous_generator.predict(continuous_X).detach().numpy()
    sec_gen   = net_stepping_manager.secondary_generator.predict(secondary_X).detach()
    E_sec_gen_t = sec_gen[:, SecondaryFeats.E_SEC_IDX]
    theta_gen = _theta_from_esec(secondary_X, E_sec_gen_t).numpy()
    E_sec_gen = E_sec_gen_t.numpy()

    # No-physics GAN ablation. Length is shared between the two presets
    # (both build from the same PhysicsLengthGenerator instance), so L_gen
    # is reused rather than recomputed.
    gan_arm = {}
    if gan_proc is not None:
        cnt_gan   = gan_proc.continuous_generator.predict(continuous_X).detach().numpy()
        sec_gan   = gan_proc.secondary_generator.predict(secondary_X).detach()
        E_sec_gan_t = sec_gan[:, SecondaryFeats.E_SEC_IDX]
        theta_gan = _theta_from_esec(secondary_X, E_sec_gan_t).numpy()
        E_sec_gan = E_sec_gan_t.numpy()
        gan_arm = {"L": [L_gen], "cnt": [cnt_gan], "sec": [E_sec_gan], "theta": [theta_gan]}
    else:
        gan_arm = {"L": [], "cnt": [], "sec": [], "theta": []}

    # --- G4 truth ---
    # The saved Y stores only E_SEC; theta is reconstructed from E_SEC via the
    # same energy-momentum conservation GEANT4 uses, so this matches angleDiscrete.
    L_g4     = length_Y.numpy().squeeze()
    cnt_g4   = continuous_Y.numpy().squeeze()
    E_sec_g4_t = secondary_Y.squeeze()
    E_sec_g4 = E_sec_g4_t.numpy()
    theta_g4 = _theta_from_esec(secondary_X, E_sec_g4_t).numpy()

    # --- Binning ---
    L_bins     = np.logspace(np.log10(L_g4.min()),     np.log10(L_g4.max()),     100)
    cnt_bins   = np.logspace(np.log10(cnt_g4.min()),   np.log10(cnt_g4.max()),   100)
    sec_bins   = np.logspace(np.log10(E_sec_g4.min()), np.log10(E_sec_g4.max()), 100)
    theta_bins = np.logspace(np.log10(theta_g4.min()), np.log10(theta_g4.max()), 100)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Order of data in list: [GEANT4, PHIN-GAN, GAN] -- length is shared
    # between the two presets, so the third entry reuses L_gen.
    plot_stylish_hist(axs[0, 0], [L_g4, L_gen] + gan_arm["L"],             L_bins,     r'$L$ [m]',             'a')
    plot_stylish_hist(axs[0, 1], [cnt_g4, cnt_gen] + gan_arm["cnt"],       cnt_bins,   r'$\Delta E^{\rm cnt}$ [eV]', 'b')
    plot_stylish_hist(axs[1, 0], [E_sec_g4, E_sec_gen] + gan_arm["sec"],   sec_bins, r'$\Delta E^{\rm sec}$ [eV]',  'c')
    plot_stylish_hist(axs[1, 1], [theta_g4, theta_gen] + gan_arm["theta"], theta_bins, r'$\theta$ [rad]',    'd')

    fig.supylabel('Probability Density', fontweight='bold')

    plt.savefig(BENCHMARKS_OUTPUTS / '1d_hists.pdf', format='pdf')
    plt.show()