"""Figure 8 -- phase-space regime map with the continuous generator's KS-distance overlay."""

from matplotlib import pyplot as plt

# --- 1. Style Configuration (Matched to Nature/Science standard) ---
plt.rcParams.update({
    'font.size': 24,#20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize': 22,#18,
    'ytick.labelsize': 22,#18,
    'legend.fontsize': 22,#16,
    'legend.frameon': False,
    'figure.dpi': 300
})

# Okabe-Ito Color Palette (Colorblind friendly)
PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',  # Changed from Vermilion to Orange for distinction
    'green': '#009E73',
    'red': '#D55E00',  # Vermilion
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'brown': '#8B4513',
}
MARKER_COLORS = [PALETTE['purple'], PALETTE['green'], PALETTE['brown'], PALETTE['orange']]


if __name__ == "__main__":
    import pickle as pkl
    import numpy as np
    import torch
    from matplotlib.colors import LogNorm
    from tqdm import tqdm

    from matplotlib import ticker

    import common.units as U
    from common.utils import PointInPhaseSpace
    from common.paths import DATASETS_DIR, BENCHMARKS_OUTPUTS, PATH_TO_GEANT4_DATA

    from physics.g4h_ionisation.generators.continuous_generator.generate_cdfs import gen_cnt_fluc_cdfs
    from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import StragglingType, ContinuousStragglingModel, \
        ContinuousLossParams
    from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material

    from benchmark.system_vs_g4.net_sampling import phin_gan_ionisation
    net_proc = phin_gan_ionisation()

    particle = proton
    straggling_model = ContinuousStragglingModel(particle, material)

    # --- FIGURE 1: Phase Space Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), constrained_layout=False)

    # Adjusted margins
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)

    # Get phase space data and plot
    params = ContinuousLossParams()
    step_lengths = np.logspace(-5, 3, 200) * U.um2cm
    kinetic_energies = np.logspace(np.log10(0.5), 2, 200) * U.MeV2eV

    straggling_type = []
    for kinetic_energy in tqdm(kinetic_energies):
        straggling_type_at_kin = []
        for step_length in step_lengths:
            params.set_state(PointInPhaseSpace(step_length, kinetic_energy))
            match params.straggling_type:
                case [StragglingType.DeterministicBetheBloch]:
                    straggling_type_at_kin.append(1)
                case [StragglingType.LowColFreqExc, StragglingType.LowColFreqIon]:
                    straggling_type_at_kin.append(2)
                case [StragglingType.LowColFreqExc, StragglingType.LowColFreqIon, StragglingType.HighColFreqIon]:
                    straggling_type_at_kin.append(3)
                case [StragglingType.HighColFreqExc, StragglingType.LowColFreqIon, StragglingType.HighColFreqIon]:
                    straggling_type_at_kin.append(4)
                case [StragglingType.ThickLimit]:
                    straggling_type_at_kin.append(5)
                case _:
                    raise ValueError(f"Unknown straggling type {params.straggling_type}")
        straggling_type.append(straggling_type_at_kin)

    # Plot phase space
    X, Y = np.meshgrid(kinetic_energies, step_lengths)
    Z_type = np.array(straggling_type).T

    # --- Text Placement ---
    region_labels = {
        1: 'Very thin limit',
        2: 'Thin limit Exc.\nThin limit Ion. w/o Gauss.',
        3: 'Thin limit Exc.\nThin limit Ion. w/Gauss.',
        4: 'Gauss. Exc.\nThin limit Ion. w/Gauss.',
        5: 'Thick Limit'
    }

    # The paper's aluminium figure keeps the full 0.5-100 MeV frame and its
    # four sample points. The referee materials' figures (appendix) show only
    # the energies GEANT4 populates -- in iron the proton's last step spends
    # everything below about 2.5 MeV in one go, so the truth has no rows
    # there -- and carry no sample points; the region labels are centred on
    # the visible part of each region.
    with open(DATASETS_DIR / f'{PATH_TO_GEANT4_DATA.stem}_continuous_X_filtered.pkl', 'rb') as file:
        X_truth = pkl.load(file)
    is_paper_beam = material.name.lower() == "aluminum"
    e_truth_min_mev = float(X_truth[:, 0].min()) * U.eV2MeV
    visible = np.ones_like(Z_type, dtype=bool) if is_paper_beam else (X * U.eV2MeV) >= e_truth_min_mev

    for region_id, label_text in region_labels.items():
        mask = (Z_type == region_id) & visible
        if np.any(mask):
            x_region = (X * U.eV2MeV)[mask]
            y_region = (Y * U.cm2um)[mask]

            center_x = 10 ** (np.mean(np.log10(x_region)))
            center_y = 10 ** (np.mean(np.log10(y_region)))

            if region_id == 1:
                center_x *= 0.5

            if region_id == 3 and is_paper_beam:
                center_x = 3e0
                center_y = 2.5e-1

            if region_id == 5:
                center_y *= 5

            match region_id:
                case 1:
                    rotation_angle = 0
                case 2:
                    rotation_angle = 15
                case 3:
                    rotation_angle = 15
                case 4:
                    rotation_angle = 0
                case 5:
                    rotation_angle = 90

            ax.text(center_x, center_y, label_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          rotation=rotation_angle,
                          fontsize=20,
                          fontweight='bold',
                          color='black',
                          bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

    contour_levels = [1.5, 2.5, 3.5, 4.5]
    ax.contour(X * U.eV2MeV, Y * U.cm2um, Z_type, levels=contour_levels, colors='black', linewidths=1.0)

    n_colors = 5
    bounds = np.arange(0.5, n_colors + 1.5, 1)
    ticks = np.arange(1, n_colors + 1)

    # Axis styling
    ax.set_xlabel("$E$ [MeV]", fontweight='bold')
    ax.set_ylabel("$L$ [μm]", fontweight='bold')
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))

    # Histogram overlay
    step_lengths = np.logspace(-5, 3, 1_00) * U.um2cm
    kinetic_energies = np.logspace(np.log10(0.5), 2, 1_00) * U.MeV2eV

    with open(DATASETS_DIR / f'{PATH_TO_GEANT4_DATA.stem}_continuous_X_filtered.pkl', 'rb') as file:
        X = pkl.load(file)
    kins, L = X.T

    # Use a single color gradient (Blues/Greens) rather than Jet for density
    h, xedges, yedges, im = ax.hist2d(kins * U.eV2MeV, L * U.m2um,
                                      bins=[kinetic_energies * U.eV2MeV, step_lengths * U.cm2um],
                                      cmap='afmhot', alpha=1, norm=LogNorm())

    # Right Colorbar
    cbar_ax_right = fig.add_axes([0.87, 0.15, 0.02, 0.8])
    cbar_hist = plt.colorbar(im, cax=cbar_ax_right)
    cbar_hist.set_label('Number of GEANT4 steps', rotation=270, labelpad=15, fontweight='bold')

    # --- Physics Model Run for KS Distance ---
    cdfs_continuous = gen_cnt_fluc_cdfs()

    Ls = []
    kins = []
    loss_vals = []
    # Reduced loop count for faster preview if needed, restore to full in prod
    for (L, kin_energy), cdf in tqdm(cdfs_continuous.items()):
        phase_space = torch.tensor([kin_energy, L], dtype=torch.float32).unsqueeze(0)
        prediction = net_proc.continuous_generator.predict(phase_space, repeat_interleave=100_000)

        pred_cdf_x = np.sort(prediction)
        pred_cdf_y = np.cumsum(np.ones_like(pred_cdf_x)) / len(pred_cdf_x)

        ys_for_comp = cdf(torch.from_numpy(pred_cdf_x)).numpy()
        loss_func_val = np.max(np.abs(ys_for_comp - pred_cdf_y))

        Ls.append(L)
        kins.append(kin_energy)
        loss_vals.append(loss_func_val)

    Ls = np.array(Ls)
    kins = np.array(kins)
    loss_vals = np.array(loss_vals)
    if not is_paper_beam:
        keep = kins * U.eV2MeV >= e_truth_min_mev
        Ls, kins, loss_vals = Ls[keep], kins[keep], loss_vals[keep]
        ax.set_xlim(e_truth_min_mev, 100)

    # Keep only KS points inside the GEANT4 envelope: at each point's energy
    # column, L must lie between the shortest and longest step the truth took
    # there. The CDF grid is gated by a convex hull in linear (E, L), which
    # bridges the dip in the step-limit ceiling at low E and admits keys no
    # GEANT4 step ever reached; those points would score the generator where
    # it has no truth. An envelope rather than a per-bin occupancy test keeps
    # the sparse very-thin fringe, where the truth populates isolated bins.
    # `h` is the histogram drawn behind the dots, so the criterion is the
    # occupancy the reader sees. The column window absorbs the E=100 MeV
    # column, which sits on the last edge, and the finite grid resolution.
    nx, ny = h.shape
    ix = np.clip(np.searchsorted(xedges, kins * U.eV2MeV, side='right') - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(yedges, Ls * U.m2um, side='right') - 1, 0, ny - 1)
    half_window = 1                                  # columns either side
    occupied = np.zeros(len(kins), dtype=bool)
    for k, (i, j) in enumerate(zip(ix, iy)):
        cols = h[max(i - half_window, 0):i + half_window + 1]
        filled = np.flatnonzero(cols.any(axis=0))
        occupied[k] = filled.size > 0 and filled[0] <= j <= filled[-1]
    print(f"[fig] dropping {int((~occupied).sum())} of {len(occupied)} KS points "
          "outside the GEANT4 step-length envelope at their energy")
    Ls, kins, loss_vals = Ls[occupied], kins[occupied], loss_vals[occupied]

    # Scatter plot with Magma (high contrast against Viridis/BuPu)
    sc = ax.scatter(kins * U.eV2MeV, Ls * U.m2um, c=loss_vals,
                    cmap='jet', norm=LogNorm(vmin=1e-2, vmax=1), alpha=1, edgecolor='none')

    # Bottom Colorbar
    cbar_ax_bottom = fig.add_axes([0.2, 0.05, 0.65, 0.02])
    cbar_scatter = plt.colorbar(sc, cax=cbar_ax_bottom, orientation='horizontal')
    cbar_scatter.set_label('KS Distance', fontweight='bold')

    # Markers: the active material's (E [MeV], L [um]) points, shared with fig 9.
    from benchmark.system_vs_g4.marked_points import marked_points as _marked_points
    marked_points = _marked_points()

    markers = ['^', '^', '^', '^']

    if is_paper_beam:
        for i, (energy_mev, step_length_um) in enumerate(marked_points):
            ax.scatter(energy_mev, step_length_um,
                       color=MARKER_COLORS[i], marker=markers[i], s=400,
                       edgecolors='black', linewidth=1.5, zorder=10,
                       label=f'({energy_mev}MeV, {step_length_um}μm)')

        ax.legend(loc='lower right', title="Sample Points")

    plt.savefig(BENCHMARKS_OUTPUTS / 'phase_space_with_markers.pdf', format='pdf', bbox_inches='tight')
    plt.show()
