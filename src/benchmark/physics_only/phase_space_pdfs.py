"""Visualizes the continuous energy-loss straggling phase space and PDFs.
Maps the straggling-type regimes (deterministic Bethe-Bloch, low/high collision
frequency) across (kinetic energy, step length) space, and plots example
ionization and excitation PDFs at selected phase-space points.
"""

from matplotlib import pyplot as plt

# --- 1. Style Configuration ---
plt.rcParams.update({
    'font.size': 24,
    # 'font.weight': 'bold',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 2.5,
    # 'axes.labelweight': 'bold',
    # 'axes.titleweight': 'bold',
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'legend.frameon': False,
    'figure.dpi': 150
})

COLORS = {
    'ion_cp': '#D55E00',
    'ion_gauss': '#009E73',
    'exc': '#0072B2',
    'total': '#000000'
}

if __name__ == "__main__":
    import numpy as np
    from matplotlib import ticker

    import common.units as U
    import common.utils as utils
    from common.utils import PointInPhaseSpace
    from common.paths import BENCHMARKS_OUTPUTS

    from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import (
        few_excitation_collisions_pdf, many_excitation_collisions_pdf,
        StragglingType, ContinuousLossParams
    )
    from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
    from physics.g4h_ionisation.explicit_physics.continuous import _ionization_straggling_function as bm

    # MC overlays sample from the gate-verified torch kernels (the production
    # samplers) — single source of truth with the runtime, per-component.
    import torch
    from physics.g4h_ionisation.explicit_physics.continuous._kernels import (
        _sample_gauss_jit,
        _sample_low_freq_excitation_jit,
        _sample_low_freq_ionisation_jit,
    )

    _MAX_GAUSS_ITER = 25  # matches PhysicsContinuousGenerator

    def _const(value: float, n: int) -> torch.Tensor:
        return torch.full((n,), float(value), dtype=torch.float64)

    def _mc_low_freq_exc(a1: float, e1: float, n: int) -> np.ndarray:
        active = torch.ones(n, dtype=torch.bool)
        return _sample_low_freq_excitation_jit(_const(a1, n), _const(e1, n), active).numpy()

    def _mc_gauss(mean: float, sig2: float, n: int) -> np.ndarray:
        active = torch.ones(n, dtype=torch.bool)
        return _sample_gauss_jit(
            _const(mean, n), _const(np.sqrt(sig2), n), active, _MAX_GAUSS_ITER,
        ).numpy()

    def _mc_compound_pois_ion(p3: float, w3: float, w: float, n: int) -> np.ndarray:
        active = torch.ones(n, dtype=torch.bool)
        return _sample_low_freq_ionisation_jit(
            _const(p3, n), _const(w3, n), _const(w, n), active,
        ).numpy()

    N = 1_000_000
    particle = proton
    continuous_loss = ContinuousLossParams(particle, material)

    # --- UPDATED FIGURE SIZE: Increased height from 10 to 18 ---
    fig = plt.figure(figsize=(28, 16), constrained_layout=True)

    # 5-column grid
    gs = fig.add_gridspec(3, 5)

    # --- Left Panel: Phase Space ---
    ax_phase = fig.add_subplot(gs[:, :2])

    # Get phase space data (step_lengths kept for plot axis bounds)
    step_lengths = np.logspace(-5, 3, 1000) * U.um2cm

    from physics.g4h_ionisation.explicit_physics.continuous.phase_space_boundaries import (
        compute_phase_space_boundaries,
    )

    print("Computing phase-space boundaries...")
    bounds = compute_phase_space_boundaries(particle, material, n_energy=500)

    E_MeV = bounds.energy_eV * U.eV2MeV
    vt_um  = bounds.boundaries["very_thin"]   * U.m2um
    ig_um  = bounds.boundaries["ion_gauss"]   * U.m2um
    ef_um  = bounds.boundaries["exc_freq"]    * U.m2um
    tl_um  = bounds.boundaries["thick_limit"] * U.m2um

    L_PLOT_MIN_um = step_lengths[0]  * U.cm2um
    L_PLOT_MAX_um = step_lengths[-1] * U.cm2um

    # Cascading fallback: when a transition boundary is absent, the simpler regime
    # below it absorbs the space that would have belonged to the more complex regime.
    #   r3_upper: ef_um if excitation-freq boundary exists, else extend teal to tl_um
    #   r2_upper: ig_um if ion-gauss boundary exists, else inherit r3_upper
    #   (NaN lower bounds in region_fills cause fill_between to skip that region)
    tl_or_max = np.where(np.isnan(tl_um), L_PLOT_MAX_um, tl_um)
    r3_upper  = np.where(np.isnan(ef_um), tl_or_max, ef_um)
    r2_upper  = np.where(np.isnan(ig_um), r3_upper,  ig_um)

    # --- Region fills (fill_between handles NaN by leaving gaps) ---
    region_fills = [
        # (lower_y,                                upper_y,    region_id)
        (np.full_like(E_MeV, L_PLOT_MIN_um),       vt_um,     1),
        (vt_um,                                     r2_upper,  2),  # absorbs 3+4 if ig_um absent
        (ig_um,                                     r3_upper,  3),  # NaN lower skips; absorbs 4 if ef_um absent
        (ef_um,                                     tl_or_max, 4),  # NaN lower skips if ef_um absent
        (tl_um,  np.full_like(E_MeV, L_PLOT_MAX_um),          5),
    ]
    colors = {r["id"]: r["color"] for r in bounds.regions}
    for lower, upper, rid in region_fills:
        ax_phase.fill_between(
            E_MeV, lower, upper,
            where=~(np.isnan(lower) | np.isnan(upper)),
            color=colors[rid], alpha=0.4, linewidth=0,
        )

    # --- Boundary lines ---
    for arr_um, label in [
        (vt_um, "very_thin"), (ig_um, "ion_gauss"),
        (ef_um, "exc_freq"),  (tl_um, "thick_limit"),
    ]:
        ax_phase.plot(E_MeV, arr_um, color="black", linewidth=1.0)

    # --- Region labels ---
    region_bounds_pairs = [
        (np.full_like(E_MeV, L_PLOT_MIN_um), vt_um),
        (vt_um,   r2_upper),
        (ig_um,   r3_upper),
        (ef_um,   tl_or_max),
        (tl_um,   np.full_like(E_MeV, L_PLOT_MAX_um)),
    ]
    region_rotations = [0, 20, 20, 0, 90]
    for (lower, upper), region, rotation in zip(
        region_bounds_pairs, bounds.regions, region_rotations
    ):
        valid = ~(np.isnan(lower) | np.isnan(upper) | (lower <= 0) | (upper <= 0))
        if not np.any(valid):
            continue
        idx = np.where(valid)[0][len(np.where(valid)[0]) // 2]
        cx = E_MeV[idx]
        cy = 10 ** (0.5 * (np.log10(lower[idx]) + np.log10(upper[idx])))
        ax_phase.text(
            cx, cy, region["label"],
            horizontalalignment="center", verticalalignment="center",
            rotation=rotation, fontsize=30, zorder=2, color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        )

    ax_phase.set_xlabel("$E$ [MeV]",
                        # fontweight='bold',
                        fontsize=50)
    ax_phase.set_ylabel("$L$ [μm]",
                        # fontweight='bold',
                        fontsize=50)
    ax_phase.set_xscale("log")
    ax_phase.set_yscale("log")
    ax_phase.set_xlim(E_MeV[0], E_MeV[-1])
    ax_phase.set_ylim(L_PLOT_MIN_um, L_PLOT_MAX_um)

    ax_phase.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax_phase.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))

    # marks = [(10, 1e-1), (10, 1e-0), (10, 1e1)]
    marks = [(10, 1e1), (10, 1e0), (10, 1e-1)]
    for i, mark in enumerate(marks):
        ax_phase.plot(mark[0], mark[1], marker='o', markersize=10,
                      markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)
        ax_phase.text(mark[0] * 1.2, mark[1], f'Panel {i + 1}',
                      # fontweight='bold',
                      va='center', fontsize=30, zorder=3)

    # --- Right Panels: Energy Loss Components ---
    # plot_params = [
    #     (0.1 * U.um2cm, "0.1 μm"),
    #     (1 * U.um2cm, "1 μm"),
    #     (10 * U.um2cm, "10 μm")
    # ]
    plot_params = [
        (10 * U.um2cm, "10 μm"),
        (1 * U.um2cm, "1 μm"),
        (0.1 * U.um2cm, "0.1 μm")
    ]

    # xlims = [(0.5e1, 1e4), (5e1, 1e4), (5e2, 1e5)]
    xlims = [(5e2, 1e5), (5e1, 1e4), (0.5e1, 1e4)]
    print("Calculating Energy Loss Distributions...")
    for i, (step_length, title_suffix) in enumerate(plot_params):
        ax = fig.add_subplot(gs[i, 2:])

        continuous_loss = ContinuousLossParams(particle, material)
        continuous_loss.set_state(PointInPhaseSpace(step_length, 10 * U.MeV2eV))

        w3 = continuous_loss.w3
        w = continuous_loss.w
        lambda_ionization = continuous_loss.p3
        n1_mean = continuous_loss.a1
        excitation_energy = continuous_loss.e1

        try:
            n_a_mean = continuous_loss.namean
            alpha = continuous_loss.alfa
            alpha1 = continuous_loss.alfa1
            E0 = continuous_loss.e0
            with_gauss = True
        except AttributeError:
            with_gauss = False
            n_a_mean = None
            alpha = None
            alpha1 = None
            E0 = None

        x, E_ion_compound_poisson_pdf_values = bm.E_ion_compound_poisson_pdf(w3, w, lambda_ionization, f_max=5, N=N * 2)

        if with_gauss:
            E_ion_gaussian_pdf_values = bm.E_ion_gaussian_pdf(x, continuous_loss.emean_ion,
                                                              np.sqrt(continuous_loss.sig2e_ion))

        if StragglingType.LowColFreqExc in continuous_loss.straggling_type:
            E_excitation_pdf_values = few_excitation_collisions_pdf(x, n1_mean, excitation_energy)
            E_excitation_mc = _mc_low_freq_exc(n1_mean, excitation_energy, N)
        else:
            E_excitation_pdf_values = many_excitation_collisions_pdf(x, n1_mean, excitation_energy)
            E_excitation_mc = _mc_gauss(continuous_loss.emean_exc, continuous_loss.sig2e_exc, N)

        E_ion_compound_poisson_mc = _mc_compound_pois_ion(lambda_ionization, w3, w, N)
        if with_gauss and n_a_mean is not None:
            E_ion_gaussian_mc = _mc_gauss(continuous_loss.emean_ion, continuous_loss.sig2e_ion, N)

        if with_gauss and E_ion_gaussian_pdf_values is not None:
            convolved_pdf = utils.convolve_3_pdfs(x, E_ion_compound_poisson_pdf_values, E_ion_gaussian_pdf_values,
                                                  E_excitation_pdf_values)
        else:
            convolved_pdf = utils.convolve_2_pdfs(x, E_ion_compound_poisson_pdf_values, E_excitation_pdf_values)

        bins = 1000

        # ax.hist(E_ion_compound_poisson_mc, bins=bins, density=True,
        #         histtype='step', linewidth=1.5, color=COLORS['ion_cp'], label=r'$\Delta E_{\rm ion}^{C.P}$ (MC)',
        #         zorder=3)
        ax.hist(E_ion_compound_poisson_mc, bins=bins, density=True,
                label=r'$\Delta E_{\rm ion}^{C.P}$ (MC)',
                histtype='stepfilled', alpha=0.1, color=COLORS['ion_cp'], zorder=1)

        # ax.hist(E_excitation_mc, bins=bins, density=True,
        #         histtype='step', linewidth=1.5, color=COLORS['exc'], label=r'$\Delta E_{\rm exc}$ (MC)', zorder=3)
        ax.hist(E_excitation_mc, bins=bins, density=True,
                label=r'$\Delta E_{\rm exc}$ (MC)',
                histtype='stepfilled', alpha=0.1, color=COLORS['exc'], zorder=1)

        if with_gauss and E_ion_gaussian_mc is not None:
            # ax.hist(E_ion_gaussian_mc, bins=bins, density=True,
            #         histtype='step', linewidth=1.5, color=COLORS['ion_gauss'],
            #         label=r'$\Delta E_{\rm ion}^{\rm Gaussian}$ (MC)', zorder=3)
            ax.hist(E_ion_gaussian_mc, bins=bins, density=True,
                    label=r'$\Delta E_{\rm ion}^{\rm Gaus}$ (MC)',
                    histtype='stepfilled', alpha=0.1, color=COLORS['ion_gauss'], zorder=1)

            total_mc = E_ion_compound_poisson_mc + E_ion_gaussian_mc + E_excitation_mc
            # ax.hist(total_mc, bins=bins, density=True,
            #         histtype='step', linewidth=1.5, color=COLORS['total'], label=r'$\Delta E_{total}$ (MC)', zorder=4)
            ax.hist(total_mc, bins=bins, density=True,
                    label=r'$\Delta E_{total}$ (MC)',
                    histtype='stepfilled', alpha=0.1, color=COLORS['total'], zorder=2)
        else:
            total_mc = E_ion_compound_poisson_mc + E_excitation_mc
            # ax.hist(total_mc, bins=bins, density=True,
            #         histtype='step', linewidth=1.5, color=COLORS['total'], label=r'$\Delta E_{total}$ (MC)', zorder=4)
            ax.hist(total_mc, bins=bins, density=True,
                    label=r'$\Delta E_{total}$ (MC)',
                    histtype='stepfilled', alpha=0.1, color=COLORS['total'], zorder=2)

        ax.plot(x, E_ion_compound_poisson_pdf_values, color=COLORS['ion_cp'], linestyle='--',
                label=r'$\Delta E_{\rm ion}^{C.P}$ (PDF)')

        ax.plot(x, E_excitation_pdf_values, color=COLORS['exc'], linestyle='--', label=r'$\Delta E_{\rm exc}$ (PDF)')

        if with_gauss and E_ion_gaussian_pdf_values is not None:
            ax.plot(x, E_ion_gaussian_pdf_values, color=COLORS['ion_gauss'], linestyle='--',
                    label=r'$\Delta E_{\rm ion}^{Gaus}$ (PDF)')

        ax.plot(x, convolved_pdf, color=COLORS['total'], linestyle='--', label='Convolved PDF')

        # ax.text(0.02, 0.95, f'Panel {i + 1}: L = {title_suffix}', transform=ax.transAxes,
        #         fontweight='bold', verticalalignment='top', fontsize=35)
        y = 0.15 if i == 0 else 0.95
        ax.text(0.02, y, f'Panel {i + 1}: $L$ = {title_suffix}', transform=ax.transAxes,
                # fontweight='bold',
                verticalalignment='top', fontsize=35)

        if i == 2:
            ax.set_xlabel(r'$\Delta E^{\rm cnt}$ [eV]',
                          # fontweight='bold',
                          fontsize=50)
        if i == 1:
            ax.set_ylabel('Probability Density',
                          # fontweight='bold',
                          fontsize=50)

        # --- LEGEND LOGIC: Top right panel ---
        if i == 0:
            ax.legend(loc='upper right',
                      ncol=3, borderaxespad=0, frameon=False, fontsize=30, labelspacing=0.1)

        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-7, 1e-1)
        # ax.set_xlim(5e0 * 10 ** i, 1e4 * 10 ** i)
        ax.set_xlim(xlims[i][0], xlims[i][1])

    # save_path = BENCHMARKS_OUTPUTS / "continuous_energy_loss_grid_plot.pdf"
    save_path = BENCHMARKS_OUTPUTS / "continuous_energy_loss_grid_plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"Plot saved to: {save_path}")
    plt.show()
