import numpy  as np
import pandas as pd
import math
import matplotlib as mpl

from matplotlib  import pyplot as plt

import common.units as U
from common.enums import G4Columns
from common.paths import VERSION_NUMBER, BENCHMARKS_OUTPUTS
from common.config import apply_project_plotting_style, MIN_ENERGY_CUTOFF, BEAM_ENERGY_eV
from common.run_context import PLOT_RANGES


apply_project_plotting_style()


def plot_kinetic_energy_vs_penetration_depth(data: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D histogram of kinetic energy vs penetration depth.
    """
    fig, ax  = plt.subplots(figsize=(5, 5), tight_layout=True)

    hist = ax.hist2d(
        (data[G4Columns.X] - data[G4Columns.X].min()) * U.m2cm,
        data[G4Columns.KineticEnergy] * U.eV2MeV,
        bins=(1000, 1000), cmap="jet", norm=mpl.colors.LogNorm(), density=False
    )
    ax.set_xlabel(r'X $[cm]$', fontsize=25)
    ax.set_ylabel(r'$E_{k} [MeV]$', fontsize=25)
    ax.axis('auto')

    fig.colorbar(hist[-1], ax=ax)

    # plt.close(fig)

    return fig, ax


def plot_stopping_power_vs_energy(steps: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D histogram of stopping power vs energy for particle steps.
    """
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

    x_range = np.logspace(start=math.log10(MIN_ENERGY_CUTOFF * U.eV2MeV), stop=2, num=300)
    y_range = np.logspace(start=-1, stop=6, num=300)
    hist = ax.hist2d(
        steps[G4Columns.KineticEnergy] * U.eV2MeV,
        np.abs(steps[G4Columns.ContinuousLoss] * U.eV2MeV) / (steps[G4Columns.StepLength] * U.m2cm),
        cmap="jet", norm=mpl.colors.LogNorm(vmin=1e0, vmax=1e4), density=False, bins=[x_range, y_range]
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Energy [MeV]', fontsize=14)
    ax.set_ylabel(r'Stopping Power $[\frac{\rm MeV}{\rm cm}]$', fontsize=14)
    ax.axis('auto')
    ax.grid(linestyle='--', alpha=0.3, color='black')
    ax.set_xlim([1e-1, 1e2])

    fig.colorbar(hist[3], ax=ax)

    # plt.close(fig)

    return fig, ax


def plot_correlation_h2d(steps: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D histograms of all ten pairwise correlations between
    the five step features: E, L, theta_discrete, dE_cnt, dE_sec.
    Arranged in a 3x5 grid with a single shared colorbar.
    """

    fig = plt.figure(figsize=(28, 18), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=6, width_ratios=[1, 1, 1, 1, 1, 0.05])

    col_2_label = {
        G4Columns.KineticEnergy: r"$E$ [eV]",
        G4Columns.ContinuousLoss: r"$\Delta E^{\rm cnt}$ [eV]",
        G4Columns.SecondaryLoss: r"$\Delta E^{\rm sec}$ [eV]",
        G4Columns.StepLength: r"$L$ [m]",
        G4Columns.angleDiscrete: r"$\theta_{\rm discrete}$ [rad]",
    }

    column_couples = [
        (G4Columns.KineticEnergy, G4Columns.ContinuousLoss),
        (G4Columns.KineticEnergy, G4Columns.SecondaryLoss),
        (G4Columns.KineticEnergy, G4Columns.StepLength),
        (G4Columns.KineticEnergy, G4Columns.angleDiscrete),
        (G4Columns.ContinuousLoss, G4Columns.SecondaryLoss),
        (G4Columns.ContinuousLoss, G4Columns.StepLength),
        (G4Columns.ContinuousLoss, G4Columns.angleDiscrete),
        (G4Columns.SecondaryLoss, G4Columns.StepLength),
        (G4Columns.SecondaryLoss, G4Columns.angleDiscrete),
        (G4Columns.StepLength, G4Columns.angleDiscrete),
    ]

    ke_bins        = np.logspace(math.log10(MIN_ENERGY_CUTOFF), math.log10(BEAM_ENERGY_eV), 100)
    l_bins         = np.logspace(*PLOT_RANGES.step_length, 100)
    theta_bins     = np.logspace(*PLOT_RANGES.theta,       100)
    cnt_bins       = np.logspace(*PLOT_RANGES.cnt_loss,    100)
    sec_bins       = np.logspace(*PLOT_RANGES.sec_loss,    100)

    bins_tuples = [
        (ke_bins, cnt_bins), (ke_bins, sec_bins), (ke_bins, l_bins), (ke_bins, theta_bins),
        (cnt_bins, sec_bins), (cnt_bins, l_bins), (cnt_bins, theta_bins),
        (sec_bins, l_bins), (sec_bins, theta_bins),
        (l_bins, theta_bins),
    ]

    axes_list = []

    for i, (col_couple, bins_tuple) in enumerate(zip(column_couples, bins_tuples)):
        row = i // 5
        col = i % 5
        ax = fig.add_subplot(gs[row, col])
        axes_list.append(ax)

        if col_couple[0] == G4Columns.KineticEnergy and (col_couple[1] == G4Columns.ContinuousLoss or
                                                         col_couple[1] == G4Columns.StepLength):
            x = steps[G4Columns.KineticEnergy].values \
                - steps[G4Columns.SecondaryLoss].values
        else:
            x = steps[col_couple[0]].values

        h = ax.hist2d(
            x=x,
            y=np.abs(steps[col_couple[1]].values),
            norm=mpl.colors.LogNorm(vmin=1e0, vmax=1e4),
            bins=bins_tuple,
            cmap="jet",
            density=False,
        )

        ax.set_xlabel(col_2_label[col_couple[0]], fontweight='bold')
        ax.set_ylabel(col_2_label[col_couple[1]], fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')

    cax = fig.add_subplot(gs[:, 5])
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=1e4)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Steps", fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)

    fig.savefig(BENCHMARKS_OUTPUTS / "correlation_h2d.pdf", format="pdf")

    return fig, ax


def plot_correlation_h2d_comparison(geant4_steps: pd.DataFrame, generator_steps: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D histograms of all ten pairwise correlations between
    the five step features: E, L, theta_discrete, dE_cnt, dE_sec.
    GEANT4 and generator overlaid panel by panel.
    """
    fig, axes = plt.subplots(3, 5, figsize=(24, 15), tight_layout=True)
    axes = axes.ravel()

    col_2_label = {
        G4Columns.KineticEnergy:  "Kinetic Energy $[eV]$",
        G4Columns.ContinuousLoss: "Continuous Energy Loss $[eV]$",
        G4Columns.SecondaryLoss:  "Secondary Energy Loss $[eV]$",
        G4Columns.StepLength:     "Step Length $[m]$",
        G4Columns.angleDiscrete:  r"$\theta_{\rm discrete}$ $[rad]$",
    }

    column_couples = [
        (G4Columns.KineticEnergy, G4Columns.ContinuousLoss),
        (G4Columns.KineticEnergy, G4Columns.SecondaryLoss),
        (G4Columns.KineticEnergy, G4Columns.StepLength),
        (G4Columns.KineticEnergy, G4Columns.angleDiscrete),
        (G4Columns.ContinuousLoss, G4Columns.SecondaryLoss),
        (G4Columns.ContinuousLoss, G4Columns.StepLength),
        (G4Columns.ContinuousLoss, G4Columns.angleDiscrete),
        (G4Columns.SecondaryLoss, G4Columns.StepLength),
        (G4Columns.SecondaryLoss, G4Columns.angleDiscrete),
        (G4Columns.StepLength, G4Columns.angleDiscrete),
    ]

    ke_bins        = np.logspace(math.log10(MIN_ENERGY_CUTOFF), math.log10(BEAM_ENERGY_eV), 100)
    l_bins         = np.logspace(*PLOT_RANGES.step_length, 100)
    theta_bins     = np.logspace(*PLOT_RANGES.theta,       100)
    cnt_bins       = np.logspace(*PLOT_RANGES.cnt_loss,    100)
    sec_bins       = np.logspace(*PLOT_RANGES.sec_loss,    100)

    bins_tuples = [
        (ke_bins, cnt_bins), (ke_bins, sec_bins), (ke_bins, l_bins), (ke_bins, theta_bins),
        (cnt_bins, sec_bins), (cnt_bins, l_bins), (cnt_bins, theta_bins),
        (sec_bins, l_bins), (sec_bins, theta_bins),
        (l_bins, theta_bins),
    ]

    for col_couple, bins_tuple, ax in zip(column_couples, bins_tuples, axes):
        if col_couple[0] == G4Columns.KineticEnergy and (col_couple[1] == G4Columns.ContinuousLoss or
                                                         col_couple[1] == G4Columns.StepLength):
            x_geant4 = geant4_steps[G4Columns.KineticEnergy].values \
                - geant4_steps[G4Columns.SecondaryLoss].values
            x_generator = generator_steps[G4Columns.KineticEnergy].values \
                - generator_steps[G4Columns.SecondaryLoss].values
        else:
            x_geant4    = geant4_steps[col_couple[0]].values
            x_generator = generator_steps[col_couple[0]].values

        geant4_hist, _, _ = np.histogram2d(
            x=x_geant4,
            y=np.abs(geant4_steps[col_couple[1]].values),
            bins=bins_tuple,
            density=False
        )
        generator_hist, _, _ = np.histogram2d(
            x=x_generator,
            y=np.abs(generator_steps[col_couple[1]].values),
            bins=bins_tuple,
            density=False
        )
        geant4_hist    = geant4_hist / x_geant4.size
        generator_hist = generator_hist / x_generator.size

        hist_relative_diff = np.abs(generator_hist - geant4_hist) / geant4_hist
        hist_relative_diff[geant4_hist < 30 / x_geant4.size] = 0
        np.save(BENCHMARKS_OUTPUTS / f"{col_couple[0]}_{col_couple[1]}_version_{VERSION_NUMBER}.npy", hist_relative_diff)

        X, Y = np.meshgrid(bins_tuple[0], bins_tuple[1])
        vmin, vmax = 1e-2, 1e0
        cax = ax.pcolormesh(
            X, Y, hist_relative_diff.T, cmap='jet',
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        )

        ax.set_xlabel(col_2_label[col_couple[0]], size=15)
        ax.set_ylabel(col_2_label[col_couple[1]], size=15)
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.colorbar(cax, ax=ax)

    return fig, axes


def steps_features_histogram(
        L: np.ndarray, theta: np.ndarray, dE_cnt: np.ndarray, dE_sec: np.ndarray
) -> tuple[plt.Figure, plt.Axes]:
    # TODO: data: pd.DataFrame as input, and extract the columns inside the function
    def get_bins(arr: np.ndarray) -> np.ndarray:
        n_bins = 100
        return np.logspace(np.log10(np.min(arr)), np.log10(np.max(arr)), n_bins)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),  tight_layout=True)
    axes = axes.ravel()

    axes[0].hist(L, bins=get_bins(L), log=True, color="blue", alpha=0.4)
    axes[0].hist(L, bins=get_bins(L), log=True, color="blue", histtype='step')
    axes[0].set_xlabel("Step Length [m]", fontsize=14)

    axes[1].hist(theta, bins=get_bins(theta), log=True, color="blue", alpha=0.4)
    axes[1].hist(theta, bins=get_bins(theta), log=True, color="blue", histtype='step')
    axes[1].set_xlabel("Deflection Angle [rad]", fontsize=14)

    axes[2].hist(dE_cnt, bins=get_bins(dE_cnt), log=True, color="blue", alpha=0.4)
    axes[2].hist(dE_cnt, bins=get_bins(dE_cnt), log=True, color="blue", histtype='step')
    axes[2].set_xlabel("Total Continuous Energy Loss [eV]", fontsize=14)

    axes[3].hist(dE_sec, bins=get_bins(dE_sec), log=True, color="blue", alpha=0.4)
    axes[3].hist(dE_sec, bins=get_bins(dE_sec), log=True, color="blue", histtype='step')
    axes[3].set_xlabel("Energy Loss via Secondary Production [eV]", fontsize=14)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.supylabel("Nuber of Steps", fontsize=14)
    plt.close()

    return fig, axes


if __name__ == "__main__":
    from data_handling.reader import read_geant4_simulation_output

    steps_lab_frame = read_geant4_simulation_output()

    plot_kinetic_energy_vs_penetration_depth(steps_lab_frame)
    plt.show()
    plot_stopping_power_vs_energy(steps_lab_frame)
    plt.show()
    plot_correlation_h2d(steps_lab_frame)
    plt.show()
    # steps_features_histogram(L=steps_lab_frame)

