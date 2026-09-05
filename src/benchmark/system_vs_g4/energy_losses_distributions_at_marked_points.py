"""Fig 9: energy-loss distributions at marked phase-space points (PHIN-GAN-only draft)."""

import numpy as np
import torch
from copy import copy

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import common.units as U
from common.utils import PointInPhaseSpace
from common.paths import BENCHMARKS_OUTPUTS

from benchmark.system_vs_g4.net_sampling import (
    generate_continuous, generate_secondary, phin_gan_ionisation,
    preset_ionisation,
)
from common.enums import SecondaryFeats

from physics.g4h_ionisation.explicit_physics.continuous.straggling_function import ContinuousStragglingModel
from physics.g4h_ionisation.explicit_physics.discrete.straggling_function import DiscreteStragglingModel
from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material


torch.manual_seed(42)

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

plt.rcParams.update({
    'font.size': 30,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'axes.labelsize': 24,
    'legend.fontsize': 24,
    'legend.frameon': False,
    'hatch.linewidth': 1.0,  # Ensures hatches are visible in PDF
    'figure.dpi': 300
})

if __name__ == "__main__":
    particle = proton
    straggling_cnt_model = ContinuousStragglingModel(particle, material)
    straggling_sec_model = DiscreteStragglingModel(particle, material)
    from benchmark.track_simulator_config import has_gan_preset
    net_proc = phin_gan_ionisation()
    # The no-physics ablation exists for aluminium only; other beams draw
    # the physics PDF and PHIN-GAN alone.
    gan_proc = preset_ionisation("gan") if has_gan_preset() else None

    # Create comparison histograms for marked points
    # Increased figure size slightly to accommodate large fonts
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    axes = axes.flatten()

    # Mark specific coordinates on the phase space plot
    # Per-material (E [MeV], L [um]) points, shared with fig 8.
    from benchmark.system_vs_g4.marked_points import marked_points as _marked_points
    marked_points = _marked_points()
    # colors = [PALETTE['red'], PALETTE['blue'], PALETTE['green'], PALETTE['orange']]
    colors = [PALETTE['purple'], PALETTE['green'], PALETTE['brown'], PALETTE['orange']]

    for i, (energy_mev, step_length_um) in enumerate(marked_points):
        # Convert to internal units
        energy_ev = energy_mev * U.MeV2eV
        step_length_cm = step_length_um * U.um2cm

        # Set up physics model for this point
        point = PointInPhaseSpace(step_length_cm, energy_ev)
        straggling_cnt_model.set_state(point)
        straggling_sec_model.set_state(point)

        # Generate samples using the ML generators
        phase_space_tensor = torch.tensor([energy_ev, step_length_cm * U.cm2m], dtype=torch.float32).unsqueeze(0)
        ke_tensor = torch.tensor([energy_ev], dtype=torch.float32)

        cnt_phin_prediction = generate_continuous(
            net_proc, copy(phase_space_tensor), correct_zero=False,
            repeat_interleave=100_000).detach().numpy()
        sec_phin_prediction = generate_secondary(
            net_proc, copy(ke_tensor), e_cnt=torch.zeros(1),
            correct_zero=False, repeat_interleave=100_000,
        ).detach().numpy()[:, SecondaryFeats.E_SEC_IDX]

        # No-physics GAN ablation curves -- corrections OFF, matching the
        # PHIN-GAN sampling above (the ablation plots the nets' raw output).
        gan_preds = ()
        if gan_proc is not None:
            cnt_gan_prediction = generate_continuous(
                gan_proc, copy(phase_space_tensor), correct_zero=False,
                correct_deterministic=False, repeat_interleave=100_000,
            ).detach().numpy()
            sec_gan_prediction = generate_secondary(
                gan_proc, copy(ke_tensor), e_cnt=torch.zeros(1),
                correct_zero=False, repeat_interleave=100_000,
            ).detach().numpy()[:, SecondaryFeats.E_SEC_IDX]
            gan_preds = (cnt_gan_prediction, sec_gan_prediction)

        # Bins span ALL arms. np.histogram silently drops samples outside
        # the bin range (and density=True then normalises over the survivors),
        # so bins sized to PHIN-GAN alone clipped the GAN ablation wherever its
        # support is wider -- visibly at (0.7 MeV, 1 um), where GAN's dE_cnt
        # runs past PHIN-GAN's maximum.
        all_preds = (cnt_phin_prediction, sec_phin_prediction) + gan_preds
        min_all = min(a.min() for a in all_preds)
        max_all = max(a.max() for a in all_preds)

        bins = np.logspace(np.log10(min_all), np.log10(max_all), 100)

        # Plot neural system (Filled with hatch)
        axes[i].hist(cnt_phin_prediction,
                     bins=bins, alpha=0.5, density=True,
                     color=colors[i],
                     hatch='/', linewidth=0.5)
        axes[i].hist(sec_phin_prediction,
                     bins=bins, alpha=0.5, density=True,
                     color=colors[i],
                     linewidth=0.5)

        # Plot GAN ablation (step lines, corrections off)
        if gan_proc is not None:
            axes[i].hist(cnt_gan_prediction,
                         bins=bins, density=True, linewidth=3, linestyle=':',
                         color=colors[i], histtype='step')
            axes[i].hist(sec_gan_prediction,
                         bins=bins, density=True, linewidth=3, linestyle='--',
                         color=colors[i], histtype='step')

        # Get physics PDF points
        loss_range = np.linspace(0, max((max(cnt_phin_prediction), max(sec_phin_prediction))), 10_000)

        # Evaluate physics PDF
        pdf_cnt_values = straggling_cnt_model.straggling_function(loss_range)
        pdf_sec_values = straggling_sec_model.straggling_function(loss_range)

        # Normalize PDFs
        pdf_cnt_values[0] = pdf_cnt_values[1]
        pdf_cnt_values /= np.trapz(pdf_cnt_values, loss_range)
        pdf_sec_values /= np.trapz(pdf_sec_values, loss_range)

        # Plot physics PDF
        axes[i].plot(loss_range, pdf_cnt_values, 'k-', linewidth=2.5, alpha=0.8)
        axes[i].plot(loss_range, pdf_sec_values, 'k-', linewidth=2.5, alpha=0.8)

        # Make custom legend
        custom_lines = [
            mlines.Line2D([0], [0], color='k', lw=2.5, label='Physics PDF'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.5, edgecolor='black',
                               label=r"PHIN-GAN ($\Delta E^{\rm sec}$)"),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.5, edgecolor='black', hatch='//',
                               label=r"PHIN-GAN ($\Delta E^{\rm cnt}$)"),
        ]
        if gan_proc is not None:
            custom_lines += [
                mlines.Line2D([0], [0], color=colors[i], lw=3, linestyle='--', label=r'GAN ($\Delta E^{\rm sec}$)'),
                mlines.Line2D([0], [0], color=colors[i], lw=3, linestyle=':', label=r'GAN ($\Delta E^{\rm cnt}$)'),
            ]

        if i==3:
            # Improved legend location and column usage
            axes[i].legend(handles=custom_lines, ncol=1, loc='upper right')

        # Title with bold font
        axes[i].set_title(fr'$E$: {energy_mev} MeV, $L$: {step_length_um} μm',
                          fontweight='bold', fontsize=28)

        axes[i].set_xscale('log')
        axes[i].set_yscale('log')

        # Subtle grid
        axes[i].grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

        # The per-panel frames are tuned to aluminium's four points; other
        # materials' points take the data-driven frame.
        match i if material.name.lower() == "aluminum" else -1:
            case 1:
                axes[i].set_xlim(100, 2.5e4)
            case 2:
                axes[i].set_xlim(950, 7e4)
            case 3:
                # upper limit follows the data so the GAN dE_cnt tail is
                # not cut at the frame edge
                axes[i].set_xlim(950, max(1.5e5, max(bins) * 1.2))
            case _:
                axes[i].set_xlim(min(bins) / 2, max(bins) * 2)

        axes[i].set_ylim(1e-7, max((max(pdf_cnt_values), max(pdf_sec_values))) * 10)  # Increased headroom

        # Ensure ticks are large
        axes[i].tick_params(axis='both', which='major')

    # Bold Sup-labels
    fig.supxlabel(r'$\Delta E$ [eV]', fontweight='bold')
    fig.supylabel('Probability Density', fontweight='bold')

    # Save
    plt.savefig(BENCHMARKS_OUTPUTS / 'energy_loss_histograms_comparison.pdf', format='pdf')
    plt.show()