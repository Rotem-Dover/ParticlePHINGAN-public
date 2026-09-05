"""
`BeamConfig` bundles the facts that together describe one run of the
pipeline: which particle, into which target material, at what beam energy,
with which tracking cutoff, and the per-feature plotting ranges that make
sense at that energy/material. Filename tokens (e.g. "300keV", "100MeV")
used under `tracks/`, `iid/`, etc. live on the config too so they cannot
drift out of sync with the energy.

Add a new run by defining a new `BeamConfig` instance below, then point
`common.run_context.ACTIVE` at it. Adding a beam is a one-liner; no branching.

`PlotRanges` carries log10 (lo, hi) bounds for each step feature so the
1D/2D histograms in `data_handling.plotter` and the `system_vs_g4`
benchmark scripts automatically retune when the beam changes. KE bounds
are derived from `min_cutoff_eV` / `energy_eV` and are not stored here.
"""
from dataclasses import dataclass

import common.units as U
from physics.definitions.material import Material, aluminum, beryllium, iron
from physics.definitions.particle import Particle, proton


@dataclass(frozen=True)
class PlotRanges:
    # All bounds are log10 of the physical value. Units: m for length,
    # rad for angles, eV for energies.
    step_length: tuple[float, float]
    theta:       tuple[float, float]
    cnt_loss:    tuple[float, float]
    sec_loss:    tuple[float, float]


@dataclass(frozen=True)
class GridBounds:
    # Bounds of the (kinetic energy, step length) phase-space grid for
    # building the analytical CDFs and the scaler normalization lookup tables.
    # The energy maximum is BeamConfig.energy_eV (not duplicated here).
    energy_min_eV:   float   # low-E floor of the grid
    step_len_min_m:  float   # short-step floor
    step_len_max_m:  float   # long-step ceiling


@dataclass(frozen=True)
class BeamConfig:
    particle:        Particle
    material:        Material
    energy_eV:       float       # nominal beam kinetic energy
    tag:             str         # filename token, e.g. "300keV" or "100MeV"
    min_cutoff_eV:   float       # tracking cutoff: steps below this are dropped
    plot_ranges:     PlotRanges  # log10 (lo, hi) for each step feature
    grid_bounds:     GridBounds  # CDF / scaler phase-space grid bounds


proton_in_aluminum_100MeV = BeamConfig(
    particle=proton, material=aluminum,
    energy_eV=100 * U.MeV2eV, tag="100MeV",
    min_cutoff_eV=0.3 * U.MeV2eV,
    # Match the paper's figure axes (Figs. 8-13) bin by bin, so generated
    # histograms overlay the printed panels directly. Ionisation-only
    # transport has no multiple-scattering tail, so a wider range (extending
    # to L~1e-12 m, theta~1e-5 rad, cnt_loss up to 1e6 eV) would leave the
    # lower half of every panel empty; these bounds are sized to the actual
    # ionisation-only step population instead.
    plot_ranges=PlotRanges(
        step_length=(-9., -4.),     # 1 nm  -> 100 um
        theta      =(-4.5, -3.),    # discrete (delta-ray) scatter
        cnt_loss   =(1., 5.),       # 10 eV -> 100 keV (per step)
        sec_loss   =(2.8, 5.5),     # ~600 eV -> ~300 keV
    ),
    grid_bounds=GridBounds(
        energy_min_eV=0.4 * U.MeV2eV,
        step_len_min_m=1e-11,
        step_len_max_m=1e-4,
    ),
)

proton_in_iron_100MeV = BeamConfig(
    particle=proton, material=iron,
    energy_eV=100 * U.MeV2eV, tag="100MeV",
    min_cutoff_eV=0.3 * U.MeV2eV,
    plot_ranges=PlotRanges(
        step_length=(-12., -3.5),   # truth span 1e-11 .. 2.23e-4 m
        theta      =(-5.5, -3.),
        cnt_loss   =(-2., 6.5),     # per-step E_cnt reaches ~3 MeV in Fe
        sec_loss   =(3.6, 5.5),     # floor = Fe's higher Tc (~5 keV)
    ),
    grid_bounds=GridBounds(
        energy_min_eV=0.4 * U.MeV2eV,
        step_len_min_m=1e-11,
        step_len_max_m=3e-4,        # truth max 2.23e-4 m; Al's 1e-4 would clip
    ),
)

proton_in_beryllium_100MeV = BeamConfig(
    particle=proton, material=beryllium,
    energy_eV=100 * U.MeV2eV, tag="100MeV",
    min_cutoff_eV=0.3 * U.MeV2eV,
    plot_ranges=PlotRanges(
        step_length=(-12., -3.5),   # truth span 1e-11 .. 1.89e-4 m
        theta      =(-5.5, -3.),
        cnt_loss   =(-2., 6.),
        sec_loss   =(2.8, 5.5),     # Be's Tc is close to Al's
    ),
    grid_bounds=GridBounds(
        energy_min_eV=0.4 * U.MeV2eV,
        step_len_min_m=1e-11,
        step_len_max_m=2e-4,        # truth max 1.89e-4 m
    ),
)

# Registry of selectable beams, keyed by preset name. common.run_context
# resolves PHINGAN_BEAM against exactly this dict -- nothing else is
# selectable, and an unknown name dies at import.
PRESETS: dict[str, BeamConfig] = {
    "proton_in_aluminum_100MeV":  proton_in_aluminum_100MeV,
    "proton_in_iron_100MeV":      proton_in_iron_100MeV,
    "proton_in_beryllium_100MeV": proton_in_beryllium_100MeV,
}
