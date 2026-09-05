"""Driver for the paper's Fig 13: macroscopic 10k-track energy-deposition
profiles + pull, PHIN-GAN vs GEANT4 truth -- and, via `--preset gan`, for
its appendix Fig E.2: the same figure for the no-physics GAN ablation arm
(same truth, same seed, same event count; only the preset differs, so the
two figures are directly comparable). Binning / cut / axis default to the
paper's parameters (1500x1500 grid, bins averaging >3 trajectories, +-5
pull axis) -- see the argparse help. Outputs default to
`energy_deposition_pull.pdf` for phin_gan and to
`energy_deposition_pull_gan.pdf` for gan.

`energy_deposition_pull_analysis` is a library (and mutates global matplotlib
rcParams at import -- everything here runs inside rc_context). The simulator
side mirrors gate T's conventions (verify_truth_parity._simulate):
E_kill_eV = MIN_ENERGY_CUTOFF so the simulated tracks stop producing rows at
the same energy floor the truth rows were filtered to, and
save_steps_states=True + E_0=1e8 (100 MeV, this beam's energy) match that
same reference call -- the brief's original snippet omitted both, which
would leave `states_and_steps_features_df` unpopulated and `run()` missing
its required `E_0` argument. Events run in chunks to bound the pandas frame:
10k events x 12k steps of NaN-padded states is ~120M rows before dropping
dead lanes.

**Coordinate-frame offset:** `TrackSimulator` places every event's injection
point at the lab-frame origin (`__initial_features_states` zero-inits
X/Y/Z), so simulated X runs 0 -> +range. The GEANT4 truth file places
injection at a FIXED X = -0.05 m instead -- verified as *exactly* -0.05 for
all 10,000 events (std 6.9e-18, i.e. bit-identical), matching the
absorber-front-face placement of the target macro that produced it. Without
correcting for this, `pull_analysis`'s 2D histogram range comes from the
truth frame (`df_1`) and every simulated X would land outside it, silently
zeroing the sim histogram (`compute_hist_stats`'s valid-range mask drops
every sim row) and producing an all-NaN pull fit independent of event count.
The fix shifts the *simulated* frame into the truth's, via `organize_df`'s
own `displace` parameter (added for exactly this kind of frame alignment),
rather than editing the pinned library.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from benchmark.system_vs_g4.pull_analysis import energy_deposition_pull_analysis as pa
from benchmark.system_vs_g4.pull_analysis.truth_null_band import (
    DEFAULT_OUT as _BAND_JSON, load_tolerance_text)
from benchmark.track_simulator_config import build_track_simulator
from common.config import MIN_ENERGY_CUTOFF
from common.enums import G4Columns, TOTAL_ENERGY_LOSS
from common.paths import BENCHMARKS_OUTPUTS, PATH_TO_GEANT4_DATA_SEED42
from common.run_context import TARGET_MATERIAL
from data_handling.reader import read_geant4_simulation_output

# Beam kinetic energy at injection -- matches verify_truth_parity._simulate's
# E_0=1e8 (100 MeV, the proton-in-aluminium beam).
_E_0_eV = 1e8

# TrackSimulator injects at lab-frame X=0; the GEANT4 truth file injects at
# X=-0.05 m (bit-identical across all 10,000 events -- measured, not
# assumed). The TRUTH is shifted by the negative of this via
# organize_df(displace=...) so both sides share the paper's frame (injection
# at z = 0) before histogramming. See the module docstring for the
# measurement and the failure a missing alignment causes.
_SIM_TO_TRUTH_X_DISPLACE_m = -0.05

_NEEDED = [G4Columns.EventNum, G4Columns.X, G4Columns.Y, TOTAL_ENERGY_LOSS]


_PRESET_LABEL = {"phin_gan": "PHIN-GAN", "gan": "GAN"}
_PRESET_SUFFIX = {"phin_gan": "", "gan": "_gan"}

# The GEANT4-vs-GEANT4 tolerance band printed under the fitted mu_0 /
# sigma_0 comes from the ten-seed truth-vs-truth study recorded by
# `truth_null_band` (measurements/pull_truth_null/truth_vs_truth_pull.json);
# `--band-json` points elsewhere; the loader refuses a study recorded at a
# different binning than the figure is drawn at. Only the PHIN-GAN figure
# (Fig 13) prints the band: the GAN ablation's Fig E.2 shows its fitted
# mu_0 / sigma_0 alone, which is what the band is for reading against.
# The study was measured on ALUMINIUM truth only (its JSON lists the ten
# 100MeVProtonInAl files), so the band is printed for the aluminium beam
# alone: the iron and beryllium figures show their fitted mu_0 / sigma_0
# without it, no separator either.
_PRESET_PRINTS_BAND = {"phin_gan": True, "gan": False}
_BAND_MATERIAL = "aluminum"


def simulate_df(n_events: int, n_steps: int, seed: int,
                device: str = "cpu", chunk: int = 1000,
                preset: str = "phin_gan") -> pd.DataFrame:
    torch.manual_seed(seed)
    chunks = []
    done = 0
    while done < n_events:
        n = min(chunk, n_events - done)
        ts = build_track_simulator(preset, device=device)
        ts.run(E_0=_E_0_eV, n_events=n, n_steps=n_steps,
               save_steps_states=True, E_kill_eV=MIN_ENERGY_CUTOFF)
        df = ts.states_and_steps_features_df
        # Dead lanes are NaN-padded -- drop them before anything else.
        df = df[np.isfinite(df[G4Columns.KineticEnergy])]
        # states_and_steps_features_df is a merge of states_along_propagation_df
        # (EventNum, StepNum, X, Y, Z, KineticEnergy) and steps_features_df
        # (EventNum, StepNum, StepLength, angleDiscrete, ContinuousLoss,
        # SecondaryLoss, geom_step_m) -- both already keyed by G4Columns
        # names, so no rename is needed here. This assert exists precisely
        # to catch it loudly if that ever stops being true.
        for col in (G4Columns.EventNum, G4Columns.X, G4Columns.Y,
                    G4Columns.ContinuousLoss, G4Columns.SecondaryLoss):
            assert col in df.columns, (
                f"states_and_steps_features_df lacks {col!r}; its columns "
                f"are {list(df.columns)} -- fix the mapping here, do not "
                f"guess downstream")
        df = df.assign(**{TOTAL_ENERGY_LOSS:
                          df[G4Columns.ContinuousLoss]
                          + df[G4Columns.SecondaryLoss]})
        df = df[_NEEDED].copy()
        df[G4Columns.EventNum] += done
        chunks.append(df)
        done += n
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-events", type=int, default=10_000)
    p.add_argument("--n-steps", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="cpu")
    p.add_argument("--preset", default="phin_gan", choices=sorted(_PRESET_LABEL),
                   help="simulator preset for the simulated arm: phin_gan "
                        "(Fig 13) or gan (Fig E.2, the no-physics ablation)")
    # Paper parameters (Sec. results, "A macroscopic comparison"): a
    # 1,500 x 1,500 y-z grid; 1-D pull histogram over bins averaging MORE
    # THAN THREE trajectories between the two samples (i.e. bins averaging
    # fewer than 4 are removed); Gaussian fit shown over the +-5 axis. These
    # are the paper's numbers, NOT `pull_analysis`'s historical defaults
    # (400 x 400, > 30) -- the defaults stay untouched in the library.
    p.add_argument("--grid", type=int, default=1500,
                   help="y-z grid, N x N bins (paper: 1500)")
    p.add_argument("--min-avg-tracks", type=int, default=3,
                   help="keep bins whose average trajectory count over the "
                        "two samples is > this (paper: 3, i.e. 'fewer than 4 "
                        "trajectories' are removed)")
    p.add_argument("--pull-range", type=float, default=5.0,
                   help="1-D pull histogram / fit half-range (paper axis: +-5)")
    p.add_argument("--out", default=None,
                   help="output PDF (default BENCHMARKS_OUTPUTS/"
                        "energy_deposition_pull[_gan].pdf)")
    p.add_argument("--band-json", type=Path, default=_BAND_JSON,
                   help="the truth-vs-truth study the printed tolerance band "
                        "is read from (written by truth_null_band)")
    p.add_argument("--no-band", action="store_true",
                   help="print only the fitted mu_0 / sigma_0, without the "
                        "tolerance band (the band is aluminium-only anyway "
                        "and is never printed for another beam)")
    args = p.parse_args()
    suffix = _PRESET_SUFFIX[args.preset]
    # Read before the simulation so a missing or mismatched study fails fast.
    prints_band = (_PRESET_PRINTS_BAND[args.preset]
                   and TARGET_MATERIAL.name.lower() == _BAND_MATERIAL
                   and not args.no_band)
    band_text = (load_tolerance_text(args.band_json, grid=args.grid,
                                     min_avg_tracks=args.min_avg_tracks)
                 if prints_band else None)

    truth_df = read_geant4_simulation_output(PATH_TO_GEANT4_DATA_SEED42)
    # _SIM_TO_TRUTH_X_DISPLACE_m is hardcoded from a one-time measurement of
    # this truth file's injection plane. If the truth is ever regenerated
    # against a different absorber geometry, this constant would silently
    # stop matching it -- organize_df would keep displacing by the wrong
    # amount and the pull fit would degrade without any loud signal. X here is in
    # metres, pre-organize_df; truth injects bit-identically across events,
    # so min() is the injection plane.
    truth_x_min = truth_df[G4Columns.X].min()
    assert abs(truth_x_min - _SIM_TO_TRUTH_X_DISPLACE_m) < 1e-9, (
        f"truth injection X ({truth_x_min}) does not match "
        f"_SIM_TO_TRUTH_X_DISPLACE_m ({_SIM_TO_TRUTH_X_DISPLACE_m}) -- the "
        f"truth file's injection plane changed (e.g. a re-generated truth "
        f"with a different absorber front face); update the constant, do "
        f"not silently keep displacing by the old value")
    sim_df = simulate_df(args.n_events, args.n_steps, args.seed, args.device,
                         preset=args.preset)

    with plt.rc_context():
        # Both arms in the paper's frame: injection at z = 0. The truth is
        # shifted by +0.05 m (its injection plane is at -0.05 m, asserted
        # above); the simulation already injects at the origin.
        truth_org = pa.organize_df(truth_df, displace=-_SIM_TO_TRUTH_X_DISPLACE_m)
        sim_org = pa.organize_df(sim_df)
        fig, _ = pa.pull_analysis(
            truth_org, sim_org, label=_PRESET_LABEL[args.preset],
            n_2d_bins=(args.grid, args.grid),
            pull_min_counts_per_bin=args.min_avg_tracks,
            pull_1d_hist_range=(-args.pull_range, args.pull_range),
            pull_fit_range=(-args.pull_range, args.pull_range),
            tolerance_text=band_text)
        out = args.out or (BENCHMARKS_OUTPUTS / f"energy_deposition_pull{suffix}.pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
