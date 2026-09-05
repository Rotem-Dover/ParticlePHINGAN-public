"""Pairwise relative-difference comparison of two GEANT4 truth datasets.

Same instrument as figure 11 (`pairwise_comparison.py`: per-row 2-D histograms
of (x, |y|), normalised per dataset, |cmp - ref| / ref, cells below 30 counts
masked, log colour scale 1e-2..1) but with two TRUTH files in the arms instead
of a generator vs truth -- for checking whether two GEANT4 truth productions
agree with each other.

Three columns:

    OLD vs NEW | OLD vs OLD (split halves) | NEW vs NEW (split halves)

The two self-comparisons are the noise reference, exactly the role
"GEANT4 vs GEANT4" plays in figure 11: any structure in the left column that
is not also in the right two is a real difference between the files. Every
arm uses N_half rows from each side, so all three columns share one
finite-sample floor.

Both inputs are read through the same pickle the runtime reader writes
(`data_handling.reader.read_geant4_simulation_output`), i.e. already
`StepNum > 0`, `KineticEnergy > MIN_ENERGY_CUTOFF`, no Transportation steps,
`SecondaryLoss = 0` where `NumOfSecondaries == 0`.

    cd src && MPLBACKEND=Agg PYTHONPATH=. python -m \\
        benchmark.system_vs_g4.truth_vs_truth_pairwise \\
        --old  /path/to/other/proton/aluminum/tracks/100MeV_1k.root \\
        --new  /path/to/resources/proton/aluminum/tracks/100MeV_1k.root

`--new` defaults to `paths.PATH_TO_GEANT4_DATA`. `--old` has no default: it
names whichever second truth production is being compared against, as a
per-invocation, explicit act. Reads only; writes the figure under
`BENCHMARKS_OUTPUTS`.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from common.enums import G4Columns
from common.paths import BENCHMARKS_OUTPUTS, PATH_TO_GEANT4_DATA
from data_handling.reader import read_geant4_simulation_output
from benchmark.system_vs_g4.pairwise_comparison import (
    COMPARISONS, plot_pairwise_relative_diff)


def truth_arrays(df) -> dict:
    E = df[G4Columns.KineticEnergy].values
    return {
        'E_l': E, 'L': df[G4Columns.StepLength].values,
        'E_cnt': E, 'L_cnt': df[G4Columns.StepLength].values,
        'E_CNT': df[G4Columns.ContinuousLoss].values,
        'E_sec': E, 'theta': df[G4Columns.angleDiscrete].values,
        'E_SEC': df[G4Columns.SecondaryLoss].values,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--old", type=Path, required=True,
                   help="the reference truth file (.root; its .pkl sibling is used if present)")
    p.add_argument("--new", type=Path, default=PATH_TO_GEANT4_DATA,
                   help="the comparison truth file (default: paths.PATH_TO_GEANT4_DATA)")
    p.add_argument("--old-label", default="old truth")
    p.add_argument("--new-label", default="new truth")
    p.add_argument("--out", type=Path, default=BENCHMARKS_OUTPUTS / "truth_vs_truth_pairwise.pdf")
    args = p.parse_args()

    old = truth_arrays(read_geant4_simulation_output(args.old, verbose=False))
    new = truth_arrays(read_geant4_simulation_output(args.new, verbose=False))
    n_old, n_new = old['E_l'].size, new['E_l'].size
    print(f"old: {n_old} rows  new: {n_new} rows")

    ARMS = [
        (f"{args.old_label}\nvs {args.new_label}", old, new),
        (f"{args.old_label}\nvs itself (split halves)", old, old),
        (f"{args.new_label}\nvs itself (split halves)", new, new),
    ]
    n_rows, n_cols = len(COMPARISONS), len(ARMS)

    fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols + 1,
                          width_ratios=[1] * n_cols + [0.05])
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    for col, (title, _, _) in enumerate(ARMS):
        axes[0, col].set_title(title, fontsize=24, fontweight='bold', pad=20)

    for row, (x_key, y_key, xb, yb, xlabel, ylabel) in enumerate(COMPARISONS):
        for col, (_, ref, cmp) in enumerate(ARMS):
            N_half = min(ref[x_key].size, cmp[x_key].size) // 2
            x_ref, y_ref = ref[x_key][:N_half], ref[y_key][:N_half]
            if ref is cmp:
                x_cmp, y_cmp = ref[x_key][N_half:2 * N_half], ref[y_key][N_half:2 * N_half]
            else:
                x_cmp, y_cmp = cmp[x_key][:N_half], cmp[y_key][:N_half]
            mesh = plot_pairwise_relative_diff(
                axes[row, col], x_ref, y_ref, x_cmp, y_cmp, xb, yb,
                ylabel=ylabel, xlabel=xlabel,
                show_ylabel=(col == 0), show_xlabel=(col == 0),
            )
            # Same summary per panel, printed: median / p90 relative
            # difference over the populated (unmasked) cells, so the three
            # columns can be compared as numbers as well as by eye.
            d = np.asarray(mesh.get_array()).ravel()
            d = d[d > 0]
            print(f"row {row} ({y_key} vs {x_key}) col {col}: cells {d.size:5d}  "
                  f"median {np.median(d):.4f}  p90 {np.percentile(d, 90):.4f}")

    norm = mpl.colors.LogNorm(vmin=1e-2, vmax=1e0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cax = fig.add_subplot(gs[:, n_cols])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Relative difference", fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=28)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format=args.out.suffix.lstrip('.'), bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.png'), dpi=110, bbox_inches='tight')
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == '__main__':
    main()
