"""Figure 14 — computation time vs number of primary particles.

- **GEANT4 (CPU)** comes from a two-column `n_events,seconds` CSV (100 MeV
  proton, G4_Al 5 cm, ionisation only, ROOT output disabled, one CPU core).
- **PHIN-GAN (GPU)** comes from `profile_runtime` JSON reports on the
  accelerated build (Triton fused MLP, torch.compile'd phases).

Run, on the committed measurement files:
    cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.plot_computation_time \
        ../measurements/computation_time/phin_gan_gpu_fused_r1.json \
        ../measurements/computation_time/phin_gan_gpu_fused_r2.json \
        --g4-csv ../measurements/computation_time/geant4_cpu_times.csv \
        [--cpu-label "Intel Xeon Gold 5320"] [--min-events 100] [--out figure.pdf]

Points below `--min-events` (default 100) are dropped from BOTH arms before
anything is plotted or fitted: at 10 primaries each arm measures its fixed
start-up cost rather than its stepping rate, so the point says nothing about
scaling and would only drag the GEANT4 fit.

Several reports are treated as REPEATS of one measurement: the plotted curve
is their mean and the per-point spread is printed, so run-to-run scatter is
reported rather than hidden behind whichever repeat was passed first.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common.paths import BENCHMARKS_OUTPUTS

# GEANT4 in black, the GPU arm in matplotlib's default orange.
C_GPU = "#d95f02"
C_G4 = "black"

# Large bold labels + inward ticks on all four spines.
PAPER_RC = {
    "font.size": 22, "axes.titlesize": 24, "axes.labelsize": 24,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "xtick.labelsize": 20, "ytick.labelsize": 20,
    "legend.fontsize": 17, "legend.frameon": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 1.2, "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
}


def load_sweep(path: Path):
    """-> (n_events, wall_s, device_name) from one profile_runtime report."""
    r = json.loads(Path(path).read_text())
    sweep = sorted(r["sweep"], key=lambda s: s["n_events"])
    n = np.array([s["n_events"] for s in sweep], dtype=float)
    t = np.array([s["wall_s"] for s in sweep], dtype=float)
    return n, t, r.get("meta", {}).get("device_name", "")


def load_repeats(paths):
    """Mean over repeats, with the spread printed rather than swallowed.

    Repeats must share an x-grid; a mismatch means the reports are not
    repeats of one measurement and averaging them would be meaningless.
    """
    loaded = [load_sweep(p) for p in paths]
    n0 = loaded[0][0]
    for path, (n, _, _) in zip(paths[1:], loaded[1:]):
        if not np.array_equal(n, n0):
            raise SystemExit(
                f"{path}: event grid {n.tolist()} != {n0.tolist()} — these "
                f"are not repeats of one measurement")
    walls = np.vstack([t for _, t, _ in loaded])
    mean = walls.mean(axis=0)
    if len(paths) > 1:
        spread = (walls.max(axis=0) - walls.min(axis=0)) / mean
        print(f"[fig] {len(paths)} repeats; per-point spread "
              + ", ".join(f"{n:g}:{s:.2%}" for n, s in zip(n0, spread)))
    device = next((d for _, _, d in loaded if d), "")
    return n0, mean, device


def build_figure(n_gpu, t_gpu, n_g4, t_g4, device_name, cpu_label):
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # GEANT4: measured points, then the fit, so the legend reads
    # points-then-fit rather than interleaving by draw order.
    ax.plot(n_g4, t_g4, "D", ms=11, color=C_G4, ls="none",
            label="GEANT4 (CPU)")
    # Linear scaling through the origin. median(t/n) rather than a least-
    # squares fit in log space: GEANT4 CPU time is linear in primaries by
    # construction, and the median is not dragged by the low-n points, whose
    # wall is dominated by the fixed /run/initialize cost sitting inside
    # TestEm5's chrono timer.
    slope = float(np.median(t_g4 / n_g4))
    n_hi = max(n_gpu.max(), n_g4.max())
    n_fit = np.array([n_g4.min() / 2.0, n_hi * 2.0])
    ax.plot(n_fit, slope * n_fit, ls="--", lw=2.2, color=C_G4,
            label="GEANT4 (Linear Fit)")
    print(f"[fig] GEANT4 linear fit: {slope * 1e3:.3f} ms/event; "
          f"extrapolated to {n_hi:g} primaries -> {slope * n_hi:.1f} s")

    ax.plot(n_gpu, t_gpu, marker="o", ms=11, lw=2.8, color=C_GPU,
            label="PHIN-GAN (GPU)")

    hw = []
    if cpu_label:
        hw.append(f"CPU: {cpu_label}")
    if device_name:
        hw.append(f"GPU: {device_name}")
    if hw:
        ax.text(0.03, 0.74, "\n".join(hw), transform=ax.transAxes,
                ha="left", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(True, which="both", ls=":", color="0.8")
    ax.set_xlabel("Number of primary particles")
    ax.set_ylabel("Computation Time [s]")
    ax.legend(loc="upper left")
    return fig


def report_crossover(n_gpu, t_gpu, n_g4, t_g4):
    """Where the GEANT4 fit crosses the PHIN-GAN curve — the figure's claim.

    Printed, not drawn: it is a derived number and belongs in the text, but
    it should not be eyeballed off a log-log plot.
    """
    slope = float(np.median(t_g4 / n_g4))
    ratio = slope * n_gpu / t_gpu          # GEANT4 / PHIN-GAN at each point
    print("[fig] GEANT4-fit / PHIN-GAN speedup: "
          + ", ".join(f"{n:g}:{r:.2f}x" for n, r in zip(n_gpu, ratio)))
    below = np.flatnonzero(ratio < 1.0)
    above = np.flatnonzero(ratio >= 1.0)
    if below.size and above.size:
        print(f"[fig] crossover lies between {n_gpu[below[-1]]:g} and "
              f"{n_gpu[above[0]]:g} primaries")
    elif not above.size:
        print("[fig] GEANT4 is cheaper across the whole measured range")
    else:
        print("[fig] PHIN-GAN is cheaper across the whole measured range")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path,
                   help="profile_runtime JSON report(s); several are averaged "
                        "as repeats of one measurement")
    p.add_argument("--g4-csv", type=Path, required=True,
                   help="two-column CSV n_events,seconds (no header) from the "
                        "saving-disabled GEANT4 proton sweep")
    p.add_argument("--cpu-label", default=None,
                   help="CPU model annotation for the GEANT4 arm")
    p.add_argument("--gpu-label", default=None,
                   help="override the GPU name read from the report meta")
    p.add_argument("--min-events", type=float, default=100,
                   help="drop points with fewer primaries than this from both "
                        "arms, the fit included (default 100)")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: BENCHMARKS_OUTPUTS/"
                        "computation_time.pdf)")
    args = p.parse_args()

    n_gpu, t_gpu, device_name = load_repeats(args.reports)
    raw = np.loadtxt(args.g4_csv, delimiter=",", ndmin=2)
    n_g4, t_g4 = raw[:, 0], raw[:, 1]

    keep_gpu, keep_g4 = n_gpu >= args.min_events, n_g4 >= args.min_events
    dropped = sorted(set(n_gpu[~keep_gpu]) | set(n_g4[~keep_g4]))
    if dropped:
        print("[fig] dropping points below --min-events "
              f"{args.min_events:g}: {', '.join(f'{n:g}' for n in dropped)}")
    n_gpu, t_gpu = n_gpu[keep_gpu], t_gpu[keep_gpu]
    n_g4, t_g4 = n_g4[keep_g4], t_g4[keep_g4]

    report_crossover(n_gpu, t_gpu, n_g4, t_g4)

    with matplotlib.rc_context(PAPER_RC):
        fig = build_figure(n_gpu, t_gpu, n_g4, t_g4,
                           args.gpu_label or device_name, args.cpu_label)
    out = args.out or (BENCHMARKS_OUTPUTS / "computation_time.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
