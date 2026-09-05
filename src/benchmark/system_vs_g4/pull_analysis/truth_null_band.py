"""The GEANT4-versus-GEANT4 pull band printed on Figs 13 and E.2.

The pull figure fits a Gaussian to the per-bin pull between a simulated and
a truth energy-deposition map. Its mu_0 and sigma_0 are only meaningful
against the scatter two *truth* samples show against each other, so this
driver scores that null: ten independently seeded 10k-event GEANT4 samples
(seeds 42..51 under `--truth-dir`), every one of the C(10,2) = 45 pairs
pulled at the figure's own binning, then

  |mu_t|  <= max over pairs of |mu|          the band's first line
  sigma_t in [min, max] over pairs of sigma  its second
  <mu>, <sigma>                              the leave-one-sample-out
                                             jackknife means, with their
                                             standard errors recorded

Each file is streamed once through the slim reader, shifted into the
figure's frame (injection at z = 0, asserted per file the way
`run_pull_figure` asserts it) and binned once; the pairs are scored from
the stored per-file histograms, which is 10 binnings instead of 90 and keeps
memory at three 1500x1500 arrays per file rather than ten 98M-row frames.
The bin range is the first file's, mirroring `pull_analysis`, whose range
comes from its truth argument.

The result is written as JSON (default `measurements/pull_truth_null/
truth_vs_truth_pull.json`), with every per-pair fit, so the four numbers
the figure prints are auditable; `run_pull_figure` reads that file through
`load_tolerance_text`.

    cd src && PYTHONPATH=. python -m \\
        benchmark.system_vs_g4.pull_analysis.truth_null_band
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmark.system_vs_g4.pull_analysis import energy_deposition_pull_analysis as pa
from common.enums import G4Columns
from common.paths import PROJECT_DIR, STORAGE
from data_handling.slim_reader import load_pull_slim_cached

log = logging.getLogger(__name__)

DEFAULT_SEEDS = tuple(range(42, 52))
# The seeded 10k productions live beside the branch's storage tree, not in
# it: `storage_paper/ResourceFiles/iid/100MeVProtonInAl_seed<N>.root`.
DEFAULT_TRUTH_DIR = STORAGE.parent / "storage_paper" / "ResourceFiles" / "iid"
DEFAULT_OUT = PROJECT_DIR / "measurements" / "pull_truth_null" / "truth_vs_truth_pull.json"
# The truth files inject at X = -0.05 m; the figure's frame puts injection
# at z = 0 (run_pull_figure._SIM_TO_TRUTH_X_DISPLACE_m).
_TRUTH_INJECTION_X_m = -0.05


@dataclass
class HistStats:
    """One sample's binned energy deposition: per-bin sum, sum of squared
    per-event deposits, and the number of events touching the bin."""
    sum: NDArray[np.float64]
    sum_sq: NDArray[np.float64]
    counts: NDArray[np.int64]


def truth_path(truth_dir: Path, seed: int) -> Path:
    return truth_dir / f"100MeVProtonInAl_seed{seed}.root"


def load_truth_hists(paths: list[Path], grid: int,
                     ) -> tuple[list[tuple[float, float]], list[HistStats]]:
    """Stream each file, shift it into the figure's frame, and bin it on the
    first file's range. Returns that range (mm) and one HistStats per file."""
    range_lims: list[tuple[float, float]] | None = None
    hists: list[HistStats] = []
    for path in paths:
        t0 = time.perf_counter()
        raw = load_pull_slim_cached(path)
        x_min = float(raw[G4Columns.X].min())
        assert abs(x_min - _TRUTH_INJECTION_X_m) < 1e-6, (
            f"{path.name}: injection X {x_min} is not {_TRUTH_INJECTION_X_m}")
        df = pa.organize_df(raw, displace=-_TRUTH_INJECTION_X_m)
        del raw
        if range_lims is None:
            range_lims = [(float(df[G4Columns.X].min()), float(df[G4Columns.X].max())),
                          (float(df[G4Columns.Y].min()), float(df[G4Columns.Y].max()))]
        h_sum, h_sum_sq, h_counts = pa.compute_hist_stats(df, range_lims, (grid, grid))
        hists.append(HistStats(h_sum, h_sum_sq, h_counts.astype(np.int64)))
        log.info("%s: %d rows binned in %.0f s", path.name, len(df),
                 time.perf_counter() - t0)
        del df
    assert range_lims is not None
    return range_lims, hists


def pair_pull(h1: HistStats, h2: HistStats, min_avg_tracks: int,
              pull_range: float) -> dict:
    """The figure's pull statistic between two binned samples: pull map,
    bins averaging more than `min_avg_tracks` events kept, Gaussian fit
    over +-pull_range (both the histogram and the fit window, as in
    `run_pull_figure`)."""
    pull = pa.compute_2d_pull(h1.sum, h1.sum_sq, h2.sum, h2.sum_sq).ravel()
    avg_count = (h1.counts.ravel() + h2.counts.ravel()) / 2
    pull[avg_count <= min_avg_tracks] = np.nan
    _, popt, pcov, chi2_ndof = pa.fit_pull_distribution(
        pull_values=pull,
        hist_range=(-pull_range, pull_range),
        fit_range=(-pull_range, pull_range))
    return {
        "mu": float(popt[1]), "sigma": float(popt[2]),
        "mu_err": float(np.sqrt(pcov[1, 1])),
        "sigma_err": float(np.sqrt(pcov[2, 2])),
        "chi2_ndof": float(chi2_ndof),
        "n_bins": int(np.isfinite(pull).sum()),
    }


def all_pairs(hists: list[HistStats], seeds: list[int], min_avg_tracks: int,
              pull_range: float) -> list[dict]:
    pairs = list(combinations(range(len(hists)), 2))
    out = []
    for k, (i, j) in enumerate(pairs):
        log.info("pair (%d, %d) [%d/%d]", seeds[i], seeds[j], k + 1, len(pairs))
        r = pair_pull(hists[i], hists[j], min_avg_tracks, pull_range)
        out.append({"i": i, "j": j, "seeds": [seeds[i], seeds[j]], **r})
    return out


def summarize(pairs: list[dict], n_batches: int) -> dict:
    """The band's four numbers plus their uncertainties. The bracketed
    means are the leave-one-sample-out jackknife of the pairwise mean
    (`pa.jackknife_pull_estimate`), which is unbiased for the pairwise mean
    and gives a standard error that respects the pairs sharing samples."""
    mus = np.array([p["mu"] for p in pairs])
    sigmas = np.array([p["sigma"] for p in pairs])
    idx = [(p["i"], p["j"]) for p in pairs]
    jk = pa.jackknife_pull_estimate(mus, sigmas, idx, n_batches)
    return {
        "mu_max_abs": float(np.max(np.abs(mus))),
        "sigma_min": float(sigmas.min()),
        "sigma_max": float(sigmas.max()),
        "mu_mean": float(jk["mu_mean"]),
        "mu_jackknife_se": float(jk["mu_se"]),
        "sigma_mean": float(jk["sigma_mean"]),
        "sigma_jackknife_se": float(jk["sigma_se"]),
        "mu_pairwise_std": float(mus.std(ddof=1)),
        "sigma_pairwise_std": float(sigmas.std(ddof=1)),
        "n_pairs": len(pairs),
    }


def tolerance_text(summary: dict) -> str:
    """The four lines the figure prints under its fitted mu_0 / sigma_0.
    The sigma range is printed to two decimals, rounded outward (floor of
    the minimum, ceiling of the maximum) so the printed band never
    understates the measured one."""
    s_lo = math.floor(summary["sigma_min"] * 100) / 100
    s_hi = math.ceil(summary["sigma_max"] * 100) / 100
    return (
        f"$|\\mu_t| \\leq {summary['mu_max_abs']:.3f}$\n"
        f"$\\sigma_t \\in [{s_lo:.2f}, {s_hi:.2f}]$\n"
        f"$\\langle\\mu\\rangle$: ${summary['mu_mean']:.3f}$\n"
        f"$\\langle\\sigma\\rangle$: ${summary['sigma_mean']:.3f}$"
    )


def load_tolerance_text(path: Path = DEFAULT_OUT, grid: int | None = None,
                        min_avg_tracks: int | None = None) -> str:
    """The band for the figure, refusing a study recorded at another binning
    than the one the figure is being drawn at."""
    doc = json.loads(path.read_text())
    meta = doc["meta"]
    if grid is not None and meta["grid"] != grid:
        raise ValueError(f"{path}: band recorded at grid {meta['grid']}, figure at {grid}")
    if min_avg_tracks is not None and meta["min_avg_tracks"] != min_avg_tracks:
        raise ValueError(f"{path}: band recorded at min_avg_tracks "
                         f"{meta['min_avg_tracks']}, figure at {min_avg_tracks}")
    return tolerance_text(doc["summary"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--grid", type=int, default=1500, help="y-z grid, N x N (figure: 1500)")
    p.add_argument("--min-avg-tracks", type=int, default=3,
                   help="keep bins averaging more than this many events (figure: 3)")
    p.add_argument("--pull-range", type=float, default=5.0,
                   help="fit and histogram half-range (figure: 5)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    paths = [truth_path(args.truth_dir, s) for s in args.seeds]
    missing = [str(q) for q in paths if not q.exists()]
    if missing:
        raise FileNotFoundError("missing truth files:\n" + "\n".join(missing))

    t0 = time.perf_counter()
    range_lims, hists = load_truth_hists(paths, args.grid)
    pairs = all_pairs(hists, args.seeds, args.min_avg_tracks, args.pull_range)
    summary = summarize(pairs, n_batches=len(hists))

    doc = {
        "meta": {
            "seeds": args.seeds,
            "files": [{"path": str(q), "bytes": q.stat().st_size,
                       "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(q.stat().st_mtime))}
                      for q in paths],
            "grid": args.grid, "min_avg_tracks": args.min_avg_tracks,
            "pull_range": args.pull_range,
            "range_mm": {"x": list(range_lims[0]), "y": list(range_lims[1])},
            "injection_x_m": _TRUTH_INJECTION_X_m,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_s": round(time.perf_counter() - t0, 1),
        },
        "summary": summary,
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(tolerance_text(summary))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
