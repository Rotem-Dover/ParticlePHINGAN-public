"""Checkpoint scan for one training run: per-checkpoint z-space statistics of
the generator against GEANT4 truth on the truth's own conditioning rows.

This is the ranking instrument behind a checkpoint pick. Every checkpoint of
the run is loaded in turn (the same scalers, the same conditioning rows, the
same noise seed, so checkpoints differ only by their weights) and scored in
the generator's scaled space, where truth is `y_scaler.scale(Y, X)`:

* continuous -- `z_bias` (mean z_gen - mean z_truth, in physics-sigma units),
  `z_std_ratio`, the two-sample KS statistic `ks`, and the tail fractions.
  The z-bias is the 1M-row proxy that ranked the aluminium continuous
  checkpoints (Spearman 0.82 against fig 12's energy distance); it is white
  noise across 100-epoch saves, so rank by it and confirm the shortlist with
  `energy_distance_analysis --score`.
* secondary -- the boundary-atom census of `secondary_endpoint_atoms`
  (mass exactly at z == 0 and z == 1) plus the tail fractions P(z > 0.99),
  P(z > 0.999) that the aluminium reflect-rule pick was made on, and `ks`.
  The generator is built at the current config kind, so an open_top
  checkpoint is scored under the reflect eval rule the runtime applies.

When the run directory holds TensorBoard event files, the physics-penalty
scalar at each checkpoint's epoch is joined into the table as `phys_reg`.

    cd src && PHINGAN_BEAM=proton_in_iron_100MeV PYTHONPATH=. python -m \
        benchmark.system_vs_g4.checkpoint_scan --stage secondary \
        --run 2026-08-22-155334_secondary_s3_hardtanh_open_top_w16_iron_6885824c7d_6f298002
    ... --stage continuous --run <run> --rows 1000000 --every 1

Writes `<BENCHMARKS_ROOT>/ckpt_scan/<beam>/<run>.csv` and a `.png` of the
metrics against epoch. Read-only on the run directory.
"""
import argparse
import csv
import pickle as pkl
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.enums import SecondaryFeats
from common.paths import (BENCHMARKS_ROOT, DATASETS_DIR, PATH_TO_GEANT4_DATA_SEED42,
                          RUNS_DIR, SCALERS_DIR)
from common.run_context import ACTIVE_NAME

_EPOCH_RE = re.compile(r"epoch=(\d+)\.ckpt$")


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no p-value)."""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def _tails(z: np.ndarray, prefix: str) -> dict:
    n = float(len(z))
    return {
        f"{prefix}_z_le_0": float((z <= 0).sum()) / n,
        f"{prefix}_z_ge_1": float((z >= 1).sum()) / n,
        f"{prefix}_z_gt_0.99": float((z > 0.99).sum()) / n,
        f"{prefix}_z_gt_0.999": float((z > 0.999).sum()) / n,
        f"{prefix}_z_max": float(z.max()),
    }


def _panel_bins(stage: str):
    """Fig 11's (conditioning E, loss) bin edges for the stage's panel."""
    import math
    from common.config import BEAM_ENERGY_eV, MIN_ENERGY_CUTOFF
    from common.run_context import PLOT_RANGES
    ke_bins = np.logspace(math.log10(MIN_ENERGY_CUTOFF), math.log10(BEAM_ENERGY_eV), 100)
    loss_bins = np.logspace(*(PLOT_RANGES.cnt_loss if stage == "continuous" else PLOT_RANGES.sec_loss), 100)
    return ke_bins, loss_bins


def _panel_reldiff(e_ref, y_ref, e_cmp, y_cmp, bins) -> tuple[float, float]:
    """Fig 11's per-bin relative difference |cmp - ref| / ref on the (E, loss)
    panel, each histogram normalised by its row count and bins with fewer than
    30 reference rows dropped: (median, fraction of bins above 0.1). KS in z
    is blind to the few-percent ripples this exposes, which is what the
    figure shows as a tinted panel."""
    hr, _, _ = np.histogram2d(e_ref, y_ref, bins=bins)
    hc, _, _ = np.histogram2d(e_cmp, y_cmp, bins=bins)
    hr /= len(e_ref)
    hc /= len(e_cmp)
    keep = hr >= 30 / len(e_ref)
    d = (np.abs(hc - hr) / np.where(keep, hr, 1.0))[keep]
    return float(np.median(d)), float((d > 0.1).mean())


def _cond_chi2(e: np.ndarray, z_gen: np.ndarray, z_truth: np.ndarray, n_e: int = 20, n_z: int = 40) -> float:
    """Reduced chi-square of the generator against truth on a conditional
    (log E, z) grid, Poisson variance from both sides, over bins with at
    least 50 truth rows. About 1 when the generator is truth-like; a real
    ripple in the spectrum raises it, while KS averages the ripple away and
    fig 11's per-bin relative difference at scan statistics is mostly its
    own Poisson noise."""
    e_edges = np.logspace(np.log10(e.min()), np.log10(e.max()), n_e + 1)
    z_edges = np.linspace(min(z_gen.min(), z_truth.min(), 0.0), max(z_gen.max(), z_truth.max(), 1.0), n_z + 1)
    hg, _, _ = np.histogram2d(e, z_gen, bins=[e_edges, z_edges])
    ht, _, _ = np.histogram2d(e, z_truth, bins=[e_edges, z_edges])
    ng, nt = len(z_gen), len(z_truth)
    keep = ht >= 50
    pg, pt = hg / ng, ht / nt
    var = hg / ng ** 2 + ht / nt ** 2
    return float(np.mean(((pg - pt) ** 2 / var)[keep]))


def _load_stage(stage: str):
    """(generator class, x_scaler, y_scaler, X, Y) for the stage, truth from
    the 10k file's filtered figure dataset."""
    stem = PATH_TO_GEANT4_DATA_SEED42.stem
    if stage == "continuous":
        from physics.g4h_ionisation.generators.continuous_generator.neural_net import ContinuousGenerator
        from physics.g4h_ionisation.generators.continuous_generator.scaler import (
            ContinuousXScaler, ContinuousYScaler)
        cls = ContinuousGenerator
        xs, ys = ContinuousXScaler.load(state_dir=SCALERS_DIR), ContinuousYScaler.load(state_dir=SCALERS_DIR)
    else:
        from physics.g4h_ionisation.generators.secondary_generator.neural_net import SecondaryGenerator
        from physics.g4h_ionisation.generators.secondary_generator.scaler import (
            SecondaryXScaler, SecondaryYScaler)
        cls = SecondaryGenerator
        xs, ys = SecondaryXScaler.load(state_dir=SCALERS_DIR), SecondaryYScaler.load(state_dir=SCALERS_DIR)
    with open(DATASETS_DIR / f"{stem}_{stage}_X_filtered.pkl", "rb") as f:
        X = torch.from_numpy(pkl.load(f)).float()
    with open(DATASETS_DIR / f"{stem}_{stage}_Y_filtered.pkl", "rb") as f:
        Y = torch.from_numpy(pkl.load(f)).float()
    return cls, xs, ys, X.reshape(len(X), -1), Y.reshape(len(Y), -1)


def _phys_reg_by_epoch(run_dir: Path, stage: str) -> dict[int, float]:
    """epoch -> physics-penalty scalar, from the run's TensorBoard events.

    The scalar is logged per optimiser step; the `epoch` scalar maps step to
    epoch, and each epoch takes the last penalty value logged within it.
    """
    tag = "Physics Regularization Continuous" if stage == "continuous" else "Physics Regularization Secondary"
    by_epoch_csv = run_dir / "scalars_by_epoch.csv"
    scalars_csv = run_dir / "scalars.csv"
    if by_epoch_csv.is_file():
        # One row per epoch, `<tag>_mean` / `<tag>_last` columns (the
        # cluster-side aggregation of the per-step scalars); the epoch mean
        # of the penalty is the smoother of the two.
        with open(by_epoch_csv, newline="") as f:
            return {int(r["epoch"]): float(r[tag + "_mean"])
                    for r in csv.DictReader(f) if r.get(tag + "_mean") not in (None, "")}
    if scalars_csv.is_file():
        # A (tag, step, wall_time, value) dump made beside the event file --
        # the event files themselves carry the logged images and run to
        # hundreds of MB, so a mirror of the run may hold only this.
        series: dict[str, list[tuple[int, float]]] = {}
        with open(scalars_csv, newline="") as f:
            for row in csv.DictReader(f):
                series.setdefault(row["tag"], []).append((int(row["step"]), float(row["value"])))
        if tag not in series or "epoch" not in series:
            return {}
        ep = sorted(series["epoch"])
        pr = sorted(series[tag])
        ep_steps, ep_vals = np.array([s for s, _ in ep]), np.array([v for _, v in ep])
        pr_steps, pr_vals = np.array([s for s, _ in pr]), np.array([v for _, v in pr])
    else:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {}
        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if tag not in tags or "epoch" not in tags:
            return {}
        ep = acc.Scalars("epoch")
        ep_steps = np.array([e.step for e in ep])
        ep_vals = np.array([e.value for e in ep])
        pr = acc.Scalars(tag)
        pr_steps = np.array([e.step for e in pr])
        pr_vals = np.array([e.value for e in pr])
    idx = np.searchsorted(ep_steps, pr_steps, side="right") - 1
    idx = np.clip(idx, 0, len(ep_vals) - 1)
    out: dict[int, float] = {}
    for e, v in zip(ep_vals[idx].astype(int), pr_vals):
        out[int(e)] = float(v)   # last value within the epoch wins
    return out


def scan(stage: str, run_dir: Path, rows: int, seed: int, every: int, out_dir: Path) -> Path:
    cls, xs, ys, X, Y = _load_stage(stage)
    g = torch.Generator().manual_seed(seed)
    if rows and rows < len(X):
        pick = torch.randperm(len(X), generator=g)[:rows]
        X, Y = X[pick], Y[pick]
    with torch.no_grad():
        x_scaled = xs.scale(X)
        z_truth = ys.scale(Y, X).flatten().numpy()
    truth_tails = _tails(z_truth, "g4")
    # Fig 11's panel metric: truth split in halves for the null, the
    # generator on the first half's conditioning against that half.
    bins = _panel_bins(stage)
    e_all = X[:, 0].numpy()
    y_all = Y[:, 0].numpy()
    half = len(X) // 2
    panel_null = _panel_reldiff(e_all[:half], y_all[:half], e_all[half:2 * half], y_all[half:2 * half], bins)
    print(f"[{ACTIVE_NAME}] {stage}: fig-11 panel null median {panel_null[0]:.4f}, "
          f"frac>0.1 {panel_null[1]:.4f}", file=sys.stderr)
    print(f"[{ACTIVE_NAME}] {stage}: {len(X)} truth rows; "
          f"truth z mean {z_truth.mean():+.4f} std {z_truth.std():.4f}; "
          + " ".join(f"{k}={v:.3e}" for k, v in truth_tails.items()), file=sys.stderr)

    ckpts = sorted(run_dir.glob("checkpoints/epoch=*.ckpt"),
                   key=lambda p: int(_EPOCH_RE.search(p.name).group(1)))[::every]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{run_dir.name}.csv"
    rows_out = []
    t0 = time.time()
    for i, ck in enumerate(ckpts):
        epoch = int(_EPOCH_RE.search(ck.name).group(1))
        gen = cls.from_checkpoint(ck, device="cpu", x_scaler=xs, y_scaler=ys)
        gen.eval()
        torch.manual_seed(seed)                      # common random numbers across checkpoints
        with torch.no_grad():
            z = gen.forward(x_scaled).reshape(len(x_scaled), -1)
            z = z[:, int(SecondaryFeats.E_SEC_IDX)] if stage == "secondary" else z[:, 0]
        with torch.no_grad():
            y_gen = ys.rescale(z[:half].reshape(half, -1), X[:half])
        y_gen = y_gen.reshape(half, -1)[:, 0].numpy()
        panel = _panel_reldiff(e_all[:half], y_all[:half], e_all[:half], y_gen, bins)
        z = z.numpy()
        rec = {
            "epoch": epoch,
            "z_bias": float(z.mean() - z_truth.mean()),
            "z_std_ratio": float(z.std() / z_truth.std()),
            "ks": _ks(z, z_truth),
            "cond_chi2": _cond_chi2(e_all, z, z_truth),
            "panel_median": panel[0],
            "panel_frac_gt_0.1": panel[1],
            "panel_null_median": panel_null[0],
            "panel_null_frac_gt_0.1": panel_null[1],
            **_tails(z, "gen"),
            **truth_tails,
            "phys_reg": float("nan"),
            "n_rows": len(z),
        }
        rows_out.append(rec)
        if i % 20 == 0 or i == len(ckpts) - 1:
            el = time.time() - t0
            print(f"  {i + 1}/{len(ckpts)} epoch={epoch} z_bias={rec['z_bias']:+.4f} ks={rec['ks']:.4f} "
                  f"panel {rec['panel_median']:.3f}/{rec['panel_frac_gt_0.1']:.3f} "
                  f"z<=0 {rec['gen_z_le_0']:.2e} z>=1 {rec['gen_z_ge_1']:.2e} z>0.99 {rec['gen_z_gt_0.99']:.2e} "
                  f"[{el:.0f}s]", file=sys.stderr)
    # Joined last so a scalar dump that lands during a long scan is still used.
    phys_reg = _phys_reg_by_epoch(run_dir, stage)
    for rec in rows_out:
        rec["phys_reg"] = phys_reg.get(rec["epoch"], float("nan"))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    _plot(rows_out, stage, out_csv.with_suffix(".png"), run_dir.name)
    return out_csv


def confirm_emd(stage: str, run_dir: Path, epochs: list[int], n_batches: int, subset: int,
                seed: int, out_dir: Path, n_jobs: int = 8) -> Path:
    """Fig 12's energy distance for a shortlist of epochs: on `n_batches`
    pairs of `subset`-row truth batches, D_E(gen | batch-1 conditioning,
    batch 2) against the truth-vs-truth null D_E(batch 1, batch 2), both in
    the generator's scaled space. Reports the median ratio and the fraction
    of batches where the generator beats the null -- the statistic the KS
    scan cannot resolve, at the cost of an O(n^2) distance per batch."""
    from joblib import Parallel, delayed
    from common.utils import earths_movers_distance

    cls, xs, ys, X, Y = _load_stage(stage)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)[:2 * n_batches * subset]
    X, Y = X[perm].reshape(n_batches, 2, subset, -1), Y[perm].reshape(n_batches, 2, subset, -1)
    with torch.no_grad():
        zt = [(ys.scale(Y[i, 0], X[i, 0]).reshape(subset, -1).numpy(),
               ys.scale(Y[i, 1], X[i, 1]).reshape(subset, -1).numpy()) for i in range(n_batches)]
        xsc = [xs.scale(X[i, 0]) for i in range(n_batches)]
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(earths_movers_distance)(a, b) for a, b in zt))
    print(f"[{ACTIVE_NAME}] {stage} EMD null over {n_batches} batches of {subset}: "
          f"median {np.median(null):.4e}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{run_dir.name}_emd.csv"
    rows_out = []
    for epoch in epochs:
        ck = run_dir / "checkpoints" / f"epoch={epoch}.ckpt"
        gen = cls.from_checkpoint(ck, device="cpu", x_scaler=xs, y_scaler=ys)
        gen.eval()
        torch.manual_seed(seed)
        with torch.no_grad():
            zg = [gen.forward(x).reshape(subset, -1).numpy() for x in xsc]
        de = np.array(Parallel(n_jobs=n_jobs)(delayed(earths_movers_distance)(a, b[1])
                                              for a, b in zip(zg, zt)))
        rec = {"epoch": epoch, "emd_null_median": float(np.median(null)),
               "emd_gen_median": float(np.median(de)),
               "ratio_median": float(np.median(de) / np.median(null)),
               "ratio_mean": float(de.mean() / null.mean()),
               "frac_gen_below_null": float((de < null).mean()),
               "n_batches": n_batches, "subset": subset}
        rows_out.append(rec)
        print(f"  epoch={epoch} D_E gen {rec['emd_gen_median']:.4e} null {rec['emd_null_median']:.4e} "
              f"ratio {rec['ratio_median']:.3f} (mean {rec['ratio_mean']:.3f}) "
              f"P(gen<null)={rec['frac_gen_below_null']:.2f}", file=sys.stderr)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return out_csv


def _plot(rows_out: list[dict], stage: str, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.array([r["epoch"] for r in rows_out])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(ep, [r["z_bias"] for r in rows_out], ".-", ms=3, lw=0.6)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("z bias (gen - G4)")
    axes[1].plot(ep, [r["ks"] for r in rows_out], ".-", ms=3, lw=0.6)
    axes[1].set_ylabel("KS(z)")
    axes[1].set_yscale("log")
    if stage == "secondary":
        axes[2].plot(ep, [r["gen_z_le_0"] for r in rows_out], ".-", ms=3, lw=0.6, label="gen z<=0")
        axes[2].plot(ep, [r["gen_z_ge_1"] for r in rows_out], ".-", ms=3, lw=0.6, label="gen z>=1")
        axes[2].axhline(rows_out[0]["g4_z_le_0"], color="C0", ls="--", lw=0.8, label="G4 z<=0")
        axes[2].axhline(rows_out[0]["g4_z_ge_1"], color="C1", ls="--", lw=0.8, label="G4 z>=1")
        axes[2].set_ylabel("atoms")
        axes[2].set_yscale("symlog", linthresh=1e-6)
        axes[3].plot(ep, [r["gen_z_gt_0.99"] for r in rows_out], ".-", ms=3, lw=0.6, label="gen z>0.99")
        axes[3].plot(ep, [r["gen_z_gt_0.999"] for r in rows_out], ".-", ms=3, lw=0.6, label="gen z>0.999")
        axes[3].axhline(rows_out[0]["g4_z_gt_0.99"], color="C0", ls="--", lw=0.8, label="G4 z>0.99")
        axes[3].axhline(rows_out[0]["g4_z_gt_0.999"], color="C1", ls="--", lw=0.8, label="G4 z>0.999")
        axes[3].set_ylabel("tail")
        axes[3].set_yscale("log")
        axes[2].legend(fontsize=7, ncol=4)
        axes[3].legend(fontsize=7, ncol=4)
    else:
        axes[2].plot(ep, [r["z_std_ratio"] for r in rows_out], ".-", ms=3, lw=0.6)
        axes[2].axhline(1, color="k", lw=0.5)
        axes[2].set_ylabel("std ratio")
        axes[3].plot(ep, [r["phys_reg"] for r in rows_out], ".-", ms=3, lw=0.6)
        axes[3].set_ylabel("phys reg (TB)")
    axes[3].set_xlabel("epoch")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["continuous", "secondary"])
    ap.add_argument("--run", required=True,
                    help="run directory name under RUNS_DIR/logger_{cnt,sec}, or an absolute path")
    ap.add_argument("--rows", type=int, default=1_000_000, help="truth rows to score on (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="score every k-th checkpoint")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--emd-epochs", default=None,
                    help="comma-separated shortlist of epochs: skip the scan and score fig 12's "
                         "energy distance for these checkpoints only")
    ap.add_argument("--emd-batches", type=int, default=100)
    ap.add_argument("--emd-subset", type=int, default=10_000)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = RUNS_DIR / ("logger_cnt" if args.stage == "continuous" else "logger_sec") / args.run
    if not (run_dir / "checkpoints").is_dir():
        sys.exit(f"no checkpoints/ under {run_dir}")
    out_dir = args.out or (BENCHMARKS_ROOT / "ckpt_scan" / ACTIVE_NAME)
    if args.emd_epochs:
        epochs = [int(e) for e in args.emd_epochs.split(",")]
        out_csv = confirm_emd(args.stage, run_dir, epochs, args.emd_batches, args.emd_subset,
                              args.seed, out_dir)
        print(out_csv)
        return 0
    out_csv = scan(args.stage, run_dir, args.rows, args.seed, args.every, out_dir)
    print(out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
