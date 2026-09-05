"""Fig E.1 -- the physics regularization along the training session of the
no-physics ablation (GAN) and the physics-informed generator (PHIN-GAN), for
the continuous and secondary stages.

Four arms: physics continuous, physics secondary, no-physics continuous,
no-physics secondary -- the exact classes and checkpoints
`benchmark.track_simulator_config`'s `PRESETS` load. Every saved checkpoint
of every arm is scored POST HOC with `KSTestPenaltyModel.physics_penalty_loss`
-- the same KS-against-theoretical-CDFs term the physics-informed trainer
adds to its generator loss (unweighted, i.e. before the lambda_physics
multiplier). It is re-evaluated rather than read off TensorBoard because the
no-physics trainer never computes it (`add_generator_loss_terms` returns
early), so a scalar-based figure could only draw two of the four curves.

Epoch 0 -- the network before its first update -- is scored too, although
no run saves a checkpoint there (the first one is `epoch=99.ckpt` at the
100-epoch cadence). Both trainers call `seed_everything(seed)` and construct
the generator immediately afterwards, before the datamodule and the critic,
so seeding with the run's seed and calling the bare constructor rebuilds the
very weights the run started from; scoring that network is the epoch-0 point
of the paper's figure, evaluated post hoc like every other one.

The walk is expensive (all four runs together hold ~2500 checkpoints, saved
every 100 epochs), so `--score` writes an INCREMENTAL per-arm cache
(`<cache-dir>/physics_losses_<arm>.npz`, resumable -- already-scored epochs
are skipped) and `--plot` draws from the caches alone. The scored curves are
committed as `measurements/physics_losses/fig_E1_physics_losses.csv`;
`--import-csv <file>` seeds the caches from that file, so a machine without
the checkpoints (or without the GPU-hours) can still add the epoch-0 points
and redraw.

    cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.physics_losses --score --device cuda
    cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.physics_losses --plot [--mark-picks] [--csv <file>]
    cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.physics_losses \
        --import-csv ../measurements/physics_losses/fig_E1_physics_losses.csv --score --plot --device cpu

The 10 %-of-keys x 10k-samples subsample inside `physics_penalty_loss` is
random per call, exactly as during training; the numpy RNG is seeded once
per arm (`--seed`) so a re-run reproduces the same curve.
"""
import argparse
import logging
import re
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

from common.config import DEVICE, apply_project_plotting_style
from common.enums import ContinuousFeats, SecondaryFeats
from common.paths import (BENCHMARKS_OUTPUTS, FIGURES_DIR, PATH_TO_CONTINUOUS_CDFS, PATH_TO_SECONDARIES_CDFS, NOPHYS_CNT_CKPT, CNT_CKPT, NOPHYS_SEC_CKPT, SEC_CKPT, NOPHYS_CNT_SCALERS_DIR, SCALERS_DIR)
from physics import penalty_models as pm
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator, NoPhysContinuousGenerator)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousXScaler, ContinuousYScaler, NoPhysContinuousYScaler)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    NoPhysSecondaryGenerator, SecondaryGenerator)
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    NoPhysSecondaryYScaler, SecondaryXScaler, SecondaryYScaler)
from physics.interfaces.generator_interface import GeneratorInterface
from physics.interfaces.scaler_interface import ScalerInterface

logger = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"^epoch=(\d+)\.ckpt$")


class Arm(NamedTuple):
    key:          str            # cache / CLI token
    label:        str            # legend text (mathtext)
    stage:        str            # "cnt" | "sec" -> which penalizer
    gen_cls:      type[GeneratorInterface]
    x_scaler_cls: type[ScalerInterface]
    y_scaler_cls: type[ScalerInterface]
    run_dir:      Path           # the run directory holding checkpoints/
    pinned_epoch: int            # the checkpoint the simulator preset loads
    color:        str
    alpha:        float
    state_dir:    Path           # scaler tree the checkpoints trained against
    init_seed:    int            # the seed the run's trainer passed to seed_everything


# The trainers' default seed (PHINGAN_SEED unset). Runs launched before the
# seed was recorded in meta.json ran at this value.
DEFAULT_TRAIN_SEED = 42


def _arm(key, label, stage, gen_cls, x_cls, y_cls, ckpt: Path, color, alpha,
         state_dir: Path = SCALERS_DIR, init_seed: int = DEFAULT_TRAIN_SEED) -> Arm:
    return Arm(key, label, stage, gen_cls, x_cls, y_cls,
               ckpt.parent.parent, int(ckpt.stem.split("=")[1]), color, alpha,
               state_dir, init_seed)


# Same classes / checkpoints as benchmark.track_simulator_config's PRESETS --
# the figure describes the training of exactly the nets the simulator ships.
ARMS: tuple[Arm, ...] = (
    # The GAN cnt arm is scored with the scaler states it trained against
    # (NOPHYS_CNT_SCALERS_DIR) -- see NOPHYS_CNT_CKPT in paths.py.
    _arm("gan_cnt", r"GAN: $\mathcal{L}^{\mathrm{cnt}}_{\mathrm{phys}}$", "cnt",
         NoPhysContinuousGenerator, ContinuousXScaler, NoPhysContinuousYScaler,
         NOPHYS_CNT_CKPT, "red", 1.0, state_dir=NOPHYS_CNT_SCALERS_DIR),
    _arm("gan_sec", r"GAN: $\mathcal{L}^{\mathrm{sec}}_{\mathrm{phys}}$", "sec",
         NoPhysSecondaryGenerator, SecondaryXScaler, NoPhysSecondaryYScaler,
         NOPHYS_SEC_CKPT, "red", 0.35),
    _arm("phingan_cnt", r"PHIN-GAN: $\mathcal{L}^{\mathrm{cnt}}_{\mathrm{phys}}$", "cnt",
         ContinuousGenerator, ContinuousXScaler, ContinuousYScaler,
         CNT_CKPT, "blue", 1.0),
    # The pinned secondary run is the seed-2 member of its sweep (meta.json
    # records "seed": 2); the other three runs predate that field and ran at
    # the trainers' default.
    _arm("phingan_sec", r"PHIN-GAN: $\mathcal{L}^{\mathrm{sec}}_{\mathrm{phys}}$", "sec",
         SecondaryGenerator, SecondaryXScaler, SecondaryYScaler,
         SEC_CKPT, "blue", 0.35, init_seed=2),
)

DEFAULT_CACHE_DIR = FIGURES_DIR / "physics_losses_cache"


# --------------------------------------------------------------------------
# checkpoint walk + cache
# --------------------------------------------------------------------------
def list_checkpoints(run_dir: Path, max_epoch: Optional[int]) -> list[tuple[int, Path]]:
    """`epoch=N.ckpt` files under <run_dir>/checkpoints, ascending in N,
    `last.ckpt` and anything else excluded, capped at max_epoch (inclusive)."""
    out = []
    for p in (run_dir / "checkpoints").iterdir():
        m = _EPOCH_RE.match(p.name)
        if m is None:
            continue
        e = int(m.group(1))
        if max_epoch is not None and e > max_epoch:
            continue
        out.append((e, p))
    return sorted(out)


def cache_path(cache_dir: Path, arm: "Arm") -> Path:
    # Keyed by the run directory's name, not just the arm: a checkpoint
    # re-pin that moves an arm to a DIFFERENT run must start a fresh cache
    # rather than silently mixing two runs' losses under one key.
    return cache_dir / f"physics_losses_{arm.key}_{arm.run_dir.name}.npz"


def load_cache(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    d = np.load(path)
    return {int(e): float(v) for e, v in zip(d["epochs"], d["losses"])}


def save_cache(path: Path, scored: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array(sorted(scored), dtype=np.int64)
    losses = np.array([scored[e] for e in epochs], dtype=np.float64)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, epochs=epochs, losses=losses)
    tmp.replace(path)     # atomic: a killed job never leaves a torn cache


def pending_checkpoints(run_dir: Path, scored: dict[int, float],
                        max_epoch: Optional[int]) -> list[tuple[int, Path]]:
    return [(e, p) for e, p in list_checkpoints(run_dir, max_epoch) if e not in scored]


def import_csv(path: Path, cache_dir: Path, arms=ARMS) -> None:
    """Seed the per-arm caches from a committed long-format CSV (the format
    `export_csv` writes). Epochs already in a cache are left as they are."""
    rows: dict[str, dict[int, float]] = {}
    with open(path) as f:
        header = f.readline().strip()
        if header != "arm,epoch,physics_regularization":
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in f:
            key, e, v = line.strip().split(",")
            rows.setdefault(key, {})[int(e)] = float(v)
    for arm in arms:
        if arm.key not in rows:
            continue
        cpath = cache_path(cache_dir, arm)
        scored = load_cache(cpath)
        new = {e: v for e, v in rows[arm.key].items() if e not in scored}
        scored.update(new)
        save_cache(cpath, scored)
        logger.info("%s: imported %d epochs from %s (%d already cached)",
                    arm.key, len(new), path, len(scored) - len(new))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def build_penalizer(stage: str) -> pm.KSTestPenaltyModel:
    if stage == "cnt":
        return pm.KSTestPenaltyModel(path_to_cdf=PATH_TO_CONTINUOUS_CDFS,
                                     columns_idx=ContinuousFeats.E_CNT_IDX)
    if stage == "sec":
        return pm.KSTestPenaltyModel(path_to_cdf=PATH_TO_SECONDARIES_CDFS,
                                     columns_idx=SecondaryFeats.E_SEC_IDX)
    raise KeyError(stage)


def initial_generator(arm: Arm, device: str,
                      x_scaler: ScalerInterface, y_scaler: ScalerInterface) -> GeneratorInterface:
    """The generator exactly as the run's trainer built it before its first
    update: `seed_everything(seed)` then the bare constructor, the same two
    calls `train_gan` makes back to back. The bare constructor is also the
    `from_checkpoint` contract, so the architecture matches the checkpoints
    scored beside it."""
    seed_everything(arm.init_seed, verbose=False)
    gen = arm.gen_cls()
    gen.freeze()
    gen.to(device)
    gen.set_scalers(x_scaler, y_scaler)
    return gen


def score_arm(arm: Arm, penalizer: pm.KSTestPenaltyModel, cache_dir: Path,
              device: str, max_epoch: Optional[int], seed: int,
              save_every: int = 10) -> dict[int, float]:
    """Walk the arm's checkpoints, scoring each not already in the cache;
    epoch 0 is scored from the untrained network (see `initial_generator`)."""
    path = cache_path(cache_dir, arm)
    scored = load_cache(path)
    todo = pending_checkpoints(arm.run_dir, scored, max_epoch)
    logger.info("%s: %d cached, %d to score under %s", arm.key, len(scored), len(todo), arm.run_dir)
    if not todo and 0 in scored:
        return scored

    x_scaler = arm.x_scaler_cls.load(state_dir=arm.state_dir).to(device)
    y_scaler = arm.y_scaler_cls.load(state_dir=arm.state_dir).to(device)

    if 0 not in scored:
        gen = initial_generator(arm, device, x_scaler, y_scaler)
        np.random.seed(seed)      # physics_penalty_loss subsamples keys via np.random
        torch.manual_seed(seed)
        with torch.no_grad():
            scored[0] = penalizer.physics_penalty_loss(generator=gen, requires_grad=False).item()
        logger.info("%s epoch=0 (untrained, init seed %d) loss=%.4e", arm.key, arm.init_seed, scored[0])
        save_cache(path, scored)

    np.random.seed(seed)      # physics_penalty_loss subsamples keys via np.random
    torch.manual_seed(seed)
    t0 = time.time()
    for i, (epoch, ckpt) in enumerate(todo, 1):
        gen = arm.gen_cls.from_checkpoint(ckpt, device=device, x_scaler=x_scaler, y_scaler=y_scaler)
        gen.eval()
        with torch.no_grad():
            loss = penalizer.physics_penalty_loss(generator=gen, requires_grad=False).item()
        scored[epoch] = loss
        logger.info("%s epoch=%d loss=%.4e (%d/%d, %.1f s/ckpt)",
                    arm.key, epoch, loss, i, len(todo), (time.time() - t0) / i)
        if i % save_every == 0:
            save_cache(path, scored)
    save_cache(path, scored)
    return scored


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_physics_losses(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                        mark_picks: bool = False,
                        ylim: tuple[float, float] = (5e-3, 3e0)) -> plt.Figure:
    """The paper's Fig E.1 layout. `curves` maps arm key -> (epochs, losses)."""
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    for arm in ARMS:
        if arm.key not in curves:
            continue
        ep, d = curves[arm.key]
        idx = np.argsort(ep)
        ep, d = np.asarray(ep)[idx], np.asarray(d)[idx]
        ax.plot(ep, d, color=arm.color, linestyle="-", linewidth=1.5, alpha=arm.alpha, label=arm.label)
        if mark_picks and arm.pinned_epoch in set(ep.tolist()):
            ax.plot([arm.pinned_epoch], [d[ep == arm.pinned_epoch][0]], marker="o",
                    markersize=9, markerfacecolor="white", markeredgecolor=arm.color,
                    markeredgewidth=2, linestyle="none", zorder=5)
    ax.set_xlabel("Epoch Number", fontsize=24, fontweight="bold")
    ax.set_ylabel("Physics Regularization", fontsize=24, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=22)
    ax.set_ylim(*ylim)
    return fig


def load_curves(cache_dir: Path, arms=ARMS, max_epoch: Optional[int] = None) -> dict:
    curves = {}
    for arm in arms:
        scored = load_cache(cache_path(cache_dir, arm))
        if max_epoch is not None:
            scored = {e: v for e, v in scored.items() if e <= max_epoch}
        if scored:
            ep = np.array(sorted(scored))
            curves[arm.key] = (ep, np.array([scored[e] for e in ep]))
        else:
            logger.warning("no cache for arm %s under %s", arm.key, cache_dir)
    return curves


def export_csv(path: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Long-format record of what was plotted, so the figure is reproducible
    from the committed numbers without the checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("arm,epoch,physics_regularization\n")
        for arm in ARMS:
            if arm.key not in curves:
                continue
            ep, d = curves[arm.key]
            for e, v in zip(ep, d):
                f.write(f"{arm.key},{int(e)},{v:.6e}\n")


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--score", action="store_true", help="walk the checkpoints and (re)fill the caches")
    p.add_argument("--plot", action="store_true", help="draw the figure from the caches")
    # The GAN arms exist for aluminium only; other beams default to the
    # PHIN-GAN pair.
    from benchmark.track_simulator_config import has_gan_preset
    p.add_argument("--arms", nargs="+", choices=[a.key for a in ARMS],
                   default=[a.key for a in ARMS if has_gan_preset() or a.key.startswith("phingan")])
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--max-epoch", type=int, default=40000,
                   help="score/plot checkpoints up to this epoch (inclusive); <0 = no cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--import-csv", type=Path, default=None,
                   help="seed the caches from a committed long-format CSV before scoring/plotting")
    p.add_argument("--out-dir", type=Path, default=BENCHMARKS_OUTPUTS)
    p.add_argument("--csv", type=Path, default=None,
                   help="also export the plotted curves (long format: arm,epoch,loss) to this CSV")
    p.add_argument("--mark-picks", action="store_true",
                   help="hollow circle on each curve at the checkpoint the simulator preset loads")
    p.add_argument("--stem", default="fig_E1_physics_losses")
    args = p.parse_args(argv)
    if not (args.score or args.plot or args.import_csv):
        p.error("give --score, --plot and/or --import-csv")
    # root at WARNING silences common.unpickle_ckpt's per-checkpoint root-logger
    # chatter; this module's own progress lines still propagate to the handler.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    max_epoch = None if args.max_epoch < 0 else args.max_epoch
    arms = [a for a in ARMS if a.key in args.arms]

    if args.import_csv is not None:
        import_csv(args.import_csv, args.cache_dir, arms)

    if args.score:
        penalizers: dict[str, pm.KSTestPenaltyModel] = {}
        for arm in arms:
            if arm.stage not in penalizers:
                logger.info("loading %s theoretical CDFs", arm.stage)
                penalizers[arm.stage] = build_penalizer(arm.stage)
            score_arm(arm, penalizers[arm.stage], args.cache_dir, args.device, max_epoch, args.seed)

    if args.plot:
        apply_project_plotting_style()
        curves = load_curves(args.cache_dir, arms, max_epoch)
        fig = plot_physics_losses(curves, mark_picks=args.mark_picks)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.stem + ("_marked" if args.mark_picks else "")
        for ext in ("pdf", "png"):
            out = args.out_dir / f"{stem}.{ext}"
            fig.savefig(out, format=ext, dpi=200)
            logger.info("wrote %s", out)
        if args.csv is not None:
            export_csv(args.csv, curves)
            logger.info("wrote %s", args.csv)


if __name__ == "__main__":
    main()
