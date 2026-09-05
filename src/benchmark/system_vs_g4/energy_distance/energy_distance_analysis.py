"""Figure 12 (energy distance): three arms per row -- GEANT4 vs GEANT4,
PHIN-GAN vs GEANT4, GAN (no-physics ablation) vs GEANT4.

Metric space: EVERY arm is scaled through the phin_gan preset's scalers
(`_proc()`) -- D_E is a distance in ONE fixed scaled space, and the arm
being evaluated only supplies the generated samples (`gen_proc`). Scaling
the GAN arm through its own no-physics proc instead would put its curve in
a different scaled space (min-max log10) than the phys-scaled G4-self
baseline it is drawn against -- visually incomparable, and once the
ablation's weights fit reasonably its curve would land LEFT of the null.

The ~50 min of sampling + EMD computation and the seconds of plotting are
split, `physics_losses.py`-style, so plot edits iterate fast: `--score`
computes all nine D_E arrays and saves them to CACHE_PATH (stamped with the
four checkpoint pins -- a re-pin invalidates the cache loudly at load rather
than silently mixing runs); `--plot` reads only the cache. No flags = score
then plot."""

import argparse
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from scipy.stats import ks_2samp
from joblib import Parallel, delayed

from common.paths import (BENCHMARKS_OUTPUTS, DATASETS_DIR, FIGURES_DIR, PATH_TO_GEANT4_DATA_SEED42, NOPHYS_CNT_CKPT, CNT_CKPT, SEC_CKPT, NOPHYS_SEC_CKPT)
from common.utils import earths_movers_distance
from benchmark.system_vs_g4.net_sampling import (
    phin_gan_ionisation, preset_ionisation)

_NET_PROC = None
_GAN_PROC = None


def _proc():
    """Build the phin_gan ionisation process once, lazily, so importing
    this module stays cheap."""
    global _NET_PROC
    if _NET_PROC is None:
        _NET_PROC = phin_gan_ionisation()
    return _NET_PROC


def _gan_proc():
    """Build the no-physics GAN ablation process once, lazily."""
    global _GAN_PROC
    if _GAN_PROC is None:
        _GAN_PROC = preset_ionisation("gan")
    return _GAN_PROC


def to_batches(tensor: torch.Tensor, subset_size: int):
    """Break tensor into equal-sized batches, discarding the remainder."""
    batched_tensor = tensor.split(subset_size)
    batched_tensor = np.array([x for x in batched_tensor if len(x) == subset_size])
    return batched_tensor


def _compute_emd_metrics(i, sample1_np, sample2_np, gen_np, indices):
    """Worker function for parallel EMD computation. Pure CPU math, no PyTorch."""
    wd1 = earths_movers_distance(sample1_np[:, indices], sample2_np[:, indices])
    wd2 = earths_movers_distance(gen_np[:, indices], sample2_np[:, indices])
    return i, wd1, wd2


def _load_and_split(x_filename: str, y_filename: str, subset_size: int = 10_000):
    """Load X/Y pickle files, ensure 2D, batch, and split into two groups."""
    with open(DATASETS_DIR / f'{PATH_TO_GEANT4_DATA_SEED42.stem}_{x_filename}', 'rb') as f:
        X = torch.from_numpy(pickle.load(f)).type(torch.float32)
    with open(DATASETS_DIR / f'{PATH_TO_GEANT4_DATA_SEED42.stem}_{y_filename}', 'rb') as f:
        Y = torch.from_numpy(pickle.load(f)).type(torch.float32)

    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    Xs = to_batches(X, subset_size)
    Ys = to_batches(Y, subset_size)

    n_batches = len(Xs) // 2 * 2  # ensure even for equal split
    Xs, Ys = Xs[-n_batches:], Ys[-n_batches:]
    data = np.concatenate([Xs, Ys], axis=-1)
    group1, group2 = np.split(data, 2)

    print(len(group1))
    return group1, group2, n_batches


def _run_emd_pipeline(group1, group2, n_batches, scale_fn, generate_fn, phase_space_cols, y_cols, indices, n_jobs=10):
    """
    Generate samples and compute EMD for one network.

    Args:
        scale_fn: callable(phase_space, samples) -> scaled numpy array
        generate_fn: callable(phase_space) -> generated tensor
        phase_space_cols: list of column indices for phase space extraction
        y_cols: slice or list for Y extraction from data
        indices: list of feature indices for EMD computation
    """
    generated_data = []
    for i in tqdm(range(n_batches // 2), desc="Generating"):
        sample1, sample2 = group1[i], group2[i]
        phase_space_1 = torch.from_numpy(sample1[:, phase_space_cols])
        phase_space_2 = torch.from_numpy(sample2[:, phase_space_cols])

        gen = generate_fn(copy(phase_space_1))
        scaled_gen = scale_fn(copy(phase_space_1), gen).cpu().numpy()

        scaled_sample1 = scale_fn(phase_space_1, torch.from_numpy(sample1[:, y_cols])).cpu().numpy()
        scaled_sample2 = scale_fn(phase_space_2, torch.from_numpy(sample2[:, y_cols])).cpu().numpy()

        generated_data.append((i, scaled_sample1, scaled_sample2, scaled_gen))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_emd_metrics)(d[0], d[1], d[2], d[3], indices)
        for d in tqdm(generated_data, desc="Calculating EMD")
    )
    results.sort(key=lambda x: x[0])

    wds1 = np.array([r[1] for r in results])
    wds2 = np.array([r[2] for r in results])
    return wds1, wds2


# ──────────────────────────── Step Length ──────────────────────────── #

def compute_step_length_emd(subset_size: int = 10_000, n_jobs: int = 10, gen_proc=None):
    """`gen_proc` supplies the GENERATED samples (default: phin_gan); the
    scaling is always the phin_gan preset's (`_proc()`) -- see module doc."""
    gen_proc = gen_proc if gen_proc is not None else _proc
    proc = _proc
    group1, group2, n_batches = _load_and_split(
        'step_length_X.pkl', 'step_length_Y.pkl', subset_size
    )

    def scale_fn(phase_space, samples):
        # scaled_ps = proc().length_generator.x_scaler.scale(copy(phase_space))
        # scaled_y = proc().length_y_scaler.scale(copy(samples)).unsqueeze(1)
        # return torch.concat([scaled_ps, scaled_y], dim=1)
        # use the scaler of the continuous module, which receives (E, L)
        e_l = torch.concat([copy(phase_space), copy(samples).unsqueeze(1)], dim=1)
        scaled_e_l = proc().continuous_generator.x_scaler.scale(e_l)
        return scaled_e_l

    def generate_fn(phase_space):
        return gen_proc().length_generator.predict(copy(phase_space))

    return _run_emd_pipeline(
        group1, group2, n_batches, scale_fn, generate_fn,
        phase_space_cols=[0], y_cols=1, indices=[0, 1], n_jobs=n_jobs,
    )


# ──────────────────────────── Continuous ──────────────────────────── #

def compute_continuous_emd(subset_size: int = 10_000, n_jobs: int = 10, gen_proc=None):
    """`gen_proc` supplies the GENERATED samples (default: phin_gan); the
    scaling is always the phin_gan preset's (`_proc()`) -- see module doc."""
    gen_proc = gen_proc if gen_proc is not None else _proc
    proc = _proc
    group1, group2, n_batches = _load_and_split(
        'continuous_X_filtered.pkl', 'continuous_Y_filtered.pkl', subset_size
    )

    def scale_fn(phase_space, samples):
        scaled_ps = proc().continuous_generator.x_scaler.scale(copy(phase_space))
        scaled_y = proc().continuous_generator.y_scaler.scale(
            copy(samples.unsqueeze(1)), phase_space=copy(phase_space)
        )
        return torch.concat([scaled_ps, scaled_y], dim=1)

    def generate_fn(phase_space):
        return gen_proc().continuous_generator.predict(copy(phase_space))

    return _run_emd_pipeline(
        group1, group2, n_batches, scale_fn, generate_fn,
        phase_space_cols=[0, 1], y_cols=2, indices=[0, 1, 2], n_jobs=n_jobs,
    )


# ──────────────────────────── Secondary ──────────────────────────── #

def compute_secondary_emd(subset_size: int = 10_000, n_jobs: int = 10, gen_proc=None):
    """`gen_proc` supplies the GENERATED samples (default: phin_gan); the
    scaling is always the phin_gan preset's (`_proc()`) -- see module doc."""
    gen_proc = gen_proc if gen_proc is not None else _proc
    proc = _proc
    group1, group2, n_batches = _load_and_split(
        'secondary_X_filtered.pkl', 'secondary_Y_filtered.pkl', subset_size
    )

    def scale_fn(phase_space, samples):
        scaled_ps = proc().secondary_generator.x_scaler.scale(copy(phase_space))
        scaled_y = proc().secondary_generator.y_scaler.scale(
            copy(samples), phase_space=copy(phase_space)
        )
        return torch.concat([scaled_ps, scaled_y], dim=1)

    def generate_fn(phase_space):
        return gen_proc().secondary_generator.predict(copy(phase_space))

    return _run_emd_pipeline(
        group1, group2, n_batches, scale_fn, generate_fn,
        phase_space_cols=[0], y_cols=slice(1, None), indices=[0, 1], n_jobs=n_jobs,
    )


# ──────────────────────────── Combined Plot ──────────────────────────── #

def plot_combined_energy_distance(
    step_length_wds: tuple[np.ndarray, ...],
    continuous_wds:  tuple[np.ndarray, ...],
    secondary_wds:   tuple[np.ndarray, ...],
    binss=None,
):
    """
    Plot a 3-row, 1-column figure of energy distance distributions.

    Each tuple is (wds_g4_vs_g4, wds_phys_vs_g4) or, with the GAN ablation
    arm included, (wds_g4_vs_g4, wds_phys_vs_g4, wds_gan_vs_g4). All three
    live in the same phys-scaled space (see the module docstring), so the
    single G4-self reference wds1 is the null for both comparisons.
    Returns the figure.
    """
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'figure.dpi': 300,
    })

    if binss is None:
        binss = [50]*3

    all_wds = [
        ("Step Length", step_length_wds),
        ("Continuous", continuous_wds),
        ("Secondaries", secondary_wds),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)

    for i, (ax, (description, wds_tuple), bins) in enumerate(zip(axes, all_wds, binss)):
        wds1, wds2 = wds_tuple[0], wds_tuple[1]
        mask = ~np.isnan(wds2)

        pval_phys = ks_2samp(wds1[mask], wds2[mask]).pvalue
        print(f"{description} — p-value (PHIN-GAN vs GEANT4): {pval_phys:.4g}")

        ax.hist(wds1[mask], bins=bins, histtype='stepfilled', color='#D55E00', label='GEANT4 vs GEANT4', alpha=0.4)
        ax.hist(wds2[mask], bins=bins, histtype='step', linewidth=1.5, color='#009E73', label='PHIN-GAN vs GEANT4', linestyle='-')

        if len(wds_tuple) > 2:
            wds3 = wds_tuple[2]
            mask3 = ~np.isnan(wds3)
            pval_gan = ks_2samp(wds1[mask3], wds3[mask3]).pvalue
            print(f"{description} — p-value (GAN vs GEANT4): {pval_gan:.4g}")
            ax.hist(wds3[mask3], bins=bins, histtype='step', linewidth=1.5, color='#0072B2', label='GAN vs GEANT4', linestyle='--')

        ax.set_yscale("log")
        # ax.set_title(description, fontweight='bold', fontsize=18)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        if i == 0:
            ax.legend(loc=(0.55, 0.45))#loc='upper right')

        ax.text(0.98, 0.97, description, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', horizontalalignment='right',
                fontname="sans-serif", fontweight='bold')

    fig.supxlabel("$D_E$", fontweight='bold', fontsize=20)
    fig.supylabel("Batches", fontweight='bold', fontsize=20)

    return fig


# ──────────────────────────── Cache & CLI ──────────────────────────── #

CACHE_PATH = FIGURES_DIR / "energy_distance_cache" / "energy_distance_wds.npz"

_ROWS = ("step_length", "continuous", "secondary")


def _current_pins() -> list[str]:
    """Identity of the four loaded checkpoints (run dir + epoch file), the
    same run-keyed convention physics_losses.py uses -- a cache scored under
    different pins must not be plotted as if it were current."""
    return [f"{p.parent.parent.name}/{p.name}" for p in
            (CNT_CKPT, SEC_CKPT,
             NOPHYS_CNT_CKPT, NOPHYS_SEC_CKPT)]


def score(cache_path=CACHE_PATH, seed=None) -> dict[str, np.ndarray]:
    """Run the full sampling + EMD computation (~50 min on 12 CPU cores) for
    all three rows x both generated arms, and persist the nine D_E arrays.

    `seed` (if given) seeds torch before any generation, making the cache a
    reproducible draw; it is recorded in the npz (-1 = unseeded). Seed it
    ONCE, in advance -- do not re-score over seeds and keep the best-looking
    p-value, which silently converts a uniform-under-H0 statistic into a
    selection maximum."""
    if seed is not None:
        torch.manual_seed(seed)
    compute_fns = (compute_step_length_emd, compute_continuous_emd,
                   compute_secondary_emd)

    arrays: dict[str, np.ndarray] = {}
    for row, fn in zip(_ROWS, compute_fns):
        arrays[f"{row}_null"], arrays[f"{row}_phys"] = fn()
    # No-physics GAN ablation arm -- same pipeline, same phys scaling, gan
    # process supplying the generated samples (paper method). Element [0]
    # of each is a second draw of the same phys-scaled G4-self reference;
    # the plot uses the phys arm's null as the one shared null.
    # The ablation was trained for aluminium only; other beams cache the
    # two-arm rows and the plot draws GEANT4-self against PHIN-GAN alone.
    from benchmark.track_simulator_config import has_gan_preset
    if has_gan_preset():
        for row, fn in zip(_ROWS, compute_fns):
            arrays[f"{row}_gan"] = fn(gen_proc=_gan_proc)[1]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, pins=np.array(_current_pins()),
             seed=np.array(seed if seed is not None else -1), **arrays)
    print(f"wrote {cache_path} (seed={seed})")
    return arrays


def load_cache(cache_path=CACHE_PATH) -> dict[str, np.ndarray]:
    """Read the scored D_E arrays; refuse a cache scored under different
    checkpoint pins rather than plotting those arms as current."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"no D_E cache at {cache_path} -- run with --score first")
    with np.load(cache_path, allow_pickle=False) as z:
        cached_pins = [str(p) for p in z["pins"]]
        if cached_pins != _current_pins():
            raise RuntimeError(
                "D_E cache is out of date: scored under checkpoint pins\n  "
                + "\n  ".join(cached_pins)
                + "\nbut the current pins are\n  "
                + "\n  ".join(_current_pins())
                + "\nre-run with --score.")
        seed = int(z["seed"]) if "seed" in z.files else -1
        print(f"D_E cache: seed={'unseeded' if seed == -1 else seed}")
        return {k: z[k] for k in z.files if k not in ("pins", "seed")}


def plot_from_cache(cache_path=CACHE_PATH,
                    out_path=BENCHMARKS_OUTPUTS / "energy_distance.pdf"):
    arrays = load_cache(cache_path)
    fig = plot_combined_energy_distance(
        *(tuple(arrays[f"{row}_{arm}"] for arm in ("null", "phys", "gan")
                if f"{row}_{arm}" in arrays)
          for row in _ROWS))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"wrote {out_path}")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true",
                        help="compute the D_E arrays and write the cache (~50 min)")
    parser.add_argument("--plot", action="store_true",
                        help="render the figure from the cache (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the generation for a reproducible cache "
                             "(pick it in advance; see score()'s docstring)")
    args = parser.parse_args(argv)
    if not args.score and not args.plot:  # historical no-flag behavior
        args.score = args.plot = True
    if args.score:
        score(seed=args.seed)
    if args.plot:
        plot_from_cache()


if __name__ == "__main__":
    main()
