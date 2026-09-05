# src/benchmark/runtime_profiling/verify_physics_gate.py
"""Optimization gate B: the OPTIMIZED build vs the EAGER build.

Gate B is eager-anchored -- only the optimized-minus-eager delta gates, the
absolutes are reported. It answers "did the Triton fused MLP or the compiled
phases shift the physics".

Both arms are built by `benchmark.runtime_profiling.simulator_builders` from
the sole `phin_gan` preset -- `build_optimized_simulator` (fused-MLP latch on,
phases compiled on CUDA) and `build_eager_simulator` (latch off, compile flags
pre-latched before the device move). On CPU both branches are dead, so the two
builders produce the same program and every gate-B delta is exactly 0.0; a
nonzero CPU delta means the builders diverged for a reason that is not
optimization.

On CUDA that same zero would be a lie: `try_enable_fused_forward` only
`log.warning`s when it refuses (no triton, params not on CUDA, unfusable
class, `build_spec` raising), so a refused fusion silently makes the optimized
arm identical to the eager one and every statistic reads as perfect agreement.
The gate therefore records both arms' ACTUAL per-stage kernel modes and
REFUSES (`passed=False`, nonzero exit, `vacuous` reason string) on CUDA when
the optimized arm fused nothing or the two arms' modes are identical. A
warning next to `"passed": true` would be read as a pass.

Statistics, both computed on `steps_features_df` columns L / THETA / E_CNT /
E_SEC:

  e2e   two-sample KS per feature, worst reported as `max_ks`. E_SEC and
        THETA carry a large exact-zero atom, and KS on a distribution with a
        Dirac atom is all-or-nothing, so those two are split: KS on the
        NONZERO continuum, plus the zero-atom mass difference, whose worst
        is `max_atom_mass_delta` and is GATED. The atom mass IS the
        secondary-generation probability -- the observable a fused kernel's
        floating-point drift would most plausibly move -- so reporting it
        without gating it left the one thing most at risk ungated.
  pull  per-feature mean shift in units of its own standard error
        (|<opt> - <eager>| / sqrt(var/n + var/n)), worst reported as
        `max_abs_mean_shift`. Normalising by the standard error is what lets
        one threshold cover four features whose units are metres, radians and
        eV.

The gate is distributional, not per-lane, and deliberately so. A particle is
alive while E > E_kill_eV, so a ~1e-6 arithmetic shift eventually flips an
alive-mask bit on some lane; from that step the lane's trajectory, step count
and downstream RNG consumption all diverge. Per-lane comparison is meaningful
for one generator call and meaningless after a few thousand steps of a track.
Do not add a seeded-exactness assertion.

    cd src && PYTHONPATH=. python -m benchmark.runtime_profiling.verify_physics_gate \
        --gate b --device cuda
    cd src && PYTHONPATH=. python -m benchmark.runtime_profiling.verify_physics_gate \
        --gate b --device cuda --calibrate
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from common.enums import G4Columns

# Physical feature -> the `steps_features_df` column that carries it,
# selected BY NAME per run (see `_run_arm`), so a frame-layout change
# surfaces as a KeyError rather than as a plausible-looking wrong number.
GATE_B_FEATURES = ("L", "THETA", "E_CNT", "E_SEC")
_SIM_NAME = {"L": G4Columns.StepLength, "THETA": G4Columns.angleDiscrete,
             "E_CNT": G4Columns.ContinuousLoss, "E_SEC": G4Columns.SecondaryLoss}

# The exact-zero-atom carriers: most steps produce no secondary, so both E_SEC
# and THETA are a Dirac atom at 0 plus a continuum. KS reads the atom's mass,
# not the continuum's shape, so these are split (continuum KS + atom mass).
_ATOM_FEATURES = ("E_SEC", "THETA")

# Minimum nonzero-continuum size for an atom feature before its KS means
# anything; an empty or near-empty continuum raises rather than returning
# 0.0 (a silent "perfect agreement").
_MIN_CONTINUUM = 100

# The working point the thresholds below were calibrated at, and the device
# the null was measured on. This is part of the calibration, not a default:
# the pull statistic is n-dependent -- a systematic floating-point bias grows
# ~sqrt(n) in pull units while the eager-vs-eager null does not -- so a null
# measured at one run size is MEANINGLESS at another. `_assert_calibration_run_size`
# refuses a gating run at any other size on the calibration device.
GATE_B_CALIBRATION_DEVICE = "cuda"
GATE_B_CALIBRATION_RUN_SIZE = {"n_events": 200, "n_steps": 12_000}

# Every statistic gate B gates. A thresholds dict must carry EXACTLY these
# keys; `None` against one of them means "report, do not gate", which is a
# decision the caller states rather than one they fall into by omission.
GATE_B_STATISTICS = ("e2e_max_ks", "pull_max_abs_mean_shift", "atom_mass_delta")

# Calibration: eager-vs-eager seed-to-seed null via gate_b_null(), seeds
# 1-4, worst of six pairs, at GATE_B_CALIBRATION_RUN_SIZE. Thresholds below
# are 2x the null. Replace the pasted measurement whenever it is
# recalibrated, so the calibration stays auditable. Never widen a threshold
# to turn a red gate green: a red gate is the finding.
#
#   "null": {
#     "e2e_max_ks": 0.0018312158233761977,
#     "pull_max_abs_mean_shift": 2.8352793270005705,
#     "atom_mass_delta": 0.0007535443505701789
#   },
#   "per_feature_null": {
#     "ks":   {"L": 0.0012200402101603114, "THETA": 0.0014492886417437623,
#              "E_CNT": 0.0013392735530280975, "E_SEC": 0.0018312158233761977},
#     "pull": {"L": 1.3446605482137564, "THETA": 1.8735518556300674,
#              "E_CNT": 2.0265125472924472, "E_SEC": 2.8352793270005705},
#     "atom_mass_delta": {"E_SEC": 0.0007535443505701789,
#                         "THETA": 0.0007530351735551832}
#   },
#   "per_feature_spread": {"ks": 1.5009471066003461, "pull": 2.108546525565094}
#
# WHY ONE SCALAR PER STATISTIC AND NOT ONE PER FEATURE: measured, not
# assumed. The KS spread (1.50x) sits inside the 2x safety factor. The pull
# spread (2.11x, E_SEC/L) sits marginally above that guideline; folding to
# one scalar is a judgment call made explicit here rather than silently --
# `run_gate_b`'s gating logic, `_validate_thresholds` and
# `GATE_B_STATISTICS` would all need restructuring to support per-feature
# keys, which is a real architecture change, not a pasted-number update.
# Gate B's own job (detecting a common-mode arithmetic shift from a
# kernel/launch-config change, not a feature-local physics regression) is
# structurally tolerant of folding to the worst feature: the cost is reduced
# sensitivity to a feature-local drift (E_SEC's own effective margin is 2x,
# L's is 2*2.835/1.345 = 4.2x), never an increased false-positive rate. A
# pull spread that grows past 2x on a future recalibration is worth
# revisiting the per-feature split, with its own dedicated review.
GATE_B_THRESHOLDS = {"e2e_max_ks": 0.0036624316467523954,
                     "pull_max_abs_mean_shift": 5.670558654001141,
                     "atom_mass_delta": 0.0015070887011403578}


def _validate_thresholds(thr: dict) -> dict:
    """Refuse a thresholds dict that does not carry exactly GATE_B_STATISTICS.

    A MISSING key is the dangerous one: `atom_mass_delta` was added in Task 15
    and every pre-Task-15 caller and doc example passes a two-key dict. Read
    with `.get(...)` that would silently ungate the atom mass -- i.e. drop the
    gate on the secondary-generation probability while still printing
    "passed": true. A silently dropped gate is exactly the failure this
    instrument exists to prevent, so a malformed dict is a loud error and
    never a quiet degradation. An UNKNOWN key is refused too: it is almost
    always a typo for a real one, which would ungate the real one.
    """
    missing = [k for k in GATE_B_STATISTICS if k not in thr]
    unknown = [k for k in thr if k not in GATE_B_STATISTICS]
    if missing or unknown:
        raise ValueError(
            f"gate B thresholds must carry exactly {list(GATE_B_STATISTICS)}; "
            f"missing={missing} unknown={unknown}. Pass an explicit None to "
            "report a statistic without gating it -- omitting the key would "
            "silently drop that gate.")
    return thr


def _assert_calibration_run_size(n_events: int, n_steps: int, what: str) -> None:
    """Refuse to use the calibrated thresholds at a run size they were not
    measured at (see GATE_B_CALIBRATION_RUN_SIZE).

    Only binds on the calibration device: on cpu both optimization branches
    are dead, every delta is exactly 0.0 by construction, and the working
    point is irrelevant to a comparison of a program against itself.
    """
    got = {"n_events": int(n_events), "n_steps": int(n_steps)}
    if got != GATE_B_CALIBRATION_RUN_SIZE:
        raise ValueError(
            f"{what}: run_size {got} != the calibration working point "
            f"{GATE_B_CALIBRATION_RUN_SIZE}. The pull statistic is "
            "n-dependent, so the calibrated thresholds do not transfer. "
            "Recalibrate with --calibrate at the new size and update BOTH "
            "GATE_B_THRESHOLDS and GATE_B_CALIBRATION_RUN_SIZE, or pass "
            "explicit thresholds.")


# ----- shared figure style ---------------------------------------------------
_C_OPT, _C_EAGER, _C_MARK = "#2a78d6", "#1baf7a", "#e34948"


def _style_axis(ax) -> None:
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _gate_style(fn):
    """Pin the rcParams every gate figure is designed against.
    energy_deposition_pull_analysis mutates GLOBAL rcParams at import
    (font.size 28 for its standalone paper figures), so any gate figure drawn
    after that module was imported would otherwise inherit poster-sized
    fonts."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # Trigger the known rcParams-mutating import BEFORE the snapshot, so
        # the pinned values below win regardless of import order.
        try:
            import benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis  # noqa: F401
        except Exception:
            pass
        with matplotlib.rc_context({
                "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                "legend.fontsize": 9, "legend.frameon": True,
                "xtick.labelsize": 9, "ytick.labelsize": 9,
                "xtick.direction": "out", "ytick.direction": "out",
                "xtick.top": False, "ytick.right": False,
                "axes.linewidth": 0.8, "figure.dpi": 100,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial"]}):
            return fn(*args, **kwargs)
    return wrapped


_FEATURE_UNITS = {"L": "m", "THETA": "rad", "E_CNT": "eV", "E_SEC": "eV"}


@_gate_style
def _plot_overlay(out_dir: Path, name: str, a: np.ndarray, b: np.ndarray,
                  ks: float | None = None) -> None:
    """Optimized-vs-eager 1-D overlay: log-spaced bins and a log x-axis for
    positive quantities spanning >3 decades; the zero/negative population
    (e.g. the E_SEC = 0 atom) is reported in the legend instead of being
    crushed into the first linear bin."""
    import matplotlib.pyplot as plt
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    unit = _FEATURE_UNITS.get(name, "")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    allv = np.concatenate([a, b])
    pos = allv[allv > 0]
    log_x = pos.size > 0 and pos.max() / pos.min() > 1e3
    if log_x:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 161)
        series = []
        for arr, base in ((a, "optimized"), (b, "eager")):
            nonpos = float((arr <= 0).mean()) if arr.size else 0.0
            label = base + (f"  (<=0: {nonpos:.1%})" if nonpos > 0 else "")
            series.append((arr[arr > 0], label))
        ax.set_xscale("log")
    else:
        lo, hi = (allv.min(), allv.max()) if allv.size else (0.0, 1.0)
        bins = np.linspace(lo, hi, 161) if hi > lo else 50
        series = [(a, "optimized"), (b, "eager")]
    ax.hist(series[0][0], bins=bins, histtype="step", density=True, lw=1.6,
            color=_C_OPT, label=series[0][1])
    ax.hist(series[1][0], bins=bins, histtype="step", density=True, lw=1.6,
            color=_C_EAGER, ls="--", label=series[1][1])
    ax.set_yscale("log")
    ax.set_xlabel(name + (f" [{unit}]" if unit else ""))
    ax.set_ylabel("density")
    title = f"gate B e2e: {name}"
    if ks is not None:
        title += f"   |   KS D = {ks:.6f}"
    ax.set_title(title, fontsize=11)
    _style_axis(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"gate_b_e2e_{name}.png", dpi=130)
    plt.close(fig)


# ----- statistics ------------------------------------------------------------
def _two_sample_ks(a: np.ndarray, b: np.ndarray) -> float:
    """D = sup |F_a - F_b|, on the two arrays as given."""
    from benchmark.runtime_profiling.gate_stats import two_sample_ks
    if a.size == 0 or b.size == 0:
        return 0.0
    return two_sample_ks(torch.from_numpy(np.ascontiguousarray(a)),
                         torch.from_numpy(np.ascontiguousarray(b)))


def _e2e_statistic(got: np.ndarray, ref: np.ndarray) -> dict:
    """Per-feature two-sample KS between the two arms, worst as `max_ks`.

    `got`/`ref` are `_run_arm` outputs: NaN-free rows, columns in
    GATE_B_FEATURES order. E_SEC/THETA are compared on their nonzero
    continuum, with the zero-atom mass difference gated SEPARATELY as
    `max_atom_mass_delta` (never folded into max_ks -- mixing an atom mass
    into a KS would hide both).
    """
    ks, atoms, n_continuum = {}, {}, {}
    for i, name in enumerate(GATE_B_FEATURES):
        a, b = got[:, i], ref[:, i]
        if name in _ATOM_FEATURES:
            atoms[name] = float(abs(_zero_fraction(a) - _zero_fraction(b)))
            a, b = a[a > 0], b[b > 0]
            n_continuum[name] = {"opt": int(a.size), "eager": int(b.size)}
            # An empty or near-empty continuum would otherwise score 0.0,
            # i.e. PERFECT agreement, which is the most dangerous possible
            # way to be wrong in a gate. Loud failure instead.
            if a.size < _MIN_CONTINUUM or b.size < _MIN_CONTINUUM:
                raise RuntimeError(
                    f"{name}: fewer than {_MIN_CONTINUUM} nonzero entries "
                    f"(opt={a.size}, eager={b.size}) -- statistic is "
                    "meaningless; raise n_events")
        ks[name] = _two_sample_ks(a, b)
    return {"max_ks": max(ks.values()), "ks": ks, "atom_mass_delta": atoms,
            "max_atom_mass_delta": max(atoms.values()) if atoms else 0.0,
            "n_continuum": n_continuum,
            "n_rows": {"opt": int(len(got)), "eager": int(len(ref))}}


def _zero_fraction(x: np.ndarray) -> float:
    return float((x == 0).mean()) if x.size else 0.0


def _pull_statistic(got: np.ndarray, ref: np.ndarray) -> dict:
    """Per-feature mean shift in units of its own standard error.

    pull = |<opt> - <eager>| / sqrt(var_opt/n_opt + var_eager/n_eager). The
    normalisation is what makes one threshold meaningful across features
    measured in metres, radians and eV; the raw means are reported too.
    Identical arms give exactly 0.0 (numerator zero short-circuits, so a
    zero-variance feature does not produce a 0/0 NaN).
    """
    pulls, means = {}, {}
    for i, name in enumerate(GATE_B_FEATURES):
        a, b = got[:, i], ref[:, i]
        means[name] = {"opt": float(a.mean()) if a.size else float("nan"),
                       "eager": float(b.mean()) if b.size else float("nan")}
        if a.size < 2 or b.size < 2:
            pulls[name] = float("nan")
            continue
        num = abs(a.mean() - b.mean())
        if num == 0.0:
            pulls[name] = 0.0
            continue
        den = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
        pulls[name] = float("inf") if den == 0.0 else float(num / den)
    # NaN propagates on purpose: an under-sampled feature must not report as
    # "small shift" -- `nan <= threshold` is False, so it fails the gate.
    worst = float(np.max(list(pulls.values())))
    return {"max_abs_mean_shift": worst, "pull": pulls, "means": means}


# ----- arms ------------------------------------------------------------------
def _run_arm(build, preset, device, n_events, n_steps, seed
             ) -> tuple[np.ndarray, dict]:
    """One simulator build + run, reduced to the four gate-B feature columns,
    plus the arm's per-stage kernel-mode report.

    Columns are selected BY NAME and dead lanes' NaN-padded rows dropped
    whole: a reduction over a NaN-padded frame silently returns NaN.

    The kernel-mode report comes from `profile_runtime._generator_report`,
    which reads the ACTUAL mode (`_fused_mlp` / `_fused_mlp_tf32`) off the
    unwrapped generator rather than the env request -- so a fusion that
    `try_enable_fused_forward` refused (no triton, params not on CUDA,
    unfusable class, `build_spec` raising) is reported as refused. Without it
    a refused fusion would make the optimized arm identical to the eager one
    and the gate would pass vacuously.
    """
    from benchmark.runtime_profiling.profile_runtime import _generator_report

    torch.manual_seed(seed)
    # `startswith`, not `== "cuda"`: torch accepts indexed device strings
    # ("cuda:0") everywhere, and an exact-string test would silently skip the
    # CUDA seeding -- leaving the two arms on different RNG streams.
    # Build-time draws inside `_probe_squeeze_out` (the fused-spec build's
    # structural probe forward) are wrapped in `torch.random.fork_rng`, so
    # they do not advance this seeded stream -- both arms draw from the same
    # stream even when the optimized arm builds a fused spec and the eager
    # arm does not.
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    simulator, sm = build(preset, device)
    simulator.run(E_0=1e8, n_events=n_events, n_steps=n_steps,
                  save_steps_states=True, E_kill_eV=0.0)
    df = simulator.steps_features_df
    arr = df[[_SIM_NAME[f] for f in GATE_B_FEATURES]].to_numpy(dtype=np.float64)
    return arr[~np.isnan(arr).any(axis=1)], _generator_report(sm)


def _vacuity_reason(device: str, opt_modes: dict, eager_modes: dict) -> str | None:
    """Why this run proves nothing, or None if it is a real comparison.

    CUDA-only. On CPU the fused and compiled branches are both dead by
    construction, so the two builders legitimately ARE the same program and a
    zero delta is the expected, meaningful result.

    Matched with `startswith`, never `== "cuda"`: an indexed device string
    ("cuda:0") is accepted by torch and by `.to()`, so an exact-string test
    would hand such a caller a vacuous `passed: True` -- the exact failure the
    check exists to refuse.
    """
    if not device.startswith("cuda"):
        return None
    if not any(info.get("fused") for info in opt_modes.values()):
        return ("vacuous: no stage of the OPTIMIZED arm reports fused=True on "
                "cuda -- try_enable_fused_forward refused (it only log.warns), "
                "so this run compared eager against eager")
    if opt_modes == eager_modes:
        return ("vacuous: the optimized and eager arms report identical kernel "
                "modes on cuda, so no optimization was exercised")
    return None


def run_gate_b(device: str, n_events: int, n_steps: int, seed: int,
               thresholds: dict | None = None, preset: str = "phin_gan",
               out_dir: Path | None = None) -> dict:
    """Optimization gate B: the OPTIMIZED build vs the EAGER build.

    This gate is eager-anchored: only the optimized-minus-eager delta gates,
    absolutes are reported. It answers "did the Triton kernel or the compiled
    phases shift the physics".

    It is distributional, not per-lane, and deliberately so. A particle is
    alive while E > E_kill_eV, so a ~1e-6 arithmetic shift eventually flips an
    alive-mask bit on some lane; from that step the lane's trajectory, step
    count and downstream RNG consumption all diverge. Per-lane comparison is
    meaningful for one generator call and meaningless after a few thousand
    steps of a track.

    A `None` threshold reports instead of gating (see GATE_B_THRESHOLDS).
    `out_dir`, when given, also writes the per-feature overlay figures.
    """
    from benchmark.runtime_profiling.simulator_builders import (
        build_eager_simulator, build_optimized_simulator)

    thr = _validate_thresholds(
        thresholds if thresholds is not None else GATE_B_THRESHOLDS)
    # The calibrated thresholds are only meaningful at the working point they
    # were measured at; see _assert_calibration_run_size.
    if (thresholds is None and device.startswith(GATE_B_CALIBRATION_DEVICE)
            and any(v is not None for v in thr.values())):
        _assert_calibration_run_size(n_events, n_steps, "run_gate_b")
    opt, opt_modes = _run_arm(build_optimized_simulator, preset, device,
                              n_events, n_steps, seed)
    eag, eag_modes = _run_arm(build_eager_simulator, preset, device,
                              n_events, n_steps, seed)

    e2e = _e2e_statistic(opt, eag)
    pull = _pull_statistic(opt, eag)
    # Refuse, never merely warn: a warning printed next to "passed": true is
    # read as a pass, and a vacuous green gate is worse than no gate.
    vacuous = _vacuity_reason(device, opt_modes, eag_modes)
    passed = vacuous is None
    if thr["e2e_max_ks"] is not None:
        passed &= e2e["max_ks"] <= thr["e2e_max_ks"]
    if thr["pull_max_abs_mean_shift"] is not None:
        passed &= pull["max_abs_mean_shift"] <= thr["pull_max_abs_mean_shift"]
    # `thr[...]`, never `thr.get(...)`: _validate_thresholds has already
    # guaranteed the key exists, and a `.get` here would resurrect the silent
    # ungating it was added to prevent.
    if thr["atom_mass_delta"] is not None:
        passed &= e2e["max_atom_mass_delta"] <= thr["atom_mass_delta"]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(GATE_B_FEATURES):
            _plot_overlay(out_dir, name, opt[:, i], eag[:, i],
                          ks=e2e["ks"][name])
    return {"e2e": e2e, "pull": pull, "passed": bool(passed),
            "device": device, "preset": preset, "thresholds": thr,
            "kernel_mode": {"optimized": opt_modes, "eager": eag_modes},
            "vacuous": vacuous,
            "run_size": {"n_events": n_events, "n_steps": n_steps, "seed": seed}}


def gate_b_null(device: str, n_events: int, n_steps: int,
                seeds=(1, 2, 3, 4), preset: str = "phin_gan") -> dict:
    """Eager-vs-eager across seeds: the scatter gate B's thresholds must sit
    strictly above. Worst statistic over all seed pairs.

    Reports, besides the three gated scalars, the PER-FEATURE worst KS and
    pull. Those are not gated -- gate B folds four features into one scalar
    -- but they are what makes the fold auditable: the fold is only
    defensible while the per-feature nulls sit
    within a factor comparable to the 2x safety factor, so that the tightest
    feature is not handed a threshold many times its own scatter. Measure the
    spread, do not assume it.

    `run_size` is emitted so the caller can assert it against
    GATE_B_CALIBRATION_RUN_SIZE -- the pull statistic is n-dependent and a
    null quoted at the wrong size is worse than no null.
    """
    from itertools import combinations

    from benchmark.runtime_profiling.simulator_builders import build_eager_simulator

    runs = {s: _run_arm(build_eager_simulator, preset, device, n_events, n_steps, s)[0]
            for s in seeds}
    worst = {"e2e_max_ks": 0.0, "pull_max_abs_mean_shift": 0.0,
             "atom_mass_delta": 0.0}
    per_ks = {f: 0.0 for f in GATE_B_FEATURES}
    per_pull = {f: 0.0 for f in GATE_B_FEATURES}
    per_atom = {f: 0.0 for f in _ATOM_FEATURES}
    for a, b in combinations(seeds, 2):
        e2e = _e2e_statistic(runs[a], runs[b])
        pull = _pull_statistic(runs[a], runs[b])
        worst["e2e_max_ks"] = max(worst["e2e_max_ks"], e2e["max_ks"])
        worst["atom_mass_delta"] = max(worst["atom_mass_delta"],
                                       e2e["max_atom_mass_delta"])
        worst["pull_max_abs_mean_shift"] = max(
            worst["pull_max_abs_mean_shift"], pull["max_abs_mean_shift"])
        for f in GATE_B_FEATURES:
            per_ks[f] = max(per_ks[f], e2e["ks"][f])
            per_pull[f] = max(per_pull[f], pull["pull"][f])
        for f in _ATOM_FEATURES:
            per_atom[f] = max(per_atom[f], e2e["atom_mass_delta"][f])
    spread = {"ks": max(per_ks.values()) / min(per_ks.values()) if min(per_ks.values()) else float("inf"),
              "pull": max(per_pull.values()) / min(per_pull.values()) if min(per_pull.values()) else float("inf")}
    return {"device": device, "seeds": list(seeds), "preset": preset,
            "run_size": {"n_events": n_events, "n_steps": n_steps},
            "calibration_run_size": dict(GATE_B_CALIBRATION_RUN_SIZE),
            "run_size_matches_calibration":
                {"n_events": int(n_events), "n_steps": int(n_steps)}
                == GATE_B_CALIBRATION_RUN_SIZE,
            "null": worst,
            "per_feature_null": {"ks": per_ks, "pull": per_pull,
                                 "atom_mass_delta": per_atom},
            "per_feature_spread": spread}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", default="b", choices=["b"],
                        help="the optimized-vs-eager gate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="on cpu both optimization branches are dead, so "
                             "gate B is a plumbing check whose deltas are "
                             "exactly zero")
    parser.add_argument("--preset", default="phin_gan")
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--calibrate", action="store_true",
                        help="print the eager-vs-eager seed-to-seed null "
                             "instead of gating")
    parser.add_argument("--out", default=None,
                        help="directory for the overlay figures and report.json")
    args = parser.parse_args()

    if args.calibrate:
        print(json.dumps(
            gate_b_null(args.device, args.n_events, args.n_steps,
                        preset=args.preset), indent=2))
        return

    out_dir = Path(args.out) if args.out else None
    r = run_gate_b(args.device, args.n_events, args.n_steps, args.seed,
                   preset=args.preset, out_dir=out_dir)
    print(json.dumps(r, indent=2, default=str))
    if out_dir is not None:
        (out_dir / "report.json").write_text(json.dumps(r, indent=2, default=str))
        print(f"report: {out_dir / 'report.json'}")
    if r["vacuous"] is not None:
        sys.exit(f"GATE B REFUSED -- {r['vacuous']}. Fix the optimized build "
                 "(triton present? params on cuda? fusable class?) and re-run.")
    if not r["passed"]:
        sys.exit("GATE B FAILED -- the optimized flow diverges from eager; "
                 "see the report before trusting optimized physics output.")
    uncalibrated = [k for k, v in r["thresholds"].items() if v is None]
    if uncalibrated:
        print(f"gate B REPORTED ONLY (uncalibrated thresholds: {uncalibrated}); "
              "measure the null with --calibrate and set them to 2x it.")
    else:
        print("gate B passed.")


if __name__ == "__main__":
    main()
