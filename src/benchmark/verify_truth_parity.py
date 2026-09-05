"""Gate T: the acceptance instrument, the simulator vs GEANT4 truth.

A pass states "the simulator differs from GEANT4 by no more than GEANT4
differs from itself at this sample size" -- the bar the trained networks
must clear.

`_simulate` builds the `phin_gan` preset and returns the step frame;
`_load_truth_sample` streams the GEANT4 reference via
`benchmark.gate_t_null.load_truth` (backed by `data_handling.truth_extract`,
never `data_handling.reader`, which caches a `.pkl` beside its source).
Column maps are asserted at import: the simulator's step frame and the truth
frame have DIFFERENT widths and NEITHER is the raw `StepFeaturesIdxes`
tensor, so indexing both sides by that enum would silently compare, say, a
step length in metres against an energy in eV.

**Run size.** 500 events x 12000 steps, chosen for its KS resolution
(~1e-3), which matters for detecting subtle degradation in a trained
network. The null below is calibrated at this same size -- a two-sample KS
statistic's scale is set by per-side n (critical value ~
1.36*sqrt((n1+n2)/(n1*n2))), so a null measured at the wrong sample size
sets a threshold a correctly-behaving gate can fail on pure sampling scatter
(see `benchmark.gate_t_null`'s docstring, which makes the same point for the
truth-vs-truth split itself).

**Termination convention.** The truth ROOT file's rows were built once with
GEANT4's own physics-region cutoff already applied: `truth_extract`'s
row-local filter drops every row with `KineticEnergy <= MIN_ENERGY_CUTOFF`.
`_simulate` therefore runs `TrackSimulator` with `E_kill_eV=MIN_ENERGY_CUTOFF`
(300 keV here) rather than the runtime's own 1 keV default, so the simulated
side stops producing rows at the same energy floor the truth side was
filtered to. Leaving the default in place would let the simulator emit
extra short, low-energy end-of-track steps the truth side never had a
chance to record, biasing L/THETA low -- matching the reference's own
termination rule, not introducing a new one.

**Vacuous-comparison refusal.** `run_truth_gate` raises `ValueError`
(message mentions "rows") if either side produces zero rows, so a caller
that accidentally asks for `n_events=0` gets a loud failure instead of a
`compare_distributions` call over two empty samples reading as a pass by
default.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.gate_t_null import _FEATURE_COL as _TRUTH_FEATURE_COL
from benchmark.gate_t_null import load_truth
from benchmark.gate_t_stats import compare_distributions
from common import paths
from common.config import MIN_ENERGY_CUTOFF
from common.enums import G4Columns

# The real GEANT4 truth: an ionisation-only file, protons in aluminium,
# produced by GEANT4 with multiple scattering disabled to match the
# package's physics scope.
DEFAULT_TRUTH_ROOT = paths.PATH_TO_GEANT4_DATA

# The gate's own run size.
DEFAULT_N_EVENTS = 500
DEFAULT_N_STEPS = 12000
DEFAULT_SEED = 20260807

_MIN_ROWS = 1000

# ---------------------------------------------------------------------------
# Column mapping -- resolved BY NAME, per frame. Indexing both sides by the
# `StepFeaturesIdxes` enum would silently compare, say, a step length in
# metres against an energy in eV: both frames here are different widths from
# the raw feature tensor and from each other.
#
# SIM side: `TrackSimulator.steps_features_df` -> EventNum,
#   StepNum, StepLength, angleDiscrete, ContinuousLoss, SecondaryLoss,
#   geom_step_m. 7 wide.
# TRUTH side: `benchmark.gate_t_null.load_truth` -> already a
#   {"L", "THETA", "E_CNT", "E_SEC"} dict, keyed by the SAME feature names
#   `compare_distributions` expects, via its own `_FEATURE_COL` map
#   (StepLength / angleDiscrete / ContinuousLoss / SecondaryLoss). No further
#   translation needed on that side; it is asserted here anyway so a rename
#   on either side fails at import, not at some downstream KS call.
# ---------------------------------------------------------------------------
SIM_COLUMNS = ("EventNum", "StepNum", "StepLength", "angleDiscrete",
               "ContinuousLoss", "SecondaryLoss", "geom_step_m")
_SIM_NAME = {"L": "StepLength", "THETA": "angleDiscrete",
             "E_CNT": "ContinuousLoss", "E_SEC": "SecondaryLoss"}
SIM_IDX = {f: SIM_COLUMNS.index(n) for f, n in _SIM_NAME.items()}

assert SIM_IDX == {"L": 2, "THETA": 3, "E_CNT": 4, "E_SEC": 5}, SIM_IDX

# The truth side's column names come from `gate_t_null._FEATURE_COL`, not
# from anything defined in this module; asserted here (not just trusted)
# so a rename on that side is caught at THIS module's import time too.
assert _TRUTH_FEATURE_COL == {
    "L": G4Columns.StepLength, "THETA": G4Columns.angleDiscrete,
    "E_CNT": G4Columns.ContinuousLoss, "E_SEC": G4Columns.SecondaryLoss,
}, _TRUTH_FEATURE_COL

_FEATURES = ("L", "THETA", "E_CNT", "E_SEC")
_ATOM_FEATURES = ("E_SEC", "THETA")

# ---------------------------------------------------------------------------
# Calibration: truth-vs-truth null, measured at the gate's own run size.
#
#   cd src && PYTHONPATH=. python -m benchmark.gate_t_null \
#     --root <DEFAULT_TRUTH_ROOT> \
#     --n-splits 6 --seed 0 --per-side-n 4901455
#
# `per_side_n` provenance: a real 500-events x 12000-steps run of `phin_gan`
# against the truth file (seed 20260807, E_kill_eV=MIN_ENERGY_CUTOFF)
# produced exactly 4,902,386 surviving rows on this machine. The truth file
# itself has only 9,802,911 rows
# total, so an independent two-way split can supply at most
# floor(9,802,911 / 2) = 4,901,455 rows per side -- 931 rows (0.019%) short
# of the measured 4,902,386. The null below is calibrated at that maximum
# feasible per-side_n (4,901,455), not the unreachable exact figure; the
# 0.019% gap is far below this statistic's own seed-to-seed scatter (see the
# `splits` list `gate_t_null --out` emits) and does not materially change
# the calibration.
#
#   {
#     "ks": {
#       "L": 0.0006842866046918861,
#       "THETA": 0.001067367175315359,
#       "E_CNT": 0.000938088792001579,
#       "E_SEC": 0.0010133696014270055
#     },
#     "atom_mass": {
#       "THETA": 0.00026094292409090114,
#       "E_SEC": 0.00026094292409090114
#     },
#     "energy_distance": 0.0008255683910146638,
#     "n_rows_total": 9802911,
#     "per_side_n": 4901455,
#     "n_splits": 6,
#     "seed": 0
#   }
#
# Thresholds below are 2x that null. A threshold at or below the
# null fails for RNG/sampling scatter, not physics -- pinned by
# `test_thresholds_exceed_the_measured_null`. Never widen a threshold to
# turn a red gate green: a red gate is the finding.
MEASURED_NULL = {
    "ks": {
        "L": 0.0006842866046918861,
        "THETA": 0.001067367175315359,
        "E_CNT": 0.000938088792001579,
        "E_SEC": 0.0010133696014270055,
    },
    "atom_mass": {
        "THETA": 0.00026094292409090114,
        "E_SEC": 0.00026094292409090114,
    },
    "energy_distance": 0.0008255683910146638,
    "n_rows_total": 9802911,
    "per_side_n": 4901455,
    "n_splits": 6,
    "seed": 0,
}

KS_THRESHOLDS = {f: 2 * v for f, v in MEASURED_NULL["ks"].items()}
ATOM_MASS_THRESHOLDS = {f: 2 * v for f, v in MEASURED_NULL["atom_mass"].items()}
ENERGY_DISTANCE_THRESHOLD = 2 * MEASURED_NULL["energy_distance"]


def _drop_nan_rows(a: np.ndarray) -> np.ndarray:
    """Dead particles' rows are NaN-padded; drop them whole."""
    return a[~np.isnan(a).any(axis=1)]


def _simulate(n_events: int, n_steps: int, seed: int, device: str = "cpu",
             _inject_e_cnt_scale: float = 1.0) -> dict:
    """Build the `phin_gan` preset and run it, returning
    `{"L": arr, "THETA": arr, "E_CNT": arr, "E_SEC": arr}`."""
    from benchmark.track_simulator_config import build_track_simulator

    torch.manual_seed(seed)
    ts = build_track_simulator("phin_gan", device=device)
    # E_kill_eV=MIN_ENERGY_CUTOFF matches the truth file's OWN row-local
    # filter (`KineticEnergy > MIN_ENERGY_CUTOFF`, `truth_extract`) so the
    # simulated side stops recording steps at the same energy floor the
    # truth side was filtered to. See module docstring.
    ts.run(E_0=1e8, n_events=n_events, n_steps=n_steps, save_steps_states=True,
           E_kill_eV=MIN_ENERGY_CUTOFF)
    cols = tuple(str(c) for c in ts.steps_features_df.columns)
    if cols != SIM_COLUMNS:
        raise RuntimeError(
            f"steps_features_df layout changed: {cols} != {list(SIM_COLUMNS)}; "
            "update SIM_COLUMNS/_SIM_NAME before trusting this gate")
    arr = ts.steps_features_df.to_numpy(dtype=np.float64)
    if _inject_e_cnt_scale != 1.0:
        arr[:, SIM_IDX["E_CNT"]] *= _inject_e_cnt_scale
    arr = _drop_nan_rows(arr)
    return {f: arr[:, idx] for f, idx in SIM_IDX.items()}


def _load_truth_sample(root_path, n_target: int, seed: int) -> dict:
    """`n_target` rows drawn without replacement from the full truth file.

    Unlike `gate_t_null._split` (which needs two mutually INDEPENDENT halves
    because both its arms are truth), only one arm here is truth -- the
    other is the simulator's own independently-drawn output -- so a single
    draw from the whole pool is sufficient; there is no second truth arm
    that could leak rows into this one.
    """
    truth = load_truth(root_path)
    n_total = len(truth["L"])
    if n_target > n_total:
        raise ValueError(
            f"gate T: asked for {n_target} truth rows but only {n_total} "
            f"survive the row-local filters in {root_path}")
    idx = np.random.default_rng(seed).choice(n_total, size=n_target, replace=False)
    return {f: arr[idx] for f, arr in truth.items()}


def run_truth_gate(n_events: int = DEFAULT_N_EVENTS,
                   n_steps: int = DEFAULT_N_STEPS,
                   seed: int = DEFAULT_SEED,
                   device: str = "cpu",
                   truth_root=None,
                   _inject_e_cnt_scale: float = 1.0) -> dict:
    """Simulate `phin_gan` and compare it against GEANT4 truth.

    Refuses a vacuous comparison: raises `ValueError` (message names "rows")
    if the simulator produces zero rows, or if the truth file cannot supply
    at least `_MIN_ROWS` rows to compare against.
    """
    if truth_root is None:
        truth_root = DEFAULT_TRUTH_ROOT

    sim = _simulate(n_events, n_steps, seed, device=device,
                    _inject_e_cnt_scale=_inject_e_cnt_scale)
    n_sim = len(sim["L"])
    if n_sim == 0:
        raise ValueError(
            f"gate T: simulator produced 0 rows at n_events={n_events}, "
            f"n_steps={n_steps} -- refusing a vacuous comparison")
    if n_sim < _MIN_ROWS:
        raise ValueError(
            f"gate T: simulator produced only {n_sim} rows (< {_MIN_ROWS}) "
            "-- refusing a comparison too small to be meaningful")

    ref = _load_truth_sample(truth_root, n_target=n_sim, seed=seed)
    n_ref = len(ref["L"])
    if n_ref < _MIN_ROWS:
        raise ValueError(
            f"gate T: truth sample has only {n_ref} rows (< {_MIN_ROWS}) "
            "-- refusing a vacuous comparison")

    out = compare_distributions(sim, ref)
    ks, atom, ed = out["ks"], out["atom_mass"], out["energy_distance"]
    passed = (all(ks[f] < KS_THRESHOLDS[f] for f in _FEATURES)
              and all(atom[f] < ATOM_MASS_THRESHOLDS[f] for f in _ATOM_FEATURES)
              and ed < ENERGY_DISTANCE_THRESHOLD)
    return {"ks": ks, "atom_mass": atom, "energy_distance": ed,
            "passed": passed, "n_rows": out["n_rows"],
            "run_size": {"n_events": n_events, "n_steps": n_steps},
            "device": device, "seed": seed, "truth_root": str(truth_root),
            "thresholds": {"ks": KS_THRESHOLDS,
                           "atom_mass": ATOM_MASS_THRESHOLDS,
                           "energy_distance": ENERGY_DISTANCE_THRESHOLD}}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gate T: the simulator vs GEANT4 truth.")
    p.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS,
                   help=f"default {DEFAULT_N_EVENTS}, gate T's own run size")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                   help=f"default {DEFAULT_N_STEPS}, gate T's own run size")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--truth-root", type=Path, default=None,
                   help=f"default {DEFAULT_TRUTH_ROOT}")
    return p


def main() -> None:
    a = _build_parser().parse_args()
    r = run_truth_gate(a.n_events, a.n_steps, a.seed, device=a.device,
                       truth_root=a.truth_root)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
