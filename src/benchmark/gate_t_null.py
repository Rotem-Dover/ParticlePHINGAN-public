"""Gate T's truth-vs-truth calibration null.

Gate T's thresholds are calibrated against a TRUTH-vs-TRUTH split:
partition the real GEANT4 sample into independent halves and measure
`gate_t_stats.compare_distributions` across them, so the threshold becomes a
multiple of GEANT4's own statistical scatter at this sample size. A pass
then means "the simulator differs from GEANT4 by no more than GEANT4
differs from itself".

Truth is loaded once via `data_handling.truth_extract.iter_truth_frames`
(streamed, writes nothing to disk -- see that module's docstring for why
`data_handling.reader` must never be used here) and materialised as plain
numpy arrays; the split itself is a random permutation per seed, never a
positional first-half/second-half split, so the two halves are genuinely
independent draws rather than an artifact of file order.

**Sample size matters.** A two-sample KS statistic's scale is set by sample
size (critical value ~ 1.36*sqrt((n1+n2)/(n1*n2))), so the null must be
measured at the SAME per-side row count gate T will actually compare at, not
at the size of the whole truth file. `_default_per_side_n` reuses
`verify_truth_parity.MEASURED_NULL["per_side_n"]` (4,901,455) -- the
MAX-FEASIBLE per-side row count, capped by the truth file's total row count
(9,802,911, so 9,802,911 // 2 = 4,901,455), not the exact row count a real
gate-T run (500 events x 12000 steps, `verify_truth_parity.DEFAULT_N_EVENTS`
/ `DEFAULT_N_STEPS`) produced on this machine -- that measured working point
is 4,902,386, so the cap sits 931 rows (0.019%) short of it. Each truth half
is then SUBSAMPLED down to that size (never rescaled analytically --
`atom_mass` is a Bernoulli-fraction difference and `energy_distance` has its
own scaling, so a single sqrt(n) correction factor would be wrong for them;
the only sound fix is to re-measure).

This module does NOT set any threshold -- it only measures and emits the
null as JSON. `verify_truth_parity.py` turns this into gate thresholds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.gate_t_stats import compare_distributions
from common.enums import G4Columns
from data_handling.truth_extract import iter_truth_frames

# Truth-file column each gate-T feature is read from.
_FEATURE_COL = {
    "L": G4Columns.StepLength,
    "THETA": G4Columns.angleDiscrete,
    "E_CNT": G4Columns.ContinuousLoss,
    "E_SEC": G4Columns.SecondaryLoss,
}

# `iter_truth_frames` needs the row-local filter columns in addition to
# whatever the caller wants back (see that module's docstring).
_COLUMNS = [
    G4Columns.StepNum, G4Columns.NumOfSecondaries, G4Columns.SecondaryLoss,
    G4Columns.KineticEnergy, G4Columns.ContinuousLoss, G4Columns.ProcessName,
    G4Columns.StepLength, G4Columns.angleDiscrete,
]


def load_truth(root_path, step_size: str = "500 MB") -> dict[str, np.ndarray]:
    """Stream `root_path` once and materialise the four gate-T features.

    Read-only: `iter_truth_frames` never opens the ROOT file for writing.
    """
    chunks: dict[str, list[np.ndarray]] = {f: [] for f in _FEATURE_COL}
    for chunk in iter_truth_frames(root_path, _COLUMNS, step_size=step_size):
        for feat, col in _FEATURE_COL.items():
            chunks[feat].append(chunk[col].to_numpy(dtype=np.float64))
    return {feat: np.concatenate(arrs) if arrs else np.empty((0,))
            for feat, arrs in chunks.items()}


def _default_per_side_n() -> int:
    """Per-side row count a real gate-T comparison draws on, taken from
    gate T's own measured run size.

    Gate T runs at 500 events x 12000 steps
    (`verify_truth_parity.DEFAULT_N_EVENTS` / `DEFAULT_N_STEPS`).
    `verify_truth_parity.MEASURED_NULL["per_side_n"]` (4,901,455) is NOT the
    exact row count a real run produced -- it is the MAX-FEASIBLE per-side
    split, capped by the truth file's total row count
    (9,802,911 // 2 = 4,901,455). The measured working point of a real run
    is 4,902,386 rows per side; the cap sits 931 rows short of it, a 0.019%
    deviation, because a real run's two sides are drawn independently rather
    than as an exact even split of the whole file. Reusing that value here,
    rather than a second hardcoded literal, keeps the two modules from being
    able to drift apart.

    Imported inside the function body rather than at module scope:
    `verify_truth_parity` imports `load_truth` and `_FEATURE_COL` FROM this
    module at ITS top level, so a top-level import here would be circular.
    """
    from benchmark.verify_truth_parity import MEASURED_NULL as _gate_t_measured_null
    return _gate_t_measured_null["per_side_n"]


def _split(truth: dict[str, np.ndarray], seed: int, per_side_n: int
          ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One independent split of `truth` into two `per_side_n`-row arms.

    Never positional: `truth` is first partitioned into two disjoint random
    halves (a permutation, not a first-half/second-half slice), and each
    half is then independently subsampled without replacement down to
    `per_side_n` rows with its OWN random generator. Two layers of
    independence -- disjoint pools, then independent draws within each pool
    -- so the two `per_side_n`-row arms share no row and no shared RNG
    stream. If any statistic on the result comes back exactly 0.0, that
    means this independence broke, not that GEANT4 matched itself exactly.
    """
    n = len(truth["L"])
    pool_perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    pool_a, pool_b = pool_perm[:half], pool_perm[half:]
    if per_side_n > min(len(pool_a), len(pool_b)):
        raise ValueError(
            f"per_side_n={per_side_n} exceeds the {min(len(pool_a), len(pool_b))}"
            " rows available per independent half of this truth file")
    idx_a = np.random.default_rng((seed, 1)).choice(
        pool_a, size=per_side_n, replace=False)
    idx_b = np.random.default_rng((seed, 2)).choice(
        pool_b, size=per_side_n, replace=False)
    a = {feat: arr[idx_a] for feat, arr in truth.items()}
    b = {feat: arr[idx_b] for feat, arr in truth.items()}
    return a, b


def truth_vs_truth_null(root_path, n_splits: int = 6, seed: int = 0,
                        per_side_n: int | None = None) -> dict:
    """The calibration: worst KS / atom-mass / energy-distance across
    `n_splits` independent draws of two `per_side_n`-row arms from the same
    GEANT4 truth file.

    `per_side_n` defaults to `_default_per_side_n()` -- the row count a real
    gate-T comparison actually uses per side -- NOT the size of the whole
    truth file. A null measured at the wrong sample size sets thresholds a
    correctly-behaving gate can fail on pure sampling scatter (see this
    module's docstring).

    Reports the worst of `n_splits` independent draws, mirroring the
    optimization gate's "worst of six pairs" convention, except each of the
    `n_splits` draws already IS an independent pair, rather than
    combinations over a handful of seeds.
    """
    if per_side_n is None:
        per_side_n = _default_per_side_n()
    truth = load_truth(root_path)
    n_rows_total = len(truth["L"])

    worst_ks: dict[str, float] = {}
    worst_atom: dict[str, float] = {}
    worst_ed = 0.0
    per_split = []

    for i in range(n_splits):
        a, b = _split(truth, seed=seed + i, per_side_n=per_side_n)
        out = compare_distributions(a, b)
        per_split.append({"ks": out["ks"], "atom_mass": out["atom_mass"],
                          "energy_distance": out["energy_distance"]})
        for feat, v in out["ks"].items():
            worst_ks[feat] = max(worst_ks.get(feat, 0.0), v)
        for feat, v in out["atom_mass"].items():
            worst_atom[feat] = max(worst_atom.get(feat, 0.0), v)
        if out["energy_distance"] is not None:
            worst_ed = max(worst_ed, out["energy_distance"])

    return {"ks": worst_ks, "atom_mass": worst_atom,
            "energy_distance": worst_ed, "n_rows_total": n_rows_total,
            "per_side_n": per_side_n, "n_splits": n_splits, "seed": seed,
            "splits": per_split}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gate T's truth-vs-truth calibration null. Read-only.")
    p.add_argument("--root", required=True, type=Path,
                   help="path to a GEANT4 truth ROOT file")
    p.add_argument("--n-splits", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-side-n", type=int, default=None,
                   help="rows per arm; defaults to "
                        "verify_truth_parity.MEASURED_NULL['per_side_n'], "
                        "the row count a real gate-T run (500x12000) "
                        "actually compares at")
    p.add_argument("--out", type=Path, default=None,
                   help="write JSON here in addition to stdout")
    return p


def main() -> None:
    a = _build_parser().parse_args()
    result = truth_vs_truth_null(a.root, n_splits=a.n_splits, seed=a.seed,
                                 per_side_n=a.per_side_n)
    text = json.dumps(result, indent=2)
    print(text)
    if a.out is not None:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
