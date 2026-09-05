"""Subsample a beam's training arrays down to a target row count, so that
different materials' training sets carry a comparable number of steps
rather than whatever count each material's raw truth happens to yield.

One uniform random mask (fixed seed, sorted so original row order survives)
is applied to ALL five arrays of the beam — secondary_X / secondary_Y /
secondary_Y_drop_theta / continuous_X / continuous_Y share one row space by
construction (build_training_datasets extracts them from the same filtered
steps), and a shared mask preserves the X<->Y row pairing in both stages.
The full arrays are NOT backed up: they are reproducible from the untouched
ROOT truth via build_training_datasets.py.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/subsample_datasets_to_reference.py \
        --material iron --target-rows 9797623 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np

ARRAYS = ("secondary_X.npy", "secondary_Y.npy", "secondary_Y_drop_theta.npy",
          "continuous_X.npy", "continuous_Y.npy")

STORAGE = Path("/Users/rotemdo/PycharmProjects/ParticlePHINGAN/storage/resources/proton")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", required=True, choices=("iron", "beryllium"))
    p.add_argument("--target-rows", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = STORAGE / args.material / "datasets"
    arrays = {name: np.load(d / name) for name in ARRAYS}
    n = {a.shape[0] for a in arrays.values()}
    assert len(n) == 1, f"row spaces differ: { {k: v.shape for k, v in arrays.items()} }"
    n = n.pop()
    assert args.target_rows < n, (args.target_rows, n)

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=args.target_rows, replace=False))

    for name, a in arrays.items():
        np.save(d / name, a[idx])
    print(f"{args.material}: {n:,} -> {args.target_rows:,} rows (seed {args.seed})")

    m_path = d / "manifest.json"
    m = json.loads(m_path.read_text())
    m["subsample"] = {
        "note": "subsampled to a target row count",
        "from_rows": int(n),
        "to_rows": int(args.target_rows),
        "seed": int(args.seed),
        "reason": ("match a target row count for comparable per-material "
                   "step counts; one shared sorted mask across all five "
                   "arrays"),
        "script": "scripts/subsample_datasets_to_reference.py",
    }
    for name in ARRAYS:
        if name in m.get("arrays", {}):
            m["arrays"][name][0] = int(args.target_rows)
    m_path.write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
