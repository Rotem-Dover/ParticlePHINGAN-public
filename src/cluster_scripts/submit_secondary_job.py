"""Submit the secondary-emission WGAN-GP training job.

Runs `physics.g4h_ionisation.generators.train_networks.secondary_generator.
train` on a GPU node, logging to `RUNS_DIR/logger_sec`. Each run gets its
own run-id subdirectory under `RUNS_DIR`.

    python src/cluster_scripts/submit_secondary_job.py
    python src/cluster_scripts/submit_secondary_job.py --resume version_0
    python src/cluster_scripts/submit_secondary_job.py --dry-run

**The preflight asserts the target width matches the architecture.**
`N_SECONDARY_FEATURES` is checkpoint-contract state: `from_checkpoint()` builds
a bare `cls()`, so that constant IS the output width. The target `.npy` is
whatever the dataset-building step last wrote. If the two disagree the job
dies inside `_prepare_data_for_training`'s reshape with a shape error that
names neither cause. Reading the `.npy` header here costs nothing and says
which one to change.

The preflight asks the DataModule which target file it will open rather
than naming it directly, so the two cannot drift apart.
"""
import sys

import numpy as np

from _training_job import build_parser, preflight, submit

from common.paths import (RUNS_DIR, DATASETS_DIR, PATH_TO_SECONDARIES_CDFS)
from common.config import N_SECONDARY_FEATURES, USE_PHYSICS


def _check_target_width(y_path) -> None:
    """Compare the on-disk Y width against the architecture constant."""
    with open(y_path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        shape, _, _ = np.lib.format._read_array_header(fh, version)

    width = 1 if len(shape) == 1 else shape[1]
    if width != N_SECONDARY_FEATURES:
        sys.exit(
            f"!! target width mismatch: {y_path.name} is (N, {width}) but "
            f"N_SECONDARY_FEATURES is {N_SECONDARY_FEATURES}.\n"
            f"   The generator's output width comes from the constant, so the\n"
            f"   two must agree. Either rebuild the target at width "
            f"{N_SECONDARY_FEATURES},\n"
            f"   or change the constant -- doing so invalidates any existing\n"
            f"   secondary checkpoint trained at a different width.")
    print(f"    ok  {'target width':24} (N, {width}) matches N_SECONDARY_FEATURES")


def main() -> None:
    args = build_parser("secondary", default_mem="16gb").parse_args()

    # Ask the DataModule which target it will open rather than naming the
    # file here, so the preflight always checks the file the job actually
    # reads. Constructing the DataModule touches no disk.
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.datamodule import DataModule
    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        SecondaryXScaler, SecondaryYScaler)
    y_path = DataModule(x_scaler_type=SecondaryXScaler,
                        y_scaler_type=SecondaryYScaler).y_path
    required = {
        "X (kinetic energy)": DATASETS_DIR / "secondary_X.npy",
        "Y (secondary target)": y_path,
    }
    if USE_PHYSICS:
        required["physics CDFs"] = PATH_TO_SECONDARIES_CDFS

    resume = args.resume.strip()
    resume_ckpt = (RUNS_DIR / "logger_sec" / resume / "checkpoints" / "last.ckpt"
                   if resume else None)

    preflight("secondary", required, resume_ckpt)
    _check_target_width(y_path)
    if resume:
        print(f">> resuming from {resume_ckpt}")
    submit("secondary", "run_on_node_secondary.sh", args)


if __name__ == "__main__":
    sys.exit(main())
