"""Submit the continuous energy-loss WGAN-GP training job.

Runs `physics.g4h_ionisation.generators.train_networks.continuous_generator.
train` on a GPU node, logging to `RUNS_DIR/logger_cnt`. Each run gets its
own run-id subdirectory under `RUNS_DIR`, so it never overwrites an existing
checkpoint directory.

    python src/cluster_scripts/submit_continuous_job.py
    python src/cluster_scripts/submit_continuous_job.py --resume version_0
    python src/cluster_scripts/submit_continuous_job.py --dry-run

**Memory default is 32gb.** The physics penalty term loads
`continuous_cdfs.pkl`, which unpickles into a dict of tensor-backed
interpolators resident alongside the training arrays -- large enough that
the secondary stage, whose CDF file is far smaller, defaults to a lower
memory allocation.
"""
import sys

from _training_job import build_parser, preflight, submit

from common.paths import (RUNS_DIR, DATASETS_DIR, PATH_INIT_DEDX, PATH_INIT_LAMBDA_ION, PATH_INIT_RANGE, PATH_INIT_SCALED_KIN, PATH_TO_CONTINUOUS_CDFS, SCALERS_DIR)
from common.config import USE_PHYSICS


def main() -> None:
    args = build_parser("continuous", default_mem="32gb").parse_args()

    required = {
        "X (E, L)": DATASETS_DIR / "continuous_X.npy",
        "Y (E_CNT)": DATASETS_DIR / "continuous_Y.npy",
        # ContinuousYScaler.load() raises FileNotFoundError without this --
        # it is the analytic mean/std lookup in portable state form, NOT
        # built at load time.
        "Y-scaler state": SCALERS_DIR / "ContinuousYScaler.state.pt",
        # This stage's `_filter_bethe_bloch_steps` constructs a
        # ContinuousStragglingModel, whose Geant4Tables base calls
        # get_init_table_set() in __init__. So the init tables are a
        # CONTINUOUS TRAINING dependency, not only a simulator one -- and a
        # missing one would otherwise surface only inside setup(), after the
        # arrays are already loaded. The secondary stage has no such filter
        # and does not need them.
        "init table: dE/dx": PATH_INIT_DEDX,
        "init table: range": PATH_INIT_RANGE,
        "init table: scaled KE": PATH_INIT_SCALED_KIN,
        "init table: lambda_ion": PATH_INIT_LAMBDA_ION,
    }
    if USE_PHYSICS:
        # ContinuousGAN's constructor raises when this is absent rather than
        # silently training without the penalty term; check it here instead.
        required["physics CDFs"] = PATH_TO_CONTINUOUS_CDFS

    resume = args.resume.strip()
    resume_ckpt = (RUNS_DIR / "logger_cnt" / resume / "checkpoints" / "last.ckpt"
                   if resume else None)

    preflight("continuous", required, resume_ckpt)
    if resume:
        print(f">> resuming from {resume_ckpt}")
    submit("continuous", "run_on_node_continuous.sh", args)


if __name__ == "__main__":
    sys.exit(main())
