"""Shared qsub plumbing for the two WGAN-GP training submitters.

`submit_continuous_job.py` and `submit_secondary_job.py` are the same job
modulo the stage name, the logger directory and the artifacts the stage
reads, so the argument parser, the preflight and the `qsub` line live here
rather than being duplicated in each submitter. Beam selection is via
`--beam`/`PHINGAN_BEAM`, resolved by `common/run_context.py`'s
`ACTIVE_NAME`.

**Why the preflight exists.** `deploy_and_train.sh` rsyncs `src/` only;
nothing syncs `storage/`. A cluster that has never had the training
artifacts staged would otherwise accept the job, burn its queue wait,
allocate a GPU and only then die inside `DataModule.setup()` on a missing
`.npy` -- or, worse for the continuous stage, well after the deterministic
Bethe-Bloch filter has already walked the dataset. Checking the paths at
submit time costs milliseconds and is the difference between a typo and a
wasted allocation.

The submitters run on the cluster head node (`deploy_and_train.sh` ssh's in
and invokes them there), so `common.paths` resolves the real `/storage`
locations and the preflight checks the files the job will actually open.
Submitting from a laptop instead still works: `ON_CLUSTER` is False there, the
paths point at the local tree, and the preflight says so rather than silently
checking the wrong filesystem.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import ON_CLUSTER  # noqa: E402

CLUSTER_PROJECT_DIR = "/srv01/agrp/rotemdo"

# Defaults shared by both stages.
#
# `io=3`: a training run writes checkpoints and figures to /storage
# periodically rather than continuously, so a low `io` allocation is enough
# to avoid the job stalling on storage I/O without over-reserving it.
#
# `walltime=72:00:00`: the queue maximum, and nowhere near enough for
# `MAX_EPOCHS = 100_000` -- see `--resume` and the note in its help text.
DEFAULTS = {
    "ncpus": "15",
    "ngpus": "1",
    "walltime": "72:00:00",
    "io": "3",
    "gputype": "A6000",
    "queue": "gpu",
}


def build_parser(stage: str, default_mem: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Submit the {stage} WGAN-GP training job to the cluster.")
    p.add_argument("--ncpus", "-nc", default=DEFAULTS["ncpus"])
    p.add_argument("--ngpus", "-ng", default=DEFAULTS["ngpus"])
    p.add_argument("--mem", "-mem", default=default_mem)
    p.add_argument("--walltime", "-wt", default=DEFAULTS["walltime"],
                   help="Queue maximum is 72:00:00. A full run needs several "
                        "of these back to back; use --resume for each one "
                        "after the first.")
    p.add_argument("--io", "-io", default=DEFAULTS["io"],
                   help="Storage I/O allocation. Do not lower below the "
                        "default -- see this module's docstring.")
    p.add_argument("--gputype", "-gt", default=DEFAULTS["gputype"])
    p.add_argument("--queue", "-q", default=DEFAULTS["queue"])
    p.add_argument("--resume", default="",
                   help="Resume from last.ckpt of this logger version "
                        "(e.g. 'version_0'). The job reads it via "
                        "PHINGAN_RESUME_VERSION; without this flag the run "
                        "starts from scratch.")
    p.add_argument("--git-sha", default="",
                   help="Recorded in the job name's environment for "
                        "provenance; set by deploy_and_train.sh.")
    p.add_argument("--run-id", default="",
                   help="TensorBoard run-directory name minted by "
                        "deploy_and_train.sh; lands in the job env as "
                        "PHINGAN_RUN_ID. When omitted the trainer mints a "
                        "local <ts>_local_<uniq> id instead.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the preflight and print the qsub line without "
                        "submitting.")
    p.add_argument("--beam", default=None,
                   help="Beam preset name (physics.definitions.beam.PRESETS); "
                        "forwarded to the job env as PHINGAN_BEAM. The "
                        "submitter itself must ALSO run under PHINGAN_BEAM "
                        "(deploy_and_train.sh sets both from one value) so "
                        "the preflight checks the same material tree the job "
                        "will read.")
    p.add_argument("--no-phys", action="store_true",
                   help="Forwarded to the job env as PHINGAN_NO_PHYS=\"1\", "
                        "which train.py's select_arm() reads to swap in the "
                        "no-physics-regularization ablation architecture. "
                        "The preflight's required-artifact set is unchanged: "
                        "a no-phys run reads the same datasets, and the "
                        "physics-only artifacts it does not read are present "
                        "cluster-side regardless.")
    p.add_argument("--seed", type=int, default=None,
                   help="Forwarded to the job env as PHINGAN_SEED, which the "
                        "trainers' select_seed() reads for seed_everything(). "
                        "Omitted => not forwarded => the trainer's default 42, "
                        "the value every run before this flag used.")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="Forwarded to the job env as PHINGAN_CKPT_EVERY, which "
                        "the trainers' select_ckpt_every() reads for "
                        "ModelCheckpoint(every_n_epochs=...). Omitted => not "
                        "forwarded => the trainer's default 100. Mind the "
                        "/storage inode quota: save_top_k=-1 keeps every file.")
    p.add_argument("--sec-output-act",
                   choices=("hardtanh", "sigmoid", "stretched_sigmoid",
                            "hardtanh_open_top", "hardtanh_reflect_top"),
                   default=None,
                   help="Secondary stage only. Forwarded to the job env as "
                        "PHINGAN_SEC_OUTPUT_ACT, which train.py's "
                        "select_output_activation() reads to pick the physics "
                        "SecondaryGenerator's terminal layer. Omitted => not "
                        "forwarded => common.config.SECONDARY_OUTPUT_ACTIVATION.")
    p.add_argument("--sec-stretch-eps", type=float, default=None,
                   help="Secondary stage only, with --sec-output-act="
                        "stretched_sigmoid. Forwarded to the job env as "
                        "PHINGAN_SEC_STRETCH_EPS (train.py select_stretch_eps()). "
                        "Omitted => config's SECONDARY_STRETCH_EPS.")
    p.add_argument("--sec-width", type=int, default=None,
                   help="Secondary stage only. Forwarded to the job env as "
                        "PHINGAN_SEC_WIDTH, the physics SecondaryGenerator's "
                        "trunk width multiplier (train.py select_width()). "
                        "Omitted => config's N_NEURON_MULTIPLIER_SEC.")
    return p


def preflight(stage: str, required: dict[str, Path], resume_ckpt: Path | None) -> None:
    """Refuse to submit when an artifact the job will open is absent.

    `required` maps a human label to the path. Every entry is checked; all
    failures are reported at once, because fixing them one submit at a time is
    the slow way to discover that three things are missing.
    """
    where = "cluster" if ON_CLUSTER else "LOCAL tree (not the cluster!)"
    print(f">> preflight for {stage}, against the {where}")

    missing = [f"    {label:24} {path}" for label, path in required.items()
               if not path.exists()]

    if resume_ckpt is not None and not resume_ckpt.exists():
        missing.append(f"    {'resume checkpoint':24} {resume_ckpt}")

    if missing:
        print(f"!! {len(missing)} required artifact(s) missing:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        print("\n   storage/ is NOT rsynced by deploy_and_train.sh. Stage the\n"
              "   artifacts first:  ./scripts/stage_artifacts.sh\n",
              file=sys.stderr)
        sys.exit(1)

    for label, path in required.items():
        size_mb = path.stat().st_size / 1e6 if path.is_file() else 0.0
        print(f"    ok  {label:24} {size_mb:9.1f} MB  {path}")


def _assemble_env(args) -> str:
    """Build the qsub `-v` env string from the parsed submitter args.

    Kept separate from `submit()` so the string can be unit-tested without
    invoking `qsub`.
    """
    run_dir = f"{CLUSTER_PROJECT_DIR}/src/"
    env = f'RUN_DIR="{run_dir}"'
    if args.resume.strip():
        env += ',PHINGAN_RESUME="1"'
        env += f',PHINGAN_RESUME_VERSION="{args.resume.strip()}"'
    if args.run_id.strip():
        env += f',PHINGAN_RUN_ID="{args.run_id.strip()}"'
    if args.git_sha.strip():
        env += f',PHINGAN_GIT_SHA="{args.git_sha.strip()}"'
    beam = (getattr(args, "beam", None) or os.environ.get("PHINGAN_BEAM", "")).strip()
    env_beam = os.environ.get("PHINGAN_BEAM", "").strip()
    if beam and env_beam and beam != env_beam:
        sys.exit(f"--beam={beam!r} disagrees with PHINGAN_BEAM={env_beam!r}; "
                 f"the preflight already ran under the env value -- refusing")
    if beam:
        env += f',PHINGAN_BEAM="{beam}"'
    no_phys = getattr(args, "no_phys", False)
    env_no_phys_raw = os.environ.get("PHINGAN_NO_PHYS", None)
    if env_no_phys_raw is not None:
        env_no_phys = env_no_phys_raw.strip() == "1"
        if no_phys != env_no_phys:
            sys.exit(f"--no-phys={no_phys} disagrees with "
                     f"PHINGAN_NO_PHYS={env_no_phys_raw!r}; the preflight "
                     f"already ran under the env value -- refusing")
    if no_phys:
        env += ',PHINGAN_NO_PHYS="1"'
    seed = getattr(args, "seed", None)
    env_seed_raw = os.environ.get("PHINGAN_SEED", "").strip()
    if seed is not None and env_seed_raw and str(seed) != env_seed_raw:
        sys.exit(f"--seed={seed} disagrees with PHINGAN_SEED={env_seed_raw!r}; "
                 f"the submitter already ran under the env value -- refusing")
    if seed is not None:
        env += f',PHINGAN_SEED="{seed}"'
    ckpt_every = getattr(args, "ckpt_every", None)
    env_ckpt_raw = os.environ.get("PHINGAN_CKPT_EVERY", "").strip()
    if ckpt_every is not None and ckpt_every < 1:
        sys.exit(f"--ckpt-every={ckpt_every}: must be a positive integer")
    if ckpt_every is not None and env_ckpt_raw and str(ckpt_every) != env_ckpt_raw:
        sys.exit(f"--ckpt-every={ckpt_every} disagrees with "
                 f"PHINGAN_CKPT_EVERY={env_ckpt_raw!r}; the submitter already "
                 f"ran under the env value -- refusing")
    if ckpt_every is not None:
        env += f',PHINGAN_CKPT_EVERY="{ckpt_every}"'
    act = getattr(args, "sec_output_act", None)
    env_act_raw = os.environ.get("PHINGAN_SEC_OUTPUT_ACT", "").strip()
    if act is not None and env_act_raw and act != env_act_raw:
        sys.exit(f"--sec-output-act={act} disagrees with "
                 f"PHINGAN_SEC_OUTPUT_ACT={env_act_raw!r}; the submitter "
                 f"already ran under the env value -- refusing")
    if act is not None:
        env += f',PHINGAN_SEC_OUTPUT_ACT="{act}"'
    eps = getattr(args, "sec_stretch_eps", None)
    env_eps_raw = os.environ.get("PHINGAN_SEC_STRETCH_EPS", "").strip()
    if eps is not None and env_eps_raw and float(eps) != float(env_eps_raw):
        sys.exit(f"--sec-stretch-eps={eps} disagrees with "
                 f"PHINGAN_SEC_STRETCH_EPS={env_eps_raw!r}; the submitter "
                 f"already ran under the env value -- refusing")
    if eps is not None:
        if not eps > 0:
            sys.exit(f"--sec-stretch-eps={eps}: must be > 0")
        env += f',PHINGAN_SEC_STRETCH_EPS="{eps!r}"'
    width = getattr(args, "sec_width", None)
    env_width_raw = os.environ.get("PHINGAN_SEC_WIDTH", "").strip()
    if width is not None and env_width_raw and str(width) != env_width_raw:
        sys.exit(f"--sec-width={width} disagrees with "
                 f"PHINGAN_SEC_WIDTH={env_width_raw!r}; the submitter "
                 f"already ran under the env value -- refusing")
    if width is not None:
        env += f',PHINGAN_SEC_WIDTH="{width}"'
    return env


def submit(stage: str, node_script: str, args, extra_env: str = "") -> None:
    script = f"{CLUSTER_PROJECT_DIR}/src/cluster_scripts/{node_script}"

    env = _assemble_env(args)
    if extra_env:
        env += f",{extra_env}"

    command = (
        # -r n: never let PBS auto-rerun a killed trainer. A requeued run
        # restarts from epoch 0 into the SAME run directory (run_version
        # resolves PHINGAN_RUN_ID), silently overwriting checkpoints. A dead
        # training job must fail visibly and be resumed deliberately with
        # --resume, never restarted implicitly.
        f"qsub -q {args.queue} -r n -N train_{stage}"
        f" -l walltime={args.walltime},mem={args.mem}"
        f",ncpus={args.ncpus},ngpus={args.ngpus},io={args.io}"
    )
    if args.gputype:
        command += f",gputype={args.gputype}"
    command += f" -v {env} {script}"

    print(command)
    if args.dry_run:
        print(">> --dry-run: not submitted")
        return
    os.system(command)
