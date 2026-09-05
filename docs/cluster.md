# Cluster

How work reaches the PBS cluster: what the deploy script sends, what has to be
staged by hand, and the rules a job must obey to produce a quotable number.

## Hosts and roots

| | |
|---|---|
| head node | `gpu2` (`CLUSTER_HOST` in `deploy_and_train.sh` and `scripts/stage_artifacts.sh`) |
| code root | `/srv01/agrp/rotemdo` — `PROJECT_DIR`, and what its presence means to `src/common/paths.py`'s `ON_CLUSTER` detection |
| storage root | `/storage/agrp/rotemdo` — `STORAGE`, holding `resources/` and `runs/` |
| interpreter | `/storage/agrp/rotemdo/venv`, sourced by every `run_on_node_*.sh` |

`src/common/paths.py` picks these automatically: if `/srv01/agrp/rotemdo` exists the
run is on the cluster, otherwise `PROJECT_DIR` is the repo and `STORAGE` is
`<repo>/storage`. No environment variable switches it.

## Deploying a training job

```bash
./deploy_and_train.sh continuous
./deploy_and_train.sh secondary --seed=7 --sec-width=16
./deploy_and_train.sh continuous --resume <run-id>
./deploy_and_train.sh secondary --dry-run
./deploy_and_train.sh secondary --gputype=A5000     # forwarded verbatim
```

The first argument is the stage (`continuous` or `secondary`); everything after it
is forwarded to that stage's submitter in `src/cluster_scripts/`. The script rsyncs
`src/` to `gpu2:/srv01/agrp/rotemdo/src/` and ssh's in to run the submitter under
the cluster venv. **It sends code only**: `storage/` is never rsynced, and `.run_id`
is excluded so a queued job's identity cannot be clobbered by the next deploy. Two
things track every deploy:

* **`RUN_ID`** = `<timestamp>_<stage-token>_<material>_<short-sha>_<uniq>`, passed
  to the job in the qsub environment as `PHINGAN_RUN_ID`, never through a file in
  the shared `src/`. The stage token absorbs `--no-phys`, `--seed`,
  `--sec-output-act`, `--sec-stretch-eps` and `--sec-width`, so a run's
  configuration is legible from its directory name.
* **A snapshot branch** `runs/<stage>/<timestamp>`, holding the exact tree that was
  sent — dirty state committed through a stash round-trip and then restored to the
  working tree. `git checkout` of that branch recovers the code behind any run. A
  same-second re-run is refused up front, before anything is touched, rather than
  colliding mid-snapshot with a stashed tree.

`--resume <run-id>` reuses the prior `RUN_ID` verbatim, so a walked-forward run
keeps logging into one directory instead of splitting its curves. `--dry-run` rsyncs
and runs the submitter's preflight anyway, but mints no snapshot branch: there is no
run to track. `--beam` selects the material tree and is exported to the submitter
process *before* Python starts, since it imports `common.paths` at module load for
its preflight. Flags, defaults and the environment variables they set are tabulated
in [training.md](training.md); the submitters refuse when a flag disagrees with a
variable already set in their own environment, rather than queueing a job whose
preflight checked one configuration and whose trainer runs another.

## Resources and the qsub line

`src/cluster_scripts/_training_job.py` assembles the submission. Its shared defaults
are `ncpus=15`, `ngpus=1`, `walltime=72:00:00`, `io=3`, `gputype=A6000`, queue
`gpu`; memory is per stage — 32 gb continuous, 16 gb secondary, because the
continuous stage's physics penalty unpickles a large CDF file alongside the training
arrays. 72 hours is the queue maximum and nowhere near `MAX_EPOCHS = 100_000`, so a
full run is several submissions walked forward with `--resume`.

Training is submitted with **`qsub -r n`**: PBS must never auto-rerun a killed
trainer. A requeued run restarts from epoch 0 into the *same* run directory, because
`PHINGAN_RUN_ID` resolves there, and silently overwrites its own checkpoints.
A dead job should fail visibly and be resumed deliberately with `--resume`. Follow a
running one with `ssh gpu2 tail -f 'train_<stage>.o<jobid>'`.

## Staging artifacts

```bash
./scripts/stage_artifacts.sh [--dry-run] [--beam=proton/iron]
```

Run this once before the first submission. It copies, one-directionally and without
`--delete`, the `datasets/`, `scalers/`, `tables/`, `cdfs/` and `init_tables/`
subtrees of `resources/<beam>/` to `gpu2:/storage/agrp/rotemdo/resources/<beam>/`.
The `cdfs/` `_xs` / `_ys` variants are left behind: they are inputs to the offline
CDF builder, and nothing on the training path opens them.

Each submitter **preflights** the files its job will open and exits nonzero listing
every missing one at once, rather than queueing a job that would wait in the queue,
allocate a GPU and only then die inside `setup()`. Verify staging with
`./deploy_and_train.sh <stage> --dry-run`. What each stage needs is listed in
[artifacts.md](artifacts.md) and [training.md](training.md).

## PBS rules

Each of these has a mechanical reason, and each has cost real time when ignored.

* **Ask for `io=16` on a measurement job.** The `io` units throttle `/storage`
  access, and a job given `io=1` spends its wall clock blocked on storage rather
  than computing. Every PBS wrapper under `measurements/` asks for `io=16` in its
  own submission line. Training runs at the lower `io=3` default, which is enough
  for a checkpoint cadence rather than continuous writing.
* **Put every arm of a comparison in one job on one card.** Job-to-job offsets are
  real and one phase can inflate while the others match, so a curve stitched from
  several jobs encodes scheduling scatter as physics. The measurement wrappers run
  all their arms — every event count, both the fused and eager arms — inside a
  single submission, alternating rather than grouping.
* **A CPU rate carries a contention band.** Nodes are shared and exclusive placement
  is not schedulable, so a single-core rate moves between jobs and drifts *within* a
  long one. Quote the conservative end and say so. Shares measured inside one run
  are immune, since everything slows together.
* **`gputype=A6000` does not name a card.** That pool resolves to several models, so
  assert the device name rather than the request: `torch.cuda.get_device_name(0)`.
  `measurements/gate_b/gate_b.pbs.sh` refuses to run unless the card matches, with
  `ALLOW_ANY_GPU=1` as a deliberate, logged override — thresholds calibrated on one
  model do not describe another.
* **Never submit to a node already running one of your jobs.** Co-scheduling onto an
  occupied node kills the incumbents.
* **A job stuck in a kill-and-requeue loop needs `qhold` before `qdel`.** A plain
  `qdel` cannot make it stay dead: the server force-requeues on a perceived node
  failure.
* **Watch the `/storage` inode quota, not the byte quota.** `save_top_k=-1` keeps
  every checkpoint file, so the `--ckpt-every` cadence sets a run's file count; the
  default of 100 is what `tests/cluster_scripts/test_submit_training_jobs.py` pins
  as safe against `MAX_EPOCHS`. Per-epoch diagnostic PNGs are off unless
  `PHINGAN_SAVE_FIGS=1` for the same reason — the figures reach TensorBoard either
  way.

## Measurement jobs

There is no deploy script for measurements. The PBS wrappers committed under
`measurements/` are the templates: `measurements/gate_b/gate_b.pbs.sh` and
`gate_b_calibrate.pbs.sh`, `measurements/fused_mlp_sweep/sweep.pbs.sh`,
`measurements/computation_time/phin_gan_timing.pbs.sh` and `geant4_timing.pbs.sh`,
and `measurements/phase_timing/phin_gan_phases.pbs.sh` and
`geant4/phase_timing.pbs.sh`. Each carries its own `qsub` line in a comment at the
top, prints the host (and a GPU job's resolved device name) before measuring, and
writes into `/storage/agrp/rotemdo`. Copy one rather than inventing a
submission. One GPU submitter exists for the fused-kernel harness,
`src/cluster_scripts/submit_fused_test_job.py`: it runs on the head node without the
venv, so it imports nothing from `src/`, takes `--tf32` and `--extra-args`, and has
its own `--dry-run`.

## See also

- [training.md](training.md) — the flags, environment variables and run identity
- [artifacts.md](artifacts.md) — what lives under `resources/` and `runs/`
- [benchmarks.md](benchmarks.md) — the gates and measurements these jobs produce
