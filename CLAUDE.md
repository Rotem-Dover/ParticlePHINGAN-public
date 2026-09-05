# CLAUDE.md

ParticlePHINGAN is a physics-informed generative emulator of GEANT4's stepping of
charged particles through matter: two WGAN-GP generators, conditioned on the
particle's state, sample the per-step continuous energy loss and the delta-ray
energy that GEANT4 would have produced, while the step length comes from an
analytic sampler and the primary's deflection from energy-momentum conservation.
The beam is a 100 MeV proton in aluminium, iron or beryllium, and ionisation is
the only physics process. Everything runs on tensors, batched across particles,
so a whole beam steps together on one GPU.

## Layout

```
src/
  common/                  config, path constants, enums, the active-beam context
  physics/definitions/     particle, material, beam presets, physical constants
  physics/g4_init_tables/  GEANT4 table builders and the runtime table reader
  physics/g4h_ionisation/  the ionisation process, its generators and training
  physics/interfaces/      generator and scaler base classes, the fused MLP
  particle_propagation/    TrackSimulator (lab frame) and SteppingManager (phases)
  benchmark/               the gates, the profiling instruments, the figures
  data_handling/           GEANT4 truth ROOT readers
  playground/              a Flask app for exploring the stages interactively
  cluster_scripts/         PBS submitters for the training and fused-kernel jobs
  build_training_datasets.py, build_figure_datasets.py   the dataset builders
```

`src/` is the source root: `pyproject.toml` sets it for pytest, and bare
interpreter invocations need `PYTHONPATH=src` (or `PYTHONPATH=.` from inside
`src/`). Imports are absolute, e.g. `from common.config import DEVICE`.

## Commands

Setup, then the test suite:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Run a simulation — both entrypoints build the `phin_gan` preset:

```bash
cd src && PYTHONPATH=. python -m particle_propagation.track_simulator
cd src && PYTHONPATH=. python -m particle_propagation.stepping_manager
```

Gate T (acceptance, against GEANT4 truth), gate B (optimised against eager), the
fused-kernel harness, the profiler and the compiled-versus-eager report:

```bash
cd src && PYTHONPATH=. python -m benchmark.verify_truth_parity [--device cuda]
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.verify_physics_gate --gate b --device cuda
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.test_fused_mlp --device cuda --n 1048576 [--sweep]
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.profile_runtime --preset phin_gan --device cuda \
    --n-steps 10250 --event-counts 1000000 --out fused.json
cd src && PYTHONPATH=. python -m benchmark.runtime_profiling.verify_compiled_parity
```

Build the artifacts a run reads, in this order (tables, then the CDFs, then the
continuous scaler's lookup, then the training datasets):

```bash
cd src                                  # one sequence, so change directory once
PYTHONPATH=. python -m physics.g4_init_tables.build_all
PYTHONPATH=. python -m \
    physics.g4h_ionisation.generators.continuous_generator.generate_cdfs
PYTHONPATH=. python -m \
    physics.g4h_ionisation.generators.secondary_generator.generate_cdfs
PYTHONPATH=. python -m \
    physics.g4h_ionisation.generators.continuous_generator.build_analytic_mean_std_table
cd .. && PYTHONPATH=src python scripts/build_continuous_scaler_state.py
cd src && PYTHONPATH=. python -m build_training_datasets
PYTHONPATH=. python -m build_figure_datasets
```

Train a stage locally, or deploy it to the cluster:

```bash
PYTHONPATH=src python -m \
    physics.g4h_ionisation.generators.train_networks.continuous_generator.train
PYTHONPATH=src python -m \
    physics.g4h_ionisation.generators.train_networks.secondary_generator.train
./deploy_and_train.sh continuous
./deploy_and_train.sh secondary --seed=7 --sec-width=16
./deploy_and_train.sh continuous --resume <run-id>   # or --dry-run
./scripts/stage_artifacts.sh                          # stage what a job reads
```

The figures, by script: each lives under `src/benchmark/system_vs_g4/` and runs
from `src/` with `PYTHONPATH=.` as a module of `benchmark.system_vs_g4`.
[docs/benchmarks.md](docs/benchmarks.md) spells out every command and output.

| fig | script |
|---|---|
| 8 | `phase_space_ks.py` |
| 9 | `energy_losses_distributions_at_marked_points.py` |
| 10 | `features_hist_comparison.py` |
| 11 | `pairwise_comparison.py` |
| 12 | `energy_distance/energy_distance_analysis.py` (`--score`, then `--plot`) |
| 13, E.2 | `pull_analysis/run_pull_figure.py` (`--preset gan` for E.2); its tolerance band comes from `pull_analysis/truth_null_band.py` |
| 14 | `plot_computation_time.py` |
| E.1 | `physics_losses.py` (`--score`, then `--plot`) |

The appendix architecture figure is drawn from the live modules, not by
hand: `PYTHONPATH=src python scripts/draw_architecture.py` (from the repository
root) instantiates both generators and both critics, walks their layers and
writes `artifacts/benchmark_outputs/architecture.{pdf,png}`.

One more gate covers the secondary generator's boundary behaviour:
`cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.secondary_endpoint_atoms`.
Checkpoints are picked with `benchmark.system_vs_g4.checkpoint_scan` (every
saved epoch of a run scored against truth; see docs/benchmarks.md). The
`phin_gan` pins live in `_PHIN_GAN_PINS` in `src/common/paths.py`, one pair per
material, and follow `PHINGAN_BEAM`; any beam other than aluminium writes its
figures under `artifacts/benchmark_outputs/<material>/`.

## Invariants

* **Never write under `storage/` from a tool.** `.claude/hooks/guard_storage.py`
  blocks writes to `storage/` and `.venv/` and to checkpoint, ROOT, pickle,
  `.npz` and `.bin` files. Those artifacts cost GPU-days or GEANT4-days to
  reproduce; build them with the commands above instead.
* **Never widen a gate threshold to turn a red gate green.** Thresholds in
  `src/benchmark/verify_truth_parity.py` and
  `src/benchmark/runtime_profiling/verify_physics_gate.py` are twice a measured
  null, pasted beside the constant it sets so the calibration stays auditable. A
  threshold at or below its own null fails on sampling scatter rather than on
  physics. A red gate is the finding; recalibrate only by re-measuring the null.
* **Enum layouts in `src/common/enums.py` are contracts.** `StepFeaturesIdxes` is
  `L, THETA, E_CNT, E_SEC, GEOM_STEP`; `SecondaryFeats` is `E_SEC_IDX` alone;
  `ContinuousFeats` is `E_CNT_IDX` alone. Reordering or extending either
  generator enum changes a network's output shape, so the checkpoints stop
  loading.
* **The architecture constants in `src/common/config.py` are the checkpoint
  contract.** `from_checkpoint` builds a bare module, so `N_EMBEDDING_*`,
  `N_NOISE_*`, `N_NEURON_MULTIPLIER_*`, the feature counts and
  `SECONDARY_OUTPUT_ACTIVATION` define the shape a stored state dict must match.
  Changing one invalidates any checkpoint trained at another value; flip it in
  the same commit as the checkpoint pins in `src/common/paths.py` (`CNT_CKPT`,
  `SEC_CKPT` and the two no-physics ones, which
  `src/benchmark/track_simulator_config.py` consumes).
* **Keep the fused kernel and its emulator in lockstep, and re-sweep on a GPU
  after any kernel change.** `src/physics/interfaces/_fused_mlp_kernel.py` and
  `_emulate_packed_forward` in `src/physics/interfaces/fused_mlp.py` are two
  independent implementations of the same arithmetic, and only the emulator runs
  in the test suite. A kernel-only edit — moving a `_LAUNCH_CONFIG` entry
  included — is caught by nothing local: treat it as unverified until the sweep
  harness and gate B have both re-run on a card.
* **The deflection angle uses `atan2`, never `acos`.** At this beam's roughly
  1e-4 rad deflections, `acos` of a ratio near one returns exactly zero in
  float32 below about 3.5e-4 rad and snaps the rest onto a coarse ladder;
  `src/physics/g4h_ionisation/kinematics.py` documents the conditioning, and
  `tests/physics/test_theta_kinematics.py` pins it with a negative control.
* **`post_step_do_it` computes the angle last, from the final secondary energy.**
  In `src/physics/g4h_ionisation/g4h_ionisation.py` the delta-ray Bernoulli mask,
  the killed-lane mask and the process-winner mask all zero `e_sec`, and the
  angle inherits every one of them through the kinematics. Move that call above
  any masking and three separate angle masks become necessary, silently.
* **No per-step host synchronisation inside a compiled phase.** A
  data-dependent `if tensor.any()` costs a device-to-host round trip every step
  and breaks the graph. The NaN guard shows the shape of the fix: the phase ORs
  its detection into a device-resident flag, and `TrackSimulator.run` reads that
  flag once per run through `check_nan_guard`, outside every compiled region.
* **Index tensors with `int(Enum.MEMBER)` inside compiled scopes.** FX codegen
  cannot serialise an `IntEnum` used directly as a tensor index, so
  `step_features[:, int(StepFeaturesIdxes.E_CNT)]` is the form used throughout
  `src/particle_propagation/track_simulator.py`.
* **Dead particles' rows are NaN-padded.** `TrackSimulator` allocates its output
  frames full of NaN and fills only live lanes, so any step-count or
  distributional statistic must drop `~isnan(KineticEnergy)` first. A NaN also
  compares false against every energy threshold, which is why the kill mask
  cannot catch one and the explicit guard exists.

## Where things are documented

* [docs/simulation.md](docs/simulation.md) — what is simulated, the three layers,
  the per-step quantities, and how to run a beam.
* [docs/physics.md](docs/physics.md) — where every number comes from: constants,
  the GEANT4 tables, the straggling models and the artifacts built from them.
* [docs/artifacts.md](docs/artifacts.md) — every file the code reads or writes,
  what builds it, and what reads it back.
* [docs/training.md](docs/training.md) — the datasets, the WGAN-GP loop, the
  physics penalty and every knob a run is launched with.
* [docs/benchmarks.md](docs/benchmarks.md) — the gates, the profiling
  instruments, the recorded numbers and the command behind every figure.
* [docs/cluster.md](docs/cluster.md) — what the deploy script sends, what is
  staged by hand, and the rules a job must obey to produce a quotable number.
* [measurements/README.md](measurements/README.md) — the recorded results
  themselves, with the hardware each ran on.

## Maintenance

This file describes current behaviour only. When a change alters layout,
entrypoints or invariants, update this file and the affected `docs/` page in the
same commit.
