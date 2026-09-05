# Training

How the continuous and secondary generators are trained: the data, the
WGAN-GP loop, the physics penalty, and every knob a run is launched with.

## The package

`src/physics/g4h_ionisation/generators/train_networks/` holds one subpackage
per stage — `continuous_generator/` and `secondary_generator/` — each with the
same four modules: `datamodule.py` (loads the stage's `(X, Y)` arrays, fits
and writes its scalers, yields scaled batches), `critic.py` (the WGAN-GP
critic, discarded after training), `wgan.py` (the `LightningModule`: loss
terms, logging, per-epoch diagnostics) and `train.py` (the runnable
entrypoint, `main()` / `__main__`).

Two shared pieces sit outside it: the `WGANInterface` base at
`src/physics/interfaces/wgan_interface.py`, alongside the other ABCs, and
`src/physics/penalty_models.py`. There is no length subpackage — that stage is
an analytic sampler, not a network ([simulation.md](simulation.md)). Nothing
trains as a side effect: no test or job calls a `main()`, and importing a
`train.py` touches no storage. Run one only after deciding to book a GPU.

## Datasets and the manifest

`python -m build_training_datasets` streams the beam's GEANT4 truth ROOT file
once and writes, under `DATASETS_DIR`, `continuous_X.npy` (`[E, L]`) /
`continuous_Y.npy` (`E_CNT`) and `secondary_X.npy` (post-continuous kinetic
energy) with both target widths — `secondary_Y.npy`, `(N, 2)`, and
`secondary_Y_drop_theta.npy`, `(N, 1)`. The secondary datamodule takes
`drop_theta=True`, so training reads the one-column form, which is what the
generator's output width means. Its row-local filters (first step in vacuum,
secondary-loss noise, the energy cutoff, transportation steps) are read off
`src/data_handling/truth_extract.py`'s executable filter list.

Alongside the arrays goes `manifest.json`: the truth file's identity, size and
mtime, the git SHA of the code that built it, the filters applied, every array
written with its shape, and the surviving row count — without which "this
checkpoint was trained on that data" is unfalsifiable.
`scripts/subsample_datasets_to_reference.py` records its trim there too.

## Datamodules and scalers

Each stage's `setup()` loads its arrays, filters them, fits the scalers it
owns and scales the data; batches are the dataset split into
`N_BATCHES = 64` pieces, so batch size follows the row count.

* **Secondary** — drops rows with `E_SEC == 0` (the network is trained on the
  nonzero continuum; the zero atom is injected at inference), fits
  `SecondaryXScaler`'s log10 min/max, and writes an *empty* state file for
  `SecondaryYScaler`, whose bounds are physics rather than data. An empty file
  is not an absent one: it records that this run fitted nothing, where a
  missing file cannot be told from a crash before saving.
* **Continuous** — additionally drops deterministic Bethe-Bloch rows by
  building a `ContinuousStragglingModel` per row, a Python loop over the whole
  dataset with no on-disk cache (a cache keyed on a file's existence would
  silently serve another dataset's result). It fits `ContinuousXScaler` and
  *loads* `ContinuousYScaler`, a function of the physics grid, not the data.

**Training reads what training writes.** The scalers are loaded in the
`LightningModule`'s `setup()` hook, not its `__init__`: Lightning runs
`DataModule.setup()` first, so by the time the model asks for a state the
datamodule has already fitted and written it. Loading in `__init__` would read
whatever the previous run left behind — or raise on a material that has never
been trained.

## Architectures

Both generators are conditioned nets: the label is embedded, concatenated with
a latent vector, and pushed through a `Linear`/`LayerNorm`/`ReLU` trunk.

* `ContinuousGenerator` embeds `[E, L]` into `N_EMBEDDING_CNT`, concatenates
  `N_NOISE_CNT` noise, runs two blocks of width
  `N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES`, then halves and reads out
  `N_CONTINUOUS_FEATURES` unclamped.
* `SecondaryGenerator` does the same from the kinetic energy alone, at
  `N_EMBEDDING_SEC` / `N_NOISE_SEC` and a trunk of `n_trunk` (default
  `N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES`), and ends in the terminal
  named by `SECONDARY_OUTPUT_ACTIVATION`.
* Each critic scores a candidate against the same embedded label through
  `LayerNorm`/`LeakyReLU` blocks of the stage's trunk width, its embedding
  width set by `N_EMBEDDING_DISC_CNT` / `N_EMBEDDING_DISC_SEC` — two widths
  that carry no checkpoint contract, the critic being discarded after
  training. Every generator constant above does; see
  [artifacts.md](artifacts.md).

`scripts/draw_architecture.py` draws all four of these for the paper's
appendix by instantiating the modules and walking their layers, so the figure
re-derives every width, norm and terminal from the constants above:

```bash
PYTHONPATH=src python scripts/draw_architecture.py     # -> artifacts/benchmark_outputs/architecture.pdf
```

The `NoPhys*` variants swap `LayerNorm` for `BatchNorm1d` (and
`InstanceNorm1d` in the embedding), and their terminals differ from each
other: `NoPhysContinuousGenerator` maps its readout through
`1.2 * sigmoid(output) - 0.1`, while `NoPhysSecondaryGenerator` ends in a
bare `Linear` and rectifies the `e_sec` column alone in `forward`, indexed
explicitly so a later column could not inherit the rectifier.

## The loop

`WGANInterface` owns the shared loop, with manual optimisation and two
`RMSprop` optimisers at `LEARNING_RATE`, alternating by batch index:
`opt_g_frequency = 1` generator step for every `opt_d_frequency = 3` critic
steps. The critic minimises `-(mean(critic(real)) - mean(critic(fake)))` plus
`LAMBDA_GP` times a gradient penalty — the squared deviation of its gradient
norm from 1 at points interpolated between real and fake samples. The
generator minimises `-mean(critic(fake))`, clipped to gradient norm 1.0, plus
whatever `add_generator_loss_terms` adds.

Each stage's `wgan.py` overrides that hook with the physics term:
`g_loss + lambda_physics * KSTestPenaltyModel.physics_penalty_loss(...)`, at
`LAMBDA_PHYSICS_CONTINUOUS` / `LAMBDA_PHYSICS_SECONDARY`. The penalty samples
a random tenth of the theoretical CDF keys, draws 10,000 samples per key
through `generator.predict` — the same scaling path the simulator uses — and
averages the Kolmogorov-Smirnov distance to the tabulated CDF. The CDF pickle
is required: both constructors raise when it is absent rather than quietly
training unregularised, and `USE_PHYSICS` gates only the loss term.
Every `EVERY_N_EPOCHS = 200` epochs, `on_train_epoch_end` runs a diagnostic
sampling pass (`SIMULATE_ON_EPOCH_END`) and logs comparison histograms to
TensorBoard.

## The no-physics arm

`PHINGAN_NO_PHYS=1` selects the ablation in `train.py::select_arm()`: it swaps
the generator class, passes the `NoPhys*` Y-scaler to both the datamodule (fit
target) and the model (load target), turns the physics term off, and renames
the logger directory `logger_cnt` -> `logger_cnt_nophys` (likewise for the
secondary), so its checkpoints cannot be mistaken for a physics run's.

## Run knobs

Every knob is an environment variable read by a `select_*` function in the
stage's `train.py`, not a config edit — the cluster's `src/` is shared with
queued jobs, and each is a per-run property. The submitter flags forward into
the job environment and refuse when they disagree with a variable already set.

| variable | flag | default | effect |
|---|---|---|---|
| `PHINGAN_SEED` | `--seed` | 42 | `seed_everything()` |
| `PHINGAN_CKPT_EVERY` | `--ckpt-every` | 100 | `ModelCheckpoint(every_n_epochs=)` |
| `PHINGAN_NO_PHYS` | `--no-phys` | unset | the ablation arm |
| `PHINGAN_SEC_OUTPUT_ACT` | `--sec-output-act` | `SECONDARY_OUTPUT_ACTIVATION` | the secondary terminal |
| `PHINGAN_SEC_STRETCH_EPS` | `--sec-stretch-eps` | `SECONDARY_STRETCH_EPS` | that terminal's stretch, for `stretched_sigmoid` |
| `PHINGAN_SEC_WIDTH` | `--sec-width` | `N_NEURON_MULTIPLIER_SEC` | secondary trunk multiplier; a positive even integer |
| `PHINGAN_BEAM` | `--beam` | the aluminium beam | which material tree the run reads and writes |
| `PHINGAN_RESUME` / `PHINGAN_RESUME_VERSION` | `--resume <id>` | unset / `version_0` | resume `last.ckpt` of that run directory |
| `PHINGAN_SAVE_FIGS` | — | unset | also write the per-epoch diagnostics as PNGs |

`PHINGAN_RESUME` exists because flipping `RESUME_FROM_CHECKPOINT` in
`src/common/config.py` also drops `LEARNING_RATE` tenfold at import; the
variable resumes without that coupling. A resumed leg logs into the *same*
directory, so its curves continue instead of splitting — and with
`MAX_EPOCHS = 100_000` against a 72-hour queue maximum, a full run is walked
forward across several submissions that way. `save_top_k=-1` keeps every
checkpoint, so the cadence of 100 holds a run's file count under the storage
inode quota.

## Run identity

`deploy_and_train.sh` mints
`RUN_ID = <timestamp>_<stage-token>_<material>_<short-sha>_<uniq>`, the stage
token picking up `_nophys`, `_s<seed>`, `_<activation>`, `_eps<value>` and
`_w<width>` from the flags above. It travels in the job environment as
`PHINGAN_RUN_ID` rather than a file in `src/`, which the next deploy would
overwrite while this job sits in the queue, and it names the run directory
(`run_version.py::get_run_version()`: a resume version, else `PHINGAN_RUN_ID`,
else a locally minted `<ts>_local_<uniq>` — never Lightning's numeric
auto-increment). The deploy also snapshots the tree it sent to a git branch.
Each run directory gets a `meta.json` — `run_id`, `git_sha`, `stage`, `beam`,
`seed`, `ckpt_every`, plus the secondary stage's activation, stretch and width
— appended to on resume, so top-level keys describe the latest leg and `legs`
is the full list.

## Running one

```bash
# Local, from the repo root.
PYTHONPATH=src python -m physics.g4h_ionisation.generators.train_networks.continuous_generator.train
PYTHONPATH=src python -m physics.g4h_ionisation.generators.train_networks.secondary_generator.train

# Cluster: deploy src/ and submit. Anything after the stage is forwarded to
# src/cluster_scripts/submit_<stage>_job.py verbatim.
./deploy_and_train.sh continuous
./deploy_and_train.sh secondary --seed=7 --sec-width=16
./deploy_and_train.sh continuous --resume <run-id>   # or --dry-run
```

The deploy sends `src/` only, so each submitter preflights the artifacts its
own job will open and exits nonzero listing every missing one at once, rather
than queueing a job that would reach a GPU and die in `setup()`.
`submit_continuous_job.py` requires that stage's two arrays,
`ContinuousYScaler.state.pt` and the four initialisation tables its
Bethe-Bloch filter builds a straggling model from; `submit_secondary_job.py`
requires only its own two arrays, plus a check of the target `.npy` header
against `N_SECONDARY_FEATURES`. Each adds its stage's CDF pickle when
`USE_PHYSICS`, since the model refuses to build without it. Stage everything
once with `./scripts/stage_artifacts.sh`.

## See also

- [artifacts.md](artifacts.md) — the datasets, scaler states and run directories
- [physics.md](physics.md) — the theoretical CDFs the penalty scores against
- [simulation.md](simulation.md) — how a trained checkpoint is loaded and run
- [cluster.md](cluster.md) — deploying, submitting and watching jobs
