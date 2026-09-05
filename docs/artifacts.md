# Artifacts

Where every file the code reads or writes lives, what builds it, and what
reads it back.

## Two roots

`src/common/paths.py` is the only place a path is spelled out. Everything
hangs off `STORAGE`, which is `/storage/agrp/rotemdo` on the cluster — chosen
when `/srv01/agrp/rotemdo` exists, which is also where the deployed `src/`
sits — and `<repo>/storage` otherwise. Under it:

* `resources/<particle>/<material>/` (`MATERIAL_DIR`) — GEANT4 truth, built
  tables, CDFs, scaler states, training datasets, trainer figures.
* `runs/<particle>/<material>/` (`RUNS_DIR`) — one directory per training
  run: checkpoints, TensorBoard event files, `meta.json`.

The particle and material segments come from the beam that
`common.run_context` resolves from `PHINGAN_BEAM`, so every constant in
`paths.py` retargets with the beam and no other module branches on the
material. See [physics.md](physics.md) for the beam registry.

`storage/` is gitignored and populated only in the main checkout, never
duplicated per checkout. A linked git working tree — one whose `.git` entry
is a file reading `gitdir: …` rather than a directory — follows that link
back to the main checkout and resolves `STORAGE` to its `storage/`, so it
reads the same tree instead of an empty local one.

## The tree

```
resources/proton/aluminum/
  tracks/                  GEANT4 truth ROOT files: <tag>_1k.root (the
                           training set) and <tag>_10k.root. Produced by the
                           external GEANT4 stepping application, not by
                           anything here; read by src/data_handling/truth_extract.py
                           Ten further seeded 10k files live outside this tree,
                           in storage_paper/ResourceFiles/iid/, read only by
                           the truth-versus-truth pull study (see
                           measurements/README.md, `pull_truth_null/`).
  init_tables/             dedx.npz, range.npz, scaled_kin_energy.npz,
                           lambda_ion.npz. Built by
                           `physics.g4_init_tables.build_all`; read at runtime
                           through src/physics/g4_init_tables/runtime.py
  tables/                  continuous_mean_std_lookup_table_steps_5000_analytic.pkl,
                           the 5000x5000x2 mean/std lookup. Built by
                           `...continuous_generator.build_analytic_mean_std_table`;
                           read only by scripts/build_continuous_scaler_state.py
  cdfs/                    continuous_cdfs.pkl / secondaries_cdfs.pkl and the
                           _xs/_ys inputs they are assembled from. Built by
                           each stage's generate_cdfs.py; read by
                           src/physics/penalty_models.py during training
  scalers/                 <ScalerClassName>.state.pt, one per scaler. Fitted
                           and written by the training datamodules, except
                           ContinuousYScaler.state.pt (built from tables/ by
                           scripts/build_continuous_scaler_state.py). Read by
                           every scaler's load(). nophys_continuous/ holds the
                           two states the no-physics continuous checkpoint was
                           trained against
  datasets/                continuous_X/Y.npy, secondary_X/Y.npy,
                           secondary_Y_drop_theta.npy + manifest.json, built by
                           `python -m build_training_datasets`; the
                           <stem>_*_filtered.pkl pairs + their manifest, built
                           by `python -m build_figure_datasets`. Read by the
                           training datamodules and the figure scripts
  validation_references/   g4_reference/ dumps and ke_to_sec_gen_frac_100.bin,
                           read by tests and diagnostics only
  figures/                 per-epoch trainer diagnostics, written only under
                           PHINGAN_SAVE_FIGS=1

runs/proton/aluminum/
  logger_cnt/<run-id>/     continuous stage: checkpoints/epoch=N.ckpt +
  logger_sec/<run-id>/     last.ckpt, TensorBoard events, meta.json
  logger_cnt_nophys/…      the same, for the no-physics ablation arms
  logger_sec_nophys/…
```

`paths.py` creates the eight `resources/` subtrees above and `RUNS_DIR`
itself on import, so a fresh clone has that much of the shape before it has
any contents; the per-run `logger_*` directories are made by the trainers.

## The pinned checkpoints

`src/benchmark/track_simulator_config.py` pins each preset stage as a
4-tuple: generator class, checkpoint, Y-scaler class, and the scaler
`state_dir` that stage's X- **and** Y-scaler load from — a checkpoint is only
meaningful with the scaler state it trained against.

| preset | stage | checkpoint constant | run directory | scaler `state_dir` |
|---|---|---|---|---|
| `phin_gan` | continuous | `CNT_CKPT` | `logger_cnt/<run-id>/checkpoints/epoch=<n>.ckpt` | `SCALERS_DIR` |
| `phin_gan` | secondary | `SEC_CKPT` | `logger_sec/<run-id>/checkpoints/epoch=<n>.ckpt` | `SCALERS_DIR` |
| `gan` | continuous | `NOPHYS_CNT_CKPT` | `logger_cnt_nophys/version_2/checkpoints/epoch=1599.ckpt` | `NOPHYS_CNT_SCALERS_DIR` |
| `gan` | secondary | `NOPHYS_SEC_CKPT` | `logger_sec_nophys/<run-id>/checkpoints/epoch=3199.ckpt` | `SCALERS_DIR` |

The `phin_gan` pins follow the beam: `_PHIN_GAN_PINS` in `common/paths.py`
holds one (run, epoch) pair per stage per material, and `CNT_CKPT` /
`SEC_CKPT` resolve from it under the active `PHINGAN_BEAM`. Every material's
secondary run is the same `hardtanh_open_top` W=16 configuration, loaded as
`hardtanh_reflect_top`. The no-physics ablation exists for aluminium only,
so the `gan` preset builds for that beam alone.

| material | continuous | secondary |
|---|---|---|
| aluminum | `2026-08-12-164426_…` epoch 23499 | `2026-08-19-192831_…s2_hardtanh_open_top_w16…` epoch 17999 |
| iron | `2026-08-10-205505_…` epoch 18899 | `2026-08-22-155334_…s3_hardtanh_open_top_w16…` epoch 29699 |
| beryllium | `2026-08-11-173628_…` epoch 37699 | `2026-08-22-155401_…s3_hardtanh_open_top_w16…` epoch 56499 |

The iron and beryllium picks came from `benchmark.system_vs_g4.checkpoint_scan`
([benchmarks.md](benchmarks.md#choosing-a-checkpoint)); the scan tables are
under `artifacts/benchmark_outputs/ckpt_scan/<beam>/`.

`<run-id>` stands for the full run-directory name, spelled out in
`src/common/paths.py`. The four constants are pinned by explicit path rather
than resolved by recency: a recency resolver keys on timestamp-prefixed run
ids and would not see a `version_N` directory at all. The length stage is
shared by both presets and loads no checkpoint.

## The checkpoint contract

`GeneratorInterface.from_checkpoint` builds a bare `cls()` and loads the
state dict into it, so the architecture constants in `src/common/config.py`
**are** the architecture:

| constant | meaning |
|---|---|
| `N_CONTINUOUS_FEATURES` / `N_SECONDARY_FEATURES` | each generator's output width, both 1 |
| `N_EMBEDDING_CNT` / `N_NOISE_CNT` | conditioning-embedding and latent widths of the continuous net, 25 each |
| `N_EMBEDDING_SEC` / `N_NOISE_SEC` | the same for the secondary net, 5 each |
| `N_NEURON_MULTIPLIER_CNT` / `N_NEURON_MULTIPLIER_SEC` | trunk-width multipliers, 50 and 16 |
| `N_NEURON_MULTIPLIER_SEC_NOPHYS` | the ablation secondary trunk's own multiplier, 32 |
| `SECONDARY_OUTPUT_ACTIVATION` | which terminal map the secondary net ends in |
| `SECONDARY_STRETCH_EPS` | that terminal's stretch, read only for `stretched_sigmoid` |

Change one without training a matching checkpoint and `load_state_dict`
raises on the shape mismatch. The two activation constants are the exception
that a shape check cannot catch — every terminal kind has the same state-dict
shape — so the net registers its choice as an `output_activation_id` buffer
and its stretch as an `output_stretch_eps` buffer, and
`SecondaryGenerator._load_from_state_dict` compares both against the net it
is loading into and appends an error naming the constant to set. One
direction is allowed deliberately: `hardtanh_open_top` and
`hardtanh_reflect_top` are the same net at training time and differ only in
the evaluation spill rule, so an open-top checkpoint loads into a reflecting
net. The reverse is refused. Flip `SECONDARY_OUTPUT_ACTIVATION` in the
same edit as `SEC_CKPT`.

The critic widths (`N_EMBEDDING_DISC_CNT`, `N_EMBEDDING_DISC_SEC`) and the
training-schedule constants below them carry no contract: the critic is
discarded after training.

## Building everything from a fresh clone

In order — each step reads what the one before it wrote. Run all of them from
`src/` with `PYTHONPATH=.` unless the line says otherwise.

```bash
# 1. GEANT4 truth. Not built here: place the beam's ROOT files in
#    resources/<particle>/<material>/tracks/ under the names paths.py expects
#    (PATH_TO_GEANT4_DATA, PATH_TO_GEANT4_DATA_SEED42).

# 2. The four initialisation tables.
python -m physics.g4_init_tables.build_all

# 3. The theoretical CDFs for the physics penalty. The continuous builder
#    also reads the truth file, to keep its keys inside its convex hull.
python -m physics.g4h_ionisation.generators.continuous_generator.generate_cdfs
python -m physics.g4h_ionisation.generators.secondary_generator.generate_cdfs

# 4. The analytic continuous mean/std lookup. Long-running: a batch job.
python -m physics.g4h_ionisation.generators.continuous_generator.build_analytic_mean_std_table

# 5. Convert that pickle into ContinuousYScaler's portable state. Nothing
#    builds it on demand: load() raises when it is absent.
cd .. && PYTHONPATH=src python scripts/build_continuous_scaler_state.py

# 6. The per-stage training arrays and their provenance manifest.
cd src && PYTHONPATH=. python -m build_training_datasets
```

Every step follows the active `PHINGAN_BEAM`; none takes a `--beam`
argument. The remaining scaler states are fitted during training itself, by
the datamodules — see [training.md](training.md).

`scripts/subsample_datasets_to_reference.py` trims one material's arrays to a
target row count with a single shared sorted mask across all five, so that
different materials train on comparable step counts; it records the
subsample in the manifest, and the full arrays are recoverable by re-running
step 6 against the untouched truth.

`./scripts/stage_artifacts.sh` copies the subset a training job opens —
`datasets/`, `scalers/`, `tables/`, the two assembled CDF pickles and
`init_tables/` — to the cluster. Nothing else syncs `storage/`: the deploy
script sends `src/` alone, which is why each submitter preflights the files
its job will open and refuses instead of queueing.

## Never write under `storage/` from a tool

The tree is gitignored, shared between checkouts, and long-running to
regenerate — the mean/std table and a training run each occupy a machine for
a stretch. `.claude/hooks/guard_storage.py` runs as a `PreToolUse` hook and
exits 2 on any `Edit`, `Write` or `NotebookEdit` whose target starts with
`storage/` or `.venv/`, or ends in `.ckpt`, `.root`, `.pkl`, `.npz` or
`.bin`. Its message points at `scripts/` and the training package as the way
to rebuild such a file. Write into `storage/` through the builders on this
page, never by hand.

## See also

- [physics.md](physics.md) — what the tables, CDFs and lookups contain
- [simulation.md](simulation.md) — the presets that load these checkpoints
- [training.md](training.md) — the runs that write under `runs/`
- [cluster.md](cluster.md) — deploying and submitting jobs
- [benchmarks.md](benchmarks.md) — the figures built from these datasets
