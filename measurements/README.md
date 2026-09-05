# Measurements

What each recorded result is, the hardware it ran on, and what reads it.

Every `.pbs.sh` beside a dataset is the job that produced it, and the profiler JSONs' `meta` blocks record their own host, device and arguments.
[../docs/benchmarks.md](../docs/benchmarks.md) quotes these numbers; [../docs/cluster.md](../docs/cluster.md) covers submission.

| file | what it is | hardware | read by |
|---|---|---|---|
| `measurements/computation_time/geant4_cpu_times.csv` | five `n_events,seconds` points from a one-core, ionisation-only GEANT4 proton sweep with saving disabled | one CPU core (model not recorded; figure 14 labels it Intel Xeon Gold 5320) | figure 14's GEANT4 arm, via `--g4-csv` |
| `measurements/computation_time/phin_gan_gpu_fused_r1.json`, `_r2.json` | `profile_runtime` scaling sweeps of the `phin_gan` preset, fused and compiled, 10,250 steps | NVIDIA RTX 6000 Ada (`meta.device_name`) | figure 14's GPU arm |
| `measurements/computation_time/phin_gan_gpu_eager_r1.json`, `_r2.json` | the same sweeps under `--eager`, neither fused nor compiled | NVIDIA RTX 6000 Ada (`meta.device_name`) | the fused-over-eager ratio |
| `measurements/phase_timing/phin_gan_fused_phases.json`, `phin_gan_eager_phases.json` | single 1e6-event runs with per-phase stage timers and peak-memory reporting | NVIDIA RTX 6000 Ada (`meta.device_name`) | the per-phase table |
| `measurements/phase_timing/geant4/g4_phase_timing.csv` | GEANT4's wall split into `hIoni` step-length, continuous-loss and secondary phases plus "other", with the timer's cost per call | one CPU core (model not recorded; `patch/ProcessTimer.cc` names a Xeon Gold 6354 as the timer-calibration host, which is not proof of the run host) | the GEANT4 phase split |
| `measurements/gate_b/report.json` | the accepted gate B run: verdict, both arms' kernel modes, statistics and thresholds | NVIDIA RTX 6000 Ada (the wrapper refuses another card) | cited by `tests/benchmark/test_verify_physics_gate.py`, which transcribes the null and pins `GATE_B_THRESHOLDS == 2 * null` |
| `measurements/fused_mlp_sweep/sweep_r1.log`, `sweep_r2.log` | kernel-versus-eager parity plus the tile and `(BLOCK, num_warps)` sweep, twice | NVIDIA RTX 6000 Ada (first log line) | `_LAUNCH_CONFIG` in `src/physics/interfaces/fused_mlp.py` |
| `measurements/physics_losses/fig_E1_physics_losses.csv` | the physics-regularisation term of all four training arms against epoch | post-hoc scoring, hardware-independent | figure E.1 |
| `measurements/pull_truth_null/truth_vs_truth_pull.json` | the truth-versus-truth pull null: all 45 pairs of ten seeded 10k-event GEANT4 aluminium samples fitted at the figure's binning, with the band's summary | CPU scoring, hardware-independent | the tolerance band printed on figure 13 |
| `measurements/checkpoint_scan/<beam>/<run-id>.csv`, `seed1/`, `*_emd.csv` | every saved epoch of the iron and beryllium `phin_gan` runs scored against GEANT4 truth (KS, z-bias, tails, atoms) at two noise seeds, plus the energy-distance confirmation of each continuous shortlist | CPU scoring, hardware-independent | the iron and beryllium pins in `_PHIN_GAN_PINS` (`src/common/paths.py`) |
| `measurements/g4_material_constants_dump.txt` | a Geant4 11.2.0 `MATPROPS` printout at a 1 um production cut for aluminium, iron and beryllium | not a timing measurement | `tests/physics/test_material_constants_match_g4_dump.py` |

## `computation_time/`

`geant4_timing.pbs.sh` calls `geant4_timing_node.sh` by name, in the cluster directory the
pair is copied to, and that script substitutes each event count into
`geant4_macro_template.mac` and runs the application once per point on one core; the macro dumps the proton's process list, so every log proves only ionisation and
transportation are registered, and the timer wraps the whole macro, which is why the figure
fits `median(t/n)`. `phin_gan_timing.pbs.sh` writes the four GPU reports with `python -m
benchmark.runtime_profiling.profile_runtime` at `--n-steps 10250` over the event-count grid,
fused arm and `--eager` arm in one job on one card. Resubmit either wrapper to reproduce.

## `phase_timing/`

`phin_gan_phases.pbs.sh` runs that profiler at one event count with `--stage-timing-events`
set, fused then eager. The timers synchronise the device around every call, so they
re-attribute time rather than create it, as each JSON says inline in `stage_timing.note`.
Under `geant4/`, `install_on_cluster.sh` applies `patch/` — a `G4WrapperProcess` subclass and a
timestamp-counter timer — to the GEANT4 application, rebuilds it and copies `macro_template.mac`
and `phase_timing_node.sh` across; `phase_timing.pbs.sh` then calls that node script, which
alternates shim-off and shim-on arms so the shim's own cost is measured in one job.

## `gate_b/`

`gate_b_calibrate.pbs.sh` measures the eager-versus-eager null at gate B's pinned working
point, and the thresholds derived from it live in
`src/benchmark/runtime_profiling/verify_physics_gate.py`. `gate_b.pbs.sh` re-checks
fused-kernel parity, then runs the gate itself, which is
`python -m benchmark.runtime_profiling.verify_physics_gate --gate b --device cuda`.

## `fused_mlp_sweep/`

`sweep.pbs.sh` runs `python -m benchmark.runtime_profiling.test_fused_mlp --device cuda
--n 1048576 --sweep` twice per tile arm on one card: a full pass and a repeat that puts the
run-to-run spread on record. Read the two logs by their agreement, never by the `BEST`
line, which cannot see its own noise.

## `pull_truth_null/`

`python -m benchmark.system_vs_g4.pull_analysis.truth_null_band` streams the ten seeded
samples (`storage_paper/ResourceFiles/iid/100MeVProtonInAl_seed{42..51}.root`, ionisation-only
productions from April 2026 that predate the multiple-scattering stage), bins each once on
the first sample's range, fits the pull of every pair and writes this file. `summary` carries
the band: the largest pairwise `|mu|`, the pairwise `sigma` range, and the
leave-one-sample-out jackknife means with their standard errors; `pairs` carries every fit.

## `checkpoint_scan/`

`python -m benchmark.system_vs_g4.checkpoint_scan --stage <continuous|secondary> --run <run-id>`
under `PHINGAN_BEAM=proton_in_iron_100MeV` or `proton_in_beryllium_100MeV` wrote these
tables (see [../docs/benchmarks.md](../docs/benchmarks.md#choosing-a-checkpoint)): one row
per saved epoch with the two-sample KS statistic in the generator's scaled space, the
z-bias, the z-std ratio, P(z > 0.99) and P(z > 0.999) beside GEANT4's, the z == 0 and
z == 1 atoms, and the run's epoch-mean physics penalty where it was joined. `seed1/` is the
same scan at noise seed 1 (the continuous pick ranks on the two-seed KS mean); `*_emd.csv`
is fig 12's energy distance for the continuous shortlist against the truth-versus-truth
null on a hundred 10 000-row batch pairs; `cond_chi2/` is the secondary scan at two
million rows with the conditional chi-square of z given E and fig 11's panel metric, on
which the secondary picks rank (the KS-best late epochs carried a chi-square near 1.9 and
a visibly tinted fig 11 panel; the picks sit at 0.96-0.98). The picks — iron continuous
epoch 18899 and secondary 29699, beryllium continuous 37699 and secondary 56499 — are the
pins in `_PHIN_GAN_PINS`; the pull figure at those pins fits mu = -0.001 +- 0.002,
sigma = 0.995 +- 0.001 in iron and mu = +0.002 +- 0.001, sigma = 1.001 +- 0.001 in
beryllium (10k events, seed 20260807), and fig 12's KS p-values against the
truth-versus-truth null are 0.96 / 0.65 / 0.38 (iron) and 0.92 / 0.45 / 0.66 (beryllium)
for step length / continuous / secondaries.

## `physics_losses/`

`python -m benchmark.system_vs_g4.physics_losses --score` walks every saved checkpoint of
all four training arms and scores it with the penalty term the physics-informed trainer
adds to its generator loss, and scores epoch 0 from the untrained network rebuilt at the
run's seed (no run checkpoints before epoch 99); `--plot --csv <file>` writes this file
beside the figure, and `--import-csv <file>` reads it back into the caches so the figure
can be redrawn without the checkpoints.
