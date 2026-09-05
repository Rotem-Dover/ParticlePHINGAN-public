# Benchmarks

How the simulator is judged: the two acceptance gates, the fused-kernel and
profiling instruments, and the command behind every figure in the paper.

Everything here lives under `src/benchmark/`. Gate T gates the physics, gate B gates
optimisations, `secondary_endpoint_atoms` gates one net's boundary behaviour; the
profiling instruments only report. Recorded results sit under `measurements/`
([index](../measurements/README.md)); every number below names the file it came from
and the hardware it ran on.

## Gate T — the simulator against GEANT4 truth

Gate T is the acceptance instrument: it runs the `phin_gan` preset against the
beam's GEANT4 truth ROOT file at 500 events x 12000 steps (`DEFAULT_N_EVENTS` /
`DEFAULT_N_STEPS`, chosen for a KS resolution near 1e-3), printing JSON and exiting
nonzero on a breach. A pass states: *the simulator differs from GEANT4 by no more
than GEANT4 differs from itself at this sample size.*

```bash
cd src && PYTHONPATH=. python -m benchmark.verify_truth_parity [--device cuda]
```

The statistics (`src/benchmark/gate_t_stats.py`) are a two-sample KS on `L`,
`THETA`, `E_CNT` and `E_SEC`, plus a 1-D energy distance on `log10(E_CNT)`. `E_SEC`
and `THETA` carry a large exact-zero Dirac atom, so each is split into KS on the
nonzero continuum plus the atom's mass gated separately: KS on a distribution with
an atom is all-or-nothing, and a spike displaced by one percent reads near 1 without
being a mis-fit. Columns map **by name, per frame**, asserted at import: the two
frames have different widths and neither is the raw `StepFeaturesIdxes` tensor, so
positional indexing would compare metres against electron-volts. Termination matches
the reference — the truth rows were filtered at `MIN_ENERGY_CUTOFF`, so the
simulated arm runs with `E_kill_eV=MIN_ENERGY_CUTOFF` rather than the runtime
default, which would add short end-of-track steps the truth never had a chance to
record. Zero simulated rows, or fewer than `_MIN_ROWS = 1000` on either side, raises
rather than passing vacuously.

Thresholds are twice a **truth-vs-truth** null, measured by `python -m
benchmark.gate_t_null --root <truth.root> --n-splits 6 --seed 0 --per-side-n
4901455`: the truth file is split into two independent halves — a random
permutation, never a positional slice — and the same statistics measured across
them, worst of six splits. The result is pasted into
`src/benchmark/verify_truth_parity.py` as `MEASURED_NULL`, the thresholds derived
from it in code, so the calibration stays auditable beside what it sets. A KS
statistic's scale is set by the per-side row count, so the null must be measured at
the count gate T compares at; `--per-side-n` therefore defaults to
`MEASURED_NULL["per_side_n"]` rather than carrying a second literal that could
drift. **Never widen a threshold to turn a red gate green** — a threshold at or
below its own null fails on sampling scatter rather than on physics, as
`tests/benchmark/test_verify_truth_parity.py` pins. A red gate is the finding.

## Gate B — optimised against eager

Gate B answers a different question: did the Triton fused MLP or the compiled phases
shift the physics? Both arms come from
`src/benchmark/runtime_profiling/simulator_builders.py` —
`build_optimized_simulator` (fused latch on, phases compiled on CUDA) against
`build_eager_simulator`. It is **eager-anchored**: only the optimised-minus-eager
delta gates, the absolutes are reported. On CPU both builders produce the same
program, so every delta must be exactly 0.0. Add `--calibrate` for the
eager-vs-eager null.

```bash
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.verify_physics_gate --gate b --device cuda
```

Three gated statistics on the `L` / `THETA` / `E_CNT` / `E_SEC` columns:
`e2e_max_ks` (worst per-feature KS, atom features split as in gate T),
`atom_mass_delta` (worst zero-atom mass shift — this *is* the secondary-generation
probability), and `pull_max_abs_mean_shift` (worst mean shift in units of its own
standard error, which is what lets one threshold cover metres, radians and
electron-volts). It is distributional by design: a 1e-6 arithmetic shift flips an
alive-mask bit and that lane's trajectory diverges, so a seeded-exactness check has
no place here.

* **The working point is pinned.** `GATE_B_CALIBRATION_RUN_SIZE` is 200 events x
  12000 steps on CUDA. The pull statistic is n-dependent, so a null does not
  transfer between sizes: `run_gate_b` raises on a gating run at any other size on
  the calibration device, while `gate_b_null` only reports the mismatch, since
  recalibrating is what produces the constant.
* **A malformed thresholds dict is refused** before any simulator is built, so a
  missing key cannot ungate a statistic while the report prints `"passed": true`.
* **Vacuity is refused on CUDA.** `try_enable_fused_forward` only logs a warning
  when it declines to fuse, which would make the optimised arm identical to the
  eager one and every statistic read as perfect agreement. The gate records both
  arms' actual per-stage kernel modes and returns `passed=False` with a `vacuous`
  reason string when the optimised arm fused nothing.

`measurements/gate_b/report.json` (RTX 6000 Ada, job wrappers `gate_b.pbs.sh` and
`gate_b_calibrate.pbs.sh` beside it) is the accepted run: `passed: true`, `vacuous:
null`, both networks fused in the optimised arm and neither in the eager arm, with
`e2e.max_ks` 0.0013518865639752242 against 0.0036624316467523954,
`max_atom_mass_delta` 0.0011226479501783848 against 0.0015070887011403578, and
`pull.max_abs_mean_shift` 0.917616343264138 against 5.670558654001141.

## The fused MLP

`src/physics/interfaces/fused_mlp.py` runs a whole generator forward pass in one
Triton kernel: activations stay in registers for the entire layer chain, and
main-memory traffic collapses to one read of the features and one write of the
output per lane. It is inference-only, opt-in via `PHINGAN_FUSED_MLP=1` on CUDA, and
checkpoint-compatible — weights are packed from the loaded module. Both trained
generators fuse; the analytic length sampler is simply absent from
`_FUSABLE_CLASS_NAMES`. That set is a **name** check and proves nothing about an
architecture: only `build_spec`'s structural validation distinguishes the nets, and
it refuses — logs, falls back to the eager forward — on any mismatch. It reads four
things off the module: the single activation kind (mixed activations are refused,
never approximated), the `(tile_a, tile_b)` pair drawn from `_TILE_LADDER = (16, 32,
64, 128)`, an 8-bits-per-layer `ln_w_mask` holding each LayerNorm's raw
normalisation width, and the readout width with its terminal epilogue. `_MIN_TILE`,
the per-class floor overriding the ladder's choice, ships empty; `_LAUNCH_CONFIG`
holds the elected launch geometry:

| net | tile pair | `(BLOCK, num_warps)` |
|---|---|---|
| `ContinuousGenerator` | (32, 64) | (64, 8) |
| `SecondaryGenerator` | (16, 16) | (128, 2) |

```bash
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.test_fused_mlp --device cuda --n 1048576 \
    [--sweep --sweep-blocks 16,32,64,128,256 --sweep-warps 2,4,8,16]
```

The harness checks the kernel against the eager forward and the pure-torch emulator, then optionally
sweeps `(BLOCK, num_warps)`, forcing the incumbent config into whatever grid is given so a candidate
has a within-job baseline. On an RTX 6000 Ada (`measurements/fused_mlp_sweep/sweep_r1.log`,
`sweep_r2.log`, job wrapper `sweep.pbs.sh`) the secondary net's parity came to `max|dz| 1.073e-06`
against a `1e-3` tolerance (the `max|dz|` line is the parity result — both logs also end in a
`FUSED-MLP PARITY FAILED:` banner, but that comes from two harness defects `test_fused_mlp.py` no
longer has, a `(B,)` vs `(B, 1)` shape mismatch and a clamp probe that could not saturate, not a
kernel mismatch), and the elected `BLOCK=128, num_warps=2` measured **0.134 ms** per launch at
1,048,576 lanes (0.135 ms on the repeat pass) against **0.518 ms** for the `(32, 16)` anchor config
in the same arm; the same log puts that net at 10.337 ms eager and 1.363 ms compiled, i.e. 19.06x
over eager and 2.51x over the compiled forward. Read a sweep by **repeat agreement, not the `BEST`
line** — a minimum over one pass cannot see its own noise — and ship a change only behind gate B:
`num_warps` reorders the floating-point reductions inside the matmuls and LayerNorms, so a
launch-config change is not bit-neutral.

**TF32 is off.** `PHINGAN_FUSED_MLP_TF32` routes the dots through the tensor cores
(`"1"` for all stages, a `':'`-separated list to scope it) and stays unset: on the
continuous net it buys negligible throughput at real accuracy risk, and on the
secondary net it speeds the phase up and passes gate B but pushes `THETA`'s KS
further from its calibrated null than IEEE does. The per-generator mode is recorded
as `_fused_mlp_tf32`, so a report cannot misstate which arm ran.

**The kernel is verified on a GPU or not at all.** The Triton kernel body and
`_emulate_packed_forward` are two independent implementations of the same arithmetic
and only the emulator runs in CI, so a kernel-only edit — moving a `_LAUNCH_CONFIG`
entry included — is caught by nothing local. Treat any change to
`src/physics/interfaces/_fused_mlp_kernel.py` as unverified until the sweep harness
and gate B have both re-run on a card, and keep the emulator in lockstep.

## Profiling and compiled parity

```bash
cd src && PHINGAN_FUSED_MLP=1 PYTHONPATH=. python -m \
    benchmark.runtime_profiling.profile_runtime --preset phin_gan --device cuda \
    --label fused --n-steps 10250 --event-counts 1000000 \
    --stage-timing-events 1000000 --torch-profile-events 0 --out fused.json
```

`profile_runtime` writes one structured JSON per invocation: beam and device
metadata, a per-stage generator report recording the class that *actually* runs and
its actual kernel mode, `torch._dynamo` counters (zero compiled graphs means
`torch.compile` never ran), the scaling sweep, and optionally per-phase stage timing
and a torch-profiler trace. `--eager` drops the compile and fusion latches; it names
no checkpoint, delegating to the preset builder and asserting every stage came back
with one. `python -m benchmark.runtime_profiling.verify_compiled_parity` runs one
seed through both builds and reports the worst step-feature delta; its `--tol`
defaults to `None`, i.e. report-only, because an uncalibrated numeric default is
indistinguishable from a threshold someone quietly loosened, so it gates only on an
explicit `--tol` backed by a measured null.

## The secondary net's boundary atoms

`cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.secondary_endpoint_atoms`
(with optional `--seed` and `--tol`) gates one property of the secondary generator:
how much mass it puts *exactly* at `z == 0` (`e_sec == Tc`) and `z == 1` (`e_sec ==
t_max(E)`), against GEANT4 truth on the same conditioning energies. It exits nonzero
when either generator atom exceeds `ATOM_TOLERANCE_X = 10` times GEANT4's, with an
absolute floor (`ATOM_ABS_FLOOR = 1e-6`) so a GEANT4 zero does not demand a
generator zero. A terminal with zero gradient outside `[0, 1]` cannot have mass
pulled back across the clamp during training, so mass accumulates there, and a `z ==
1` atom shows up downstream as a low-`THETA` shelf: at the kinematic endpoint the
deflection goes to zero. See [physics.md](physics.md) for the endpoint,
[training.md](training.md) for the selectable terminals.

## Choosing a checkpoint

`benchmark.system_vs_g4.checkpoint_scan` scores every saved epoch of one
training run against GEANT4 truth on the truth's own conditioning rows, in
the generator's scaled space (where truth is `y_scaler.scale(Y, X)`), with
the same scalers, rows and noise seed for every checkpoint:

```bash
cd src && PHINGAN_BEAM=proton_in_iron_100MeV PYTHONPATH=. python -m \
    benchmark.system_vs_g4.checkpoint_scan --stage continuous --run <run-id> [--seed 1]
cd src && PHINGAN_BEAM=proton_in_iron_100MeV PYTHONPATH=. python -m \
    benchmark.system_vs_g4.checkpoint_scan --stage continuous --run <run-id> \
    --emd-epochs 18899,25699,31499
```

The scan writes `artifacts/benchmark_outputs/ckpt_scan/<beam>/<run-id>.csv`
(and a `.png`): per epoch the two-sample KS statistic, the z-bias, the
z-std ratio, `cond_chi2` (a reduced chi-square of the generator against
truth on a 20 x 40 grid of log E by z, about 1 for a truth-like generator),
fig 11's per-bin relative difference on the stage's (E, loss) panel beside
the truth-vs-truth null at the same statistics, the tail fractions
P(z > 0.99) and P(z > 0.999) beside GEANT4's, the boundary atoms at z == 0
and z == 1, and the run's physics-penalty scalar when a `scalars_by_epoch.csv`
sits beside the checkpoints. Runs are five to fifteen minutes for a thousand
checkpoints on a laptop CPU at one to two million rows. Rank the secondary
stage on `cond_chi2`: KS averages away the few-percent ripples in the
delta-ray spectrum that fig 11 exposes as a tinted panel, and late epochs of
the referee-material runs had the best KS with a chi-square near 1.9 while
mid-run epochs sat at 1.0. Among the chi-square-flat window keep the
checkpoints whose two tail fractions sit within roughly ±10 % of GEANT4's
and prefer the smaller z == 0 atom, then run `secondary_endpoint_atoms` on
the pin. For the continuous stage rank on KS averaged over two seeds and
confirm the shortlist with `--emd-epochs`, which computes fig 12's energy
distance against the truth-vs-truth null on a hundred 10 000-row batch pairs
(the z-bias alone is white noise across 100-epoch saves).

## The figures

`cd src && PYTHONPATH=. python -m build_figure_datasets` writes the six pickled
arrays the phase-space figures read — raw step-length pairs plus filtered continuous
and secondary pairs — under `DATASETS_DIR` with a provenance manifest beside them;
`src/benchmark/system_vs_g4/net_sampling.py` is the shared sampling surface over a
preset's ionisation process. Figures write into `BENCHMARKS_OUTPUTS`
(`artifacts/benchmark_outputs/`) unless the command says otherwise; commands run
from `src/` with `PYTHONPATH=.`, and the script column is relative to
`src/benchmark/system_vs_g4/`.

Every figure follows `PHINGAN_BEAM`. For the referee materials
(`PHINGAN_BEAM=proton_in_iron_100MeV`, `proton_in_beryllium_100MeV`) the same
commands write under `artifacts/benchmark_outputs/<material>/`, read that
material's truth, datasets, scalers and pins, and draw two arms — GEANT4 and
PHIN-GAN — because the no-physics ablation was trained for aluminium only
(`has_gan_preset()` in `benchmark/track_simulator_config.py`; Fig E.1 defaults
to its two PHIN-GAN arms, Fig E.2 has no counterpart). Fig 9 samples the
material's own four phase-space points, from
`benchmark/system_vs_g4/marked_points.py`: aluminium's coordinates leave the
explicit-physics models' domain in iron (no delta rays below about 2.5 MeV,
where Tc exceeds t_max) and at (10 MeV, 0.5 um) in beryllium. Fig 8 for these
materials (an appendix figure) draws no sample points and clips its energy
axis and KS grid at the lowest energy the truth populates. For every beam, Fig 8
draws a KS point only inside the GEANT4 step-length envelope: in the background
histogram, its L must lie between the shortest and longest step the truth took in
its energy column (one column either side). The CDF grid is gated by a convex hull
in linear (E, L), which bridges the dip in the step-limit ceiling at low energy and
admits keys the truth never reaches (67 of 1265 in aluminium); an envelope rather
than a per-bin occupancy test keeps the sparse very-thin fringe, and the script
prints how many points it drops. Fig 13's
tolerance band is the aluminium truth-versus-truth null and is printed for
the aluminium beam only; no band has been measured for the other materials,
so their pull panels show the fitted mu_0 / sigma_0 alone (`--no-band` does
the same for aluminium).

| fig | script | command | output |
|---|---|---|---|
| 8 | `phase_space_ks.py` | `python -m benchmark.system_vs_g4.phase_space_ks` | `phase_space_with_markers.pdf` |
| 9 | `energy_losses_distributions_at_marked_points.py` | `python -m benchmark.system_vs_g4.energy_losses_distributions_at_marked_points` | `energy_loss_histograms_comparison.pdf` |
| 10 | `features_hist_comparison.py` | `python -m benchmark.system_vs_g4.features_hist_comparison` | `1d_hists.pdf` |
| 11 | `pairwise_comparison.py` | `python -m benchmark.system_vs_g4.pairwise_comparison` | `pairwise_comparison.pdf` |
| 12 | `energy_distance/energy_distance_analysis.py` | `python -m benchmark.system_vs_g4.energy_distance.energy_distance_analysis --score`, then `--plot` | `energy_distance.pdf` |
| 13 | `pull_analysis/run_pull_figure.py` | `python -m benchmark.system_vs_g4.pull_analysis.run_pull_figure` | `energy_deposition_pull.pdf` |
| 14 | `plot_computation_time.py` | see below | `computation_time.pdf` |
| E.1 | `physics_losses.py` | `python -m benchmark.system_vs_g4.physics_losses --score --device cpu`, then `--plot [--mark-picks]` | `fig_E1_physics_losses.pdf` / `.png` |
| E.2 | `pull_analysis/run_pull_figure.py` | `python -m benchmark.system_vs_g4.pull_analysis.run_pull_figure --preset gan` | `energy_deposition_pull_gan.pdf` |

**Fig 12** caches its nine energy-distance arrays under `--score`, stamped with the
four checkpoint pins so a re-pin invalidates the cache loudly instead of mixing
runs; every arm is scaled through the `phin_gan` preset's scalers, so the distance
lives in one fixed space. **Figs 13 and E.2** are one figure on two presets — same
truth, seed and event count — at the paper's binning: a 1500x1500 grid, bins
averaging more than three trajectories, a `+-5` pull axis. The tolerance band
Fig 13 prints under the fitted `mu_0` / `sigma_0` (Fig E.2 shows the fit alone)
is the truth-versus-truth null:
`python -m benchmark.system_vs_g4.pull_analysis.truth_null_band` pulls every
pair of ten independently seeded 10k-event GEANT4 samples at that same binning
and records the 45 fits in `measurements/pull_truth_null/truth_vs_truth_pull.json`;
the figure reads the band from that file (`--band-json`) and refuses one recorded
at another binning. **Fig E.1** scores all
four arms' checkpoints post hoc with `KSTestPenaltyModel.physics_penalty_loss`,
since the no-physics trainer never computes that term; its curves are committed as
`measurements/physics_losses/fig_E1_physics_losses.csv`. No run saves a
checkpoint before `epoch=99`, so the epoch-0 point of each curve is the
untrained network: the trainers call `seed_everything(seed)` and construct the
generator right after, so seeding with the run's seed (the pinned secondary
run's `meta.json` records 2, the other three ran at the default 42) and calling
the bare constructor rebuilds the weights the run started from. The value is
seed-dependent (an order of magnitude of scatter across seeds for the secondary
arms), which is why the run's own seed is wired per arm rather than a
convenient one. `--import-csv` seeds the caches from the committed CSV so the
epoch-0 points can be added, and the figure redrawn, without the checkpoints.

### Figure 14 and its numbers

```bash
cd src && PYTHONPATH=. python -m benchmark.system_vs_g4.plot_computation_time \
    ../measurements/computation_time/phin_gan_gpu_fused_r1.json \
    ../measurements/computation_time/phin_gan_gpu_fused_r2.json \
    --g4-csv ../measurements/computation_time/geant4_cpu_times.csv \
    --cpu-label "Intel Xeon Gold 5320"
```

The GEANT4 arm, `measurements/computation_time/geant4_cpu_times.csv`, is five
`n_events,seconds` points from a one-core, ionisation-only, saving-disabled sweep
(`geant4_timing.pbs.sh` and its node script beside it). The script drops every
point below `--min-events` (default 100) from both arms before plotting or fitting:
at 10 primaries each arm measures its fixed start-up cost, not its stepping rate.
The fit is `median(t/n)` over the surviving points, not least squares: the timer
wraps the whole macro, so `/run/initialize` sits inside every point and the low-`n`
ones are overhead-dominated. That median is **11.46 ms/event**, extrapolating to
11,464 s at 1e6 primaries. Several GPU reports count as repeats of one measurement,
so that curve is their mean and the per-point spread is printed.

The GPU arm is the four `phin_gan_gpu_*.json` reports in the same directory, all on
one RTX 6000 Ada (`meta.device_name`) at 10,250 steps. At 1e6 events
`phin_gan_gpu_fused_r1.json` / `_r2.json` read **35.4515 s** and **35.5310 s**;
`phin_gan_gpu_eager_r1.json` / `_r2.json` read **405.2859 s** and **405.4382 s**.
Against the 11,464 s GEANT4 fit that is **323.0x** for the fused arm (mean 35.49 s)
and 28.3x for the eager one; fused over eager is **11.42x**. The GPU curve is flat
to 1e5 — 6.3 to 7.0 s at every point there, not monotone — then bends at 1e6, since
`steps_executed` is about 10,000 at every event count: the plateau is the fixed step
loop, and cheaper steps push the knee right.

### Where the time goes

In `measurements/phase_timing/`, `phin_gan_fused_phases.json` and
`phin_gan_eager_phases.json` are single-point 1e6-event runs on one card, timers on:

| phase | fused | share | eager | ratio |
|---|---|---|---|---|
| `along_step_do_it` (continuous net) | 17.1972 s | 67.8 % | 285.3677 s | 16.6x |
| `post_step_do_it` (secondary net) | 5.0676 s | 20.0 % | 106.9858 s | 21.1x |
| `compute_step_limit` | 3.1020 s | 12.2 % | 6.6467 s | 2.1x |
| attributed total | 25.3667 s | 100 % | 399.0001 s | 15.7x |

Peak CUDA memory over the same pair is 402.7 MB fused against 1278.1 MB eager, a
3.17x reduction. The continuous net is the phase that matters: removing the
secondary net entirely would buy 1.25x. `measurements/phase_timing/geant4/` splits
the GEANT4 arm the same way, with a `G4WrapperProcess` timing shim around `hIoni` —
verified as a pure forwarder by bit-identical ntuples. Its `g4_phase_timing.csv`
runs one core, alternating shim-off and shim-on arms at 1e4 plus one at 1e5; that
1e5 row reads 75.6764 s step length, 257.3934 s continuous, 213.9668 s secondary and
719.2464 s other against a 1266.2830 s wall, over ~9.8e8 calls per phase at a
recorded timer cost of 15.7 ns/call. Subtracting that cost puts `hIoni` at **40.3%**
of the wall (GEANT4's framework, transportation and tracking are the other 59.7%),
split **9 / 50 / 41%** across step length, continuous loss and secondary generation,
against the fused pipeline's 12 / 68 / 20%.

### Reading a timing number

* **Per-phase timers synchronise CUDA around every call**, so they re-attribute
  time rather than create it. Quote the un-instrumented sweep wall (35.49 s at
  1e6) and take *shares* and *ratios* from the phase table; never sum the buckets
  and call the sum a wall. Each JSON says so inline, as `stage_timing.note`.
* **Only within-job ratios are trustworthy.** Job-to-job offsets are real and one
  phase can inflate while the others match, so every arm of a comparison belongs
  in one job on one card — hence the alternating arms above. CPU rates carry a
  contention band on top of that: nodes are shared, exclusive placement is not
  schedulable here, and a single-core rate both moves between jobs and drifts
  *within* a long one. Quote the conservative end and say so; shares inside one
  run are immune, since everything slows together.
* **Name the arm in every ratio.** Fusion and compilation are separate
  configuration boundaries; a ratio crossing both at once compares neither.

## See also

- [simulation.md](simulation.md) — the presets and pipeline being measured
- [physics.md](physics.md) — the models the truth and the tables come from
- [training.md](training.md) — the runs whose checkpoints these gates score
- [artifacts.md](artifacts.md) — where truth, datasets and checkpoints live
- [cluster.md](cluster.md) — submitting the jobs behind these measurements
