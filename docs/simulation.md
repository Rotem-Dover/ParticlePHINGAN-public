# Simulation

How a beam of particles is stepped through matter: the layers, the per-step
quantities, and how to run one.

## What is simulated

The pipeline transports 100 MeV protons through an unbounded homogeneous
medium of one material — aluminium, iron or beryllium, set by the `PHINGAN_BEAM`
environment variable (`src/physics/definitions/beam.py`). Ionisation is the
only physics process: a proton loses energy continuously along each step and,
at the end of some steps, ejects a delta ray (a secondary electron) and is
deflected. Each step therefore carries three sampled quantities — the step
length `L`, the continuous energy loss `E_CNT`, and the secondary energy loss
`E_SEC` — plus the primary's scattering angle `THETA`, which is computed from
`E_SEC` rather than sampled. Two of the three are produced by WGAN-GP
generators trained on GEANT4 truth; the step length comes from an analytic
Monte Carlo sampler of GEANT4's own stepping limits.

## The three layers

```
TrackSimulator            src/particle_propagation/track_simulator.py
  owns lab-frame state (X, Y, Z, kinetic energy) and the alive mask;
  per step calls the stepping manager, rotates the particle-frame result
  into the lab frame, updates positions and energies
        |
        v
SteppingManager           src/particle_propagation/stepping_manager.py
  runs each physics process's three GEANT4-mirroring phases and stacks
  the result into one per-step feature tensor
        |
        v
G4hIonisation             src/physics/g4h_ionisation/g4h_ionisation.py
  the one process: three generators (length / continuous / secondary)
  plus the built GEANT4 tables (InitTableSet)
```

The frame rotation is a closed-form Rodrigues apply,
`track_simulator.rotate_from_x_axis`: the rotation is defined by mapping
x-hat onto the pre-step direction, so the lab displacement is exactly
`geom_step * pre_step_dir` and no 3x3 matrix is materialised.

## Phases and the step-feature layout

`SteppingManager.__call__` takes a kinetic-energy tensor and runs, in order:

1. **Step limit.** Every process proposes a step length via
   `compute_step_limit`; the minimum over processes wins. When two or more
   processes compete for the discrete vertex, the winner masks are computed
   here and passed to their post-step phase as `won`; with a single
   competitor the mask is `None`, meaning always eligible.
2. **Along step.** `along_step_do_it` accumulates the continuous energy loss
   `e_cnt` over the chosen step length.
3. **Post step.** `post_step_do_it` accumulates the discrete secondary energy
   loss `e_sec` and the primary's deflection `theta`.

The return is one `(n_events, 5)` tensor whose columns are fixed by
`StepFeaturesIdxes` (`src/common/enums.py`):

| index | name | meaning |
|---|---|---|
| 0 | `L` | step length [m] |
| 1 | `THETA` | primary scattering angle [rad] |
| 2 | `E_CNT` | continuous energy loss [eV] |
| 3 | `E_SEC` | secondary energy loss [eV] |
| 4 | `GEOM_STEP` | geometric step length [m] |

That layout is a contract — the generators, the geometry transform and the
output frames all index it positionally. `GEOM_STEP` is a value copy of `L`
(nothing shortens a true step into a shorter geometric one), kept as its own
column so the geometry code can read it unconditionally.

## The generators

Each stage implements `GeneratorInterface`
(`src/physics/interfaces/generator_interface.py`). `predict()` is wrapped by
`@scaled_forward`, which scales the conditioning inputs, runs `forward()` in
scaled space and rescales the output, so every call site works in physical
units.

* **Length** — `PhysicsLengthGenerator`
  (`src/physics/g4h_ionisation/generators/length_generator/physics_mc.py`), an
  analytic Monte Carlo of GEANT4's two stepping limits: the deterministic
  energy-loss limit built from the range table with `dRoverRange = 0.1` and a
  100 µm final range, and the discrete limit `-log(u) / sigma` drawn against
  the biased ionisation cross-section. The minimum of the two is the step. It
  overrides `predict()` to bypass scaling — it already works in physical
  units. There is no trained length network.
* **Continuous** — `ContinuousGenerator`
  (`.../continuous_generator/neural_net.py`), a WGAN-GP generator conditioned
  on `(kinetic energy, step length)` and emitting `E_CNT`. Its trunk width is
  `N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES`; the embedding and latent
  widths are `N_EMBEDDING_CNT` and `N_NOISE_CNT` (`src/common/config.py`).
* **Secondary** — `SecondaryGenerator`
  (`.../secondary_generator/neural_net.py`), a WGAN-GP generator conditioned
  on kinetic energy alone and emitting `E_SEC` as its single column
  (`SecondaryFeats`). Its terminal activation is selected by
  `SECONDARY_OUTPUT_ACTIVATION` and recorded in the checkpoint as an
  `output_activation_id` buffer, so a checkpoint refuses to load into a net
  built with a different terminal.
* **Theta** — not sampled. `compute_primary_theta`
  (`src/physics/g4h_ionisation/kinematics.py`) reconstructs it from
  energy-momentum conservation once `e_sec` is known: the delta ray's momentum
  and emission angle follow from its kinetic energy, and the primary's
  residual transverse and longitudinal momenta give
  `theta = atan2(p1_perp, p1_z)`. `atan2` rather than `acos(p1_z / |p1|)`:
  at this beam's deflections the cosine differs from 1.0 by less than one
  float32 ulp, so `acos` returns exactly 0 and the distribution collapses.
  `atan2`'s two arguments carry no such cancellation.

`post_step_do_it` computes theta **last**, from the final `e_sec`. Every
masking above it — the Bernoulli secondary-generation draw, the
negative-energy guard, the `won` mask — sets `e_sec` to zero, and a zero
`e_sec` makes `theta` exactly zero on its own, which is why there are no
separate theta masks.

## The deterministic regime

Below a threshold the GEANT4 fluctuation model returns a mean loss with no
fluctuation at all, and the network must not be consulted. `_along_step_core`
handles this with a chain of `torch.where` over disjoint masks rather than
boolean indexing (a `tensor[mask]` gather runs `aten::nonzero`, a device-host
sync, in the hot loop):

```python
e_cnt = torch.where(is_deterministic, average_loss.to(e_cnt.dtype), e_cnt)
```

`is_deterministic` is the OR of two predicates, excluding lanes that deposit
their whole energy: this process's own `average_loss < min_energy_loss`, and
the continuous Y-scaler's `lookup_is_nan(continuous_input)`, which reports
whether the mean/std lookup table stores NaN for that cell. The table's
builder writes NaN over the region it calls deterministic, and its lookup is a
nearest-cell gather with no interpolation, so a lane can sit exactly where the
table's cell is NaN as well while `average_loss` has already crossed the
threshold. ORing the table's own mask in means the two predicates can never
disagree, and the NaN never reaches the network or the guard below.

## The NaN guard

`along_step_do_it` is a thin, uncompiled wrapper around `_along_step_core`. It
ORs `torch.isnan(e_cnt).any()` into a device-resident 0-d bool latch:

* **Accumulate unconditionally.** The detection runs on every device, in both
  the eager and compiled arms, with no tracing or device condition. `|=` into a
  device tensor is a pure device-side op.
* **Read once per run.** `check_nan_guard()` performs the single host read and
  raises a `RuntimeError` if the latch is set — an explicit `raise`, not an
  `assert`, since asserts are stripped under `python -O` and this is the
  mechanism's only read site. `TrackSimulator.run()` calls
  `reset_nan_guard()` before its step loop and `check_nan_guard()` after it,
  both through `SteppingManager`'s fan-out methods. A caller driving
  `along_step_do_it` directly owns those two calls itself.
* **Accumulate outside the compiled core.** Only
  `compute_step_limit`, `_along_step_core` and `post_step_do_it` are compiled
  on CUDA. Folding the `.any()` reduction into the core lets the compiler fuse
  it with the whole phase body into one kernel whose launch geometry is baked
  from the shape that first triggered compilation. Outside, the phase math
  fuses into ordinary pointwise kernels and the guard costs one extra small
  reduction launch per step — not a synchronisation.

The latch is lossless: a bad lane at any step raises at the end of the run, at
the cost of the message naming the run rather than the step.

## Fixed-shape stepping

The whole batch, dead lanes included, goes through the stepping manager every
step; dead lanes' outputs are discarded by a `torch.where` against the alive
mask (not a multiplication — a masked lane may legitimately carry NaN, and
`0 * nan == nan`). Dead lanes are stepped at a sentinel energy,
`max(E_kill_eV, GRID_ENERGY_MIN_eV)`, so log-space scalers and spline tables
never see a non-positive energy.

Compaction happens on a window, not per step. Every
`_ALL_DEAD_CHECK_INTERVAL = 64` steps one device sync serves both the all-dead
early exit and a rebuild of the active index set: alive lanes first, padded
with dead lanes up to the next power-of-two bucket (floor
`_MIN_COMPACTION_BUCKET = 1024`), returning `None` for a full-width bucket.
Power-of-two buckets keep the set of distinct batch shapes bounded, so
`torch.compile` does not re-specialise per shape.

A particle is alive while its kinetic energy exceeds `run()`'s `E_kill_eV`
(default 1e3 eV). `alive_counts_per_step` records the pre-step alive count
on-device and is transferred once, at the end of the run.

## Presets and how to run one

`src/benchmark/track_simulator_config.py` holds the two presets, pinned by
explicit checkpoint path (`src/common/paths.py`):

* **`phin_gan`** — the physics-informed configuration: analytic-MC length,
  `ContinuousGenerator` (`CNT_CKPT`), `SecondaryGenerator` (`SEC_CKPT`), all
  scalers loaded from `SCALERS_DIR`.
* **`gan`** — the no-physics ablation: the same pipeline and length stage, with the `NoPhys*`
  generators (`NOPHYS_CNT_CKPT`, `NOPHYS_SEC_CKPT`) — `BatchNorm1d` in place of `LayerNorm`, a
  `1.2 * sigmoid(x) - 0.1` readout on the continuous net, a bare `Linear` readout with `e_sec` rectified by `ReLU` on the secondary net, and its own scaler state `NOPHYS_CNT_SCALERS_DIR`.

Any other name raises `KeyError`. `PHINGAN_COMPILE_MODE` picks the `torch.compile` mode for
CUDA physics-phase compilation (`TORCH_COMPILE_MODE`), default `"default"`.

```python
from benchmark.track_simulator_config import build_track_simulator

sim = build_track_simulator("phin_gan", device="cpu")
sim.run(E_0=100e6, n_events=4, n_steps=6, save_steps_states=True)
print(sim.steps_features_df.head())
```

```
   EventNum  StepNum    StepLength  ...  ContinuousLoss  SecondaryLoss   geom_step_m
0         0        0  1.471613e-05  ...    13909.576172    3524.371826  1.471613e-05
1         0        1  9.179720e-07  ...     1939.020508    7919.918945  9.179720e-07
```

`build_stepping_manager(name, device)` returns just the manager, for stepping
a batch of energies without the track geometry. Both entrypoints also run as
scripts: `python -m particle_propagation.track_simulator` and
`python -m particle_propagation.stepping_manager`.

## Outputs

`run(save_steps_states=True)` fills device tensors that are exposed as cached
pandas frames:

| frame | columns |
|---|---|
| `states_along_propagation_df` | `EventNum`, `StepNum`, `X`, `Y`, `Z`, `KineticEnergy` |
| `steps_features_df` | `EventNum`, `StepNum`, `StepLength`, `angleDiscrete`, `ContinuousLoss`, `SecondaryLoss`, `geom_step_m` |
| `steps_df_lab_frame` | `EventNum`, `StepNum`, `dX`, `dY`, `dZ`, `ContinuousLoss`, `SecondaryLoss` |
| `states_and_steps_features_df` | the first two merged on `(EventNum, StepNum)` |
| `states_and_lab_frame_steps_df` | the first and third merged the same way |

Each is a `cached_property`, and `run()` drops all five caches on entry, so
re-running one simulator instance never serves the previous run's frames.

**Dead particles' rows are NaN-padded.** The state and step tensors are
allocated full of NaN and written only for lanes alive at that step, so a
track that stops at step 300 of 10,000 leaves NaN in every later row. Drop
`~isnan(KineticEnergy)` before any step-count statistic, or the padding is
counted as physics.

## See also

- [physics.md](physics.md) — the constants, tables and straggling models the
  process is built from
- [artifacts.md](artifacts.md) — where checkpoints, tables and scaler states live
- [training.md](training.md) — how the two generators are trained
- [benchmarks.md](benchmarks.md) — comparing the simulator against GEANT4 truth
