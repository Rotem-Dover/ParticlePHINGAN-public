# Physics

Where every number in the simulation comes from: constants, the GEANT4 tables,
the straggling models, and the artifacts built from them.

## One constant set, one active beam

`src/physics/definitions/physical_constants.py` is the only set of fundamental
constants in the repository. Its values match CLHEP/Geant4 11.2.0 rather than
current CODATA, deliberately: the electron radius (via `alpha` and `hbarc`) and
the atomic mass both enter the ionisation cross-section directly, so a mismatch
in either is a systematic offset in every cross-section computed from them.
`alpha` and `hbarc` are derived from the same SI inputs CLHEP uses, so
`electron_radius_cm` is the one CLHEP computes.

`src/physics/definitions/beam.py` bundles a run into one `BeamConfig`:
particle, material, beam energy, filename tag, tracking cutoff, plot ranges and
the `GridBounds` of the phase-space grid the CDFs and lookup tables are built
over. Three are registered in `PRESETS`, all 100 MeV protons at a 0.3 MeV
tracking cutoff:

| preset | material | `energy_min_eV` | `step_len_min_m` | `step_len_max_m` |
|---|---|---|---|---|
| `proton_in_aluminum_100MeV` | aluminium | 0.4e6 | 1e-11 | 1e-4 |
| `proton_in_iron_100MeV` | iron | 0.4e6 | 1e-11 | 3e-4 |
| `proton_in_beryllium_100MeV` | beryllium | 0.4e6 | 1e-11 | 2e-4 |

Iron and beryllium need a wider step-length ceiling: their GEANT4 truth reaches
longer steps than aluminium's would admit. `src/common/run_context.py` resolves the `PHINGAN_BEAM` environment variable
against that registry once, at import: unset selects the aluminium beam, an
unknown name raises `RuntimeError` rather than selecting physics silently, and
the selection is frozen for the process. `src/common/paths.py` derives its whole
material subtree from it, so switching beams retargets every artifact path with
no code change. Import `PARTICLE` / `TARGET_MATERIAL` from `common.run_context`
(or the `common.config` re-exports), never `physics.definitions.particle`
directly. Adding a beam is one `BeamConfig` and one `PRESETS` entry.

## Material constants

`src/physics/definitions/material.py` holds a `Material` dataclass — the
delta-ray production threshold `Tc`, mean excitation energy `I`, atomic mass `A`,
atomic number `Z`, density `rho`, radiation length — plus derived
`number_of_atoms_per_volume` / `electron_density`. Every float of the **iron
and beryllium** instances is pasted verbatim from the Geant4 11.2.0
`InitTablesDump` MATPROPS printout committed at
`measurements/g4_material_constants_dump.txt`, and
`tests/physics/test_material_constants_match_g4_dump.py` re-parses that file to
pin both. **The dump tool's cut argument is in nanometres**:
`./InitTablesDump <outDir> 1000.0 G4_<Al|Fe|Be>` is the 1 µm production cut the
reference tables were produced with, while `1.0` is a 1 nm cut that clamps `Tc`
to Geant4's lowest tabulated threshold.

**Aluminium's floats are not the dump's** and are not pinned to it: its `Tc`,
`rho` and radiation length differ from the MATPROPS line at the ~1e-13 relative
level — floating-point noise between Geant4 builds, not a physics difference,
as the dump file notes beneath its MATPROPS block. What pins aluminium is
downstream: `tests/physics/g4_init_tables/test_vs_g4_reference.py` rebuilds the
four initialisation tables from these constants and compares them node by node,
at roughly 1 ULP, against binaries dumped straight out of Geant4's own
`G4PhysicsVector` objects (`common.paths.G4_REFERENCE_DIR`) — for all three
materials. A drifted constant moves a table and fails there.

`scripts/extract_g4_material_constants.py` extracts the Sternheimer
density-correction and stopping-power rows from a local Geant4 source tree.

## The four initialisation tables

`src/physics/g4_init_tables/` builds the tables GEANT4 computes at
initialisation, from the documented models rather than from a dump:

| table | file | model |
|---|---|---|
| dE/dx | `dedx.npz` | `models/bragg.py` below 2 MeV (PSTAR data for NIST-named materials, the ICRU49 analytic fit otherwise), `models/bethe_bloch.py` above, with Barkas/high-order (`models/em_corrections.py`), shell (`models/shell_correction.py`) and Sternheimer density corrections |
| CSDA range | `range.npz` | `build_range.py`: the running integral of 1/(dE/dx) over the dE/dx spline, 100 substeps per bin |
| inverse range | `scaled_kin_energy.npz` | the range table swapped, range [cm] -> energy [eV] |
| ionisation cross-section | `lambda_ion.npz` | `build_lambda_ion.py`: the Berger-Seltzer integral of dsigma/dT from `Tc` to `T_max`, times the electron density |

The grid matches the reference simulation: 10 eV to 10 TeV at
`G4_BINS_PER_DECADE = 10` (`grid.py`). Build them for the active beam with

```bash
cd src && PYTHONPATH=. python -m physics.g4_init_tables.build_all
```

which writes into `common.paths.INIT_TABLES_DIR` for the active `PHINGAN_BEAM`
(the builder takes no `--beam` argument).

At runtime `src/physics/g4_init_tables/runtime.py::InitTableSet` loads the four
`.npz` files into `G4PhysicsVector`s and is the single table source for
`G4hIonisation` and the length sampler. Interpolation is the **cubic spline
GEANT4 itself evaluates**: `src/physics/g4_init_tables/interpolator.py` reproduces
`G4PhysicsVector::Value` including not-a-knot second derivatives, is
branch-free so it never forces a device-host sync inside the step loop, and
evaluates in float64. Two subtleties are encoded once, in `InitTableSet`: the
ionisation vector splines sigma(E) with the zero node at the delta-ray threshold
as a knot (lambda = 1/sigma is recovered at evaluation), and `sigma_ion_biased`
applies the `G4_LAMBDA_FACTOR = 0.8` biased-denominator rule, pivoting on the
cross-section maximum.

## The analytical straggling models

`src/physics/g4h_ionisation/explicit_physics/` implements the energy-loss
distributions analytically, in NumPy, one `PointInPhaseSpace` at a time — what
the networks are fit against. Nothing on the per-step path calls it.

* **Continuous** (`continuous/straggling_function.py`) — GEANT4's
  modified-Urban fluctuation scheme. The loss splits into an excitation part
  and an ionisation part in a fixed ratio (`ionization_rate = 0.56`), each
  with a low- and a high-collision-frequency form selected against a
  threshold of 8 ionisation events; the two parts are convolved into the
  combined PDF. Excitation (`_excitation_straggling_function.py`) is a Poisson
  sum over excitation quanta at low frequency and a truncated Gaussian at high
  frequency. Ionisation (`_ionization_straggling_function.py`) inverts the
  compound-Poisson characteristic function by FFT, which needs three things
  together, none sufficient alone: a Lanczos sigma taper (`sinc(f/f_max)`),
  suppressing the Gibbs ringing that would otherwise leave a spurious density
  floor at unreachable energies; an `ifftshift`, because the frequency array is
  centred on zero while the FFT expects the zero-frequency sample at index 0;
  and the real part rather than the absolute value — skipping the shift while
  taking the real part turns the sign alternation into a comb (odd bins zero,
  even bins doubled), while the absolute value hides the alternation and makes
  the shift look inert. The regime is labelled by a `StragglingType`:
  `DeterministicBetheBloch`, `ThickLimit`, `LowColFreqExc`/`HighColFreqExc`,
  `LowColFreqIon`/`HighColFreqIon`.
* **Discrete** (`discrete/straggling_function.py`) — the energy transferred to
  a single knocked-out electron above the production threshold `Tc`; one regime,
  `SecondaryGeneration`. `set_state` asserts `t_max >= Tc`, a real boundary:
  below a material-dependent primary energy there is no phase space for
  delta-ray production at all.
* **Step length** (`step_length/straggling_function.py`) — `min(L_discrete,
  L_eloss)`, with `L_discrete` an exponential variate of the macroscopic
  cross-section and `L_eloss` the deterministic limiter built from
  `DR_OVER_RANGE = 0.1` and `FINAL_RANGE_CM = 0.01`. Its labels are a
  `StepLengthType`: `Deterministic` (`t_max <= Tc`, no discrete interactions)
  and `TruncatedExponential`.

`GeantConstants` (in `continuous/straggling_function.py`) carries the model's
scalars — the 10 eV outer-shell ionisation threshold `e0`, the 8-collision
threshold, `linear_loss_limit = 0.01`, `min_energy_loss = 10 eV` — and both
`G4hIonisation` and `interp_kernels` import it, so runtime and analytics agree.

## The continuous mean/std lookup

`ContinuousYScaler` z-scores `E_CNT` against a per-`(E, L)` mean and standard
deviation gathered from a 5000x5000x2 table, built analytically:

```bash
cd src && PYTHONPATH=. python -m \
  physics.g4h_ionisation.generators.continuous_generator.build_analytic_mean_std_table
```

The builder computes exact log10 moments of the analytical CDF on a 500x500
coarse grid (`n_jobs=-1`, all cores) and bilinearly upsamples to the fine grid,
then re-computes exact moments for the band of fine cells hugging the stopping
wall `L = range(E)`, where the surface kinks and bilinear interpolation would
cut the corner. Bilinear upsampling, unlike bicubic, cannot ring near the
boundary where the surface transitions to NaN. It is the most expensive build
here — long-running, so run it once per material, as a batch job — and lowering
`--n-coarse` to save time changes the z-space.

**The table is NaN over the deterministic region, by design.** Cells whose mean
loss falls below the model's minimum loss (or above the primary energy) are
stored as NaN rather than given a placeholder standard deviation, which would
bleed an arbitrary value into neighbouring stochastic cells through the
interpolation; geometrically that region is a contiguous small-`L` prefix of
every kinetic-energy row. Nothing normalises against it:
`ContinuousYScaler.lookup_is_nan(phase_space)` reports the gather's NaN mask per
lane and `G4hIonisation._along_step_core` ORs that into `is_deterministic`, so
the table's own mask is the authority. See [simulation.md](simulation.md).

`ContinuousYScaler.load()` does not read that pickle; convert it into the state
the scaler restores from — `ContinuousYScaler.state.pt`, under `SCALERS_DIR`:

```bash
PYTHONPATH=src python scripts/build_continuous_scaler_state.py
```

## The theoretical CDFs and the physics penalty

Each network stage has a `generate_cdfs.py` tabulating the analytical CDF over
that stage's conditioning grid, pickled as `Interp1dLinear` lookups under
`common.paths.CDFS_DIR`:

* `src/physics/g4h_ionisation/generators/continuous_generator/generate_cdfs.py` — keyed by
  `(step length, kinetic energy)` over a ~20 %-spaced log grid inside
  `GridBounds`, filtered to the convex hull of the GEANT4 truth so no key falls
  where there is no data to compare against.
* `src/physics/g4h_ionisation/generators/secondary_generator/generate_cdfs.py`
  — by kinetic energy alone, starting at `max(0.5e6, _heavy_min_ke(Tc * 1.01))`,
  the closed-form inverse of `Particle.t_max`: below that energy a proton cannot
  produce a delta ray in a material of that `Tc` at all, so the discrete model
  has no distribution to tabulate.

`src/physics/penalty_models.py::KSTestPenaltyModel` consumes those pickles:
`physics_penalty_loss` samples the generator at each CDF key, forms the
Kolmogorov-Smirnov statistic between the sampled and the theoretical CDF, and
returns it as an extra term in the generator's loss — the physics-informed half
of the method. It samples through `predict`, so it sees the runtime's scaling.
See [training.md](training.md).

## The zero-loss atom and the delta-ray endpoint

Two exact-value populations are handled explicitly rather than left to a
network. **`compute_zero_prob`** is a closed-form `G4UniversalFluctuation`
expression over material constants, no table lookup: the probability that a
step deposits exactly zero continuous energy.
`G4hIonisation.compute_zero_prob` calls `_compute_zero_prob_jit`
(`src/physics/g4h_ionisation/interp_kernels.py`) with that module's own
constants (`_MATERIAL_TC`, `_MATERIAL_I`, `_IONIZATION_RATE`, …). The networks are
trained on nonzero losses only and emit a continuous density, so the along-step
phase injects this atom itself, drawing `u < zero_prob` per lane — unless the
generator advertises `samples_zero_loss = True`, in which case a second
injection would double-count the atom and deplete the small-loss region.

**The `z == 0` atom of `SecondaryYScaler`.** That scaler is a log10 min-max
between `Tc` and the kinematic `t_max(E)`, so a network output of exactly zero
rescales to `10 ** log10(Tc)` — one repeated constant on every such lane,
independent of energy. `rescale_e_secondary` routes it around the exponential
with `torch.where(is_atom, self._Tc, ...)`, `_Tc` being `float32` of the active
material's `Tc` itself: exponentiating instead would make that value's last bit
depend on which kernel evaluated the power, displacing the whole atom between
an eager and a compiled arm. `SecondaryYScaler` carries no fitted state at all
— its bounds are physics, defined at class level.

## The straggling reference values

`tests/physics/test_straggling_reference_values.py` pins all three models
numerically. `tests/data/straggling_reference.npz` holds, for 121
(kinetic energy, step length) points, every scalar of the continuous model's
state, its regime label, and 512-point samples of the four distributions; the
test compares at `rtol=1e-12`, so drift in `explicit_physics`, the init tables
it reads, or the beam constants fails there. It is meaningful only for the
100 MeV proton-in-aluminium beam, which the capture script asserts. Regenerate:

```bash
PYTHONPATH=src python scripts/capture_straggling_reference.py \
    --out tests/data/straggling_reference.npz
```

The grid starts at 0.5 MeV, not the beam's 0.4 MeV floor: at 0.4 MeV the
discrete model's `t_max >= Tc` assertion fails for every step length, so the
grid is nudged above that genuine boundary rather than the assertion weakened.

## See also

- [simulation.md](simulation.md) — how the process consumes this per step
- [artifacts.md](artifacts.md) — where each table, CDF and scaler state lives
- [training.md](training.md) — the WGAN-GP training the CDFs regularise
- [benchmarks.md](benchmarks.md) — comparison against GEANT4 truth
