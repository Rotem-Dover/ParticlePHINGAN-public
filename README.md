# ParticlePHINGAN

Code accompanying the paper

> **Passage of particles through matter and the effective straggling-function: High-fidelity accelerated simulation via Physics-Informed Machine Learning**
> Oleksandr Borysov, Rotem Dover, Eilam Gross, Nilotpal Kakati and Noam Tal Hod
> Department of Particle Physics and Astrophysics, Weizmann Institute of Science. Under peer review.
> Corresponding author: Rotem Dover, rotem.dover@weizmann.ac.il

PHIN-GAN is a physics-informed generative emulator of GEANT4's stepping of
charged particles through matter. Two WGAN-GP generators, conditioned on the
particle's state, sample the per-step continuous energy loss and the delta-ray
energy that GEANT4 would have produced. The step length comes from an analytic
sampler of GEANT4's own stepping limits and the primary's deflection from
energy-momentum conservation. The beam is a 100 MeV proton in aluminium, iron
or beryllium, and ionisation is the only physics process. Everything runs on
tensors, batched across particles, so a whole beam steps together on one GPU.

The best way to explore the trained system is the **playground**, a local web
application that runs the emulator, overlays it on GEANT4 truth and on the
analytical straggling functions, and lets you switch every stage between its
neural generator and its physics Monte Carlo baseline. The rest of this README
gets you there; the scripts behind the paper's figures come after.

## Quick start

Python 3.12 and PyTorch 2.5.1. A CUDA GPU is optional for the playground and
the tests, and required for the timing figures.

```bash
git clone https://github.com/Rotem-Dover/ParticlePHINGAN-public.git
cd ParticlePHINGAN-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the data archive from Zenodo, https://doi.org/10.5281/zenodo.22386940, and unpack
it at the repository root. `phingan_runtime_artifacts.tar` holds the trained
checkpoints, scaler states and stepping tables for all three materials;
`phingan_geant4_truth_aluminum.tar` holds the aluminium GEANT4 truth the
comparisons are made against.

```bash
tar -xf phingan_runtime_artifacts.tar        # about 1 GB: everything the emulator needs
tar -xf phingan_geant4_truth_aluminum.tar    # about 10 GB: GEANT4 truth, for the comparisons
```

Then launch the playground and open http://127.0.0.1:5050 in a browser:

```bash
PYTHONPATH=src python -m playground.server
```

`PORT=5060` changes the port and `PHINGAN_BEAM=proton_in_iron_100MeV` (or
`proton_in_beryllium_100MeV`) starts on another material; the material can
also be switched from inside the app.

## The playground

The top bar shows the active beam and, for each of the three stages (step
length, continuous loss, delta-ray energy), a dropdown that picks either a
trained checkpoint or the physics Monte Carlo baseline. Every tab below uses
that mix.

- **G4 Browser + Generators.** A 2D phase-space histogram of the GEANT4 truth
  (kinetic energy against step length by default). Clicking pins a point or a
  bin; the right-hand panels then overlay, at that point, the neural
  generator's samples, the physics Monte Carlo sampler, the analytical
  straggling-function PDF and the GEANT4 rows inside the bin. This is the
  paper's Figure 9, interactive.
- **Trajectory.** Runs the emulator on a beam and renders the 3D tracks with a
  per-step feature summary.
- **Pull.** Runs the emulator and compares its energy-deposition map with the
  GEANT4 reference as a pull distribution, the paper's Figure 13.
- **Pairwise.** Picks two step features and shows emulator and GEANT4 as 2D
  histograms with their difference maps, the paper's Figure 11.
- **Table.** The raw per-step table of the most recent run, either the
  emulator's or the GEANT4 tracks it was compared with.
- **Cluster.** A browser for checkpoints on the authors' PBS cluster. It needs
  SSH access to that cluster and is of no use elsewhere.

Without the truth archive the Trajectory tab and the generator overlays still
work; the G4 histograms, Pull and Pairwise tabs need it.

## Running without the browser

The same emulator runs from the command line:

```bash
cd src && PYTHONPATH=. python -m particle_propagation.track_simulator
```

The acceptance gate against GEANT4 truth, the tests, and the script behind
each figure of the paper are documented in [docs/benchmarks.md](docs/benchmarks.md);
[CLAUDE.md](CLAUDE.md) has a one-page map from figure number to script.
Retraining a generator is covered in [docs/training.md](docs/training.md).
The recorded measurements the paper quotes, with the hardware each ran on,
are in [measurements/README.md](measurements/README.md).

```bash
pytest tests/
```

Without the data archive about 380 tests pass and about 70 fail or error,
every one of them on a missing file under `storage/`. With the runtime and
truth archives unpacked the whole suite passes on CPU.

## What is here and what is not

This repository is a snapshot of the code at the state the paper describes;
the tag `v1.0-paper` marks it. It holds the source, the tests, the
documentation and the recorded measurements behind the figures and gate
thresholds. The trained checkpoints, the GEANT4 truth samples, the training
datasets and the lookup tables live in the Zenodo archive above.
[docs/artifacts.md](docs/artifacts.md) lists every file the code reads, what
builds it and what reads it back, so the archive can also be rebuilt from
scratch. Iron and beryllium GEANT4 truth are available from the corresponding
author on request.

The cluster scripts and deployment helpers refer to the authors' own cluster
paths and hosts. They are kept for completeness and will need adapting to
another site.

## Documentation

* [docs/simulation.md](docs/simulation.md) — what is simulated and how to run a beam
* [docs/physics.md](docs/physics.md) — where every number comes from
* [docs/artifacts.md](docs/artifacts.md) — every file read or written
* [docs/training.md](docs/training.md) — datasets, the WGAN-GP loop, the physics penalty
* [docs/benchmarks.md](docs/benchmarks.md) — gates, profiling, the command behind every figure
* [docs/cluster.md](docs/cluster.md) — the PBS cluster workflow; site-specific

## Citing

See [CITATION.cff](CITATION.cff), or cite the paper above.

## License

MIT, see [LICENSE](LICENSE).
