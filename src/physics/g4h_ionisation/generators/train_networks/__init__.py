"""The WGAN-GP training stack for the continuous and secondary generators.

One subpackage per generator stage -- `continuous_generator/` and
`secondary_generator/` -- each holding the same four modules: `critic.py`,
`datamodule.py`, `wgan.py` (the `LightningModule`) and `train.py` (the
entrypoint). There is no `length_generator/` here: the length stage is the
analytic MC sampler `generators/length_generator/physics_mc.py`, not a
network.

Nothing trains automatically. No test or CI job invokes a `main()`, and each
stage's `train.main()` must be run explicitly, only after a deliberate
decision to book a GPU.

Everything here reads and writes `common.paths.DATASETS_DIR` /
`SCALERS_DIR` / `RUNS_DIR`.

Two pieces of the stack live outside this package because they are shared with
code that is not stage-specific: the `WGANInterface` base at
`physics/interfaces/wgan_interface.py`, alongside the other training/runtime
ABCs, and `physics/penalty_models.py`, used by both stages.
"""
