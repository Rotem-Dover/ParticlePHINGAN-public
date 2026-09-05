"""
Single source of truth for "which run is the pipeline modelling?"

`ACTIVE` is a `BeamConfig` chosen from `physics.definitions.beam.PRESETS` by
the `PHINGAN_BEAM` environment variable at import time. Unset selects the
proton-in-aluminum beam, so every default flow needs no environment setup; an
unknown value raises immediately rather than selecting physics silently. The
selection is frozen per process (read once, at import), and the deploy
scripts carry it into cluster jobs through the qsub env (PHINGAN_BEAM, next
to PHINGAN_RUN_ID) -- never through a shared file, so two jobs deployed at
once cannot race on a shared setting.

`PARTICLE`, `TARGET_MATERIAL`, `BEAM_ENERGY_eV`, `BEAM_ENERGY_TAG`,
`MIN_ENERGY_CUTOFF_eV`, `GRID_BOUNDS`, `GRID_ENERGY_MIN_eV`,
`GRID_STEP_LEN_MIN_m`, and `GRID_STEP_LEN_MAX_m` are re-exported as
module-level constants so that `common.config` and `common.paths` (and any
downstream code) can keep reading them directly without an import cycle.
"""
import os

from physics.definitions.beam import BeamConfig, PRESETS

_DEFAULT_BEAM = "proton_in_aluminum_100MeV"

ACTIVE_NAME: str = os.environ.get("PHINGAN_BEAM", "").strip() or _DEFAULT_BEAM
if ACTIVE_NAME not in PRESETS:
    raise RuntimeError(
        f"PHINGAN_BEAM={ACTIVE_NAME!r} is not a registered beam preset; "
        f"valid values: {sorted(PRESETS)}"
    )

ACTIVE: BeamConfig = PRESETS[ACTIVE_NAME]

PARTICLE             = ACTIVE.particle
TARGET_MATERIAL      = ACTIVE.material
BEAM_ENERGY_eV       = ACTIVE.energy_eV
BEAM_ENERGY_TAG      = ACTIVE.tag
MIN_ENERGY_CUTOFF_eV = ACTIVE.min_cutoff_eV
PLOT_RANGES          = ACTIVE.plot_ranges

GRID_BOUNDS          = ACTIVE.grid_bounds
GRID_ENERGY_MIN_eV   = ACTIVE.grid_bounds.energy_min_eV
GRID_STEP_LEN_MIN_m  = ACTIVE.grid_bounds.step_len_min_m
GRID_STEP_LEN_MAX_m  = ACTIVE.grid_bounds.step_len_max_m
