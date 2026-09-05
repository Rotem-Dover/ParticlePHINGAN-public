"""
Global configuration for the ParticlePHINGAN project: the WGAN-GP
generators' architecture constants and checkpoint contract, the WGAN-GP
training-schedule constants, the energy cutoff, device and torch.compile
selection, and a shared matplotlib plotting style.

The network constants in the architecture section are a CHECKPOINT
CONTRACT: `from_checkpoint` builds a bare `cls()` and loads a state dict
into it, so a constant here must match the shape the pinned checkpoint
(`common.paths`) was trained at -- changing one without retraining breaks
loading. The training-schedule constants below them configure
`physics/g4h_ionisation/generators/train_networks/`'s trainers directly and
carry no such contract -- they can be changed freely, at the cost of needing
a fresh training run.
"""
import os
import torch
from matplotlib import pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from common.paths import ON_CLUSTER
from common.run_context import (
    PARTICLE, TARGET_MATERIAL, BEAM_ENERGY_eV, BEAM_ENERGY_TAG, MIN_ENERGY_CUTOFF_eV,
    GRID_BOUNDS, GRID_ENERGY_MIN_eV, GRID_STEP_LEN_MIN_m, GRID_STEP_LEN_MAX_m,
)

# ------------------------------- data ------------------------------- #
# Per-run cutoff lives in common.run_context (scales with the active particle
# / beam). Re-exported here so `from common.config import MIN_ENERGY_CUTOFF`
# keeps working.
MIN_ENERGY_CUTOFF = MIN_ENERGY_CUTOFF_eV

# ------------------------------- Generator architecture ---------------- #
# `from_checkpoint` builds a bare `cls()`, so every constant in this section
# IS the architecture: changing one without retraining the corresponding
# checkpoint (`common.paths.CNT_CKPT` / `SEC_CKPT` / `NOPHYS_CNT_CKPT` /
# `NOPHYS_SEC_CKPT`) breaks `load_state_dict`.
#
# N_CONTINUOUS_FEATURES / N_SECONDARY_FEATURES: output width of each
# generator. The continuous generator emits `[e_cnt]`; the secondary
# generator emits `[e_sec]` alone (`common.enums.SecondaryFeats`) -- the
# primary's scattering angle is reconstructed from energy-momentum
# conservation (`physics.g4h_ionisation.kinematics`) rather than sampled, so
# the network does not need to emit it.
#
# N_SECONDARY_FEATURES is also the secondary trunk's width knob: the trunk is
# built `N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES` wide, so at one
# output column the multiplier IS the width. The secondary problem is
# conditioned on kinetic energy alone and spans one smooth PDF family, unlike
# the continuous net's two-dimensional domain and multiple straggling
# regimes, so a narrow trunk suffices.
N_CONTINUOUS_FEATURES = 1
N_SECONDARY_FEATURES = 1

# Per-stage conditioning-embedding and latent-noise widths.
N_EMBEDDING_CNT = 25
N_NOISE_CNT     = 25
N_EMBEDDING_SEC = 5
N_NOISE_SEC     = 5

# Hidden-width multipliers for each generator's trunk.
N_NEURON_MULTIPLIER_CNT = 50
N_NEURON_MULTIPLIER_SEC = 16

# The `gan` (no-physics ablation) preset's secondary trunk width
# (NoPhysSecondaryGenerator, NOPHYS_SEC_CKPT) -- a separate constant because
# the ablation checkpoint's shape need not track N_NEURON_MULTIPLIER_SEC.
N_NEURON_MULTIPLIER_SEC_NOPHYS = 32

# The secondary generator's terminal activation. Every kind maps a
# hidden-layer output through a mostly-[0, 1] map onto the scaled secondary
# target z:
#   "hardtanh"           -- Hardtanh(0, 1): a hard clamp.
#   "sigmoid"             -- a smooth logistic squash; cannot reach exact 0
#                            or 1.
#   "stretched_sigmoid"   -- StretchedSigmoid, widens the sigmoid's range by
#                            SECONDARY_STRETCH_EPS before clamping so the
#                            logistic tail can still reach 1 in eval.
#   "hardtanh_open_top"   -- clamps the lower bound at 0 during training but
#                            leaves the top open; eval clamps the top to 1.
#   "hardtanh_reflect_top" -- same training rule as hardtanh_open_top, but
#                            eval reflects spill above 1 back down (z -> 2 -
#                            z) instead of clamping it, so mass that would
#                            pile on a z == 1 atom spreads into the tail
#                            instead. A checkpoint trained under
#                            "hardtanh_open_top" loads under this kind (see
#                            the compatibility note in
#                            secondary_generator/neural_net.py).
# All kinds share the same state-dict shape, so this constant (together with
# the state dict's own `output_activation_id` buffer, which `load_state_dict`
# checks for a mismatch) is what tells `from_checkpoint` which map to build.
# Must be flipped together with `common.paths.SEC_CKPT`.
SECONDARY_OUTPUT_ACTIVATION = "hardtanh_reflect_top"

# Stretch of the "stretched_sigmoid" terminal: z = (1 + 2*eps) * sigmoid(x) -
# eps, clamped to [0, 1] at inference (see StretchedSigmoid in
# secondary_generator/neural_net.py). Read, and recorded in the checkpoint's
# state dict, only when SECONDARY_OUTPUT_ACTIVATION == "stretched_sigmoid";
# every other kind records 0.0 and ignores it.
SECONDARY_STRETCH_EPS = 0.01

# Per-stage WGAN-GP critic embedding widths. Training-only: the critic is
# discarded after training and plays no part in the generator checkpoint
# contract.
N_EMBEDDING_DISC_CNT = 25
N_EMBEDDING_DISC_SEC = 5

# ------------------------------- Training (WGAN-GP) ----------------------- #
# Hyperparameters for `physics/g4h_ionisation/generators/train_networks/`.
N_BATCHES  = 64            # events per training batch.
MAX_EPOCHS = 100_000       # trainer stop condition; a run is walked forward
                           # with --resume rather than raising this number.
RESUME_FROM_CHECKPOINT = False   # set True (or use PHINGAN_RESUME) to resume
                                  # a run instead of starting fresh.
LAMBDA_GP      = 1e-2      # WGAN-GP gradient-penalty weight.
LEARNING_RATE  = 1e-5 if not RESUME_FROM_CHECKPOINT else 1e-6  # dropped 10x
                           # on resume, since a resumed run restarts from a
                           # trained state rather than from scratch.
LAMBDA_PHYSICS_SECONDARY = 1e-1   # weight of the physics (KS-test-vs-CDF)
                                  # regularization term in SecondaryGAN's
                                  # generator loss.
LAMBDA_PHYSICS_CONTINUOUS = 1e-1  # same, for ContinuousGAN.
# Gates the physics (KS-test-vs-CDF) regularization term in SecondaryGAN's /
# ContinuousGAN's generator loss.
USE_PHYSICS = True
SIMULATE_ON_EPOCH_END = True   # run a validation simulation pass at the end
                                # of every epoch (diagnostic figures/logging).
EVERY_N_EPOCHS = 200            # cadence, in epochs, of that validation pass.

# torch.compile mode for the runtime physics-phase compilation on CUDA
# (G4hIonisation / TrackSimulator geometry). "default" is the safe inductor
# mode; "reduce-overhead" additionally wraps the compiled subgraphs in CUDA
# graphs (cudagraph trees re-record per concrete shape — cheap — while the
# compiled code is shared), attacking the per-step kernel-launch floor.
# Env-overridable so a profiling job can A/B the modes without a source edit:
# PHINGAN_COMPILE_MODE=reduce-overhead python ...
TORCH_COMPILE_MODE = os.environ.get("PHINGAN_COMPILE_MODE", "default").strip() or "default"

# # ------------------------------- physics ------------------------------- #
# PARTICLE / TARGET_MATERIAL / BEAM_ENERGY_* live in common.run_context;
# re-exported here so `from common.config import ...` keeps working.
__all__ = [
    "PARTICLE", "TARGET_MATERIAL",
    "BEAM_ENERGY_eV", "BEAM_ENERGY_TAG",
    "MIN_ENERGY_CUTOFF", "MIN_ENERGY_CUTOFF_eV",
    "GRID_BOUNDS", "GRID_ENERGY_MIN_eV", "GRID_STEP_LEN_MIN_m", "GRID_STEP_LEN_MAX_m",
]


# DEVICE = 'cuda' if ON_CLUSTER else 'cpu'
if ON_CLUSTER:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEVICE = 'cpu'


def apply_project_plotting_style():
    # --- Nature Communications Styling (High Visibility) ---
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'axes.labelsize': 24,
        'legend.fontsize': 20,
        'legend.frameon': False,
        # 'figure.dpi': 300
    })
