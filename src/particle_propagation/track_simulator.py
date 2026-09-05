"""
Full track simulator that propagates a beam of particles through the active
target material (see `common.run_context`) step by step. Uses
SteppingManager to produce per-step physics quantities, applies 3D rotation
from the particle frame to the lab frame, updates positions and energies, and
records the full trajectory. Provides pandas DataFrames of states and step
features for downstream analysis and visualisation.
"""
import torch
import numpy as np
import pandas as pd
from typing    import Optional, Callable
from functools import cached_property
from tqdm      import tqdm
from enum import IntEnum

from common.config import DEVICE, GRID_ENERGY_MIN_eV
from common.enums import G4Columns, StepFeaturesIdxes

# How often (in steps) the all-particles-dead early-exit is evaluated.
# The check reads a GPU boolean back to Python, i.e. a full device sync;
# doing it every step stalled the pipeline. Stepping an all-dead batch is a
# no-op on the state (dead lanes are masked), so a late exit only costs up
# to N-1 wasted steps at the very end of the run. The active-set
# re-compaction rides the same boundary sync.
_ALL_DEAD_CHECK_INTERVAL = 64

# Smallest re-compaction bucket. Below ~this width every kernel is at its
# launch-latency floor anyway, so smaller buckets buy nothing and just add
# distinct shapes.
_MIN_COMPACTION_BUCKET = 1024

from particle_propagation.stepping_manager import SteppingManager


class StateFeaturesIdxes(IntEnum):
    X  = 0
    Y  = 1
    Z  = 2
    E  = 3


class LabFrameStepsFeaturesIdxes(IntEnum):
    dX = 0
    dY = 1
    dZ = 2
    E_CNT = 3
    E_SEC = 4


def rotate_from_x_axis(d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply the Rodrigues rotation that maps x-hat onto `d` to the vector
    `u`, in closed form: with v = x-hat x d = (0, -dz, dy) and c = dx,

        R u = u + v x u + (v (v.u) - u |v|^2) / (1 + c)

    This is the elementwise application of the Rodrigues matrix
    R = I + [v]x + [v]x^2 / (1 + c), with the anti-parallel denominator
    clamped to 1 below 1e-6 (branch-free; d ~ -x-hat never occurs in
    stepping, where directions change by small deflections) and |v|^2 kept
    explicit (NOT 1 - c^2, which diverges from the clamped path at
    d ~ -x-hat). An independent matrix-form reference implementation lives
    in tests/particle_propagation/test_track_simulator.py and pins this
    function, including the clamp branch. Deliberately plain Python (no
    @torch.jit.script): scripted functions graph-break out of
    torch.compile, while this inlines and fuses into the caller's inductor
    graph with no B x 3 x 3 intermediates.
    """
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    denom = 1.0 + dx
    denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)

    v_dot_u = -dz * uy + dy * uz
    v_sq = dy * dy + dz * dz

    # v x u = (-dz*uz - dy*uy, dy*ux, dz*ux)
    out_x = ux + (-dz * uz - dy * uy) + (-ux * v_sq) / denom
    out_y = uy + dy * ux + (-dz * v_dot_u - uy * v_sq) / denom
    out_z = uz + dz * ux + (dy * v_dot_u - uz * v_sq) / denom

    return torch.stack([out_x, out_y, out_z], dim=1)


class TrackSimulator:

    def __init__(
            self,
            stepping_manager:   SteppingManager,
            device:           str  = DEVICE,
            dtype:            torch.dtype = torch.float32#torch.bfloat16
    ):
        self.stepping_manager = stepping_manager
        self.device = device
        self.dtype = dtype
        self._geometry_compiled = False

        self.to(device, dtype)

        self.particles_state:          Optional[torch.Tensor] = None  # 1 sample:          X, Y, Z, E
        self.states_along_propagation: Optional[torch.Tensor] = None  # n_steps+1 samples: Xs, Ys, Zs, Es
        self.steps_features:           Optional[torch.Tensor] = None  # n_steps samples:   L, theta, dE_cnt, dE_sec, geom_step
        self.delta_state:              Optional[torch.Tensor] = None  # n_steps samples:   dX, dY, dZ (lab frame), dE_cnt, dE_sec
        self.momentum_direction:       Optional[torch.Tensor] = None
        self.steps_info_lab_frame:     Optional[torch.Tensor] = None  # n_steps samples:   dXs, dYs, dZs, dE_cnt, dE_secs
        self.alive_particles:          Optional[torch.Tensor] = None  # n_events
        self.alive_counts_per_step:    Optional[torch.Tensor] = None  # executed steps: pre-step alive count (device tensor)
        # Overwritten per run() with max(E_kill_eV, grid floor); initialized
        # here so __generate_step never sees an unset attribute.
        self._dead_lane_sentinel_eV:   float = float(GRID_ENERGY_MIN_eV)
        # Window-compacted active index set (None -> full batch); managed by run().
        self._active_idx:              Optional[torch.Tensor] = None

    def to(self, device: str, dtype: torch.dtype = None) -> None:
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.stepping_manager.to(device, dtype=self.dtype)
        if device == 'cuda' and not self._geometry_compiled:
            # Fuse the per-step geometry (frame rotations, direction
            # normalization, delta assembly) the same way the process phases
            # are compiled. The rotation math is the plain-Python closed form
            # rotate_from_x_axis (no @torch.jit.script — scripted functions
            # would graph-break out of this region), so the whole geometry
            # step inlines into one inductor graph with no B x 3 x 3
            # intermediates. Bound-method shadow, see G4hIonisation.to().
            from common.config import TORCH_COMPILE_MODE
            self._get_delta_state_and_post_dir = torch.compile(
                self._get_delta_state_and_post_dir, dynamic=True, mode=TORCH_COMPILE_MODE)
            self._geometry_compiled = True

    # Every cached_property below stores its value in the instance __dict__;
    # run() must drop them or a reused simulator serves the previous run's
    # frames (playground preset fast-path, pull chunk loop).
    _FRAME_CACHE_ATTRS = (
        "states_and_lab_frame_steps_df",
        "states_and_steps_features_df",
        "states_along_propagation_df",
        "steps_df_lab_frame",
        "steps_features_df",
    )

    def _invalidate_frame_caches(self) -> None:
        for attr in self._FRAME_CACHE_ATTRS:
            self.__dict__.pop(attr, None)

    @cached_property
    def states_and_lab_frame_steps_df(self) -> pd.DataFrame:  # EventNum, StepNum, X, Y, Z, KineticEnergy, dX, dY, dZ, dE_cnt, dE_sec
        return pd.merge(
            self.states_along_propagation_df,
            self.steps_df_lab_frame,
            on=[G4Columns.EventNum, G4Columns.StepNum],
            how='inner'
        )

    @cached_property
    def states_and_steps_features_df(self) -> pd.DataFrame:  # EventNum, StepNum, X, Y, Z, KineticEnergy, L, theta, dE_cnt, dE_sec
        return pd.merge(
            self.states_along_propagation_df,
            self.steps_features_df,
            on=[G4Columns.EventNum, G4Columns.StepNum],
            how='inner'
        )

    @cached_property
    def states_along_propagation_df(self) -> pd.DataFrame:  # EventNum, StepNum, X, Y, Z, KineticEnergy
        states_arr = self.states_along_propagation.detach().cpu().float().numpy()
        n_events, n_steps_plus_one, n_features = states_arr.shape

        flat_states = states_arr.reshape(-1, n_features)
        event_nums = np.repeat(np.arange(n_events), n_steps_plus_one)
        step_nums = np.tile(np.arange(n_steps_plus_one), n_events)

        df = pd.DataFrame(flat_states, columns=[
            G4Columns.X, G4Columns.Y, G4Columns.Z,
            G4Columns.KineticEnergy
        ])
        df.insert(0, G4Columns.StepNum, step_nums)
        df.insert(0, G4Columns.EventNum, event_nums)

        return df

    @cached_property
    def steps_df_lab_frame(self) -> pd.DataFrame:  # EventNum, StepNum, dX, dY, dZ, dE_cnt, dE_sec
        return self.__steps_df(
            steps_info=self.steps_info_lab_frame,
            columns=["dX", "dY", "dZ",
                     G4Columns.ContinuousLoss, G4Columns.SecondaryLoss]
        )

    @cached_property
    def steps_features_df(self) -> pd.DataFrame:
        return self.__steps_df(
            steps_info=self.steps_features,
            columns=[G4Columns.StepLength, G4Columns.angleDiscrete,
                     G4Columns.ContinuousLoss, G4Columns.SecondaryLoss,
                     "geom_step_m"]
        )

    @staticmethod
    def __steps_df(steps_info: torch.Tensor, columns: list[str]) -> pd.DataFrame:
        steps_arr = steps_info.detach().cpu().float().numpy()
        n_events, n_steps, n_features = steps_arr.shape

        flat_steps = steps_arr.reshape(-1, n_features)
        event_nums = np.repeat(np.arange(n_events), n_steps)
        step_nums = np.tile(np.arange(n_steps), n_events)

        df = pd.DataFrame(flat_steps, columns=columns)
        df.insert(0, G4Columns.StepNum, step_nums)
        df.insert(0, G4Columns.EventNum, event_nums)

        df[G4Columns.ContinuousLoss] = df[G4Columns.ContinuousLoss].abs()
        df[G4Columns.SecondaryLoss] = df[G4Columns.SecondaryLoss].abs()
        
        return df

    @torch.inference_mode()
    def run(self, E_0: float, n_events: int, n_steps: int,
            save_steps_states: bool = False, E_kill_eV: float = 1e3,
            progress_cb: Optional[Callable[[int, int], None]] = None) -> None:
        """Propagate `n_events` particles for up to `n_steps`. A particle is
        considered alive while its kinetic energy exceeds `E_kill_eV` (default
        1 keV — matches the order of magnitude of G4's proton tracking
        cutoff). Below that, the proton is treated as stopped."""
        """
        Simulates steps and chain them one after another chronologically.
        """
        self._invalidate_frame_caches()
        # Clear the processes' device-side diagnostic latches so this run
        # cannot inherit a previous run's NaN escape (and read them once,
        # after the loop below).
        self.stepping_manager.reset_nan_guard()
        self.alive_particles = torch.ones(n_events, dtype=torch.bool, device=self.device)
        x_0 = self.__initial_features_states(E_0, n_events)
        current_energy = x_0[:, StateFeaturesIdxes.E]

        self.momentum_direction = torch.zeros((n_events, 3), dtype=x_0.dtype, device=self.device)
        self.momentum_direction[:, StateFeaturesIdxes.X] = 1.0

        if save_steps_states:
            self.__init_states_along_propagation(n_steps, x_0)
            self.__init_steps_info(n_steps, n_events)
        else:
            self.particles_state      = x_0
            self.steps_info_lab_frame = None
            self.steps_features       = None

        # Dead lanes keep being stepped at a fixed batch shape (their outputs
        # are discarded); they are fed this in-domain sentinel energy so
        # log-space scalers and physics tables never see E <= 0.
        self._dead_lane_sentinel_eV = max(float(E_kill_eV), float(GRID_ENERGY_MIN_eV))

        # Pre-step alive counts, recorded on-device (no per-step sync); a
        # single transfer at the end of the run reads the whole decay curve.
        self.alive_counts_per_step = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        executed_steps = 0

        # Windowed re-compaction: the physics processes step only the lanes
        # in _active_idx (alive lanes plus dead-lane padding up to a
        # power-of-two bucket). The index set is rebuilt once per
        # _ALL_DEAD_CHECK_INTERVAL window, at the boundary where the
        # all-dead check already syncs the device — so the compaction's
        # host round-trip is amortized 1/N instead of paid per step, and
        # the power-of-two buckets keep the set of distinct batch shapes
        # bounded (no per-shape JIT re-specialization churn).
        self._active_idx = None  # None -> full batch

        iterable = range(1, n_steps + 1)
        if n_steps > 1:
            iterable = tqdm(iterable, desc="Simulating steps")

        # CUDA-graph trees (torch.compile mode="reduce-overhead") reuse
        # static output buffers across replays; marking each loop iteration
        # as a new "step" declares the previous iteration's graph outputs
        # dead — true here: everything that persists across steps lives in
        # particles_state / momentum_direction / freshly-allocated tensors.
        # No-op under the default compile mode.
        _mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None) \
            if self.device == 'cuda' else None

        for i in iterable:
            if _mark_step is not None:
                _mark_step()
            self.alive_counts_per_step[i - 1] = self.alive_particles.sum()
            step_features = self.__generate_step(current_energy)

            # momentum_direction is always a fresh eager tensor (built by the
            # torch.where below, never a compiled-graph output buffer), so it
            # is safe to feed into the compiled geometry without a clone —
            # CUDA-graph replay copies inputs into its static buffers.
            self.delta_state, post_step_momentum_dir = self._get_delta_state_and_post_dir(
                step_features, self.momentum_direction
            )

            self._update_particles_state()

            # Update momentum direction for alive particles only.
            # torch.where instead of boolean-index assignment: `t[mask] = ...`
            # runs aten::nonzero, a GPU->CPU sync, every step.
            self.momentum_direction = torch.where(
                self.alive_particles.unsqueeze(1), post_step_momentum_dir, self.momentum_direction
            )

            if save_steps_states:
                self._chain_step_to_track(i)
                self.__update_steps_info(i - 1, step_features)

            self.alive_particles = self.particles_state[:, StateFeaturesIdxes.E] > E_kill_eV

            # Update current_energy for the next iteration
            current_energy = self.particles_state[:, StateFeaturesIdxes.E]

            if progress_cb is not None:
                progress_cb(i, n_steps)

            executed_steps = i
            # Window boundary: one device sync serves both the all-dead
            # early exit and the active-set re-compaction.
            if i % _ALL_DEAD_CHECK_INTERVAL == 0 or i == n_steps:
                n_alive = int(self.alive_particles.sum())
                if n_alive == 0:
                    if progress_cb is not None:
                        progress_cb(n_steps, n_steps)
                    break
                if i < n_steps:
                    self._active_idx = self._build_active_idx(n_alive, n_events)

        self.alive_counts_per_step = self.alive_counts_per_step[:executed_steps]

        # The single host read of the along-step NaN latch, for the whole run.
        # Outside the loop and outside every compiled region by construction:
        # one sync per run instead of one per step, on any device. It comes
        # last so a run that raises here has still left its outputs intact for
        # inspection.
        self.stepping_manager.check_nan_guard()

    def _build_active_idx(self, n_alive: int, n_events: int) -> Optional[torch.Tensor]:
        """Alive lanes first (stable original order), padded with dead lanes
        up to the next power-of-two bucket (capped at n_events). Padding uses
        real dead lanes — their indices are disjoint from the alive ones, so
        the per-step index_copy_ scatter never has duplicate targets, and
        their outputs are zeroed by the same valid-mask machinery as the
        fixed-shape path. Returns None for a full-width bucket (fast path)."""
        bucket = 1 << max(0, (n_alive - 1)).bit_length()
        bucket = max(bucket, _MIN_COMPACTION_BUCKET)
        if bucket >= n_events:
            return None
        order = torch.argsort((~self.alive_particles).to(torch.int8), stable=True)
        return order[:bucket]

    def __initial_features_states(self, E_0: float, n_events: int) -> torch.Tensor:
        """
        Initial position and energy of particles
        """
        # Features (lab frame): 'X', 'Y', 'Z', 'kineticEnergy'
        x_0 = torch.zeros(
            (n_events, len(StateFeaturesIdxes)), device=self.device, dtype=self.dtype, requires_grad=False
        )
        x_0[:, StateFeaturesIdxes.E]  = E_0
        return x_0

    def __init_states_along_propagation(self, n_steps: int, x_0: torch.Tensor) -> None:
        # del self.states_along_propagation

        n_events         = x_0.shape[0]
        n_state_features = len(StateFeaturesIdxes)

        self.states_along_propagation = torch.full(
            fill_value    = float("nan"),
            size          = (n_events, n_steps + 1, n_state_features),
            device        = self.device,
            dtype         = self.dtype,
            requires_grad = False,
        )
        self.states_along_propagation[:, 0, :] = x_0
        self.particles_state = x_0

    def __init_steps_info(
            self,
            n_steps:  int,
            n_events: int,
    ) -> None:
        # del self.lab_frame_steps_info

        n_steps_features = len(LabFrameStepsFeaturesIdxes)
        n_step_features  = len(StepFeaturesIdxes)

        self.steps_info_lab_frame = torch.full(
            fill_value    = float("nan"),
            size          = (n_events, n_steps, n_steps_features),
            device        = self.device,
            dtype         = self.dtype,
            requires_grad = False,
        )

        self.steps_features = torch.full(
            fill_value=float("nan"),
            size=(n_events, n_steps, n_step_features),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )

    def __generate_step(self, current_energy: torch.Tensor) -> torch.Tensor:
        """
        Generate a step and return step features.

        Args:
            current_energy (torch.Tensor): Current kinetic energy of particles.

        Returns:
            torch.Tensor: Step step length (L), deflection angle (theta),
             continuous energy loss (dE_CONTINUOUS), secondary energy (dE_SECONDARY).
        """
        # Fixed-shape stepping: the whole batch (dead lanes included) goes
        # through the stepping manager every step, and dead lanes' outputs
        # are discarded. A per-step compaction `x[alive]` would run
        # aten::nonzero (a GPU->CPU host sync) once per step; compaction is
        # done instead only at the _ALL_DEAD_CHECK_INTERVAL-step window
        # boundary (_build_active_idx), which already pays a sync for the
        # all-dead check. Fixed shapes also keep the graph static for
        # torch.compile / CUDA graphs.
        idx = self._active_idx
        if idx is None:
            # Full-width bucket: step everything, mask dead lanes.
            valid = self.alive_particles
            energy = current_energy
        else:
            # Compacted bucket: gather by the window's index set. index_select
            # with a precomputed index tensor is sync-free (unlike boolean
            # masking, whose data-dependent output shape runs aten::nonzero).
            # `valid` is re-gathered from the CURRENT alive mask so lanes that
            # died mid-window are masked immediately, not at the boundary.
            valid = self.alive_particles.index_select(0, idx)
            energy = current_energy.index_select(0, idx)

        # Dead lanes are stepped at an in-domain sentinel energy so log-space
        # scalers and spline tables stay in-bounds (E can reach 0 at the end
        # of a track; log10(0) would inject inf/NaN into their lanes).
        sentinel = torch.full_like(energy, self._dead_lane_sentinel_eV)
        safe_energy = torch.where(valid, energy, sentinel)

        features = self.stepping_manager(safe_energy)

        # torch.where, NOT multiplication by the mask: masked-out lanes may
        # legitimately carry inf/NaN intermediates and 0 * nan == nan.
        features = torch.where(valid.unsqueeze(1), features, torch.zeros_like(features))

        if idx is None:
            return features
        # Scatter back to full width. Zeros for non-active lanes reproduce
        # the fixed-shape semantics (their delta is a state no-op); idx has
        # no duplicate targets (alive lanes + disjoint dead-lane padding).
        out = torch.zeros((current_energy.shape[0], features.shape[1]),
                          dtype=features.dtype, device=features.device)
        out.index_copy_(0, idx, features)
        return out

    def _get_delta_state_and_post_dir(
            self, step_features: torch.Tensor, pre_step_pX_pY_pZ: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract state difference based on step features and calculate post-step momentum direction.
        The geometric path length is encoded in step_features by the
        SteppingManager; since nothing shortens the true step into a shorter
        geometric one, it always equals the true step length L.

        The particle-frame displacement is [geom_step, 0, 0] and the frame
        rotation R is DEFINED by R x-hat = pre_step_dir, so the lab
        displacement is simply geom_step * pre_step_dir — no rotation matrix
        is ever materialized. The post-step direction is one
        rotate-from-x-axis apply (particle frame -> lab frame), a closed-form
        elementwise expression that inductor fuses; rotations are orthogonal,
        so a single normalize at the end suffices in place of a per-frame
        normalize at each stage.
        """
        post_dir_particle_frame = self._get_post_dir_particle_frame(step_features)

        # int(): FX codegen inside the compiled geometry graph cannot
        # serialize IntEnum indices (see G4hIonisation.post_step_do_it).
        geom_step = step_features[:, int(StepFeaturesIdxes.GEOM_STEP)]
        dEs = step_features[:,
              [int(StepFeaturesIdxes.E_CNT), int(StepFeaturesIdxes.E_SEC)]
              ]

        # transform displacement: R @ [g, 0, 0]^T = g * (R x-hat) = g * pre_dir
        dX_dY_dZ_lab_frame = geom_step.unsqueeze(1) * pre_step_pX_pY_pZ
        # transform post-step direction:
        post_dir_lab_frame = rotate_from_x_axis(pre_step_pX_pY_pZ, post_dir_particle_frame)

        # normalize to be safe (branch-free: boolean-index division would
        # sync via aten::nonzero every step; zero-norm rows fall back to the
        # pre-step direction, since a zero-length post-step direction carries
        # no information to rotate toward)
        post_dir_lab_frame = torch.nan_to_num(post_dir_lab_frame, nan=0.0)
        norm = torch.linalg.norm(post_dir_lab_frame, dim=1, keepdim=True)
        post_dir_lab_frame = torch.where(
            norm > 0,
            post_dir_lab_frame / torch.where(norm > 0, norm, torch.ones_like(norm)),
            pre_step_pX_pY_pZ,
        )

        # restack distance changes and energy changes:
        delta_state_lab_frame = torch.cat((dX_dY_dZ_lab_frame, dEs), dim=1)

        return delta_state_lab_frame, post_dir_lab_frame

    @staticmethod
    def _get_post_dir_particle_frame(step_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the (un-normalized) post-step direction in the particle frame.

        There is no multiple-scattering rotation to compose here: the
        post-step particle-frame direction is the discrete deflection
        `theta` about a uniformly drawn azimuth, `(cos t, sin t cos phi,
        sin t sin phi)` with `phi ~ U(-pi, pi)` drawn as `2*pi*rand - pi`.
        Normalization happens once, in the lab frame.
        """
        n = step_features.shape[0]
        dtype = step_features.dtype
        device = step_features.device

        # int(): FX codegen cannot serialize IntEnum indices in the compiled
        # geometry graph.
        theta = step_features[:, int(StepFeaturesIdxes.THETA)]

        phi = 2.0 * torch.pi * torch.rand(n, dtype=dtype, device=device) - torch.pi
        sin_theta = torch.sin(theta)
        return torch.stack([
            torch.cos(theta),
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
        ], dim=1)

    def _update_particles_state(self):
        # StateFeaturesIdxes X..Z and LabFrameStepsFeaturesIdxes dX..dZ are
        # both contiguous prefixes — sliced views instead of list fancy
        # indexing (which dispatches index_put_ with an index tensor per step).
        self.particles_state[:, int(StateFeaturesIdxes.X):int(StateFeaturesIdxes.Z) + 1] += \
            self.delta_state[:, int(LabFrameStepsFeaturesIdxes.dX):int(LabFrameStepsFeaturesIdxes.dZ) + 1]

        self.particles_state[:, int(StateFeaturesIdxes.E)] = \
            self.particles_state[:, int(StateFeaturesIdxes.E)] - \
            self.delta_state[:, int(LabFrameStepsFeaturesIdxes.E_CNT)] - \
            self.delta_state[:, int(LabFrameStepsFeaturesIdxes.E_SEC)]

    def _chain_step_to_track(self, i) -> None:
        self.states_along_propagation[self.alive_particles, i, :] = self.particles_state[self.alive_particles]

    def __update_steps_info(self, i, step_features: torch.Tensor) -> None:
        self.steps_info_lab_frame[self.alive_particles, i, :] = self.delta_state[self.alive_particles]
        self.steps_features[self.alive_particles, i, :]       = step_features[self.alive_particles]


if __name__ == "__main__":
    import time

    from common.config import DEVICE

    from benchmark.track_simulator_config import build_track_simulator

    device = 'cpu'

    simulator = build_track_simulator("phin_gan", device=device)

    E_0_, N_events, N_steps = 99.99e6, int(1e3), 10_250
    print(f"\nSimulating {N_events} events with {N_steps} steps each...")
    t_0 = time.time()
    simulator.run(E_0=E_0_, n_events=N_events, n_steps=N_steps, save_steps_states=False)
    print(f"\tSimulation took {time.time() - t_0:.2f} seconds to complete.")
    print("Done!")
