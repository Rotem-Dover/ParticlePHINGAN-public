"""
Tests for the TrackSimulator and its rotation-matrix helper. Covers coordinate-frame
rotations (particle frame to lab frame and back), state updates (position and energy),
track chaining, momentum-direction tracking, and conversion of simulation output to
a pandas DataFrame.
"""
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from particle_propagation.stepping_manager import SteppingManager
from particle_propagation.track_simulator import (
    rotate_from_x_axis, TrackSimulator,
    LabFrameStepsFeaturesIdxes, StateFeaturesIdxes,
)
from common.enums import G4Columns, StepFeaturesIdxes


def _random_unit_vectors(n: int, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g, dtype=dtype)
    return v / torch.linalg.norm(v, dim=1, keepdim=True)


def compute_rotation_matrices(
        input_dir:  torch.Tensor,
        target_dir: torch.Tensor
) -> torch.Tensor:
    """Reference oracle: batched Rodrigues rotation matrices mapping each
    `input_dir` row onto the matching `target_dir` row,

        R = I + [v]x + [v]x^2 / (1 + c),   v = input x target, c = input . target,

    with the anti-parallel denominator clamped to 1 below 1e-6 — the same
    guard `rotate_from_x_axis` uses, so the two paths must agree everywhere
    including at target ~ -input. This matrix form is an independent
    derivation, kept as the oracle `rotate_from_x_axis`'s closed-form
    elementwise apply is checked against.
    """
    v   = torch.cross(input_dir, target_dir, dim=1)
    cos = torch.einsum('ij, ij->i', input_dir, target_dir)

    identity_mat = torch.eye(3, dtype=input_dir.dtype, device=input_dir.device).repeat(input_dir.shape[0], 1, 1)

    # Skew-symmetric cross-product matrix: [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
    skew_symmetric = torch.zeros((input_dir.shape[0], 3, 3), dtype=input_dir.dtype, device=input_dir.device)
    skew_symmetric[:, 0, 1] = -v[:, 2]
    skew_symmetric[:, 0, 2] = v[:, 1]
    skew_symmetric[:, 1, 0] = v[:, 2]
    skew_symmetric[:, 1, 2] = -v[:, 0]
    skew_symmetric[:, 2, 0] = -v[:, 1]
    skew_symmetric[:, 2, 1] = v[:, 0]

    denom = 1 + cos
    parallel_mask = denom < 1e-6
    denom[parallel_mask] = 1.0

    return identity_mat + skew_symmetric + (skew_symmetric @ skew_symmetric) / denom.unsqueeze(1).unsqueeze(1)


class TestRotateFromXAxis:
    """The closed-form rotate-from-x-axis apply must reproduce the Rodrigues
    matrix path (compute_rotation_matrices with input_dir = x-hat) exactly,
    since it replaces it inside the compiled geometry graph."""

    def test_matches_rodrigues_matrix_for_random_directions(self):
        N = 500
        d = _random_unit_vectors(N, dtype=torch.float64, seed=1)
        u = _random_unit_vectors(N, dtype=torch.float64, seed=2)
        x_hat = torch.tensor([1., 0., 0.], dtype=torch.float64).repeat(N, 1)

        rot_mat = compute_rotation_matrices(input_dir=x_hat, target_dir=d)
        expected = torch.einsum('ijk,ik->ij', rot_mat, u)

        result = rotate_from_x_axis(d, u)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_maps_x_axis_to_target(self):
        N = 100
        d = _random_unit_vectors(N, dtype=torch.float32, seed=3)
        x_hat = torch.zeros(N, 3)
        x_hat[:, 0] = 1.0
        assert torch.allclose(rotate_from_x_axis(d, x_hat), d, atol=1e-6)

    def test_antiparallel_guard_matches_matrix_path(self):
        # d ~ -x-hat trips the denom < 1e-6 guard; the closed form must
        # reproduce the guarded matrix result, not the analytic limit.
        d = torch.tensor([
            [-1.0, 0.0, 0.0],
            [-0.9999999, 1e-7, -1e-7],
        ], dtype=torch.float64)
        u = _random_unit_vectors(2, dtype=torch.float64, seed=4)
        x_hat = torch.tensor([1., 0., 0.], dtype=torch.float64).repeat(2, 1)

        rot_mat = compute_rotation_matrices(input_dir=x_hat, target_dir=d)
        expected = torch.einsum('ijk,ik->ij', rot_mat, u)
        assert torch.allclose(rotate_from_x_axis(d, u), expected, atol=1e-12)


class TestComputeRotationMatrices:

    def test_get_rotation_matrices_torch(self):
        # Seeded: an unseeded draw occasionally lands near the anti-parallel
        # guard band, where the guarded fp32 matrix legitimately exceeds atol.
        N = 100
        target_dir = torch.tensor([1., 0., 0.]).repeat(N, 1)
        xyz_dir = _random_unit_vectors(N, seed=8)

        rot_mat = compute_rotation_matrices(xyz_dir, target_dir)
        xyz_dir_in_particle_frame = torch.einsum('ijk,ik->ij', rot_mat, xyz_dir)

        # Due to floating point precision, use allclose
        assert torch.allclose(xyz_dir_in_particle_frame, torch.tensor([1., 0., 0.]), atol=1e-5)

    def test_get_rotation_matrices_forward_backward_torch(self):
        N = 100
        target_dir = torch.tensor([1., 0., 0.]).repeat(N, 1)
        xyz_dir = _random_unit_vectors(N, seed=9)

        rot_mat_to_particle_frame = compute_rotation_matrices(input_dir=target_dir, target_dir=xyz_dir)
        xyz_dir_in_particle_frame = torch.einsum('ijk,ik->ij', rot_mat_to_particle_frame, xyz_dir)

        rot_mat_to_lab_frame = compute_rotation_matrices(input_dir=xyz_dir, target_dir=target_dir)
        xyz_dir_in_lab_frame = torch.einsum('ijk,ik->ij', rot_mat_to_lab_frame, xyz_dir_in_particle_frame)

        assert torch.allclose(xyz_dir_in_lab_frame, xyz_dir, atol=1e-4)


class TestTrackSimulator:
    @pytest.fixture
    def stepping_manager(self):
        # Mock SteppingManager to avoid loading actual models
        gen = MagicMock(spec=SteppingManager)
        gen.to.return_value = gen
        return gen

    @pytest.fixture
    def simulator(self, stepping_manager):
        sim = TrackSimulator(stepping_manager=stepping_manager, device='cpu')
        return sim

    @pytest.fixture
    def delta_step_particle_frame(self):
        N = 100
        features = torch.rand(N, len(StepFeaturesIdxes), dtype=torch.float32)
        features[:, StepFeaturesIdxes.L] = features[:, StepFeaturesIdxes.L] * 10.0 + 0.1
        features[:, StepFeaturesIdxes.THETA] = features[:, StepFeaturesIdxes.THETA] * np.pi
        # Nothing shortens the true step into a geometric one, so
        # geom_step is always the true step.
        features[:, StepFeaturesIdxes.GEOM_STEP] = features[:, StepFeaturesIdxes.L].clone()
        return features

    def test_displacement_is_geom_step_times_pre_dir(self, simulator: TrackSimulator, delta_step_particle_frame):
        """The particle-frame displacement is [geom_step, 0, 0] and the frame
        rotation R maps x-hat to pre_step_dir by construction, so the lab
        displacement must be exactly geom_step * pre_step_dir — for ANY
        direction, not just the axis-aligned ones."""
        N = delta_step_particle_frame.shape[0]
        pre_dir = _random_unit_vectors(N, seed=5)

        delta_state, _ = simulator._get_delta_state_and_post_dir(delta_step_particle_frame, pre_dir)

        geom_step = delta_step_particle_frame[:, StepFeaturesIdxes.GEOM_STEP]
        expected = geom_step.unsqueeze(1) * pre_dir
        assert torch.allclose(delta_state[:, :3], expected, atol=1e-5)

    def test_post_dir_opening_angle_matches_discrete_theta(self, simulator: TrackSimulator):
        """The only deflection this simulator applies is the discrete
        `theta` about a uniformly drawn azimuth. The azimuth is drawn
        inside the simulator, so what is pinned here is the rotation
        invariant: the post-step lab direction is a unit vector whose
        opening angle against the pre-step direction is exactly theta.
        With theta = 0 it must reduce to the pre-step direction."""
        N = 200
        features = torch.zeros((N, len(StepFeaturesIdxes)), dtype=torch.float32)
        g = torch.Generator().manual_seed(6)
        theta = torch.rand(N, generator=g) * 0.5
        features[:, StepFeaturesIdxes.THETA] = theta
        features[:, StepFeaturesIdxes.GEOM_STEP] = 1.0
        pre_dir = _random_unit_vectors(N, seed=7)

        _, post_dir = simulator._get_delta_state_and_post_dir(features, pre_dir)

        norms = torch.linalg.norm(post_dir, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        cos_open = torch.einsum('ij,ij->i', post_dir, pre_dir)
        assert torch.allclose(cos_open, torch.cos(theta), atol=1e-5)

        # theta = 0 -> identity, compared against the matrix reference.
        zero = torch.zeros((N, len(StepFeaturesIdxes)), dtype=torch.float32)
        zero[:, StepFeaturesIdxes.GEOM_STEP] = 1.0
        _, post_dir_0 = simulator._get_delta_state_and_post_dir(zero, pre_dir)
        x_hat = torch.tensor([1., 0., 0.]).repeat(N, 1)
        rot = compute_rotation_matrices(input_dir=x_hat, target_dir=pre_dir)
        expected = torch.einsum('ijk,ik->ij', rot, x_hat)
        assert torch.allclose(post_dir_0, expected, atol=1e-5)

    def test_single_event_batch_shapes(self, simulator: TrackSimulator):
        """n_events == 1 must not collapse the batch dim (the old
        .squeeze() before hstack broke this)."""
        features = torch.zeros((1, len(StepFeaturesIdxes)), dtype=torch.float32)
        features[:, StepFeaturesIdxes.GEOM_STEP] = 2.0
        pre_dir = torch.tensor([[0., 1., 0.]])

        delta_state, post_dir = simulator._get_delta_state_and_post_dir(features, pre_dir)

        assert delta_state.shape == (1, len(LabFrameStepsFeaturesIdxes))
        assert post_dir.shape == (1, 3)
        assert torch.allclose(delta_state[0, :3], torch.tensor([0., 2., 0.]), atol=1e-6)

    def test_post_dir_particle_frame_is_unit_up_to_fp(self, simulator: TrackSimulator, delta_step_particle_frame):
        post_dir = simulator._get_post_dir_particle_frame(delta_step_particle_frame)

        # Composed rotations of unit vectors stay unit (normalization happens
        # once, later, in the lab frame)
        assert post_dir.shape == (100, 3)
        dir_magnitude = torch.linalg.norm(post_dir, dim=1)
        assert torch.allclose(dir_magnitude, torch.ones_like(dir_magnitude), atol=1e-5)

    def test_get_delta_step_lab_frame(self, simulator: TrackSimulator, delta_step_particle_frame):
        N = delta_step_particle_frame.shape[0]

        pre_step_dirs = {
            'x': torch.tensor([[1., 0., 0.]] * N, dtype=torch.float32),
            'y': torch.tensor([[0., 1., 0.]] * N, dtype=torch.float32),
            'z': torch.tensor([[0., 0., 1.]] * N, dtype=torch.float32)
        }

        for axis, pre_step_dir in pre_step_dirs.items():
            delta_step_lab, post_dir = simulator._get_delta_state_and_post_dir(delta_step_particle_frame, pre_step_dir)

            assert delta_step_lab.shape == (N, len(LabFrameStepsFeaturesIdxes))

            assert torch.allclose(
                delta_step_lab[:, [LabFrameStepsFeaturesIdxes.E_CNT, LabFrameStepsFeaturesIdxes.E_SEC]],
                delta_step_particle_frame[:, [StepFeaturesIdxes.E_CNT, StepFeaturesIdxes.E_SEC]]
            )

            geom_step = delta_step_particle_frame[:, StepFeaturesIdxes.GEOM_STEP]

            # Displacement is along pre-step momentum direction (geom_step along that axis)
            if axis == 'x':
                lab_long = delta_step_lab[:, LabFrameStepsFeaturesIdxes.dX]
            elif axis == 'y':
                lab_long = delta_step_lab[:, LabFrameStepsFeaturesIdxes.dY]
            elif axis == 'z':
                lab_long = delta_step_lab[:, LabFrameStepsFeaturesIdxes.dZ]

            assert torch.allclose(lab_long, geom_step, atol=1e-4)

    def test_update_particles_state(self, simulator: TrackSimulator):
        N_PARTICLES = 10

        simulator.particles_state = torch.zeros((N_PARTICLES, len(StateFeaturesIdxes)), dtype=torch.float32)
        # Set initial E
        simulator.particles_state[:, StateFeaturesIdxes.E] = 100.0

        # Mock delta state
        simulator.delta_state = torch.zeros((N_PARTICLES, len(LabFrameStepsFeaturesIdxes)), dtype=torch.float32)
        # Move in X by 5.0
        simulator.delta_state[:, LabFrameStepsFeaturesIdxes.dX] = 5.0
        # Lose energy: 10 cnt, 5 sec
        simulator.delta_state[:, LabFrameStepsFeaturesIdxes.E_CNT] = 10.0
        simulator.delta_state[:, LabFrameStepsFeaturesIdxes.E_SEC] = 5.0

        simulator._update_particles_state()

        # Check positions
        assert torch.allclose(simulator.particles_state[:, StateFeaturesIdxes.X], torch.tensor([5.0]*N_PARTICLES))
        assert torch.allclose(simulator.particles_state[:, StateFeaturesIdxes.Y], torch.tensor([0.0]*N_PARTICLES))
        assert torch.allclose(simulator.particles_state[:, StateFeaturesIdxes.Z], torch.tensor([0.0]*N_PARTICLES))

        # Check energy (100 - 10 - 5 = 85)
        assert torch.allclose(simulator.particles_state[:, StateFeaturesIdxes.E], torch.tensor([85.0]*N_PARTICLES))

    def test_chain_step_to_track(self, simulator: TrackSimulator):
        N_PARTICLES = 5
        N_STEPS = 2

        simulator.alive_particles = torch.ones(N_PARTICLES, dtype=torch.bool)

        # Manually init states_along_propagation
        simulator.states_along_propagation = torch.full(
            (N_PARTICLES, N_STEPS + 1, len(StateFeaturesIdxes)),
            float("nan"), dtype=torch.float32
        )

        # Some arbitrary state
        simulator.particles_state = torch.tensor([[1., 2., 3., 4.]] * N_PARTICLES)

        # Chain to step 1
        simulator._chain_step_to_track(1)

        assert torch.allclose(
            simulator.states_along_propagation[:, 1, :],
            simulator.particles_state
        )
        # Verify index 0 and 2 are still nan
        assert torch.isnan(simulator.states_along_propagation[:, 0, :]).all()
        assert torch.isnan(simulator.states_along_propagation[:, 2, :]).all()

    def test_momentum_direction_tracking(self, simulator: TrackSimulator):
        N_PARTICLES = 5
        simulator.alive_particles = torch.ones(N_PARTICLES, dtype=torch.bool)
        simulator.particles_state = torch.zeros((N_PARTICLES, len(StateFeaturesIdxes)), dtype=torch.float32)

        # Initial momentum direction is along X
        simulator.momentum_direction = torch.zeros((N_PARTICLES, 3), dtype=torch.float32)
        simulator.momentum_direction[:, 0] = 1.0

        expected = torch.zeros((N_PARTICLES, 3))
        expected[:, 0] = 1.0
        assert torch.allclose(simulator.momentum_direction, expected)

        # After updating momentum direction (simulating what run() does)
        normalized = 1.0 / np.sqrt(2.0)
        new_dir = torch.zeros((N_PARTICLES, 3), dtype=torch.float32)
        new_dir[:, 0] = normalized
        new_dir[:, 1] = normalized
        simulator.momentum_direction[simulator.alive_particles] = new_dir[simulator.alive_particles]

        assert torch.allclose(simulator.momentum_direction[:, 0], torch.tensor(normalized, dtype=torch.float32))
        assert torch.allclose(simulator.momentum_direction[:, 1], torch.tensor(normalized, dtype=torch.float32))
        assert torch.allclose(simulator.momentum_direction[:, 2], torch.tensor(0.0, dtype=torch.float32))

    def test_states_along_propagation_df(self, simulator: TrackSimulator):
        N_PARTICLES = 2
        N_STEPS = 2

        # Mock data
        # (Events, Steps, Features)
        data_np = np.zeros((N_PARTICLES, N_STEPS + 1, 4))  # 4 features X,Y,Z,E

        # Event 0:
        # Step 0: 0,0,0,100
        # Step 1: 1,0,0,90
        # Step 2: 2,0,0,80
        data_np[0, 0, :] = [0, 0, 0, 100]
        data_np[0, 1, :] = [1, 0, 0, 90]
        data_np[0, 2, :] = [2, 0, 0, 80]

        # Event 1:
        data_np[1, 0, :] = [0, 0, 0, 100]
        data_np[1, 1, :] = [0, 1, 0, 90]
        data_np[1, 2, :] = [0, 2, 0, 80]

        simulator.states_along_propagation = torch.tensor(data_np, dtype=torch.float32)

        df = simulator.states_along_propagation_df

        # Columns expected: EventNum, StepNum, X, Y, Z, KineticEnergy
        assert G4Columns.EventNum in df.columns
        assert G4Columns.StepNum in df.columns
        assert G4Columns.X in df.columns

        # Check row count: N_PARTICLES * (N_STEPS + 1) = 2 * 3 = 6
        assert len(df) == 6

        # Check specific values
        # Event 0, Step 1 -> X=1
        row = df[(df[G4Columns.EventNum] == 0) & (df[G4Columns.StepNum] == 1)]
        assert row[G4Columns.X].values[0] == 1.0
        assert row[G4Columns.KineticEnergy].values[0] == 90.0

    def test_run_straight_line_no_scatter(self, stepping_manager):
        """Characterization of the run() loop: with no scattering and a fixed
        continuous loss per step, particles march along +x and lose energy
        linearly. Guards the loop bookkeeping (state update, momentum update,
        alive accounting) across refactors."""
        N_EVENTS, N_STEPS = 8, 5
        STEP_LEN, E_CNT_LOSS, E_0 = 1.5, 100.0, 1e6

        class _FakeSteppingManager:
            def to(self, device, dtype=None):
                return self

            # Part of the SteppingManager interface TrackSimulator.run()
            # calls: the per-run reset/read of the processes' NaN latches.
            # No-ops here -- this double emits fixed features and runs no
            # process. Deliberately present rather than absent, so run()
            # can keep calling them unguarded (a typo'd name on the real
            # SteppingManager must raise, not silently ungate the guard).
            def reset_nan_guard(self):
                pass

            def check_nan_guard(self):
                pass

            def __call__(self, energy):
                n = energy.shape[0]
                features = torch.zeros(n, len(StepFeaturesIdxes), dtype=energy.dtype)
                features[:, StepFeaturesIdxes.L] = STEP_LEN
                features[:, StepFeaturesIdxes.GEOM_STEP] = STEP_LEN
                features[:, StepFeaturesIdxes.E_CNT] = E_CNT_LOSS
                return features

        sim = TrackSimulator(stepping_manager=_FakeSteppingManager(), device='cpu')
        sim.run(E_0=E_0, n_events=N_EVENTS, n_steps=N_STEPS, save_steps_states=True)

        expected_x = torch.full((N_EVENTS,), N_STEPS * STEP_LEN)
        assert torch.allclose(sim.particles_state[:, StateFeaturesIdxes.X], expected_x, atol=1e-4)
        assert torch.allclose(sim.particles_state[:, StateFeaturesIdxes.Y], torch.zeros(N_EVENTS), atol=1e-5)
        assert torch.allclose(sim.particles_state[:, StateFeaturesIdxes.Z], torch.zeros(N_EVENTS), atol=1e-5)
        expected_e = torch.full((N_EVENTS,), E_0 - N_STEPS * E_CNT_LOSS)
        assert torch.allclose(sim.particles_state[:, StateFeaturesIdxes.E], expected_e, atol=1e-2)

        # Momentum stays unit-norm and along +x
        assert torch.allclose(sim.momentum_direction, torch.tensor([[1., 0., 0.]]).repeat(N_EVENTS, 1), atol=1e-5)
        # Decay curve: everyone alive at every executed step
        assert sim.alive_counts_per_step.tolist() == [N_EVENTS] * N_STEPS

