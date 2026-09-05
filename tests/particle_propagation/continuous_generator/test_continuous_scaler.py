"""Tests for ContinuousXScaler / ContinuousYScaler.

Both scalers are DATA-FITTED, not derived from BeamConfig: their state is a
fitted min/max (X) and a physics mean/std lookup table (Y), restored by
`load()` from `<ClassName>.state.pt`. Numerical values are pinned by
`tests/physics/test_scalers.py`; these tests cover the structural contract
(round-trip invertibility, non-mutation, device move).
"""
import pytest
import torch

from physics.g4h_ionisation.generators.continuous_generator.scaler import ContinuousXScaler, ContinuousYScaler


class TestContinuousXScaler:

    @pytest.fixture
    def scaler(self):
        return ContinuousXScaler.load()

    @pytest.fixture
    def phase_space(self, scaler):
        # Draw inside the scaler's own fitted bounds: column 0 is log10(E),
        # column 1 is |log10(L)| -- L is stored with its sign flipped so
        # its skew runs the same direction as E's.
        torch.manual_seed(42)
        n = 100
        lo, hi = scaler.min_post_log, scaler.max_post_log
        ke = torch.pow(10, torch.rand(n) * (hi[0] - lo[0]) + lo[0])
        sl = torch.pow(10, -(torch.rand(n) * (hi[1] - lo[1]) + lo[1]))
        return torch.stack([ke, sl], dim=1)

    def test_bounds_come_from_the_fitted_state(self, scaler):
        # A fitted min/max, not derived from BeamConfig. Both features must
        # be a proper, finite, non-degenerate interval or every scaled
        # value is garbage.
        assert scaler.min_post_log.shape == (2,)
        assert scaler.max_post_log.shape == (2,)
        assert torch.isfinite(scaler.min_post_log).all()
        assert torch.isfinite(scaler.max_post_log).all()
        assert torch.all(scaler.max_post_log > scaler.min_post_log)

    def test_scale_rescale_roundtrip(self, scaler, phase_space):
        scaled = scaler.scale(phase_space)
        assert torch.all(scaled >= -1e-5)
        assert torch.all(scaled <= 1.0 + 1e-5)
        rescaled = scaler.rescale(scaled)
        assert torch.allclose(phase_space, rescaled, atol=1e-3, rtol=1e-3)

    def test_scale_does_not_mutate_the_caller_tensor(self, scaler, phase_space):
        before = phase_space.clone()
        scaler.scale(phase_space)
        assert torch.equal(phase_space, before)

    def test_device_move(self, scaler):
        scaler.to('cpu')
        assert scaler.min_post_log.device.type == 'cpu'


class TestContinuousYScaler:

    @pytest.fixture
    def scaler(self):
        return ContinuousYScaler.load()

    @pytest.fixture
    def phase_space(self):
        torch.manual_seed(42)
        n = 100
        ke = torch.pow(10, torch.rand(n) * 2 + 6)
        sl = torch.pow(10, torch.rand(n) * 2 - 6)
        return torch.stack([ke, sl], dim=1)

    @pytest.fixture
    def y_data(self):
        torch.manual_seed(42)
        return torch.pow(10, torch.rand(100) * 3 + 2).unsqueeze(1)

    def test_lookup_state_is_restored(self, scaler):
        assert scaler.lookup_table.shape == (scaler.kin_table.shape[0],
                                             scaler.step_table.shape[0], 2)
        assert scaler._kin_max_idx == scaler.kin_table.shape[0] - 1
        assert scaler._step_max_idx == scaler.step_table.shape[0] - 1
        # The lookup table stores NaN over the deterministic region
        # (~29.2% of cells), where the along-step guard overwrites the
        # network's output with the analytic mean rather than reading this
        # table. The structural expectation is therefore not that every
        # cell is finite, but that the finite complement is finite and
        # both channels (mean, std) share one NaN mask.
        nan = torch.isnan(scaler.lookup_table)
        assert 0.2 < nan.float().mean() < 0.4, nan.float().mean()
        assert torch.equal(nan[..., 0], nan[..., 1])
        assert torch.isfinite(scaler.lookup_table[~nan]).all()

    def test_scale_rescale_roundtrip(self, scaler, y_data, phase_space):
        scaled = scaler.scale(y_data, phase_space=phase_space)
        assert scaled.shape == y_data.shape
        assert not torch.isnan(scaled).any()
        rescaled = scaler.rescale(scaled, phase_space=phase_space)
        assert torch.allclose(y_data, rescaled, rtol=1e-3)

    def test_device_move(self, scaler):
        scaler.to('cpu')
        assert scaler.kin_table.device.type == 'cpu'
