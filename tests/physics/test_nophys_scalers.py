"""NoPhys Y-scalers: the no-physics ablation arm's scalers, using the
portable-state convention every scaler uses. Secondary is (N, 1),
energy-only, matching the single-column SecondaryFeats layout."""
import pytest
import torch

from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    NoPhysContinuousYScaler,
)
from physics.g4h_ionisation.generators.secondary_generator.scaler import (
    NoPhysSecondaryYScaler,
)


def _write_state(tmp_path, cls_name, lo, hi):
    torch.save({"min_post_log": torch.tensor(lo), "max_post_log": torch.tensor(hi)},
               tmp_path / f"{cls_name}.state.pt")


def test_continuous_roundtrip(tmp_path):
    # abs(log10(MeV)) space: 100 eV = 1e-4 MeV -> 4.0 ; 1 MeV -> 0.0
    _write_state(tmp_path, "NoPhysContinuousYScaler", 0.0, 6.0)
    s = NoPhysContinuousYScaler.load(state_dir=tmp_path)
    x = torch.tensor([100.0, 1e4, 1e6])          # eV
    z = s.scale(x.clone())
    assert torch.allclose(s.rescale(z.clone()), x, rtol=1e-5)
    assert (z >= 0).all() and (z <= 1).all()


def test_secondary_is_one_column_energy_only(tmp_path):
    # log10(MeV) space (no abs fold): 1 keV = 1e-3 MeV -> -3.0
    _write_state(tmp_path, "NoPhysSecondaryYScaler",
                 torch.tensor([-4.0]), torch.tensor([0.0]))
    s = NoPhysSecondaryYScaler.load(state_dir=tmp_path)
    y = torch.tensor([[1e3], [1e4], [1e5]])      # eV, (N, 1)
    z = s.scale(y.clone())
    assert z.shape == (3, 1)
    assert torch.allclose(s.rescale(z.clone()), y, rtol=1e-5)
    # non-differentiable path must agree with the differentiable one
    assert torch.allclose(s.rescale(z.clone(), differentiable_ops=False), y,
                          rtol=1e-5)


def test_load_raises_without_state(tmp_path):
    with pytest.raises(FileNotFoundError):
        NoPhysContinuousYScaler.load(state_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        NoPhysSecondaryYScaler.load(state_dir=tmp_path)


def test_state_dict_tensors_matches_load_format(tmp_path):
    _write_state(tmp_path, "NoPhysContinuousYScaler", 0.0, 6.0)
    s = NoPhysContinuousYScaler.load(state_dir=tmp_path)
    d = s.state_dict_tensors()
    assert set(d) == {"min_post_log", "max_post_log"}
    torch.save(d, tmp_path / "NoPhysContinuousYScaler.state.pt")
    s2 = NoPhysContinuousYScaler.load(state_dir=tmp_path)
    assert torch.equal(s2.min_post_log, s.min_post_log)
