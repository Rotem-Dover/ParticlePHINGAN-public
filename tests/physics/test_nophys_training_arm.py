"""The USE_PHYSICS=False arm: selected per-run by PHINGAN_NO_PHYS, never a
config constant (config.py is value-pinned). Fit arithmetic writes the same
.state.pt the scalers' load() reads."""
import numpy as np
import torch

from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    ContinuousGenerator, NoPhysContinuousGenerator,
)
from physics.g4h_ionisation.generators.continuous_generator.scaler import (
    ContinuousYScaler, NoPhysContinuousYScaler,
)
from physics.g4h_ionisation.generators.train_networks.continuous_generator import (
    train as cnt_train,
)


def test_default_arm_is_physics(monkeypatch):
    monkeypatch.delenv("PHINGAN_NO_PHYS", raising=False)
    gen, y_cls, use_phys, logger = cnt_train.select_arm()
    assert isinstance(gen, ContinuousGenerator)
    assert y_cls is ContinuousYScaler
    assert use_phys is True
    assert logger == "logger_cnt"


def test_nophys_arm_selected_by_env(monkeypatch):
    monkeypatch.setenv("PHINGAN_NO_PHYS", "1")
    gen, y_cls, use_phys, logger = cnt_train.select_arm()
    assert isinstance(gen, NoPhysContinuousGenerator)
    assert y_cls is NoPhysContinuousYScaler
    assert use_phys is False
    assert logger == "logger_cnt_nophys"


def test_nophys_fit_writes_state_load_reads(tmp_path):
    # The datamodule fit arithmetic, exercised via its public helper: fit on
    # a small sample, write state, load through the scaler class, and check
    # scale() of the training extremes lands exactly on [0, 1].
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import (
        fit_nophys_y_scaler,
    )
    y = torch.tensor([100.0, 1e4, 1e6])          # eV
    fit_nophys_y_scaler(y, state_dir=tmp_path)
    s = NoPhysContinuousYScaler.load(state_dir=tmp_path)
    z = s.scale(y.clone())
    assert torch.isclose(z.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(z.max(), torch.tensor(1.0), atol=1e-6)


def test_nophys_fit_writes_state_load_reads_secondary(tmp_path):
    # Mirror of the continuous fit-writes/load-reads test, for the secondary
    # stage's (N, 1)-shaped Y and per-column min/max.
    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        NoPhysSecondaryYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.datamodule import (
        fit_nophys_y_scaler as fit_nophys_y_scaler_sec,
    )
    y = torch.tensor([[100.0], [1e4], [1e6]])     # eV, (N, 1)
    fit_nophys_y_scaler_sec(y, state_dir=tmp_path)
    s = NoPhysSecondaryYScaler.load(state_dir=tmp_path)
    z = s.scale(y.clone())
    assert torch.isclose(z.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(z.max(), torch.tensor(1.0), atol=1e-6)


def test_filter_nan_lookup_rows_short_circuits_for_nophys_without_touching_disk(tmp_path):
    # `_filter_nan_lookup_rows` must short-circuit by class before calling
    # `self.y_scaler_type.load()`, since the NoPhys arm's state may not
    # have been fit-and-written yet, and an eager load would raise
    # FileNotFoundError. Exercised here with an empty, nonexistent
    # scaler_dir, so a regression back to an eager load fails loudly (as a
    # real exception, not a silent no-op) rather than needing a real
    # .state.pt fixture on disk.
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import (
        DataModule,
    )

    assert not (tmp_path / "NoPhysContinuousYScaler.state.pt").exists()

    dm = DataModule(
        x_scaler_type=object,
        y_scaler_type=NoPhysContinuousYScaler,
        data_dir=tmp_path,
        scaler_dir=tmp_path,
    )
    dm.X_real = torch.tensor([[1e6, 1e-6], [5e5, 1e-7]])
    dm.Y_real = torch.tensor([100.0, 50.0])

    dm._filter_nan_lookup_rows()   # must not raise

    assert dm.scaler_y is None     # never touched -- the whole point of the short-circuit
    assert dm.X_real.shape[0] == 2 and dm.Y_real.shape[0] == 2  # no rows dropped


def test_continuous_gan_signature_pins_y_scaler_type_with_phys_default():
    # Import-only check (Finding 2): ContinuousGAN's constructor now takes a
    # y_scaler_type parameter defaulting to the phys ContinuousYScaler, so the
    # phys path stays behaviorally identical to before this fix. Deliberately
    # does NOT construct a ContinuousGAN instance -- its __init__ eagerly
    # unpickles the ~2 GB continuous_cdfs.pkl via PhysicsPenaltyModel, which is
    # far too heavy for a unit test.
    import inspect

    from physics.g4h_ionisation.generators.train_networks.continuous_generator.wgan import (
        ContinuousGAN,
    )

    sig = inspect.signature(ContinuousGAN.__init__)
    assert "y_scaler_type" in sig.parameters
    assert sig.parameters["y_scaler_type"].default is ContinuousYScaler


def test_secondary_gan_signature_pins_y_scaler_type_with_phys_default():
    # Mirror of the above for SecondaryGAN. Its __init__ is likewise heavy
    # (unpickles secondaries_cdfs.pkl), so this stays signature-only too.
    import inspect

    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        SecondaryYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.wgan import (
        SecondaryGAN,
    )

    sig = inspect.signature(SecondaryGAN.__init__)
    assert "y_scaler_type" in sig.parameters
    assert sig.parameters["y_scaler_type"].default is SecondaryYScaler
