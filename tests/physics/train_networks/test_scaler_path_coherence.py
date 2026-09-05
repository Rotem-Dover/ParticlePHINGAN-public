"""A fresh material tree has no scaler state under SCALERS_DIR until a
DataModule fits and writes it. The GANs must therefore (a) not touch
scaler state in __init__, and (b) load it in the Lightning setup() hook --
which runs AFTER DataModule.setup() has fitted and written the states --
from SCALERS_DIR."""
import torch

from common import paths


def test_continuous_gan_constructs_without_any_scaler_state(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCALERS_DIR", tmp_path / "empty")
    from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
        ContinuousGenerator,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import (
        Critic,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.wgan import (
        ContinuousGAN,
    )

    gan = ContinuousGAN(generator=ContinuousGenerator(), critic=Critic(),
                        datamodule=None, use_physics_regularization=False)
    assert gan.x_scaler is None and gan.y_scaler is None  # nothing read yet


def test_continuous_gan_setup_reads_from_scalers_dir(tmp_path, monkeypatch):
    from physics.g4h_ionisation.generators.continuous_generator.scaler import (
        ContinuousXScaler,
        ContinuousYScaler,
    )

    scaler_dir = tmp_path / "scalers"
    scaler_dir.mkdir()
    torch.save({"min_post_log": torch.tensor([0.0, 0.0]),
                "max_post_log": torch.tensor([1.0, 1.0])},
               scaler_dir / "ContinuousXScaler.state.pt")
    monkeypatch.setattr(paths, "SCALERS_DIR", scaler_dir)

    x = ContinuousXScaler.load(paths.SCALERS_DIR)
    assert torch.equal(x.max_post_log, torch.tensor([1.0, 1.0]))

    # y-scaler: stub .load so the test does not need the 200 MB lookup state
    loaded_from = {}

    def fake_load(cls, state_dir=None):
        loaded_from["dir"] = state_dir
        obj = cls()
        # Stub just enough state for .to(device) to succeed -- the real
        # `.load()` populates these from the 200 MB lookup pickle, which
        # this test deliberately avoids building.
        obj.kin_table = torch.tensor([0.0, 1.0])
        obj.step_table = torch.tensor([0.0, 1.0])
        obj.lookup_table = torch.zeros(2, 2, 2)
        return obj

    monkeypatch.setattr(ContinuousYScaler, "load", classmethod(fake_load))

    from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
        ContinuousGenerator,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import (
        Critic,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.wgan import (
        ContinuousGAN,
    )

    gan = ContinuousGAN(generator=ContinuousGenerator(), critic=Critic(),
                        datamodule=None, use_physics_regularization=False)
    gan.setup("fit")
    assert gan.x_scaler is not None and gan.y_scaler is not None
    assert loaded_from["dir"] == paths.SCALERS_DIR


def test_secondary_gan_constructs_without_any_scaler_state(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCALERS_DIR", tmp_path / "empty")
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
        SecondaryGenerator,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import (
        Critic,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.wgan import (
        SecondaryGAN,
    )

    gan = SecondaryGAN(generator=SecondaryGenerator(), critic=Critic(),
                       datamodule=None, use_physics_regularization=False)
    assert gan.x_scaler is None and gan.y_scaler is None  # nothing read yet


def test_secondary_gan_setup_reads_from_scalers_dir(tmp_path, monkeypatch):
    from physics.g4h_ionisation.generators.secondary_generator.scaler import (
        SecondaryXScaler,
        SecondaryYScaler,
    )

    scaler_dir = tmp_path / "scalers"
    scaler_dir.mkdir()
    torch.save({"min_post_log": torch.tensor([0.0]),
                "max_post_log": torch.tensor([1.0])},
               scaler_dir / "SecondaryXScaler.state.pt")
    torch.save({}, scaler_dir / "SecondaryYScaler.state.pt")
    monkeypatch.setattr(paths, "SCALERS_DIR", scaler_dir)

    x = SecondaryXScaler.load(paths.SCALERS_DIR)
    assert torch.equal(x.max_post_log, torch.tensor([1.0]))

    from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
        SecondaryGenerator,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import (
        Critic,
    )
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.wgan import (
        SecondaryGAN,
    )

    gan = SecondaryGAN(generator=SecondaryGenerator(), critic=Critic(),
                       datamodule=None, use_physics_regularization=False)
    gan.setup("fit")
    assert gan.x_scaler is not None and gan.y_scaler is not None
