"""The continuous training stack must import, construct, compose generator +
critic through one real optimisation step, and set up its DataModule against
actually-built arrays -- without a GPU and without touching real storage
outside the frozen, read-only scaler/CDF namespace those components already
depend on.

The secondary stage's equivalent integration test
(`tests/test_build_training_datasets.py::
test_datamodule_setup_succeeds_against_the_built_secondary_target`)
constructs the real DataModule and calls `setup()` for real; this file
mirrors that bar for the continuous stage, constructing the real DataModule
rather than only comparing arrays to a manifest from the same run.
"""
import torch

import build_training_datasets as brd
from tests.helpers_synthetic_g4 import write_synthetic_root


def test_training_modules_import():
    from physics.g4h_ionisation.generators.train_networks.continuous_generator import (  # noqa: F401
        critic, datamodule, wgan, train,
    )


def test_critic_forward_shape():
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import Critic
    from common.config import N_CONTINUOUS_FEATURES

    c = Critic()
    n = 32
    x = torch.rand(n, N_CONTINUOUS_FEATURES)
    labels = torch.rand(n, 2)
    out = c(x, labels)
    assert out.shape[0] == n


def test_generator_and_critic_compose_for_one_step():
    # The point: prove the pieces fit before booking a GPU. A shape mismatch
    # between generator output and critic input is the classic way a
    # training run dies twenty minutes in. Mirrors the secondary stage's
    # equivalent test (tests/physics/train_networks/test_training_stack_imports.py
    # ::test_generator_and_critic_compose_for_one_step).
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.critic import Critic
    from physics.g4h_ionisation.generators.continuous_generator.neural_net import ContinuousGenerator

    g, c = ContinuousGenerator(), Critic()
    n = 64
    labels = torch.rand(n, 2) * torch.tensor([1e8, 1e-3])  # [kinetic_energy, step_length]
    fake = g(labels)
    assert fake.shape == (n,)

    score = c(fake, labels)
    assert score.shape[0] == n

    score.mean().backward()
    assert any(p.grad is not None for p in c.parameters())
    assert any(p.grad is not None for p in g.parameters())


def test_datamodule_setup_succeeds_against_the_built_continuous_target(tmp_path):
    # Construct the real DataModule against the arrays build_training_datasets
    # actually writes, and call setup() for real -- not a manifest-shape
    # comparison. Do not reduce this to array/shape asserts against the
    # same run's own manifest.
    from physics.g4h_ionisation.generators.continuous_generator.scaler import (
        ContinuousXScaler, ContinuousYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import DataModule

    root = tmp_path / "synthetic.root"
    write_synthetic_root(root, n_events=60, steps_per_event=10, seed=11)
    out = tmp_path / "out"
    brd.build(root, out)

    dm = DataModule(
        ContinuousXScaler, ContinuousYScaler,
        data_dir=out, scaler_dir=tmp_path / "scalers",
    )
    dm.setup()

    assert dm.X is not None and dm.Y is not None
    assert dm.X.shape[1] == 2
    assert dm.X.shape[0] == dm.Y.shape[0]
    assert dm.scaler_x is not None and dm.scaler_y is not None
    # ContinuousYScaler.load() must have populated its physics lookup table
    # (see continuous_generator/datamodule.py's docstring point 4: it is loaded, not
    # fit-from-data).
    assert dm.scaler_y.lookup_table is not None

    # setup() is memoized; a second call must be a no-op, not refit its
    # scalers again.
    scaler_x_before = dm.scaler_x
    dm.setup()
    assert dm.scaler_x is scaler_x_before


def test_a_nan_lookup_row_is_filtered_out_of_the_training_set(tmp_path):
    # The analytic lookup writes NaN on a THREE-way OR, while
    # _filter_bethe_bloch_steps tests only one of the three. A survivor would
    # abort training on ContinuousYScaler._assert_lookup_finite, so the
    # datamodule filters on the gather itself.
    from physics.g4h_ionisation.generators.continuous_generator.scaler import (
        ContinuousXScaler, ContinuousYScaler,
    )
    from physics.g4h_ionisation.generators.train_networks.continuous_generator.datamodule import DataModule

    dm = DataModule(
        ContinuousXScaler, ContinuousYScaler,
        data_dir=tmp_path, scaler_dir=tmp_path / "scalers",
    )
    # Row 0 is ordinary physics; row 1 is a short step that gathers NaN (the
    # analytic table's interior deterministic region).
    dm.X_real = torch.tensor([[1.0e8, 1.0e-5], [1.0e8, 1.0e-10]], dtype=torch.float32)
    dm.Y_real = torch.tensor([1.0e3, 1.0e3], dtype=torch.float32)

    dm._filter_nan_lookup_rows()

    assert dm.X_real.shape[0] == 1, dm.X_real
    assert abs(float(dm.X_real[0, 1]) / 1.0e-5 - 1.0) < 1e-6  # float32
    # And the survivor genuinely does not trip the guard the filter protects.
    dm.scaler_y.scale(dm.Y_real.reshape(-1, 1), phase_space=dm.X_real)
