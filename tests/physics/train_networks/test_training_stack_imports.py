"""The training stack must import, construct, and run one optimisation step on
a tiny synthetic batch -- without a GPU and without touching real storage."""
import numpy as np
import pytest
import torch


def test_training_modules_import():
    from physics.g4h_ionisation.generators.train_networks.secondary_generator import (  # noqa: F401
        critic, datamodule, wgan, train,
    )
    from physics.interfaces import wgan_interface  # noqa: F401
    from physics import penalty_models  # noqa: F401


def test_critic_forward_shape():
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import Critic
    from common.config import N_SECONDARY_FEATURES

    c = Critic()
    n = 32
    x = torch.rand(n, N_SECONDARY_FEATURES)
    labels = torch.rand(n)
    out = c(x, labels)
    assert out.shape[0] == n


def test_generator_and_critic_compose_for_one_step():
    # The point: prove the pieces fit before booking a GPU. A shape mismatch
    # between generator output and critic input is the classic way a training
    # run dies twenty minutes in.
    from physics.g4h_ionisation.generators.train_networks.secondary_generator.critic import Critic
    from physics.g4h_ionisation.generators.secondary_generator.neural_net import SecondaryGenerator

    g, c = SecondaryGenerator(), Critic()
    labels = torch.rand(64) * 1e8
    fake = g(labels)
    score = c(fake, labels)
    assert score.shape[0] == 64
    score.mean().backward()
    assert any(p.grad is not None for p in c.parameters())
