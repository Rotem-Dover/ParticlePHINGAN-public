"""NoPhys generators: the no-physics ablation architectures, built at the
per-stage constants in common.config; secondary emits (N, 1)."""
import torch

from common.config import (
    N_CONTINUOUS_FEATURES, N_SECONDARY_FEATURES,
)
from physics.g4h_ionisation.generators.continuous_generator.neural_net import (
    NoPhysContinuousGenerator,
)
from physics.g4h_ionisation.generators.secondary_generator.neural_net import (
    NoPhysSecondaryGenerator,
)


def test_continuous_forward_shape_and_range():
    g = NoPhysContinuousGenerator().eval()
    out = g.forward(torch.tensor([[5e6, 1e-6], [2e6, 1e-7]]))
    assert out.shape == (2,)
    # paper's output map is 1.2*sigmoid - 0.1 -> open interval (-0.1, 1.1)
    assert (out > -0.1).all() and (out < 1.1).all()


def test_secondary_forward_is_one_column():
    g = NoPhysSecondaryGenerator().eval()
    out = g.forward(torch.tensor([5e6, 2e6, 1e6]))
    assert out.shape == (3, N_SECONDARY_FEATURES)
    assert N_SECONDARY_FEATURES == 1


def test_bare_ctor_is_the_checkpoint_contract():
    # from_checkpoint builds cls() with no args -- the config constants ARE
    # the architecture. A bare ctor must therefore succeed and produce a
    # net whose parameter shapes are reproducible.
    a = NoPhysContinuousGenerator()
    b = NoPhysContinuousGenerator()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and pa.shape == pb.shape
