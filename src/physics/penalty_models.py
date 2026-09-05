"""
Defines physics-informed penalty models used during GAN training. Provides a
Kolmogorov-Smirnov test penalty that compares the generator's output distribution
against precomputed theoretical CDFs, encouraging the generated samples to match
the analytical energy-loss or step-length distributions.

Consumes `common.paths.PATH_TO_SECONDARIES_CDFS`, a pickle of theoretical CDFs
built offline, via
`g4h_ionisation/generators/train_networks/secondary_generator/wgan.py`'s
`SecondaryGAN`; nothing in this module writes that file.

A trained checkpoint's pickled `penalty_model` hyperparameter is a class
reference into this exact module path
(`physics.penalty_models.KSTestPenaltyModel`), so `common/unpickle_ckpt.py`'s
absent-module stub for `physics.penalty_models` never needs to fire — the
real module resolves the reference. Do not move this module without reading
that one's docstring.
"""
import torch
import numpy as np
import pickle as pkl

from pathlib import Path
from abc import ABC, abstractmethod

from common.utils import Interp1dLinear
from physics.interfaces.generator_interface import GeneratorInterface


class PhysicsPenaltyModel(ABC):
    def __init__(self, path_to_cdf: Path, columns_idx: int):
        with open(path_to_cdf, "rb") as f:
            self.physics_cdfs: dict[tuple, Interp1dLinear] = pkl.load(f)

        for key in self.physics_cdfs:
            self.physics_cdfs[key] = self.physics_cdfs[key].to(torch.float32).to('cpu')

        self.columns_idx = columns_idx

        # Detect CDF key format: (step_length, kin_energy) vs (kin_energy,)
        first_key = next(iter(self.physics_cdfs))
        self.e_only = len(first_key) == 1

        self.post_init()

    @abstractmethod
    def post_init(self):
        pass

    @abstractmethod
    def physics_penalty_loss(self, generator: GeneratorInterface, requires_grad: bool = True) -> torch.Tensor:
        pass


class KSTestPenaltyModel(PhysicsPenaltyModel):
    def post_init(self):
        pass

    def physics_penalty_loss(
            self,
            generator: GeneratorInterface,
            requires_grad: bool = True,
    ) -> torch.Tensor:
        device = generator.device
        loss = torch.tensor(0., device=device, requires_grad=requires_grad)

        physics_cdfs = list(self.physics_cdfs.keys())
        indexes = np.random.randint(len(physics_cdfs), size=int(len(physics_cdfs) / 10))
        physics_cdfs_keys = np.asarray(physics_cdfs, dtype=object)[indexes].tolist()

        n_samples_per_point = 10_000

        if self.e_only:
            # E-only CDFs: keys are (kin_energy,), labels are [kin_energy]
            labels = torch.tensor(
                [k[0] for k in physics_cdfs_keys], dtype=torch.float32, device=device
            )
        else:
            # (step_length, kin_energy) CDFs: labels are [kin_energy, step_length]
            labels = torch.tensor(physics_cdfs_keys, dtype=torch.float32, device=device)[:, [1, 0]]

        fake_steps = generator.predict(labels, repeat_interleave=n_samples_per_point).to('cpu')

        if fake_steps.ndim == 1:
            # Single-output generator (e.g., ContinuousGenerator)
            energy_losses = fake_steps
        elif requires_grad:
            mask_energy_loss = torch.zeros_like(fake_steps)
            mask_energy_loss[:, self.columns_idx] += 1.
            energy_losses = (fake_steps * mask_energy_loss).sum(dim=1)
        else:
            energy_losses = fake_steps[:, self.columns_idx]

        n_cdfs = len(physics_cdfs_keys)
        for i, key in enumerate(physics_cdfs_keys):
            cdf_key = tuple(key) if isinstance(key, list) else key
            slice_ = slice(i * n_samples_per_point, (i + 1) * n_samples_per_point)
            loss = loss + kolmogorov_smirnov_distance(
                data=energy_losses[slice_], cdf=self.physics_cdfs[cdf_key]) / n_cdfs

        return loss


def kolmogorov_smirnov_distance(data: torch.Tensor, cdf: Interp1dLinear) -> torch.Tensor:
    sorted_data = data.view(-1).sort().values
    theoretical_cdf = cdf(sorted_data)
    empirical_cdf_values = _get_empirical_cdf_values(sorted_data)

    return torch.max(torch.abs(theoretical_cdf - empirical_cdf_values))


def _get_empirical_cdf_values(data: torch.Tensor) -> torch.Tensor:
    return torch.arange(1, len(data) + 1, device=data.device, dtype=data.dtype) / len(data)
