"""
Critic (discriminator) network for the secondary energy loss WGAN-GP.
Discriminates E_SEC values conditioned on kinetic energy using an embedding
layer and fully connected blocks with LayerNorm and LeakyReLU.
"""

import torch.nn as nn
import torch

from common.config import N_EMBEDDING_DISC_SEC, N_SECONDARY_FEATURES, N_NEURON_MULTIPLIER_SEC


class Critic(nn.Module):
    """
    Critic for secondary generator.
    Discriminates E_SEC (N_SECONDARY_FEATURES features) conditioned on kinetic_energy (1 label).
    """
    def __init__(
            self,
            n_embedding: int = N_EMBEDDING_DISC_SEC,
    ):
        super(Critic, self).__init__()
        self.n_embedding = n_embedding
        self.embed = self.embedding()

        self.model = nn.Sequential(
            self._block(
                N_SECONDARY_FEATURES + self.n_embedding,
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_SEC * N_SECONDARY_FEATURES,
                N_NEURON_MULTIPLIER_SEC // 2 * N_SECONDARY_FEATURES,
                bias=True,
            ),
            nn.Linear(
                N_NEURON_MULTIPLIER_SEC // 2 * N_SECONDARY_FEATURES,
                1,
                bias=False
            ),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, bias: bool = True):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2)
        )

    def embedding(self):
        return nn.Sequential(
            nn.Linear(1, self.n_embedding, bias=True),
            nn.LayerNorm(self.n_embedding),
            nn.ReLU(),
            nn.Linear(self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, x, labels) -> torch.Tensor:
        labels = self.embed(labels.unsqueeze(1))
        while x.ndim < 2:
            x = x.unsqueeze(1)
        disc_input = torch.cat([x, labels], dim=-1)

        del x, labels

        return self.model(disc_input).squeeze(1)
