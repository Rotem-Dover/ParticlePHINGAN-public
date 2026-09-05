"""
Critic (discriminator) network for the continuous energy loss WGAN-GP.
Discriminates continuous energy loss values conditioned on kinetic energy and
step length using an embedding layer and fully connected blocks with
LayerNorm and LeakyReLU.
"""

import torch.nn as nn
import torch

from common.config import N_NEURON_MULTIPLIER_CNT, N_EMBEDDING_DISC_CNT, N_CONTINUOUS_FEATURES


class Critic(nn.Module):
    """
    Critic for continuous energy loss generator (default architecture).
    Discriminates E_CNT (1 feature) conditioned on [kinetic_energy, step_length] (2 labels).
    Uses LayerNorm + LeakyReLU in every hidden block.
    """
    def __init__(
            self,
            n_embedding: int = N_EMBEDDING_DISC_CNT,
    ):
        super(Critic, self).__init__()
        self.n_embedding = n_embedding
        self.embed = self.embedding()

        self.model = nn.Sequential(
            self._block(
                N_CONTINUOUS_FEATURES + self.n_embedding,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=True,
            ),
            self._block(
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES,
                bias=True,
            ),
            nn.Linear(
                N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES,
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
        size = 1
        return nn.Sequential(
            nn.Linear(2, size * self.n_embedding, bias=True),
            nn.LayerNorm(size * self.n_embedding),
            nn.ReLU(),
            nn.Linear(size * self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, x, labels) -> torch.Tensor:
        labels = self.embed(labels.unsqueeze(1))
        while x.ndim < 2:
            x = x.unsqueeze(1)
        disc_input = torch.cat([x, labels], dim=-1)

        del x, labels

        return self.model(disc_input).squeeze(1)
