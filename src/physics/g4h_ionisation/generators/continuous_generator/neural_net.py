"""
Neural network generators for the continuous energy-loss stage. ContinuousGenerator
is the physics-informed WGAN-GP generator conditioned on (kinetic energy, step length).
NoPhysContinuousGenerator is a variant using BatchNorm and sigmoid output activation
instead of LayerNorm, serving as a non-physics baseline.
"""
import torch

from torch import nn

from common.config import N_NEURON_MULTIPLIER_CNT, N_EMBEDDING_CNT, N_NOISE_CNT, N_CONTINUOUS_FEATURES

from physics.interfaces.generator_interface import GeneratorInterface


class NoPhysContinuousGenerator(GeneratorInterface):
    """
    Non-physics continuous energy loss generator.
    Uses BatchNorm1d instead of LayerNorm, InstanceNorm1d in embedding.
    Input: [kinetic_energy, step_length] (2 features)
    Output: E_CNT (1 feature)
    """
    def __init__(
            self,
            n_embedding:   int  = N_EMBEDDING_CNT,
            n_noise:       int  = N_NOISE_CNT,
    ):
        super(NoPhysContinuousGenerator, self).__init__()
        self.n_noise     = n_noise
        self.n_embedding = n_embedding
        self.embed       = self.embedding()
        self.model = nn.Sequential(
            self._block(
                n_noise + self.n_embedding,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=False
            ),
            self._block(
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=False
            ),
            self._last_block()
        )

    @staticmethod
    def _block(
            in_channels:  int,
            out_channels: int,
            bias:         bool = True
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels, affine=True),
            nn.ReLU()
        )

    @staticmethod
    def _last_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES, N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES, bias=False),
            nn.BatchNorm1d(N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES, affine=True),
            nn.ReLU(),
            nn.Linear(
                N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES,
                N_CONTINUOUS_FEATURES,
                bias=True
            ),
        )

    def embedding(self) -> nn.Sequential:
        size = 1
        return nn.Sequential(
            nn.Linear(2, size * self.n_embedding, bias=True),
            nn.InstanceNorm1d(1, affine=True),
            nn.ReLU(),
            nn.Linear(size * self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        assert len(labels.shape) == 2, f"Shape should be 'number of predictions' x 2. Got instead: {labels.shape}"
        assert labels.shape[1]   == 2, f"Number of input features should be 2. Got instead: {labels.shape[1]}"

        latent_vector = torch.randn(labels.shape[0], self.n_noise, device=self.device, dtype=labels.dtype)

        labels = self.embed(labels.unsqueeze(1))

        gen_input = torch.cat([labels, latent_vector], dim=-1)

        output = self.model(gen_input)

        output = 1.2 * torch.sigmoid(output) - 0.1

        return output.flatten()


class ContinuousGenerator(GeneratorInterface):
    """
    Generator for continuous energy loss (E_CNT).
    Input: [kinetic_energy, step_length] (2 features)
    Output: E_CNT (1 feature)
    """
    def __init__(
            self,
            n_embedding:   int  = N_EMBEDDING_CNT,
            n_noise:       int  = N_NOISE_CNT,
    ):
        super(ContinuousGenerator, self).__init__()
        self.n_noise     = n_noise
        self.n_embedding = n_embedding
        self.embed       = self.embedding()
        self.model = nn.Sequential(
            self._block(
                n_noise + self.n_embedding,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=False
            ),
            self._block(
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES,
                bias=False
            ),
            self._last_block()
        )

    @staticmethod
    def _block(
            in_channels:  int,
            out_channels: int,
            bias:         bool = True
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    @staticmethod
    def _last_block() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(N_NEURON_MULTIPLIER_CNT * N_CONTINUOUS_FEATURES, N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES, bias=False),
            nn.LayerNorm(N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES),
            nn.ReLU(),
            nn.Linear(
                N_NEURON_MULTIPLIER_CNT // 2 * N_CONTINUOUS_FEATURES,
                N_CONTINUOUS_FEATURES,
                bias=True
            ),
        )

    def embedding(self) -> nn.Sequential:
        size = 1
        return nn.Sequential(
            nn.Linear(2, size * self.n_embedding, bias=True),
            nn.LayerNorm(size * self.n_embedding),
            nn.ReLU(),
            nn.Linear(size * self.n_embedding, self.n_embedding, bias=True),
            nn.Flatten()
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        assert len(labels.shape) == 2, f"Shape should be 'number of predictions' x 2. Got instead: {labels.shape}"
        assert labels.shape[1]   == 2, f"Number of input features should be 2. Got instead: {labels.shape[1]}"

        latent_vector = torch.randn(labels.shape[0], self.n_noise, device=self.device, dtype=labels.dtype)

        labels = self.embed(labels.unsqueeze(1))

        gen_input = torch.cat([labels, latent_vector], dim=-1)

        output = self.model(gen_input)

        return output.flatten()
