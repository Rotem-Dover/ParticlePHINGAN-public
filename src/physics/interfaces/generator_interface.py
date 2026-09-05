"""
Abstract base class for all GAN generators used in particle propagation.
Provides a common interface for forward pass, checkpoint loading, freeze/unfreeze,
and automatic input/output scaling via the @scaled_forward decorator so that
inference code can work in physical units while the network operates in scaled space.
"""
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Self, Optional

import torch
from torch import nn

from common.unpickle_ckpt import load_checkpoint
from common.config import DEVICE
from physics.interfaces.scaler_interface import ScalerInterface


def scaled_forward(func):
    """
    Decorator to apply scaling to the input and output of the generator
    if scalers are available.
    """
    @wraps(func)
    def wrapper(self, labels: torch.Tensor, repeat_interleave: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        # If scalers are not set, run the raw function (useful for training/debugging)
        if self.x_scaler is None or self.y_scaler is None:
            if repeat_interleave is not None:
                labels = labels.repeat_interleave(repeat_interleave, dim=0)
            return func(self, labels, *args, **kwargs)

        # Apply Input Scaling
        scaled_labels = self.x_scaler.scale(labels)

        # Apply repeat_interleave after scaling if specified
        if repeat_interleave is not None:
            scaled_labels = scaled_labels.repeat_interleave(repeat_interleave, dim=0)
            labels = labels.repeat_interleave(repeat_interleave, dim=0)

        # Run the Generator (Forward Pass)
        generated_data = func(self, scaled_labels, *args, **kwargs)

        # Apply Output Rescaling (passing original labels as phase_space context)
        return self.y_scaler.rescale(generated_data, phase_space=labels)

    return wrapper


class GeneratorInterface(ABC, nn.Module):
    """
    Interface for the generators used in the particle propagation system.
    Includes built-in support for input/output scaling.
    """

    # Whether the generator's output distribution already contains the
    # exact-zero atom (e.g. the physics MC samplers, whose Poisson channels
    # produce loss == 0 at the G4 rate). NN generators are trained on
    # filtered nonzero data and output a continuous density, so the caller
    # (G4hIonisation.along_step_do_it) re-injects P(loss=0) externally —
    # but must skip that injection when this flag is True, otherwise the
    # zero atom is double-counted and nonzero small losses are depleted.
    samples_zero_loss: bool = False

    def __init__(self):
        super().__init__()
        self.checkpoint: Optional[Path] = None
        self.x_scaler: Optional[ScalerInterface] = None
        self.y_scaler: Optional[ScalerInterface] = None

    def set_scalers(self, x_scaler: ScalerInterface, y_scaler: ScalerInterface):
        """
        Inject scalers into the generator instance.
        """
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    @abstractmethod
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (typically operates in scaled/latent space).
        """
        raise NotImplementedError

    @scaled_forward
    def predict(self, labels: torch.Tensor, repeat_interleave: Optional[int] = None) -> torch.Tensor:
        """
        Inference method that automatically handles scaling via the @scaled_forward decorator.
        Calls self.forward() internally.

        Args:
            labels: Input labels tensor.
            repeat_interleave: If specified, repeats each label this many times after scaling
                               but before prediction. Useful for generating many samples per label.

        Returns:
            Generated output tensor.
        """
        return self.forward(labels)

    @abstractmethod
    def embedding(self) -> nn.Sequential:
        """
        Create the embedding layer.

        Returns:
            nn.Sequential: Embedding layer.
        """
        raise NotImplementedError

    @property
    def device(self) -> str:
        return next(self.parameters()).device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @classmethod
    def from_checkpoint(
            cls, checkpoint: Path, device: str = DEVICE,
            x_scaler: Optional[ScalerInterface] = None, y_scaler: Optional[ScalerInterface] = None,
            infer_arch: bool = False,
    ) -> Self:
        # Safe checkpoint loading
        state_dict = load_checkpoint(checkpoint)

        # Filter out the generator state dictionary
        generator_state_dict = {
            k.replace("generator.", ""): v for k, v in state_dict.items() if "generator" in k
        }

        # Construct the generator. By default the bare constructor is used —
        # config constants ARE the architecture, and a shape mismatch must
        # fail loud (a silently-adapted net would mask config drift on the
        # runtime path). `infer_arch=True` is the opt-in for tooling (the
        # playground's checkpoint browser) that must load checkpoints
        # trained at a different width than the current config constants;
        # it asks the class to recover its constructor kwargs from the
        # state dict shapes, where the class supports that.
        ctor_kwargs = {}
        if infer_arch:
            infer = getattr(cls, "ctor_kwargs_from_state_dict", None)
            if infer is not None:
                ctor_kwargs = infer(generator_state_dict)
        generator = cls(**ctor_kwargs)
        generator.load_state_dict(generator_state_dict)

        generator.freeze()

        generator.to(device)

        generator.checkpoint = checkpoint
        generator.set_scalers(x_scaler, y_scaler)

        return generator


class DummyGenerator(GeneratorInterface):
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.zeros(labels.shape[0], device=labels.device)

    def embedding(self) -> nn.Sequential:
        return nn.Sequential()
