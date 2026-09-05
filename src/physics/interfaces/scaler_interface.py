"""
Abstract base class for data scalers that pre-process inputs and post-process
outputs of the ML generators. Supports device transfer and dtype conversion.
Concrete scalers implement scale/rescale for each generator stage.
"""
import torch
import logging

from typing_extensions  import Self
from abc                import ABC, abstractmethod

log = logging.getLogger(__name__)


class ScalerInterface(ABC):
    """Strategy scaler for data pre- and post-processing to the ML model."""

    def to_dtype(self, dtype: torch.dtype) -> Self:
        """Convert all tensor attributes to the given dtype."""
        for attr_name, attr_val in vars(self).items():
            if isinstance(attr_val, torch.Tensor):
                setattr(self, attr_name, attr_val.to(dtype))
        return self

    @abstractmethod
    def to(self, device: str) -> Self:
        raise NotImplementedError

    @abstractmethod
    def scale(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def rescale(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DummyScaler(ScalerInterface):
    def to(self, device: str) -> Self:
        return self

    def scale(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor

    def rescale(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor
