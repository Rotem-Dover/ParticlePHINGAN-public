"""
Kolmogorov-Smirnov distance utility for benchmarking and the playground.
"""
import torch


def kolmogorov_smirnov_distance(data: torch.Tensor, cdf) -> torch.Tensor:
    sorted_data = data.view(-1).sort().values
    theoretical_cdf = cdf(sorted_data)
    empirical_cdf_values = _get_empirical_cdf_values(sorted_data)

    return torch.max(torch.abs(theoretical_cdf - empirical_cdf_values))


def _get_empirical_cdf_values(data: torch.Tensor) -> torch.Tensor:
    return torch.arange(1, len(data) + 1, device=data.device, dtype=data.dtype) / len(data)
