"""
Utility functions for the continuous straggling model: a truncated Gaussian PDF
and a special normalization routine that partitions probability between a zero-loss
delta component (exp(-lambda)) and the continuous loss tail.
"""
import math

import numpy as np

from common.utils import simple_pdf_normalization


def truncated_gaussian(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Compute the PDF of the truncated Gaussian distribution.
    """
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    pdf[(x < 0) | (x > 2 * mean)] = 0
    pdf = simple_pdf_normalization(x, pdf)
    return pdf


def special_normalization(x, pdf: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Normalize the PDF, with special treatment for the zero-energy loss case.
    """
    diff_x = np.diff(x)

    norm_zero = math.exp(-lmbda)
    norm_non_zero = 1 - norm_zero

    pdf *= (norm_non_zero / np.sum(diff_x[1:] * pdf[1:-1]))

    pdf[0] = norm_zero / (x[1] - x[0])
    return pdf
