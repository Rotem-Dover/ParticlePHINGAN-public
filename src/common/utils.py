"""
Shared utility classes and functions: Interval for energy-bin bookkeeping,
Interp1dLinear for GPU-friendly piecewise-linear interpolation (used by
physics tables), PointInPhaseSpace dataclass, PDF normalisation and
convolution helpers, and an optional Numba-accelerated Earth Mover's Distance.
"""
from dataclasses import dataclass, field

import torch
import numpy as np
from numpy._typing import NDArray
from scipy.signal import fftconvolve


class Interval:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end   = end

    @classmethod
    def from_str(cls, string: str):
        return cls(*string.split("_to_"))

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2

    @property
    def log_center(self) -> float:
        return 10 ** ((np.log10(self.start) + np.log10(self.end)) / 2)

    def __contains__(self, item: float) -> bool:
        return self.start <= item <= self.end

    def __repr__(self):
        return f"Interval({self.start: .2e}, {self.end: .2e})"

    def __str__(self):
        return f"{self.start: .2e} to {self.end: .2e}"

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        return (self.start < vector) & (vector <= self.end)


def sort_intervals(intervals: list[Interval]) -> list[Interval]:
    return sorted(intervals, key=lambda interval: interval.center)


class Interp1dLinear:
    def __init__(self,
                 x: torch.Tensor, y: torch.Tensor, bounds_error: bool = False,
                 clamp_values: tuple[float, float] = (float('nan'), float('nan')), do_clamp: bool = False,
                 fill_values:  tuple[float, float] = (float('nan'), float('nan')), do_fill:  bool = False
                 ):
        self.x = x
        self.y = y

        # Sort the samples by x
        self.x, indices = torch.sort(self.x)
        self.y = self.y[indices]

        self.bounds_error = bounds_error
        self.clamp_values = clamp_values
        self.do_clamp     = do_clamp
        self.fill_values  = fill_values
        self.do_fill      = do_fill

        if not np.isnan(clamp_values[0]) or not np.isnan(clamp_values[1]):
            self.do_clamp = True

        if not np.isnan(fill_values[0]) or not np.isnan(fill_values[1]):
            self.do_fill = True

        # Pre-compute boundary slopes to avoid recomputing every call
        self._left_slope = self.left_side_slope(self.x, self.y)
        self._right_slope = self.right_side_slope(self.x, self.y)

    def __call__(self, x_new: torch.Tensor) -> torch.Tensor:
        x, y = self.x, self.y
        x_min, x_max = x[0], x[-1]

        if self.bounds_error and ((x_new < x_min) | (x_new > x_max)).any():
            raise ValueError("A value in x_new is outside the interpolation range.")

        # Find the indices of the points to the left and right of x_new
        left_idx = torch.searchsorted(x, x_new, right=True) - 1
        right_idx = left_idx + 1

        # Clamp the indices to the range of the data
        left_idx  = torch.clamp(left_idx,  0, len(x) - 1)
        right_idx = torch.clamp(right_idx, 0, len(x) - 1)

        # Get the x and y values at the indices
        x_left  = x[left_idx]
        x_right = x[right_idx]
        y_left  = y[left_idx]
        y_right = y[right_idx]

        # Linear interpolation: compute slope, avoiding division by zero
        denom = x_right - x_left
        same_idx = denom == 0
        denom[same_idx] = 1.0  # prevent div-by-zero; slope will be overwritten
        slope = (y_right - y_left) / denom

        # For edge points where left==right, extrapolate using boundary slopes
        if same_idx.any():
            slope[same_idx & (x_new <= x_min)] = self._left_slope
            slope[same_idx & (x_new >= x_max)] = self._right_slope

        y_new = y_left + slope * (x_new - x_left)

        # Apply fill values for out-of-bounds inputs (no tensor allocation)
        if self.do_fill:
            below = x_new < x_min
            above = x_new > x_max
            if below.any():
                y_new[below] = self.fill_values[0]
            if above.any():
                y_new[above] = self.fill_values[1]

        if self.do_clamp:
            y_new = torch.clamp(y_new, self.clamp_values[0], self.clamp_values[1])

        return y_new

    def to(self, device: torch.device | str):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if isinstance(self._left_slope, torch.Tensor):
            self._left_slope = self._left_slope.to(device)
        if isinstance(self._right_slope, torch.Tensor):
            self._right_slope = self._right_slope.to(device)
        return self

    @property
    def device(self):
        return self.x.device

    @staticmethod
    def left_side_slope(x: torch.Tensor, y: torch.Tensor):
        i = 1
        while (x[i] == x[0]) and i < 10_000:
            i += 1
        return (y[1] - y[0]) / (x[i] - x[0])

    @staticmethod
    def right_side_slope(x: torch.Tensor, y: torch.Tensor):
        i = -2
        while (x[i] == x[-1]) and i > -10_000:
            i -= 1
        return (y[-1] - y[-2]) / (x[-1] - x[i])


@dataclass
class PointInPhaseSpace:
    step_length:    float = field(default=None)  # cm
    primary_energy: float = field(default=None)  # eV

    def is_none(self) -> bool:
        return all([value is None for value in self.__dict__.values()])


def simple_pdf_normalization(x, pdf: NDArray[float]) -> NDArray[float]:
    # Normalize the PDF
    pdf /= (np.sum(np.diff(x) * pdf[:-1]))
    return pdf


# ------------------------------ Convolution of PDFs ------------------------------

def convolve_3_pdfs(x: NDArray[float], pdf1: NDArray[float], pdf2: NDArray[float], pdf3: NDArray[float]) -> NDArray[float]:
    """
    Convolve 3 PDFs.
    args:
        x:    NDArray[float] - Values at which to evaluate the PDF.
        pdf1: NDArray[float] - PDF of the first  distribution.
        pdf2: NDArray[float] - PDF of the second distribution.
        pdf3: NDArray[float] - PDF of the third  distribution.
    """
    # Convolve the first two PDFs
    convolved_pdf = convolve_2_pdfs(x, pdf1, pdf2)

    # Convolve with the third PDF
    convolved_pdf = convolve_2_pdfs(x, convolved_pdf, pdf3)

    return convolved_pdf


def convolve_2_pdfs(x: NDArray[float], pdf1: NDArray[float], pdf2: NDArray[float]) -> NDArray[float]:
    """
    Convolve 2 PDFs.
    args:
        x:    NDArray[float] - Values at which to evaluate the PDF.
        pdf1: NDArray[float] - PDF of the first  distribution.
        pdf2: NDArray[float] - PDF of the second distribution.
    """
    # Convolve the first two PDFs
    convolved_pdf = fftconvolve(pdf1, pdf2, mode='full')
    convolved_pdf = __fix_convolution(x, convolved_pdf)

    return convolved_pdf


def __fix_convolution(x: NDArray[float], convolved_pdf: NDArray[float]) -> NDArray[float]:
    """
    Fix the x-axis of the convolved PDF, and normalize the convolved-pdf to 1.
    args:
        x:   NDArray[float] - Values at which to evaluate the PDF.
        pdf: NDArray[float] - PDF of the distribution.
    """
    # Subsample the convolved PDF
    extended_x    = np.linspace(2 * x[0], 2 * x[-1], len(convolved_pdf))
    convolved_pdf = np.interp(x, extended_x, convolved_pdf, left=0, right=0)

    convolved_pdf = np.where(convolved_pdf >= 0, convolved_pdf, 0.)

    # Normalize the convolved PDF
    convolved_pdf /= np.sum(convolved_pdf[:-1] * np.diff(x))

    return convolved_pdf

try:
    import numba
    @numba.njit()
    def earths_movers_distance(sample1: NDArray[float], sample2: NDArray[float]) -> float:
        """
        Compute the Earth's Movers Distance between two samples.
        """
        n1 = len(sample1)
        n2 = len(sample2)

        n_dim = len(sample1[0])

        first_term = 0
        for i in range(n1):
            for j in range(n2):
                distance = 0
                for k in range(n_dim):
                    distance += (sample1[i][k] - sample2[j][k]) ** 2
                distance = distance ** 0.5
                first_term += distance
        first_term *= (2 / (n1 * n2))

        second_term = 0
        for i in range(n1):
            for j in range(n1):
                distance = 0
                for k in range(n_dim):
                    distance += (sample1[i][k] - sample1[j][k]) ** 2
                distance = distance ** 0.5
                second_term += distance
        second_term *= (-1 / (n1 ** 2))

        third_term = 0
        for i in range(n2):
            for j in range(n2):
                distance = 0
                for k in range(n_dim):
                    distance += (sample2[i][k] - sample2[j][k]) ** 2
                distance = distance ** 0.5
                third_term += distance
        third_term *= (-1 / (n2 ** 2))

        return first_term + second_term + third_term
except ImportError as e:
    print(e)
    def earths_movers_distance(sample1: NDArray[float], sample2: NDArray[float]) -> float:
        raise ImportError("Numba is not installed. Please install numba to use the Earth's Movers Distance function.")