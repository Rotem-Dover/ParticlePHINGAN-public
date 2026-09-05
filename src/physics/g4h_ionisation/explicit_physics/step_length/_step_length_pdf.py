"""
Analytical PDFs for the step-length distribution. Provides a truncated exponential
PDF (exponential body with a point mass at L_eloss) for the general case and a
deterministic delta-function PDF for when no discrete interactions are possible.
"""
import numpy as np
from numpy.typing import NDArray

from common.utils import simple_pdf_normalization


def _point_mass_bin_width(x: NDArray[float], idx: int) -> float:
    """Local bin width at index `idx`, matching the left-Riemann convention in
    `simple_pdf_normalization` (integral weight = diff(x)[idx] = x[idx+1]-x[idx]).

    Representing a Dirac point mass as an in-bin density requires the LOCAL bin
    width, not the first bin width `x[1]-x[0]`. On a uniform grid the two are
    equal (so the CDF path is unchanged); on a log-spaced grid the first bin can
    be millions of times narrower, which over-weights the point mass and squashes
    the continuous body after normalization.
    """
    n = len(x)
    if n < 2:
        return 1.0
    if idx < n - 1:
        return x[idx + 1] - x[idx]
    return x[idx] - x[idx - 1]


def truncated_exponential_pdf(x: NDArray[float], sigma: float, L_eloss: float) -> NDArray[float]:
    """
    PDF of the step length L = min(Exp(1/Sigma), L_eloss).

    The step length is the minimum of an exponential random variable
    (from discrete PostStep interaction sampling) and a deterministic
    energy-loss step limit (from AlongStep).

    The distribution has:
    - A continuous part: f(L) = Sigma * exp(-Sigma * L)  for 0 <= L < L_eloss
    - A point mass: P(L = L_eloss) = exp(-Sigma * L_eloss)

    For binned representation, the point mass is placed in the bin
    containing L_eloss with appropriate density.

    Parameters
    ----------
    x : NDArray[float]
        Step length values in cm at which to evaluate the PDF.
    sigma : float
        Macroscopic cross-section Sigma(E) in 1/cm.
    L_eloss : float
        Deterministic energy-loss step limit in cm.

    Returns
    -------
    NDArray[float]
        Probability density values.
    """
    pdf = np.zeros_like(x)

    if sigma <= 0:
        # Pure deterministic: delta function at L_eloss
        idx = np.argmin(np.abs(x - L_eloss))
        pdf[idx] = 1.0 / _point_mass_bin_width(x, idx)
        return pdf

    # Continuous exponential part: f(L) = Sigma * exp(-Sigma * L) for L < L_eloss
    mask = x < L_eloss
    pdf[mask] = sigma * np.exp(-sigma * x[mask])

    # Point mass at L_eloss: P = exp(-Sigma * L_eloss)
    # Represented as a spike in the bin containing L_eloss (local bin width so
    # the spike integrates to P on any grid — see _point_mass_bin_width).
    point_mass = np.exp(-sigma * L_eloss)
    idx = np.argmin(np.abs(x - L_eloss))
    pdf[idx] += point_mass / _point_mass_bin_width(x, idx)

    return simple_pdf_normalization(x, pdf)


def deterministic_pdf(x: NDArray[float], L_eloss: float) -> NDArray[float]:
    """
    PDF when step length is purely deterministic (delta function at L_eloss).

    Used when T_max <= Tc (no delta-ray production possible),
    so Sigma = 0 and the step is always the energy-loss limit.

    Parameters
    ----------
    x : NDArray[float]
        Step length values in cm.
    L_eloss : float
        Deterministic energy-loss step limit in cm.

    Returns
    -------
    NDArray[float]
        Probability density values (delta approximation).
    """
    pdf = np.zeros_like(x)
    idx = np.argmin(np.abs(x - L_eloss))
    pdf[idx] = 1.0 / _point_mass_bin_width(x, idx)
    return pdf
