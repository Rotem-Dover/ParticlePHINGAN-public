"""
Analytical PDFs for discrete (secondary electron) energy-loss distributions. Includes
the Rutherford-like secondary straggling function, the single-collision PDF, the
truncated f-variable distribution, and the associated Bernoulli distribution for
the rejection-sampling loop count.
"""
import numpy as np

from common.utils import simple_pdf_normalization


# TODO: Add documentation and changed functions naming


def secondary_straggling_function(x: np.ndarray, w3: float, w: float, beta: float) -> np.ndarray:
    pdf = (w3 - beta**2 * x) / ((w - beta**2 * np.log(1 + w)) * x**2)
    min_loss = w3 / (1 + w)
    max_loss = w3
    pdf[np.logical_or(x < min_loss, x > max_loss)] = 0
    pdf = simple_pdf_normalization(x, pdf)
    return pdf


def moller_straggling_function(x: np.ndarray, primary_energy: float, Tc: float,
                               gamma: float) -> np.ndarray:
    """
    Møller (e-e-) delta-ray energy-transfer straggling PDF for an electron
    primary — the analytical counterpart of the Møller branch of the rejection
    sampler in `_kernels._sample_moller_fraction_jit`.

    The relativistic Møller cross-section, after collecting kinematic factors,
    depends on the energy fraction `xf = E / T0` through `z(xf)/xf²` with

        z(xf) = 1 − g·xf + xf²·(1 − g + (1 − g·y)/y²),   y = 1 − xf,
        g     = (2γ − 1)/γ²,

    over `xf ∈ [Tc/T0, 1/2]` (lower bound: the production cut; upper bound: the
    identical-particle convention T_max = T0/2). Expanding z(xf)/xf² gives the
    manifestly xf ↔ (1−xf) symmetric form

        f(xf) ∝ 1/xf² − g/xf + (1 − g) + 1/(1−xf)² − g/(1−xf).

    See moller_mc_derivation.pdf, Eqs. (4)/(17). The two `1/x²` poles describe
    soft scattering of either outgoing electron and lift the tail toward
    E → T0/2 relative to the heavy-particle Rutherford form, which instead
    carries a (1 − β² E/T_max) suppression that vanishes there.

    Unlike `secondary_straggling_function`, the spectrum is parameterised by the
    primary energy and γ directly (the kinematics are not captured by w3/w/β).
    """
    xf = x / primary_energy
    y = 1.0 - xf
    g = (2.0 * gamma - 1.0) / gamma ** 2
    pdf = 1.0 / xf ** 2 - g / xf + (1.0 - g) + 1.0 / y ** 2 - g / y
    min_loss = Tc
    max_loss = 0.5 * primary_energy
    pdf[np.logical_or(x < min_loss, x > max_loss)] = 0
    pdf = simple_pdf_normalization(x, pdf)
    return pdf


def collision_pdf(x: np.ndarray, w3: float, w: float) -> np.ndarray:
    return w3 / (w*x**2)


def f_pdf(x: np.ndarray, w: float, beta: float) -> np.ndarray:
    return beta ** 2 / (w * (1 - x)**2)


def f_trunc_pdf(x: np.ndarray, w3: float, w: float, beta: float) -> np.ndarray:
    return beta**2 / (w - beta**2 * np.log(1+w)) * (x / (1-x)**2)


def bernouli_distribution(w: float, beta: float, n: int | np.ndarray) -> float:
    """
    Bernoulli distribution
    """
    prob = 1 - (beta ** 2 / w) * np.log(1 + w)
    return prob * (1 - prob) ** (n-1)