"""
Analytical PDFs for the excitation component of continuous energy loss. Provides
a Poisson-based discrete PDF for the low-collision-frequency regime and a truncated
Gaussian PDF for the high-collision-frequency regime.
"""
import numpy as np
from scipy.special import factorial

from physics.g4h_ionisation.explicit_physics.continuous._utils import truncated_gaussian, special_normalization


def few_excitation_collisions_pdf(x: np.ndarray, mean_num_of_excitations: int | float, excitation_energy: float) -> np.ndarray:
    """
    Compute the PDF of the energy loss due to low frequency excitation collisions.
    In the code, mean_number_of_col is equivalent to poisson parameter and n_excitations is equivalent to k in the formula.
    args:
        x: np.ndarray - Values at which to evaluate the PDF.
        mean_num_of_excitations: int - Mean number of excitations.
        excitation_energy: float - Excitation energy.
    """
    assert np.isclose(x[0], 0), f"The first element of x must be 0. Got {x[0]} instead."

    n_excitations = x // excitation_energy  # number of excitations is a round off of energy loss divided by excitation energy
    n_excitations[x < 0] -= 1

    pdf = np.zeros_like(x)

    # Overflow protection
    n_excitations_threshold = 100
    overflow_protection = n_excitations < n_excitations_threshold

    cond1 = (n_excitations > 0)  & overflow_protection
    cond2 = (n_excitations > -1) & overflow_protection
    # In the following, we take +0, +1 instead of -1, +0 cause of n_excitations[x < 0] -= 1
    pdf[cond1] += mean_num_of_excitations ** (n_excitations[cond1] + 0) / factorial(n_excitations[cond1] + 0)  # First part in the sum
    pdf[cond2] += mean_num_of_excitations ** (n_excitations[cond2] + 1) / factorial(n_excitations[cond2] + 1)  # Second part in the sum
    pdf *= 0.5 * np.exp(-mean_num_of_excitations) / excitation_energy

    pdf = special_normalization(x, pdf, mean_num_of_excitations)

    assert np.all(pdf >= 0), "PDF contains negative values, which is not allowed."

    return pdf


def many_excitation_collisions_pdf(x: np.ndarray, mean_num_of_excitations: int | float, excitation_energy: float) -> np.ndarray:
    """
    Compute the PDF of the energy loss due to high frequency excitation collisions.
    args:
        x: np.ndarray - Values at which to evaluate the PDF.
        mean_num_of_excitations: int - Mean number of excitations.
        excitation_energy: float - Excitation energy.
    """
    mean_energy_loss = mean_num_of_excitations * excitation_energy
    std = np.sqrt(mean_num_of_excitations * excitation_energy ** 2)
    return truncated_gaussian(x, mean_energy_loss, std)
