"""
Analytical PDFs for the ionization component of continuous energy loss. Derives the
compound-Poisson distribution via inverse Fourier transform of the characteristic
function for the low-collision-frequency regime, and convolves it with a Gaussian
correction for the high-collision-frequency regime.
"""
import numpy as np

from numpy.typing  import NDArray
from scipy.special import sici
from scipy.fft     import fft, fftfreq, ifftshift
from scipy.interpolate import interp1d

from common.utils import simple_pdf_normalization, convolve_2_pdfs
from physics.g4h_ionisation.explicit_physics.continuous._utils import truncated_gaussian, special_normalization


def few_ionization_collisions_pdf(
        x: NDArray[float], w3: float, w: float, lmbda: float, f_max: float = 20., N: int = 2_000_000
) -> NDArray[float]:
    assert x[0] == 0
    x_, pdf = E_ion_compound_poisson_pdf(w3, w, lmbda, f_max, N)
    pdf = interp1d(x_, pdf, kind='linear', bounds_error=False, fill_value=(0., 0.))(x)
    pdf[x < w3] = 0
    pdf = special_normalization(x, pdf, lmbda)  # NOTE! Since normalization is a must here, this function should be called only with a full range of x
    assert np.all(pdf >= 0), "PDF contains negative values, which is not allowed."
    return pdf


def many_ionization_collisions_pdf(
    x: NDArray[float],
    emean_ion: float, sig2e_ion: float,
    w3: float, w: float, lmbda: float, f_max: float = 20., N: int = 2_000_000
) -> NDArray[float]:
    x_, pdf1 = E_ion_compound_poisson_pdf(w3, w, lmbda, f_max, N)
    pdf2     = E_ion_gaussian_pdf(x_, emean_ion, np.sqrt(sig2e_ion))

    pdf = interp1d(x_, convolve_2_pdfs(x_, pdf1, pdf2), kind='linear', bounds_error=False, fill_value=(0., 0.))(x)
    pdf = np.where(pdf >= 0, pdf, 0.)
    return simple_pdf_normalization(x, pdf)


def E_ion_gaussian_pdf(x: NDArray[float], mean: float, std: float) -> NDArray[float]:
    """
    Compute the PDF of the gaussian term in E_ion.
    """
    return truncated_gaussian(x, mean, std)


def E_ion_compound_poisson_pdf(w3: float, w: float, lmbda: float, f_max: float = 20., N: int = 2_000_000) -> tuple[NDArray[float], NDArray[float]]:
    """
    Compute the PDF of the compound poisson process term in E_ion using the inverse Fourier transform of the
    characteristic function.
    args:
        w3: float - Scaling constant.
        w: float - Scaling constant.
        lmbda: float - Poisson parameter.
        f_max: float - Maximum frequency range.
        N: int - Number of points for Fourier transform.
    returns:
        x: NDArray[float] - Real domain values where PDF is evaluated.
        E_loss_pdf_t: NDArray[float] - PDF of the energy loss.
    """
    # Define frequency domain with N points, symmetric around 0
    fs = np.linspace(-f_max, f_max, N)

    # Calculate the characteristic function in the frequency domain
    E_ion_characteristic_values = E_ion_characteristic(fs, w3, w, lmbda)

    # Reconstructing a PDF from a truncated, discretely sampled characteristic
    # function needs three things working together, none of which is
    # sufficient alone:
    #
    # 1. Lanczos sigma taper. The characteristic function is truncated at
    #    +-f_max, so a raw inversion suffers Gibbs ringing of amplitude
    #    comparable to the true density -- a spurious floor at energies the
    #    particle cannot physically reach in one step. Multiplying by
    #    sinc(f/f_max) (=1 at f=0, ->0 at the cutoff) suppresses that ringing
    #    at the source and is the only step that removes the floor's
    #    magnitude; the other two below only decide the *sign* of what is left.
    # 2. ifftshift. `fs` is centred on 0 (zero frequency at the array MIDDLE),
    #    but np.fft.fft expects the zero-frequency sample at index 0. Feeding
    #    the centred array directly multiplies the output by (-1)^n, an
    #    adjacent-point sign alternation.
    # 3. Taking the real part rather than np.abs() below. These two choices
    #    are only correct together: with np.abs(), the (-1)^n alternation from
    #    skipping ifftshift is invisible (|Re| is the same either way), so
    #    ifftshift alone would look like a no-op; conversely, dropping
    #    np.abs() without ifftshift turns the alternation into a comb -- odd
    #    bins exactly zero, even bins at twice the correct density -- because
    #    np.abs() is the only thing that rectifies every other bin's sign back
    #    to positive. ifftshift removes the alternation so the real part is the
    #    density directly, with no comb and no rectified floor.
    E_ion_characteristic_values = E_ion_characteristic_values * np.sinc(fs / f_max)
    E_ion_characteristic_values = ifftshift(E_ion_characteristic_values)

    # Perform the inverse Fourier transform
    E_ion_pdf_values = fft(E_ion_characteristic_values).real / (2 * np.pi)

    dx = 2 * np.pi * N / (fs.max() - fs.min())

    # Real domain (x) range
    x = fftfreq(N, d=1 / dx)

    # Return positive x values for PDF
    x = x[:N // 2]
    E_ion_pdf_values = E_ion_pdf_values[:N // 2]

    # Fix numerical errors - except from x=0, pdf below w3 is 0.
    # Delta function at 0 is added at few_ionization_collisions_pdf
    E_ion_pdf_values[x < w3] = 0

    # Ensure the PDF is non-negative (clips residual ringing negatives to zero).
    E_ion_pdf_values = np.where(E_ion_pdf_values >= 0, E_ion_pdf_values, 0.)

    # Normalize the PDF
    E_ion_pdf_values = special_normalization(x, E_ion_pdf_values, lmbda)

    return x, E_ion_pdf_values


def E_ion_characteristic(t: NDArray[float], w3: float, w: float, lmbda: float) -> NDArray[float]:
    """
    Compute the characteristic function of the compound poisson process term in E_ion.
    args:
        t: NDArray[float] - Values at which to evaluate the characteristic function.
        w3: float - Scaling constant.
        w: float - Scaling constant.
        lmbda: float - Poisson parameter
    """
    E_ion_characteristic_t = E_i_ion_characteristic(t, w3, w)
    return np.exp(lmbda * (E_ion_characteristic_t - 1))


def E_i_ion_pdf(x: NDArray[float], w3: float, w: float) -> NDArray[float]:
    """
    Compute the PDF of E_i_ion fluctuations.
    args:
        x: NDArray[float] - Values at which to evaluate the PDF.
        w3: float - Scaling constant.
        w: float - Scaling constant.
    """
    return w3 / (w * x ** 2)


def E_i_ion_characteristic(t: NDArray[float], w3: float, w: float) -> NDArray[float]:
    """
    Compute the characteristic function of E_i_ion fluctuations.
    args:
        t: NDArray[float] - Values at which to evaluate the characteristic function.
        w3: float - Scaling constant.
        w: float - Scaling constant
    """
    # Compute the sine and cosine integrals
    a, b = w3, w3 / (1 - w)
    si_a, ci_a = sici(t * a)
    si_b, ci_b = sici(t * b)

    real_part = (w3 / w) * (np.cos(a * t) / a - np.cos(b * t) / b + t * si_a - t * si_b)
    imag_part = (w3 / w) * (-t * ci_a + t * ci_b + np.sin(a * t) / a - np.sin(b * t) / b)

    return real_part + 1j * imag_part
