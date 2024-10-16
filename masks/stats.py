import numpy as np
import numpy.typing as npt
from scipy.signal import correlate


def variance(detector: npt.NDArray, decoder: npt.NDArray) -> npt.NDArray:
    """
    :param detector:
    :param decoder:
    :return: sky cross-correlation variance map
    """
    n, m = detector.shape
    return correlate(np.square(pad(decoder)), detector)[n - 1 : -n + 1, m - 1 : -m + 1]


def significance(n: int, b: int) -> float:
    """
    Standard deviation significance assuming Poisson (by Wilk's theorem).

    :param n: number of source photons
    :param b: number of background photons
    :return: significance in standard deviations
    """
    return np.sqrt(2 * (n * np.log(n / b) - (n - b)))
