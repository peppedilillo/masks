from numpy import typing as npt
import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from .utils import is_prime


def ura(r: int, s: int, m: int = 1) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Generates a URA mask and its decoding array.

    :param r: number of rows
    :param s: number of columns
    :param m: number of tiles
    :return: the mask array and its decoding array.
    """
    assert is_prime(r)
    assert is_prime(s)
    assert r - s == 2

    c_r_i = np.zeros(r) - 1
    c_s_j = np.zeros(s) - 1
    for x in range(1, r):
        c_r_i[x**2 % r] = 1
    for y in range(1, s):
        c_s_j[y**2 % s] = 1

    _a_ij = np.zeros([r, s])
    for i in range(r):
        for j in range(s):
            if i == 0:
                _a_ij[i, j] = 0
            elif j == 0:
                _a_ij[i, j] = 1
            elif c_r_i[i] * c_s_j[j] == 1:
                _a_ij[i, j] = 1

    a_ij = np.zeros([m * r, m * s])
    for i in range(m * r):
        for j in range(m * s):
            a_ij[i, j] = _a_ij[i % r, j % s]
    a_ij = np.roll(a_ij, int((r + 1) / 2), axis=0)
    a_ij = np.roll(a_ij, int((s + 1) / 2), axis=1)

    g_ij = a_ij.copy()
    g_ij[g_ij == 0] = -1
    return a_ij, g_ij


def pad(mask: npt.NDArray) -> npt.NDArray:
    """
    Pads a mask by wrapping around border
    :param mask:
    :return:
    """
    ws = [int(dim / 2) for dim in mask.shape]
    # noinspection PyTypeChecker
    return np.pad(
        mask,
        pad_width=[(int(dim / 2), int(dim / 2)) for dim in mask.shape],
        mode="wrap",
    )


def encode(sky: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Encodes a sky count map through a coded mask.

    :param sky: a sky count map
    :param mask:
    :return: a detector map
    """
    n, m = sky.shape
    norm = np.sum(mask[mask == 1])
    detector = correlate(pad(mask), sky)[n - 1 : -n + 1, m - 1 : -m + 1] / norm
    return detector
