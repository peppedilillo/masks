import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from . import encode
from . import pad
from .stats import variance


def cross_correlation(detector: npt.NDArray, decoder: npt.NDArray) -> npt.NDArray:
    """
    Performs cross-correlation of detector image.

    :param detector: detector hit map
    :param decoder:
    :return: sky cross-correlation map
    """
    n, m = detector.shape
    return correlate(pad(decoder), detector)[n - 1 : -n + 1, m - 1 : -m + 1]


def mlem(detector: npt.NDArray, mask: npt.NDArray, niters: int = 32):
    """
    Maximum-likelihood reconstruction

    :param detector: detector hit map
    :param mask:
    :param niters: number of iterations
    :return: sky probability map
    """

    def _mlem_prob(_mask):
        n, m = _mask.shape
        p = np.zeros((n, m) * 2)
        for i in range(n):
            for j in range(m):
                p[i, j, :] = pad(_mask)[i : i + n, j : j + m]
        p /= np.sum(_mask)
        return p

    prob = _mlem_prob(mask)
    n, m = detector.shape
    y0 = np.ones((n, m)) * np.mean(detector)
    y_j = np.zeros((n, m))
    y_i = y0
    for _ in range(niters):
        y_j = y_i * np.sum(
            detector
            * prob
            / np.sum(
                y_i * prob,
                axis=(2, 3),
            ),
            axis=(2, 3),
        )
        y_i = y_j.copy()
    return y_j


def mem(
    detector: npt.NDArray,
    mask: npt.NDArray,
    decoder: npt.NDArray,
    maxdepth=15000,
    step=0.1,
) -> tuple[npt.NDArray, list]:
    """
    Maximum-entropy reconstruction

    :param detector:
    :param mask:
    :param decoder:
    :param maxdepth:
    :param step:

    :return: sky mem map, chisquared trace
    """
    var = variance(detector, decoder)
    trace = [np.inf]
    lm = 0
    y = np.zeros(detector.shape)
    for i in range(maxdepth):
        res = encode(y, mask) - detector
        y_ = np.exp(-1 - lm * cross_correlation(res, decoder) / var)
        chisq = np.sum(np.square(res) / var)
        lm += step
        if chisq > trace[-1]:
            break
        y = y_
        trace.append(chisq)
    trace.pop(0)
    return y / np.sum(y), trace


def iros(
        detector: npt.NDArray,
        mask: npt.NDArray,
        decoder: npt.NDArray,
        threshold: float=5.0,
):
    def shadowgram(x, y, counts, mask):
        s = np.zeros(mask.shape)
        s[x, y] = counts
        return encode(s, mask)

    def record_source(source, skymap_cc):
        x, y, counts = *source, skymap_cc[*source]
        return {
            "x": source[0].item(),
            "y": source[1].item(),
            "counts": counts.item(),
        }

    detector = detector.copy()
    nphotons = np.sum(detector)
    cleaned_sources = []
    skymap_cc = cross_correlation(detector, decoder)
    snr_cc = skymap_cc / np.sqrt(variance(detector, decoder))
    targets = np.argwhere(snr_cc > threshold)
    for target in targets:
        cleaned_sources.append(record_source(target, skymap_cc))
        shadow = shadowgram(*target, skymap_cc[*target], mask)
        detector = detector - shadow
        skymap_cc = cross_correlation(detector, decoder)

    brate = (nphotons - sum([source["counts"] for source in cleaned_sources])) / np.prod(mask.shape)
    skymap_iros = np.ones(mask.shape) * brate
    for source in cleaned_sources:
        skymap_iros[source["x"], source["y"]] = source["counts"]
    return skymap_iros, cleaned_sources
