from typing import Sequence

import numpy as np
import numpy.typing as npt

from .. import pad


def absorbed(photons: npt.NDArray, mask: npt.NDArray) -> npt.NDArray[np.bool]:
    """
    Given a list of photons (direction/impact site tuples) returns a mask selecting
    the ones not going through.

    :param photons:
    :param mask:
    :return: a boolean mask
    """
    _A = pad(mask)
    dirx, diry, impx, impy = photons.T
    # noinspection PyTypeChecker
    return _A[dirx + impx, diry + impy] == 0


def transport(photons: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Given a list of photons (direction/impact site tuples) returns the resulting
    detector image.

    :param photons:
    :param mask:
    :return: detector hitmap
    """
    mask_size = mask.shape
    transmitted = photons[~absorbed(photons, mask)]
    detector, *_ = np.histogram2d(
        x=transmitted[:, 2],
        y=transmitted[:, 3],
        bins=[np.arange(mask_size[0] + 1), np.arange(mask_size[1] + 1)],
    )
    return detector


def transmit(photons: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Ground truth. Returns a count map of the sky directions of the photons which
    made it through the mask.

    :param photons:
    :param mask:
    :return: sky count map
    """
    mask_size = mask.shape
    transmitted = photons[~absorbed(photons, mask)]
    counts, *_ = np.histogram2d(
        x=transmitted[:, 0],
        y=transmitted[:, 1],
        bins=[np.arange(mask_size[0] + 1), np.arange(mask_size[1] + 1)],
    )
    return counts


def source(photons: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Ground truth. Returns a count map of the sky directions of all photons, including
    thouse stopped by the mask.

    :param photons:
    :param mask:
    :return: sky count map
    """
    mask_size = mask.shape
    counts, *_ = np.histogram2d(
        x=photons[:, 0],
        y=photons[:, 1],
        bins=[np.arange(mask_size[0] + 1), np.arange(mask_size[1] + 1)],
    )
    return counts


def source_photons(size: int, shape: tuple[int, int], direction: npt.NDArray):
    """
    Generates a photon list from a unique direction.

    :param size:
    :param shape: the mask's shape.
    :param direction: source direction.
    :return:
    """
    n, m = shape
    ps = np.dstack(
        (
            *(direction * np.ones((size, 2))).T,
            np.random.randint(n, size=size),
            np.random.randint(m, size=size),
        )
    )[0]
    return ps.astype(int)


def background_photons(size: int, shape: tuple[int, int]):
    """
    Generates a photon list from a random directions.

    :param size:
    :param shape: the mask's shape.
    :return:
    """
    n, m = shape
    ps = np.dstack(
        (
            np.random.randint(n, size=size),
            np.random.randint(m, size=size),
            np.random.randint(n, size=size),
            np.random.randint(m, size=size),
        )
    )[0]
    return ps.astype(int)


def random_photons(
    fsources: Sequence,
    brate: float,
    shape: tuple[int, int],
    dsources: Sequence | None = None,
) -> tuple[npt.NDArray, dict]:
    """
    Simulates a photon list.

    :param fsources: number of photons to simulate for each source.
    :param brate: number of photon expected from background per detector/mask element.
    :param shape: detector shape.
    :param dsources: optional. directions of the sources.
    :return: photon list and a dictionary of simulations info
    """
    info = {}
    n, m = shape
    bphotons = background_photons(int(brate * n * m), shape)
    if dsources is None:
        dsources = np.dstack(
            [
                np.random.randint(n, size=len(fsources)),
                np.random.randint(m, size=len(fsources)),
            ]
        )[0]
        info = {"source_directions": dsources}
    sphotons = np.concatenate(
        [
            source_photons(nsignal, shape, signal_dir)
            for (nsignal, signal_dir) in zip(fsources, dsources)
        ]
    )

    photon_list = np.concatenate((bphotons, sphotons))
    return photon_list, info
