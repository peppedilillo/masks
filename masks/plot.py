import matplotlib.pyplot as plt
import numpy.typing as npt


def plotmaps(
        annotated_maps: list[tuple[npt.NDArray, str]],
        ncols: int = 3,
        **kwargs,
):
    """
    A function to plot multiple matrices together.

    :param annotated_maps:
    :param ncols:
    :param kwargs: passed to `plt.subplots()`.
    :return:
    """
    nmaps = len(annotated_maps)
    if nmaps < ncols:
        ncols = nmaps
    nrows = (nmaps - 1) // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, **kwargs)
    axs = axs.flatten() if hasattr(axs, "__len__") else [axs]
    for ax, (arr, title) in zip(axs, annotated_maps):
        c0 = ax.imshow(arr)
        fig.colorbar(c0, ax=ax, location="bottom", pad=0.075)
        ax.set_title(title)
    for ax in axs[nmaps:]:
        ax.axis("off")
    return fig, axs
