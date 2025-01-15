import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_multi_legs(
    time_series_block,
    timestep,
    spacing=10,
    legs=("LF", "LM", "LH", "RF", "RM", "RH"),
    ax=None,
):
    """Plot a time series of scores for multiple legs.

    Parameters
    ----------
    time_series_block : np.ndarray
        Time series of scores for multiple legs. The shape of the array
        should be (n, m), where n is the number of time steps and m is the
        length of ``legs``.
    timestep : float
        Timestep of the time series in seconds.
    spacing : float, optional
        Spacing between the time series of different legs. Default: 10.
    legs : list[str], optional
        List of leg names. Default: ["LF", "LM", "LH", "RF", "RM", "RH"].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    t_grid = np.arange(time_series_block.shape[0]) * timestep
    spacing *= -1
    offset = np.arange(6)[np.newaxis, :] * spacing
    score_hist_viz = time_series_block + offset
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    for i in range(len(legs)):
        ax.axhline(offset.ravel()[i], color="k", linewidth=0.5)
        ax.plot(t_grid, score_hist_viz[:, i])
    ax.set_yticks(offset[0], legs)
    ax.set_xlabel("Time (s)")
    return ax
