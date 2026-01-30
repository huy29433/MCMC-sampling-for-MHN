import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from cycler import cycler


mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["font.size"] = 6.02249



okabe_ito_colors ={
    # "yellow": "#f0e442",
    "green": "#009e74",
    "blue": "#56b3e9",
    # "pink": "#cc79a7",
    "orange": "#d55e00",
    # "eggyolk": "#e69f00",
}
cmap = mpl.colors.ListedColormap(okabe_ito_colors.values())
mpl.rcParams["axes.prop_cycle"] = cycler("color", okabe_ito_colors.values())

def plot_theta_dist(
    quantiles: np.ndarray,
    tile_len: int,
    n_events: int,
    cmhn: bool = False,
    border_len: float = 0.1,
    br_ir_sep: float = 0.2,
    abs_max_ir: Optional[float] = None,
    min_br: Optional[float] = None,
    max_br: Optional[float] = None,
    events: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    linewidth: float = 0.3,
):

    assert quantiles.shape[0] == n_events if cmhn else n_events + 1
    assert quantiles.shape[1] == n_events * tile_len

    br_mask = np.repeat(np.eye(n_events if cmhn else n_events +
                        1, n_events, dtype=bool), tile_len, axis=1)

    if abs_max_ir is None:
        abs_max_ir = np.abs(quantiles[~br_mask]).max()
    if min_br is None:
        min_br = quantiles[br_mask].min()
    if max_br is None:
        max_br = quantiles[br_mask].max()
    if events is None:
        events = np.arange(n_events).tolist()

    if ax is None:
        _, ax = plt.subplots()

    for i in range(n_events + 1 if not cmhn else n_events):
        for j in range(n_events):
            if i == j:
                ax.imshow(
                    quantiles[i, j * tile_len: (j+1)
                              * tile_len].reshape(1, -1),
                    extent=(0, 1, j + j * border_len, j+1 + j * border_len),
                    cmap="Greens", vmin=min_br, vmax=max_br,
                    interpolation="None")
                ax.add_patch(mpl.patches.Rectangle(
                    (0,
                     j + j * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black", facecolor="none")
                )
            else:
                ax.imshow(
                    quantiles[i, j * tile_len: (j+1)
                              * tile_len].reshape(1, -1),
                    extent=(
                        j+1 + br_ir_sep + (j+1) * border_len,
                        j+2 + br_ir_sep + (j+1) * border_len,
                        i + i * border_len,
                        i+1 + i * border_len
                    ),
                    cmap="RdBu_r",
                    vmin=-abs_max_ir,
                    vmax=abs_max_ir,
                    interpolation="None"
                )
                ax.add_patch(mpl.patches.Rectangle(
                    (j+1 + br_ir_sep + (j+1) * border_len,
                     i + i * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black", facecolor="none")
                )

    ax.set_xlim(0 - border_len, (border_len + 1) * (n_events + 1) + br_ir_sep)
    ax.set_ylim((n_events + 1) * (border_len + 1) + border_len, 0 - border_len)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_theta_var(
    variances: np.ndarray,
    n_events: int,
    cmhn: bool=False,
    border_len: float = 0.1,
    br_ir_sep: float = 0.2,
    max_var: Optional[float] = None,
    events: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    linewidth: float = 0.3,
):

    if max_var is None:
        max_var = variances.max()

    if events is None:
        events = np.arange(n_events).tolist()

    if ax is None:
        _, ax = plt.subplots()

    cmap = mpl.cm.Greys

    for i in range(n_events + 1 if not cmhn else n_events):
        for j in range(n_events):
            if i == j:
                ax.add_patch(mpl.patches.Rectangle(
                    (0,
                     j + j * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black", facecolor=cmap(variances[i, j]/max_var))
                )
            else:
                ax.add_patch(mpl.patches.Rectangle(
                    (j+1 + br_ir_sep + (j+1) * border_len,
                     i + i * border_len),
                    1, 1, linewidth=linewidth,
                    edgecolor="black", facecolor=cmap(variances[i, j]/max_var))
                )

    ax.set_xlim(0 - border_len, (border_len + 1) * (n_events + 1) + br_ir_sep)
    ax.set_ylim((n_events + 1) * (border_len + 1) + border_len, 0 - border_len)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def min_max_br(quantiles: np.ndarray, tile_len: int) -> tuple[float, float]:
    """Calculate the base rate minimum and maximum from quantiles.

    Args:
        quantiles (np.ndarray): Array of quantiles with shape (n_events (+1), n_events * 20).
        tile_len (int): Number of tiles used in quantile calculation.

    Returns:
        tuple[float, float]: Base rate minimum and base rate maximum.
    """
    br_mask = np.eye(quantiles.shape[0],
                     quantiles.shape[1] // tile_len, dtype=bool)
    br_mask = np.repeat(br_mask, tile_len, axis=1)
    br_min = quantiles[br_mask].min()
    br_max = quantiles[br_mask].max()
    return br_min, br_max


def max_abs_ir(quantiles: np.ndarray, tile_len: int) -> float:
    """Calculate the maximum absolute influence factor from quantiles.

    Args:
        quantiles (np.ndarray): Array of quantiles with shape (n_events (+1), n_events * 20).
        tile_len (int): Number of tiles used in quantile calculation.

    Returns:
        float: Maximum absolute influence factor.
    """
    br_mask = np.eye(quantiles.shape[0],
                     quantiles.shape[1] // tile_len, dtype=bool)
    br_mask = np.repeat(br_mask, tile_len, axis=1)
    
    return np.abs(quantiles[~br_mask]).max()

