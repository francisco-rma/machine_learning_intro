from typing import Callable, Sequence
import matplotlib.pyplot as plt
from linear_models.data import Data
import numpy as np


def scatter_plot(
    data: list[Data], h: Callable, file: str, params: Sequence[float], title: str, show=False
):
    thresholds = data[:, 0]
    a_scores = data[:, 1]
    b_scores = data[:, 2]

    labels = [h(d) >= 0.0 for d in data]
    colors = ["tab:blue" if label else "tab:orange" for label in labels]
    plt.figure(figsize=(6, 6))
    plt.scatter(a_scores, b_scores, c=colors, alpha=0.6, edgecolor="k")

    threshold_vals = np.linspace(min(thresholds), max(thresholds), 100)
    wealth_vals = np.linspace(min(a_scores), max(a_scores), 100)
    profile_vals = list(
        map(
            lambda threshold, wealth: (-params[0] * threshold - params[1] * wealth) / params[2],
            threshold_vals,
            wealth_vals,
        )
    )

    plt.plot(
        wealth_vals,
        profile_vals,
        color="green",
        linestyle="--",
        label=f"{params[0]:.2f} = {params[1]:.2f}*a_score + {params[2]:.2f}*b_score",
    )

    plt.xlabel("a Score")
    plt.ylabel("b Score")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sum >= 0",
            markerfacecolor="tab:blue",
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sum < 0",
            markerfacecolor="tab:orange",
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            lw=2,
            linestyle="--",
            label=f"{params[0]:.2f} = {params[1]:.2f}*wealth_score + {params[2]:.2f}*profile_score",
        ),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(file)
    if show:
        plt.show(block=False)
    # plt.close()


def result_plot(data: list[Data], result: list[float], file: str, title: str, show=False):
    a_scores = data[:, 1]
    b_scores = data[:, 2]
    labels = [res >= 0.0 for res in result]
    colors = ["tab:blue" if label else "tab:orange" for label in labels]
    plt.figure(figsize=(6, 6))
    plt.scatter(a_scores, b_scores, c=colors, alpha=0.6, edgecolor="k")

    plt.xlabel("a Score")
    plt.ylabel("b Score")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sum >= 0",
            markerfacecolor="tab:blue",
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sum < 0",
            markerfacecolor="tab:orange",
            markersize=8,
            markeredgecolor="k",
        ),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(file)

    if show:
        plt.show(block=False)
