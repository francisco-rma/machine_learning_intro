from typing import Callable, Sequence
import matplotlib.pyplot as plt
from linear_models.data import Credit
import numpy as np


def scatter_plot(data: list[Credit], h: Callable, file: str, params: Sequence[float], title: str):
    thresholds = [d.threshold for d in data]
    wealth_scores = [d.wealth_score for d in data]
    profile_scores = [d.profile_score for d in data]
    labels = [h(d) >= 0.0 for d in data]
    colors = ["tab:blue" if label else "tab:orange" for label in labels]
    plt.figure(figsize=(6, 6))
    plt.scatter(wealth_scores, profile_scores, c=colors, alpha=0.6, edgecolor="k")

    threshold_vals = np.linspace(min(thresholds), max(thresholds), 100)
    wealth_vals = np.linspace(min(wealth_scores), max(wealth_scores), 100)
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
        label=f"{params[0]:.2f} = {params[1]:.2f}*wealth_score + {params[2]:.2f}*profile_score",
    )

    plt.xlabel("Wealth Score")
    plt.ylabel("Profile Score")
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
    # plt.show()
    plt.close()
