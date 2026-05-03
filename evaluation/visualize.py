"""
Visualisation utilities for Pareto Set Learning experiments.

Three plots:
  - 3D Pareto surface over the simplex grid
  - 2D pairwise projections (safety/speed, safety/comfort, speed/comfort)
  - Adaptation plot: bar chart of per-objective costs at extreme λ presets,
    showing that the SAME network changes behaviour as λ changes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


OBJ_LABELS = ["f_safety", "f_speed", "f_comfort"]
OBJ_COLORS = ["tab:red", "tab:blue", "tab:green"]


def plot_pareto_3d(grid_result: dict, title: str = "PSL Pareto surface"):
    """
    3D scatter of the learned Pareto surface.

    grid_result : dict from evaluate_policy_grid()
    """
    costs = grid_result["costs"]
    lambdas = grid_result["lambdas"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Colour each point by which objective dominates its preference vector.
    dominant = lambdas.argmax(axis=1)
    colors = np.array(OBJ_COLORS)[dominant]

    sc = ax.scatter(costs[:, 0], costs[:, 1], costs[:, 2],
                    c=colors, s=50, alpha=0.85, edgecolors="k", linewidths=0.4)

    try:
        ax.plot_trisurf(costs[:, 0], costs[:, 1], costs[:, 2],
                        alpha=0.15, color="grey")
    except Exception:
        pass

    ax.set_xlabel("f_safety");  ax.set_ylabel("f_speed");  ax.set_zlabel("f_comfort")
    ax.set_title(title)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=c, label=lab)
        for c, lab in zip(OBJ_COLORS, ["safety-leaning λ", "speed-leaning λ", "comfort-leaning λ"])
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_pareto_projections(grid_result: dict,
                            baseline_points: dict | None = None,
                            title_prefix: str = "PSL"):
    """
    Three pairwise scatter plots — useful when 3D is hard to read.
    Optionally overlay baseline points (e.g. fixed-λ scalarized agents).

    baseline_points : optional dict {label: cost_vector_3d} to overlay as stars.
    """
    costs = grid_result["costs"]
    lambdas = grid_result["lambdas"]
    dominant = lambdas.argmax(axis=1)
    colors = np.array(OBJ_COLORS)[dominant]

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(costs[:, i], costs[:, j], c=colors, s=40,
                   alpha=0.85, edgecolors="k", linewidths=0.3, label="PSL grid")
        ax.set_xlabel(OBJ_LABELS[i]); ax.set_ylabel(OBJ_LABELS[j])
        ax.set_title(f"{title_prefix}: {OBJ_LABELS[i]} vs {OBJ_LABELS[j]}")
        ax.grid(alpha=0.3)

        if baseline_points:
            for label, vec in baseline_points.items():
                ax.scatter(vec[i], vec[j], marker="*", s=180, edgecolors="k",
                           linewidths=0.8, label=label, zorder=5)
            ax.legend(fontsize=7)

    plt.tight_layout()
    return fig, axes


def plot_lambda_adaptation(adaptation_results: dict,
                           title: str = "Single PSL policy adapting to different λ"):
    """
    Show that the same network changes behaviour when λ changes.

    adaptation_results : dict {label: cost_vector_3d}
        e.g. {"safety-only λ=(1,0,0)": [0.05, 0.7, 0.4], ...}
    """
    labels = list(adaptation_results.keys())
    costs  = np.stack([adaptation_results[k] for k in labels])  # (N, 3)
    n_obj  = costs.shape[1]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (lab, color) in enumerate(zip(OBJ_LABELS, OBJ_COLORS)):
        ax.bar(x + (i - 1) * width, costs[:, i], width, label=lab, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean per-step cost (lower = better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig, ax
