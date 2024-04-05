import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from pandas import DataFrame


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42


def make_sample_time_series_plot(
    predictions: Dict[str, np.ndarray],
    performances: Dict[str, Dict[str, float]],
    unmasked_datasets: Dict[str, Dataset],
    output_path: Path,
    dof_subset_tag: Optional[str] = None,
):
    fig, axs = plt.subplots(
        3, 2, figsize=(8, 6), tight_layout=True, sharex=True, sharey=True
    )
    for i, (gait, ds) in enumerate(unmasked_datasets.items()):
        t_grid = (np.arange(len(ds)) + ds.ignore_first_n) * 1e-4
        for j, dof in enumerate(["roll", "pitch"]):
            axs[i, j].plot(
                t_grid,
                np.rad2deg(unmasked_datasets[gait].roll_pitch_ts[:, j]),
                lw=1,
                color="black",
                label="Actual",
            )
            axs[i, j].plot(
                t_grid,
                np.rad2deg(predictions[gait][:, j]),
                lw=1,
                color="tab:red",
                label="Predicted",
                linestyle="--",
            )
            axs[i, j].text(
                1.0,
                0.01,
                f"$R^2$={performances[gait][dof]:.2f}",
                ha="right",
                va="bottom",
                transform=axs[i, j].transAxes,
            )
            axs[i, j].set_ylim(-15, 15)
            axs[i, j].set_xlim(51, 52)
            if i == 0:
                axs[i, j].set_title(rf"Thorax {dof}", size=12)
            if j == 0:
                axs[i, j].text(
                    -0.3,
                    0.5,
                    f"{gait.title()} gait",
                    size=12,
                    va="center",
                    rotation=90,
                    transform=axs[i, j].transAxes,
                )
            if i == 2:
                axs[i, j].set_xlabel("Time [s]")
            if j == 0:
                axs[i, j].set_ylabel(r"Angle [$^\circ$]")
            if i == 0 and j == 0:
                axs[i, j].legend(frameon=False)
            sns.despine(ax=axs[i, j])
    if dof_subset_tag is not None:
        fig.suptitle(f"DoF selection: {dof_subset_tag}", fontweight="bold")
    fig.savefig(output_path)
    plt.close(fig)


def make_feature_selection_summary_plot(
    performances_df: DataFrame, dof_subset_tags: List[str], output_path: Path
):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    bar_width = 0.4
    category_base = np.arange(len(dof_subset_tags))
    color_config = {
        "roll": ("royalblue", "midnightblue"),
        "pitch": ("peru", "saddlebrown"),
    }
    marker_config = {
        "tripod": "^",
        "tetrapod": "s",
        "wave": "d",
    }
    for dof in ["roll", "pitch"]:
        color_light, color_dark = color_config[dof]
        xs = category_base + (0.5 * bar_width * (-1 if dof == "roll" else 1))
        mean_r2s = [
            performances_df.loc[cat, :, dof]["r2_score"].mean()
            for cat in dof_subset_tags
        ]
        ax.bar(xs, mean_r2s, width=bar_width, color=color_light)
        for gait in ["tripod", "tetrapod", "wave"]:
            ys = [
                performances_df.loc[cat, gait, dof]["r2_score"]
                for cat in dof_subset_tags
            ]
            ax.scatter(xs, ys, color=color_dark, marker=marker_config[gait], s=5)
    ax.set_ylabel("$R^2$")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylim(0, 1)
    ax.set_xticks(category_base)
    ax.set_xticklabels(dof_subset_tags)
    ax.set_xlim(-0.5, category_base[-1] + 0.5)
    legend_elements = [
        Patch(facecolor="royalblue", label="Thorax roll"),
        Patch(facecolor="peru", label="Thorax pitch"),
        Line2D(
            [],
            [],
            color="black",
            marker="^",
            markersize=5,
            linestyle="None",
            label="Tripod gait",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="s",
            markersize=5,
            linestyle="None",
            label="Tetrapod gait",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="d",
            markersize=5,
            linestyle="None",
            label="Wave gait",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), frameon=False
    )
    sns.despine(ax=ax)
    fig.savefig(output_path)
    plt.close(fig)
