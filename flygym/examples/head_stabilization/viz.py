import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Callable
from pandas import DataFrame
from sklearn.metrics import r2_score
from flygym.examples.head_stabilization import WalkingDataset


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42


_color_config = {
    "roll": ("royalblue", "midnightblue"),
    "pitch": ("peru", "saddlebrown"),
}
_marker_config = {
    "tripod": "^",
    "tetrapod": "s",
    "wave": "d",
}


def visualize_one_dataset(
    model: Callable,
    test_datasets: Dict[str, Dict[str, Dict[str, WalkingDataset]]],
    output_path: Path,
    joint_angles_mask: Optional[np.ndarray] = None,
    subset_tag: Optional[str] = None,
    dn_drive: str = "0.94_1.02",
):
    fig, axs = plt.subplots(
        3, 2, figsize=(9, 6), tight_layout=True, sharex=True, sharey=True
    )
    for i, gait in enumerate(["tripod", "tetrapod", "wave"]):
        for j, terrain in enumerate(["flat", "blocks"]):
            # Collect data
            ds = test_datasets[gait][terrain][dn_drive]
            joint_angles = ds.joint_angles
            if joint_angles_mask is not None:
                joint_angles = joint_angles.copy()
                joint_angles = joint_angles[:, joint_angles_mask]
            contact_mask = ds.contact_mask
            y_true = ds.roll_pitch_ts

            # Make predictions
            x = np.concatenate([joint_angles, contact_mask], axis=1)
            y_pred = model(torch.tensor(x[None, ...])).detach().cpu().numpy().squeeze()

            # Evaluate performance
            perf = {}
            for k, dof in enumerate(["roll", "pitch"]):
                perf[dof] = r2_score(y_true[:, k], y_pred[:, k])

            # Visualize
            ax = axs[i, j]
            t_grid = (np.arange(len(ds)) + ds.ignore_first_n) * 1e-4
            for k, dof in enumerate(["roll", "pitch"]):
                color_light, color_dark = _color_config[dof]
                ax.plot(
                    t_grid,
                    np.rad2deg(y_true[:, k]),
                    linestyle="--",
                    lw=1,
                    color=color_light,
                    label=f"Actual {dof}",
                )
                ax.plot(
                    t_grid,
                    np.rad2deg(y_pred[:, k]),
                    linestyle="-",
                    lw=1,
                    color=color_dark,
                    label=f"Predicted {dof}",
                )
                axs[i, j].text(
                    1.0,
                    0.01 if k == 0 else 0.1,
                    f"{dof.title()}: $R^2$={perf[dof]:.2f}",
                    ha="right",
                    va="bottom",
                    transform=axs[i, j].transAxes,
                    color=color_dark,
                )
            if i == 0 and j == 1:
                ax.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
            if i == 0:
                axs[i, j].set_title(rf"{terrain.title()} terrain", size=12)
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
            ax.set_xlim(0.5, 1.5)
            ax.set_ylim(-45, 45)
            sns.despine(ax=ax)
    if subset_tag is not None:
        fig.suptitle(f"DoF selection: {subset_tag}", fontweight="bold")
    fig.savefig(output_path)
    plt.close(fig)


def make_feature_selection_summary_plot(
    performances_df: DataFrame, dof_subset_tags: List[str], output_path: Path
):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    bar_width = 0.4
    category_base = np.arange(len(dof_subset_tags))
    for dof in ["roll", "pitch"]:
        color_light, color_dark = _color_config[dof]
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
            ax.scatter(xs, ys, color=color_dark, marker=_marker_config[gait], s=5)
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
