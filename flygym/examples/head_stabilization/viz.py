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
    dof_subset_tag: Optional[str] = None,
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
                joint_angles[:, ~joint_angles_mask] = 0
            contact_mask = ds.contact_mask
            y_true = ds.roll_pitch_ts

            # Make predictions
            x = np.concatenate([joint_angles, contact_mask], axis=1)
            x = torch.tensor(x[None, ...], device=model.device)
            y_pred = model(x).detach().cpu().numpy().squeeze()

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

    if dof_subset_tag is not None:
        fig.suptitle(f"DoF selection: {dof_subset_tag}", fontweight="bold")
    fig.savefig(output_path)
    plt.close(fig)


def make_feature_selection_summary_plot(
    test_performance_df: DataFrame, output_path: Path, title: str = None
):
    dof_subset_tags = test_performance_df["dof_subset_tag"].unique()
    dof_subset_tags_basex = {tag: i * 3 for i, tag in enumerate(dof_subset_tags)}

    fig, ax = plt.subplots(figsize=(9, 3), tight_layout=True)
    ax.axhline(0, color="black", lw=0.5)
    for i, dof in enumerate(["roll", "pitch"]):
        df_copy = test_performance_df.copy()
        x_lookup = {k: v + i for k, v in dof_subset_tags_basex.items()}
        df_copy["_x"] = df_copy["dof_subset_tag"].map(x_lookup)
        color_light, color_dark = _color_config[dof]
        sns.swarmplot(
            data=df_copy,
            x="_x",
            y=f"r2_{dof}",
            ax=ax,
            color=color_dark,
            dodge=True,
            order=list(range(len(dof_subset_tags) * 3 - 1)),
            size=1.5,
        )
        sns.boxplot(
            data=df_copy,
            x="_x",
            y=f"r2_{dof}",
            ax=ax,
            dodge=True,
            fliersize=0,
            boxprops={"facecolor": "None", "edgecolor": "k", "linewidth": 0.5},
            order=list(range(len(dof_subset_tags) * 3 - 1)),
            linewidth=1,
        )
    legend_elements = [
        Line2D(
            [],
            [],
            color=_color_config["roll"][1],
            marker=".",
            markersize=5,
            linestyle="None",
            label="Roll",
        ),
        Line2D(
            [],
            [],
            color=_color_config["pitch"][1],
            marker=".",
            markersize=5,
            linestyle="None",
            label="Pitch",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0, 0.2),
        frameon=False,
    )
    if min(df_copy["r2_roll"].min(), df_copy["r2_pitch"].min()) < -0.26:
        raise ValueError(
            "Lowest R2 score is below the display limit. Some data not shown in figure."
        )
    ax.set_ylim(-0.26, 1)
    ax.set_xticks(np.array(list(dof_subset_tags_basex.values())) + 0.5)
    ax.set_xticklabels(dof_subset_tags)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel("")
    ax.set_ylabel("$R^2$")
    if title is not None:
        ax.set_title(title)
    sns.despine(ax=ax, bottom=True)
    fig.savefig(output_path)
    plt.close(fig)
