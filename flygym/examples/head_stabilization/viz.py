import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sys import stderr
from tqdm import trange
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Callable
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
    test_datasets: dict[str, dict[str, dict[str, WalkingDataset]]],
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


def closed_loop_comparison_video(
    data: dict[tuple[bool, str], list[np.ndarray]],
    fps: int,
    video_path: Path,
    run_time: float,
    action_range: tuple[float, float] = (-20, 20),
    dpi: int = 300,
):
    fig, axs = plt.subplots(
        2,
        4,
        figsize=(11.2, 6.3),
        gridspec_kw={"width_ratios": [1, 1, 0.85, 1.5]},  # 'wspace':0.1, 'hspace':0.1},
        layout="compressed",
        # tight_layout=True,
    )

    plot_elements = {}

    def init():
        # Turn off all
        #  borders
        for ax in axs.flat:
            ax.axis("off")

        # Initialize views
        for i, stabilization_on in enumerate([False, True]):
            for j, view in enumerate(
                ["birdeye", "zoomin", "raw_vision", "neck_actuation"]
            ):
                if view == "raw_vision":
                    vmin, vmax = 0, 1
                    cmap = "gray"
                else:
                    vmin, vmax = 0, 255
                    cmap = None

                if view in ["birdeye", "zoomin"]:
                    img = np.zeros_like(data[(stabilization_on, view)][0])
                elif view == "neck_actuation":
                    pass
                else:
                    img = np.zeros_like(data[(stabilization_on, view)][0][0, ...])

                ax = axs[i, j]

                if view == "neck_actuation":
                    ax.axis("on")  # Enable axis for actuation data
                    ax.set_xlim(0 - 0.05, run_time + 0.05)
                    ax.set_ylim(
                        action_range
                    )  # Assuming actuation ranges, adjust as necessary
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel(r"Actuation signal [$^\circ$]")
                    for dof in ["roll", "pitch"]:
                        for version in ["true", "pred"]:
                            key = (stabilization_on, view, f"{dof}_{version}")
                            label = (
                                f"{'Predicted' if version == 'pred' else 'Optimal'} "
                                f"{dof}"
                            )
                            plot_elements[key] = ax.plot(
                                [],
                                [],
                                label=label,
                                color=_color_config[dof][0 if version == "true" else 1],
                                ls="--" if version == "true" else "-",
                                lw=1,
                            )[0]
                    if i == 0:
                        ax.legend(frameon=False, loc="upper right", fontsize=7, ncols=2)
                        ax.set_title(f"Neck actuation")
                    sns.despine(ax=ax)
                else:
                    plot_elements[(stabilization_on, view)] = ax.imshow(
                        img,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                    )

        # Panel titles
        axs[0, 0].set_title("Bird’s-eye view")
        axs[0, 1].set_title("Zoomed-in view")
        axs[0, 2].set_title("Left eye raw vision")
        axs[0, 3].set_title(f"Neck actuation")
        axs[0, 0].text(
            -0.2,
            0.5,
            f"No head stabilization",
            size=12,
            va="center",
            rotation=90,
            transform=axs[0, 0].transAxes,
        )
        axs[1, 0].text(
            -0.2,
            0.5,
            f"Head stabilization",
            size=12,
            va="center",
            rotation=90,
            transform=axs[1, 0].transAxes,
        )

        return list(plot_elements.values())

    def update(frame_id):
        for i, stabilization_on in enumerate([False, True]):
            for j, view in enumerate(
                ["birdeye", "zoomin", "raw_vision", "neck_actuation"]
            ):
                if view in ["birdeye", "zoomin"]:
                    img = data[(stabilization_on, view)][frame_id]
                    img = img.astype(np.float32) / 255
                elif view == "neck_actuation":
                    # Update line data for pitch and yaw
                    frame_data = data[(stabilization_on, view)][frame_id]
                    roll_true, pitch_true, roll_pred, pitch_pred, t = frame_data
                    last_x = plot_elements[
                        (stabilization_on, view, "roll_true")
                    ].get_xdata()
                    if len(last_x) > 0 and t < last_x[-1]:
                        # reset all the plot elements
                        for dof in ["roll", "pitch"]:
                            for version in ["true", "pred"]:
                                key = (stabilization_on, view, f"{dof}_{version}")
                                plot_elements[key].set_data([], [])

                    data_to_add = {
                        "roll_true": roll_true,
                        "pitch_true": pitch_true,
                        "roll_pred": roll_pred,
                        "pitch_pred": pitch_pred,
                    }
                    for dof in ["roll", "pitch"]:
                        for version in ["true", "pred"]:
                            key = (stabilization_on, view, f"{dof}_{version}")
                            new_y_val = data_to_add[f"{dof}_{version}"]
                            plot_elements[key].set_data(
                                np.append(plot_elements[key].get_xdata(), t),
                                np.append(plot_elements[key].get_ydata(), new_y_val),
                            )
                elif view == "raw_vision":
                    img = data[(stabilization_on, view)][frame_id][0, ...]
                    img[img == 0] = np.nan
                else:
                    raise ValueError(f"Unexpected view: {view}")
                if not view == "neck_actuation":
                    plot_elements[(stabilization_on, view)].set_data(img)
        return list(plot_elements.values())

    animation = FuncAnimation(
        fig,
        update,
        frames=trange(len(data[True, "birdeye"]), file=stderr),
        init_func=init,
        blit=False,
    )

    video_path.parent.mkdir(exist_ok=True, parents=True)
    animation.save(video_path, writer="ffmpeg", fps=fps, dpi=dpi)


def plot_rotation_time_series(
    rotation_data: dict[str, np.ndarray], output_path: Path, dt: float = 1e-4
):
    fig, axs = plt.subplots(
        2, 2, figsize=(6, 3), tight_layout=True, sharex=True, sharey=True
    )
    for idof, dof in enumerate(["roll", "pitch"]):
        for iterrain, terrain_type in enumerate(["flat", "blocks"]):
            ax = axs[idof, iterrain]
            head_angle = rotation_data[terrain_type]["head"][:, idof]
            thorax_angle = rotation_data[terrain_type]["thorax"][:, idof]
            t_grid = np.arange(len(head_angle)) * dt
            ax.axhline(0, color="black", lw=1)
            ax.plot(t_grid, np.rad2deg(head_angle), label="Head", color="tab:red", lw=1)
            ax.plot(
                t_grid, np.rad2deg(thorax_angle), label="Thorax", color="tab:blue", lw=1
            )
            ax.set_xlim(0.5, 1)
            ax.set_ylim(-20, 20)
            sns.despine(ax=ax, bottom=True)
            if idof == 0:
                ax.set_title(f"{terrain_type.title()} terrain")
            if idof == 1:
                ax.set_xlabel("Time [s]")
            if iterrain == 0:
                ax.set_ylabel(rf"{dof.title()} [$^\circ$]")
            if idof == 0 and iterrain == 1:
                ax.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(output_path)


def plot_activities_std(
    activities_std_data, output_path, vmin=0, vmax=0.3, cmap="inferno"
):
    fig, axs = plt.subplots(
        2, 3, figsize=(7, 6), tight_layout=True, gridspec_kw={"width_ratios": [3, 3, 1]}
    )
    for i, stabilization_on in enumerate([False, True]):
        for j, terrain_type in enumerate(["flat", "blocks"]):
            std_img = activities_std_data[(terrain_type, stabilization_on)][:, :, 0]
            ax = axs[i, j]
            ax.imshow(std_img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
            if stabilization_on:
                stab_str = "Head stabilization"
            else:
                stab_str = "No head stabilization"
            ax.set_title(f"{terrain_type.title()} terrain, {stab_str}")
            ax.set_aspect("equal")

    # Draw colorbar manually
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    fig.colorbar(
        sm,
        ax=axs[0, 2],
        shrink=0.6,
        aspect=5,
        label="Standard deviation (AU)",
    )
    axs[0, 2].axis("off")
    axs[1, 2].axis("off")
    fig.savefig(output_path)
