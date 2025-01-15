import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.core import ObsType
from pathlib import Path
from typing import Any


def get_data_path(package: str, file: str) -> Path:
    """Given the names of the package and a file (or directory) included as
    package data, return the absolute path of it in the installed package.
    This wrapper handles the ``pkg_resources``-to-``importlib.resources``
    API change in Python."""
    if sys.version_info >= (3, 9):
        import importlib.resources

        return importlib.resources.files(package) / file
    else:
        import pkg_resources

        return Path(pkg_resources.resource_filename(package, file)).absolute()


def load_config() -> dict[str, Any]:
    """Load the YAML configuration file as a dictionary."""
    with open(get_data_path("flygym", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def plot_mujoco_rollout(
    obs_list: list[ObsType], timestep: float, out_dir: Path
) -> None:
    """Plot the fly position and joint angle time series of a simulation
    and save the image to file. This function is used for debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot joint observations
    num_joints_to_plot = 7
    fig, axs = plt.subplots(3, 1, figsize=(4, 6), tight_layout=True)
    axs[0].plot([x["joints"][0, :num_joints_to_plot] for x in obs_list])
    axs[0].set_title("Joint positions")
    vel_from_pos = (
        np.diff([x["joints"][0, :num_joints_to_plot] for x in obs_list], axis=0)
        / timestep
    )
    axs[1].plot(vel_from_pos, marker="+", linestyle="None", color="gray")
    axs[1].plot([x["joints"][1, :num_joints_to_plot] for x in obs_list])
    axs[1].set_title("Joint velocities")
    axs[2].plot([x["joints"][2, :num_joints_to_plot] for x in obs_list])
    axs[2].set_title("Joint forces")
    plt.savefig(out_dir / "joints_ts.png")
    plt.close(fig)

    # Plot fly position and orientation
    fig, axs = plt.subplots(2, 2, figsize=(6, 4), tight_layout=True)
    axs[0, 0].plot([x["fly"][0, :] for x in obs_list])
    axs[0, 0].set_title("Cartesian position")
    lin_vel_from_pos = np.diff([x["fly"][0, :] for x in obs_list], axis=0) / timestep
    axs[0, 1].plot(lin_vel_from_pos, marker="+", linestyle="None", color="gray")
    axs[0, 1].plot([x["fly"][1, :] for x in obs_list])
    axs[0, 1].set_title("Cartesian velocity")
    axs[1, 0].plot([x["fly"][2, :] for x in obs_list])
    axs[1, 0].set_title("Angular position (Euler)")
    ang_vel_from_pos = np.diff([x["fly"][2, :] for x in obs_list], axis=0) / timestep
    axs[1, 1].plot(ang_vel_from_pos, marker="+", linestyle="None", color="gray")
    axs[1, 1].plot([x["fly"][3, :] for x in obs_list])
    axs[1, 1].set_title("Angular velocity (Euler)")
    plt.savefig(out_dir / "fly_pos_ts.png")
    plt.close(fig)
