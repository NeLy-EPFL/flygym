import yaml
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.core import ObsType
from pathlib import Path
from typing import List
from flygym.common import get_data_path


def load_config():
    with open(get_data_path("flygym.mujoco", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def plot_mujoco_rollout(obs_list: List[ObsType], timestep: float, out_dir: Path):
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
