import pickle
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt, savgol_filter, hilbert
from scipy.ndimage import minimum_filter
from pathlib import Path
from typing import Dict
from tqdm import trange

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

data_basedir = Path("./outputs/path_integration/")
model_basedir = data_basedir / "models"
model_basedir.mkdir(exist_ok=True, parents=True)
gaits = ["tripod", "tetrapod", "wave"]
num_trials_per_gait = 15


# Load random exploration data
def load_trial_data(trial_dir: Path) -> Dict[str, np.ndarray]:
    with open(trial_dir / "sim_data.pkl", "rb") as f:
        sim_data = pickle.load(f)
    obs_hist = sim_data["obs_hist"]
    action_hist = sim_data["action_hist"]

    end_effector_pos_ts = np.array(
        [obs["stride_diff_unmasked"] for obs in obs_hist], dtype=np.float32
    )

    contact_force_ts = np.array(
        [obs["contact_forces"] for obs in obs_hist], dtype=np.float32
    )
    contact_force_ts = np.linalg.norm(contact_force_ts, axis=2)  # calc force magnitude
    contact_force_ts = contact_force_ts.reshape(-1, 6, 6).sum(axis=2)  # total per leg
    contact_force_ts = np.array(
        [medfilt(arr, kernel_size=11) for arr in contact_force_ts.T]
    ).T
    contact_force_ts = savgol_filter(
        contact_force_ts, window_length=21, polyorder=3, axis=0
    )
    contact_force_ts = np.array(
        [minimum_filter(arr, size=21, mode="nearest") for arr in contact_force_ts.T]
    ).T

    dn_drive_ts = np.array(action_hist, dtype=np.float32)

    fly_orientation_ts = np.array(
        [obs["fly_orientation"][:2] for obs in obs_hist], dtype=np.float32
    )

    fly_pos_ts = np.array([obs["fly"][0, :2] for obs in obs_hist], dtype=np.float32)

    del sim_data
    gc.collect()

    return {
        "end_effector_pos": end_effector_pos_ts,
        "contact_force": contact_force_ts,
        "dn_drive": dn_drive_ts,
        "fly_orientation": fly_orientation_ts,
        "fly_pos": fly_pos_ts,
    }


trial_data = {}
for gait in ["tripod"]:  # gaits:
    for seed in [0]:  # trange(num_trials_per_gait, desc=f"Loading {gait} gait trials"):
        trial_dir = data_basedir / f"random_exploration/seed={seed}_gait={gait}"
        trial_data[(gait, seed)] = load_trial_data(trial_dir)


# Visualize contact forces for each pair of legs
seed = 0
force_thresholds = (0.5, 1, 3)
fig, axs = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True, sharex=True)
t_grid = np.arange(trial_data[("tripod", 0)]["contact_force"].shape[0]) * 1e-4
for i, pos in enumerate(["fore", "mid", "hind"]):
    ax = axs[i]
    ax.plot(t_grid, trial_data[("tripod", 0)]["contact_force"][:, i], lw=1)
    ax.plot(t_grid, trial_data[("tripod", 0)]["contact_force"][:, 3 + i], lw=1)
    ax.axhline(force_thresholds[i], color="black", lw=1)
    ax.set_xlim(10, 10.5)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Contact force [mN]")
    ax.set_title(f"{pos.title()} legs")
    if i == 2:
        ax.set_xlabel("Time [s]")
    sns.despine(ax=ax)
fig.savefig(model_basedir / "contact_forces_min.pdf")
