import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.arena.mujoco_arena import OdorArena
from flygym.state import stretched_pose
from flygym.util.config import all_leg_dofs

# Initialize simulation
run_time = 1

sim_params = MuJoCoParameters(
    timestep=1e-4, render_mode="saved", render_playspeed=0.1, enable_olfaction=True
)
odor_source = [[2000, -500, 0], [5000, 500, 0]]
peak_intensity = [[80, 0], [0, 100]]
arena = OdorArena(
    odor_source=odor_source,
    peak_intensity=peak_intensity,
    diffuse_func=lambda x: (x / 1000) ** -2,
)
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
    arena=arena,
    init_pose=stretched_pose,
    actuated_joints=all_leg_dofs,
)

# Load recorded data
data_path = Path(pkg_resources.resource_filename("flygym", "data"))
with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
    data = pickle.load(f)

# Interpolate 5x
num_steps = int(run_time / nmf.timestep)
data_block = np.zeros((len(nmf.actuated_joints), num_steps))
measure_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
interp_t = np.arange(num_steps) * nmf.timestep
for i, joint in enumerate(nmf.actuated_joints):
    data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

# Run simulation
obs_list = []
for i in trange(num_steps):
    joint_pos = data_block[:, i]
    action = {"joints": joint_pos}
    obs, reward, terminated, truncated, info = nmf.step(action)
    nmf.render()
    obs_list.append(obs)
nmf.close()
nmf.save_video(Path("odor_arena.mp4"))

fly_pos = np.array([obs["fly"] for obs in obs_list])[:, 0, :]
fly_orient = np.array([obs["fly"] for obs in obs_list])[:, 2, :]
odor = np.array([obs["odor_intensity"] for obs in obs_list])

# fig, axs = plt.subplots(
#     3, 1, figsize=(5, 5), tight_layout=True, height_ratios=[3, 1, 1], sharex=True
# )
# axd = plt.figure(constrained_layout=True, figsize=(9, 6)).subplot_mosaic(
#     """
#     AAADDD
#     AAADDD
#     BBBDDD
#     CCCXXX
#     """
# )
axd = plt.figure(constrained_layout=True, figsize=(9, 3)).subplot_mosaic(
    """
    AAADD
    """
)
t = np.arange(num_steps) * nmf.timestep

ax = axd["A"]
ax.plot(t, odor[:, 0, 0], linestyle="--", color="tab:blue", label="odor 1 (L)")
ax.plot(t, odor[:, 0, 1], linestyle=":", color="tab:blue", label="odor 1 (R)")
ax.plot(t, odor[:, 1, 0], linestyle="--", color="tab:orange", label="odor 2 (L)")
ax.plot(t, odor[:, 1, 1], linestyle=":", color="tab:orange", label="odor 2 (R)")
ax.legend(loc="upper right")
ax.set_ylabel("Odor intensity (a.u.)")
ax.set_xlabel("Time (s)")

# ax = axd["B"]
# ax.plot(t, fly_pos[:, 0] / 1000, color="tab:red", label="-")
# ax.set_ylabel("Position (mm)")

# ax = axd["C"]
# ax.plot(t, np.zeros_like(t), color="tab:brown", linewidth=0.5)
# ax.plot(t, fly_orient[:, 0] + np.pi / 2, color="tab:brown", label="-.")
# ax.set_ylim([-np.pi - 0.1, np.pi + 0.1])
# ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
# ax.set_yticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])
# ax.set_ylabel("Heading (rad.)")
# ax.set_xlabel("Time (s)")

ax = axd["D"]
max_extent = max(fly_pos[:, :2].max(), arena.odor_source[:, :2].max()) * 1.1 / 1000
ax.plot(fly_pos[:, 0] / 1000, fly_pos[:, 1] / 1000, color="black", label="walk path")
ax.scatter([0], [0], marker="o", color="gray", label="origin")
ax.scatter(
    [odor_source[0][0] / 1000],
    [odor_source[0][1] / 1000],
    color="tab:blue",
    marker="D",
    label="odor source 1",
)
ax.scatter(
    [odor_source[1][0] / 1000],
    [odor_source[1][1] / 1000],
    color="tab:orange",
    marker="D",
    label="odor source 2",
)
ax.legend(loc="upper left")
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)
ax.set_aspect("equal")
ax.set_title("Arena layout")

# axd["X"].axis("off")

plt.show()
