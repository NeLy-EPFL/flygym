"""Demo script for NeuroMechFlyMuJoCo environment:
Execute an environment where all leg joints of the fly repeat a sinusoidal
motion. The output will be saved as a video."""

import numpy as np
import pkg_resources
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.state import stretched_pose
from flygym.util.config import all_leg_dofs


# First, we initialize simulation
run_time = 1
sim_params = MuJoCoParameters(timestep=1e-4, render_mode="saved", render_playspeed=0.1)
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
    init_pose=stretched_pose,
    actuated_joints=all_leg_dofs,
)

# Define the frequency, phase, and amplitude of the sinusoidal waves
freq = 20
phase = 2 * np.pi * np.random.rand(len(nmf.actuators))
amp = 0.9

obs_list = []  # keep track of the observed states
num_steps = int(run_time / nmf.timestep)
for i in trange(num_steps):
    joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
    action = {"joints": joint_pos}
    obs, reward, terminated, truncated, info = nmf.step(action)
    nmf.render()
    obs_list.append(obs)
nmf.close()

nmf.save_video(Path("sine_wave.mp4"))

# Visualize joint angles, velocities, and forces over time
num_joints_to_plot = 7
fig, axs = plt.subplots(3, 1, figsize=(4, 6), tight_layout=True)
axs[0].plot([x["joints"][0, :num_joints_to_plot] for x in obs_list])
axs[0].set_title("Joint positions")
axs[1].plot([x["joints"][1, :num_joints_to_plot] for x in obs_list])
axs[1].set_title("Joint velocities")
axs[2].plot([x["joints"][2, :num_joints_to_plot] for x in obs_list])
axs[2].set_title("Joint forces")
plt.show()

# Visualize fly's global position and orientation in the arena over time
fig, axs = plt.subplots(2, 2, figsize=(6, 4), tight_layout=True)
axs[0, 0].plot([x["fly"][0, :] for x in obs_list])
axs[0, 0].set_title("Cartesian position")
axs[0, 1].plot([x["fly"][1, :] for x in obs_list])
axs[0, 1].set_title("Cartesian velocity")
axs[1, 0].plot([x["fly"][2, :] for x in obs_list])
axs[1, 0].set_title("Angular position (Euler)")
axs[1, 1].plot([x["fly"][3, :] for x in obs_list])
axs[1, 1].set_title("Angular velocity (Euler)")
plt.show()
