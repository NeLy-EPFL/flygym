"""Demo script for NeuroMechFlyMuJoCo environment:
Execute an environment where all leg joints of the fly repeat a sinusoidal
motion. The output will be saved as a video."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo

# First, we initialize simulation
run_time = 1
out_dir = Path('mujoco_basic_untethered_sinewave')
nmf = NeuroMechFlyMuJoCo(render_mode='saved', output_dir=out_dir,
                         init_pose='stretch',
                         render_config={'playspeed': 0.2})

# Define the frequency, phase, and amplitude of the sinusoidal waves
freq = 20
phase = 2 * np.pi * np.random.rand(len(nmf.actuators))
amp = 0.9

obs_list = []    # keep track of the observed states
while nmf.curr_time <= run_time:    # main loop
    joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
    action = {'joints': joint_pos}
    obs, info = nmf.step(action)
    nmf.render()
    obs_list.append(obs)
nmf.close()

# Visualize joint angles, velocities, and forces over time
num_joints_to_plot = 7
fig, axs = plt.subplots(3, 1, figsize=(4, 6), tight_layout=True)
axs[0].plot([x['joints'][0, :num_joints_to_plot] for x in obs_list])
axs[0].set_title('Joint positions')
axs[1].plot([x['joints'][1, :num_joints_to_plot] for x in obs_list])
axs[1].set_title('Joint velocities')
axs[2].plot([x['joints'][2, :num_joints_to_plot] for x in obs_list])
axs[2].set_title('Joint forces')
plt.show()

# Visualize fly's global position and orientation in the arena over time
fig, axs = plt.subplots(2, 2, figsize=(6, 4), tight_layout=True)
axs[0, 0].plot([x['fly'][0, :] for x in obs_list])
axs[0, 0].set_title('Cartesian position')
axs[0, 1].plot([x['fly'][1, :] for x in obs_list])
axs[0, 1].set_title('Cartesian velocity')
axs[1, 0].plot([x['fly'][2, :] for x in obs_list])
axs[1, 0].set_title('Angular position (Euler)')
axs[1, 1].plot([x['fly'][3, :] for x in obs_list])
axs[1, 1].set_title('Angular velocity (Euler)')
plt.show()