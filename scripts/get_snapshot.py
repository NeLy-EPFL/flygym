"""Demo script for NeuroMechFlyMuJoCo environment:
Execute an environment where all leg joints of the fly repeat a sinusoidal
motion. The output will be saved as a video."""

import numpy as np
import pkg_resources
import pickle
from pathlib import Path
from tqdm import trange
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.state import stretched_pose
from flygym.util.config import all_leg_dofs
import cv2

# Initialize simulation
run_time = 0.01

sim_params = MuJoCoParameters(timestep=1e-4, render_mode="saved", render_playspeed=0.1)
nmf = NeuroMechFlyMuJoCo(
    sim_params=sim_params,
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
    obs_list.append(obs)

nmf.render()
snapshot_side = nmf.get_last_frame()
nmf.sim_params.render_camera = "Animat/camera_front"
nmf.render()
snapshot_front = nmf.get_last_frame()

"""width, height = nmf.sim_params.render_window_size
snapshot_front = nmf.physics.render(
    width=width,
    height=height,
    camera_id="Animat/camera_left"
    )
"""

snapshot = np.concatenate((snapshot_side, snapshot_front), axis=1)
snapshot = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)
cv2.imshow("nmf_snapshot", snapshot)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("nmf_snapshot.png", snapshot)

nmf.close()
