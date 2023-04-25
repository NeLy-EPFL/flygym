"""Demo script for NeuroMechFlyMuJoCo environment:
Execute an environment where all leg joints of the fly repeat a sinusoidal
motion. The output will be saved as a video."""

import numpy as np
import pkg_resources
import pickle
from pathlib import Path
from tqdm import trange
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.util.config import all_leg_dofs

# Initialize simulation
run_time = 1
out_dir = Path('kin_replay')
nmf = NeuroMechFlyMuJoCo(render_mode='saved',
                         output_dir=out_dir,
                         timestep=1e-4,
                         render_config={'playspeed': 0.1},
                         init_pose='stretch',
                         actuated_joints=all_leg_dofs)

# Load recorded data
data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
with open(data_path / 'behavior' / '210902_pr_fly1.pkl', 'rb') as f:
    data = pickle.load(f)

# Interpolate 5x
num_steps = int(run_time / nmf.timestep)
data_block = np.zeros((len(nmf.actuated_joints), num_steps))
measure_t = np.arange(len(data['joint_LFCoxa'])) * data['meta']['timestep']
interp_t = np.arange(num_steps) * nmf.timestep
for i, joint in enumerate(nmf.actuated_joints):
    data_block[i, :] = np.interp(interp_t, measure_t, data[joint])

# Run simulation
obs_list = []
for i in trange(num_steps):
    joint_pos = data_block[:, i]
    action = {'joints': joint_pos}
    obs, info = nmf.step(action)
    nmf.render()
    obs_list.append(obs)
nmf.close()