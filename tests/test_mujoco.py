import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import flygym
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo


random_state = np.random.RandomState(0)


_temp_base_dir = Path(tempfile.gettempdir()) / 'flygym_test'
print(f'Test output data will be saved to {_temp_base_dir}')


def test_basic_untethered_sinewave():
    out_dir = _temp_base_dir / 'mujoco_basic_untethered_sinewave'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    nmf = NeuroMechFlyMuJoCo(render_mode='saved', output_dir=out_dir)
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
    amp = 0.9
    
    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        # nmf.render()
        obs_list.append(obs)
    nmf.close()
    
    # Plot joint observations
    num_joints_to_plot = 7
    fig, axs = plt.subplots(3, 1, figsize=(4, 6), tight_layout=True)
    axs[0].plot([x['joints'][0, :num_joints_to_plot] for x in obs_list])
    axs[0].set_title('Joint positions')
    vel_from_pos = np.diff([x['joints'][0, :num_joints_to_plot]
                            for x in obs_list], axis=0) / nmf.timestep
    axs[1].plot(vel_from_pos, marker='+', linestyle='None', color='gray')
    axs[1].plot([x['joints'][1, :num_joints_to_plot] for x in obs_list])
    axs[1].set_title('Joint velocities')
    axs[2].plot([x['joints'][2, :num_joints_to_plot] for x in obs_list])
    axs[2].set_title('Joint forces')
    plt.savefig(out_dir / 'joints_ts.png')
    plt.close(fig)
    
    # Plot fly position and orientation
    fig, axs = plt.subplots(2, 2, figsize=(6, 4), tight_layout=True)
    axs[0, 0].plot([x['fly'][0, :] for x in obs_list])
    axs[0, 0].set_title('Cartesian position')
    lin_vel_from_pos = np.diff([x['fly'][0, :] for x in obs_list],
                                axis=0) / nmf.timestep
    axs[0, 1].plot(lin_vel_from_pos, marker='+', linestyle='None',
                    color='gray')
    axs[0, 1].plot([x['fly'][1, :] for x in obs_list])
    axs[0, 1].set_title('Cartesian velocity')
    axs[1, 0].plot([x['fly'][2, :] for x in obs_list])
    axs[1, 0].set_title('Angular position (Euler)')
    ang_vel_from_pos = np.diff([x['fly'][2, :] for x in obs_list],
                                axis=0) / nmf.timestep
    axs[1, 1].plot(ang_vel_from_pos, marker='+', linestyle='None',
                    color='gray')
    axs[1, 1].plot([x['fly'][3, :] for x in obs_list])
    axs[1, 1].set_title('Angular velocity (Euler)')
    plt.savefig(out_dir / 'fly_pos_ts.png')
    plt.close(fig)

def test_gapped_terrain():
    out_dir = _temp_base_dir / 'mujoco_gapped_terrain'
    nmf = NeuroMechFlyMuJoCo(render_mode='headless', output_dir=out_dir,
                                terrain='gapped')
    nmf.close()

def test_blocks_terrain():
    out_dir = _temp_base_dir / 'mujoco_blocks_terrain'
    nmf = NeuroMechFlyMuJoCo(render_mode='headless', output_dir=out_dir,
                                terrain='blocks')
    nmf.close()