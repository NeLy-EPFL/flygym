import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym.mujoco import NeuroMechFlyMuJoCo
from flygym.mujoco.util import plot_mujoco_rollout


def test_stretched_pose():
    random_state = np.random.RandomState(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    nmf = NeuroMechFlyMuJoCo(init_pose="stretch")
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf._actuators))
    amp = 0.9

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        obs_list.append(obs)
    nmf.close()

    out_dir = temp_base_dir / "mujoco_stretched_pose"
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir)


def test_zero_pose():
    random_state = np.random.RandomState(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    nmf = NeuroMechFlyMuJoCo(init_pose="zero", spawn_pos=(0, 0, 3))
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf._actuators))
    amp = 0.9

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        obs_list.append(obs)
    nmf.close()

    out_dir = temp_base_dir / "mujoco_zero_pose"
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir)


def test_tripod_pose():
    random_state = np.random.RandomState(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    nmf = NeuroMechFlyMuJoCo(init_pose="tripod")
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf._actuators))
    amp = 0.9

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        obs_list.append(obs)
    nmf.close()

    out_dir = temp_base_dir / "mujoco_tripod_pose"
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir)
