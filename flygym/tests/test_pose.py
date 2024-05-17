import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym import Fly, SingleFlySimulation
from flygym.util import plot_mujoco_rollout


def test_stretched_pose():
    np.random.seed(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    fly = Fly(init_pose="stretch")
    sim = SingleFlySimulation(fly=fly)
    run_time = 0.01
    freq = 20
    amp = np.pi / 2

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    obs_list = []
    while sim.curr_time <= run_time:
        joint_pos = fly_init_pos + amp * np.sin(freq * sim.curr_time)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
    sim.close()

    out_dir = temp_base_dir / "mujoco_stretched_pose"
    plot_mujoco_rollout(obs_list, sim.timestep, out_dir)


def test_zero_pose():
    # The fly is spawn from high up, so it will not collide with the floor
    # Contact with the floor with straight leg zeros pose is strenuous on the physics
    np.random.seed(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    fly = Fly(init_pose="zero", spawn_pos=(0, 0, 3))
    sim = SingleFlySimulation(fly=fly)
    run_time = 0.01
    freq = 80
    amp = np.pi / 2

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    obs_list = []
    while sim.curr_time <= run_time:
        joint_pos = fly_init_pos + amp * np.sin(freq * sim.curr_time)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
    sim.close()

    out_dir = temp_base_dir / "mujoco_zero_pose"
    plot_mujoco_rollout(obs_list, sim.timestep, out_dir)


def test_tripod_pose():
    np.random.seed(0)
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")

    fly = Fly(init_pose="tripod")
    sim = SingleFlySimulation(fly=fly)
    run_time = 0.01
    freq = 80
    amp = np.pi / 2

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    obs_list = []
    while sim.curr_time <= run_time:
        joint_pos = fly_init_pos + amp * np.sin(freq * sim.curr_time)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
    sim.close()

    out_dir = temp_base_dir / "mujoco_tripod_pose"
    plot_mujoco_rollout(obs_list, sim.timestep, out_dir)
