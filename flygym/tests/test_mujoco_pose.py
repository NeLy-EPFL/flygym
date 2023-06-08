import numpy as np

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.tests import temp_base_dir
from flygym.tests.common import plot_mujoco_rollout


random_state = np.random.RandomState(0)


def test_stretched_pose():
    from flygym.state import stretched_pose

    nmf = NeuroMechFlyMuJoCo(render_mode="saved", init_pose=stretched_pose)
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
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
    from flygym.state import zero_pose

    nmf = NeuroMechFlyMuJoCo(render_mode="saved", init_pose=zero_pose)
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
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


def test_walking_pose():
    from flygym.state import walking_pose

    nmf = NeuroMechFlyMuJoCo(render_mode="saved", init_pose=walking_pose)
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
    amp = 0.9

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        obs_list.append(obs)
    nmf.close()

    out_dir = temp_base_dir / "mujoco_walking_pose"
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir)
