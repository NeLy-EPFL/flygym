import pickle

import flygym.mujoco.preprogrammed
import numpy as np
import pytest
from flygym.common import get_data_path
from flygym.mujoco import NeuroMechFly, Parameters
from flygym.mujoco.core import NeuroMechFlyV0


@pytest.mark.skip(
    reason="github actions runner doesn't have a display; render will fail"
)
def test_new_nmf_same_as_nmf_v0():
    timestep = 1e-4
    actuated_joints = flygym.mujoco.preprogrammed.all_leg_dofs
    data_path = get_data_path("flygym", "data")
    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = pickle.load(f)

    run_time = 1
    target_num_steps = int(run_time / timestep)
    data_block = np.zeros((len(actuated_joints), target_num_steps))
    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * timestep
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(output_t, input_t, data[joint])

    sim_params = Parameters(
        timestep=1e-4,
        render_mode="saved",
        render_playspeed=0.2,
        draw_contacts=True,
        draw_adhesion=True,
        draw_gravity=True,
        draw_sensor_markers=True,
        align_camera_with_gravity=True,
        camera_follows_fly_orientation=True,
        enable_adhesion=True,
    )

    nmf_v0 = NeuroMechFlyV0(
        sim_params=sim_params,
        init_pose="stretch",
        actuated_joints=actuated_joints,
        control="position",
    )

    nmf = NeuroMechFly(
        sim_params=sim_params,
        init_pose="stretch",
        actuated_joints=actuated_joints,
        control="position",
    )

    nmf_v0.reset()
    nmf.reset()

    for i in range(target_num_steps):
        joint_pos = data_block[:, i]

        action_v0 = {"joints": joint_pos.copy(), "adhesion": np.ones(6) * (i % 2)}
        obs_v0, reward_v0, terminated_v0, truncated_v0, info_v0 = nmf_v0.step(action_v0)
        im_v0 = nmf_v0.render()

        action = {"joints": joint_pos.copy(), "adhesion": np.ones(6) * (i % 2)}
        obs, reward, terminated, truncated, info = nmf.step(action)
        im = nmf.render()

        assert np.all(im_v0 == im)

        for key in obs_v0:
            assert np.all(obs_v0[key] == obs[key])

        for key in info_v0:
            assert np.all(info_v0[key] == info[key])
