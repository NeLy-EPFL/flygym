import numpy as np
import scipy.stats
import pytest

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.tests import temp_base_dir
from flygym.util.config import (
    raw_img_width_px,
    raw_img_height_px,
    num_ommatidia_per_eye,
)
from flygym.util.vision import visualize_visual_input


random_state = np.random.RandomState(0)


@pytest.mark.skip(
    reason="github actions runner doesn't have a display; render will fail"
)
def test_vision_dimensions():
    # Initialize simulation
    num_steps = 100
    sim_params = MuJoCoParameters(
        enable_olfaction=True, enable_vision=True, render_raw_vision=True
    )
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params)

    # Run simulation
    obs_list = []
    for i in range(num_steps):
        joint_pos = np.zeros(len(nmf.actuated_joints))
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        # nmf.render()
        obs_list.append(obs)
    nmf.close()

    # Check dimensionality
    assert len(obs_list) == num_steps
    assert nmf.vision_update_mask.shape == (num_steps,)
    assert nmf.vision_update_mask.sum() == int(
        num_steps * sim_params.timestep * sim_params.vision_refresh_rate)
    assert obs["raw_vision"].shape == (2, raw_img_height_px, raw_img_width_px, 3)
    assert obs["vision"].shape == (2, num_ommatidia_per_eye, 2)

    print(temp_base_dir)

    # Test postprocessing
    visualize_visual_input(
        output_path=temp_base_dir / "vision/eyes.mp4",
        vision_data_li=[x["vision"] for x in obs_list],
        raw_vision_data_li=[x["raw_vision"] for x in obs_list],
        vision_update_mask=nmf.vision_update_mask,
        vision_refresh_rate=sim_params.vision_refresh_rate,
        playback_speed=0.1,
    )
