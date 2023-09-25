import numpy as np
import tempfile
import pytest
from pathlib import Path

from flygym.mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from flygym.mujoco.util import load_config
from flygym.mujoco.vision import visualize_visual_input


random_state = np.random.RandomState(0)


@pytest.mark.skip(
    reason="github actions runner doesn't have a display; render will fail"
)
def test_vision_dimensions():
    # Load config
    config = load_config()

    # Initialize simulation
    num_steps = 100
    sim_params = MuJoCoParameters(
        enable_olfaction=True, enable_vision=True, render_raw_vision=True
    )
    nmf = NeuroMechFlyMuJoCo(sim_params=sim_params)

    # Run simulation
    obs_list = []
    info_list = []
    for i in range(num_steps):
        joint_pos = np.zeros(len(nmf.actuated_joints))
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        # nmf.render()
        obs_list.append(obs)
        info_list.append(info)
    nmf.close()

    # Check dimensionality
    assert len(obs_list) == num_steps
    assert nmf.vision_update_mask.shape == (num_steps,)
    assert nmf.vision_update_mask.sum() == int(
        num_steps * sim_params.timestep * sim_params.vision_refresh_rate
    )
    height = config["vision"]["raw_img_height_px"]
    width = config["vision"]["raw_img_width_px"]
    assert info["raw_vision"].shape == (2, height, width, 3)
    assert obs["vision"].shape == (2, config["vision"]["num_ommatidia_per_eye"], 2)

    print((obs["vision"][:, :, 0] > 0).sum(), (obs["vision"][:, :, 1] > 0).sum())

    # Test postprocessing
    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    visualize_visual_input(
        nmf.retina,
        output_path=temp_base_dir / "vision/eyes.mp4",
        vision_data_li=[x["vision"] for x in obs_list],
        raw_vision_data_li=[x["raw_vision"] for x in info_list],
        vision_update_mask=nmf.vision_update_mask,
        vision_refresh_rate=sim_params.vision_refresh_rate,
        playback_speed=0.1,
    )
