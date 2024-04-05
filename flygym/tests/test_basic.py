import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym import NeuroMechFly
from flygym.util import plot_mujoco_rollout


def test_basic_untethered_sinewave():
    np.random.seed(0)

    nmf = NeuroMechFly()
    run_time = 0.01
    freq = 20
    amp = np.pi / 2

    obs, _ = nmf.reset()
    fly_init_pos = obs["joints"][0]

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = fly_init_pos + amp * np.sin(freq * nmf.curr_time)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        obs_list.append(obs)
    nmf.close()

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")
    out_dir = temp_base_dir / "mujoco_basic_untethered_sinewave"
    # nmf.save_video(out_dir / "video.mp4")
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir / "plot.png")
