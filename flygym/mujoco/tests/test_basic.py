import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym.mujoco import NeuroMechFly
from flygym.mujoco.util import plot_mujoco_rollout


def test_basic_untethered_sinewave():
    random_state = np.random.RandomState(0)

    nmf = NeuroMechFly()
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf._actuators))
    amp = 0.9

    obs_list = []
    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = nmf.step(action)
        # nmf.render()
        obs_list.append(obs)
    nmf.close()

    # These should be deterministic
    # print(obs_list[-1]["fly"].sum())
    assert np.isclose(obs_list[-1]["fly"].sum(), -257.14624, rtol=0.03)

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")
    out_dir = temp_base_dir / "mujoco_basic_untethered_sinewave"
    # nmf.save_video(out_dir / "video.mp4")
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir / "plot.png")
