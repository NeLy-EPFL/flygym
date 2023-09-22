import numpy as np
import tempfile
from pathlib import Path

from flygym.mujoco import NeuroMechFlyMuJoCo
from flygym.mujoco.util import plot_mujoco_rollout

random_state = np.random.RandomState(0)


def test_basic_untethered_sinewave():
    nmf = NeuroMechFlyMuJoCo()
    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
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
    # print(obs_list[-1]["joints"].sum())
    # print(obs_list[-1]["fly"].sum())
    # print(obs_list[-1]["contact_forces"].sum())
    # print(obs_list[-1]["fly_orientation"].sum())
    # print(obs_list[-1]["end_effectors"].sum())
    assert np.isclose(obs_list[-1]["joints"].sum(), 606.5673288891932)
    assert np.isclose(obs_list[-1]["fly"].sum(), -80.72276400618352)
    assert np.isclose(obs_list[-1]["contact_forces"].sum(), 0)
    assert np.isclose(obs_list[-1]["fly_orientation"].sum(), 0.5435753368176544)
    assert np.isclose(obs_list[-1]["end_effectors"].sum(), 17.853036591414362)

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    out_dir = temp_base_dir / "mujoco_basic_untethered_sinewave"
    # nmf.save_video(out_dir / "video.mp4")
    plot_mujoco_rollout(obs_list, nmf.timestep, out_dir / "plot.png")
