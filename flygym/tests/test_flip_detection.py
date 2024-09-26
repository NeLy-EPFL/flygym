import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym import Fly, SingleFlySimulation, is_rendering_skipped


def test_fly_flipped():
    np.random.seed(0)

    print("is_rendering_skipped: ", is_rendering_skipped)

    cameras = [] if is_rendering_skipped else None  # None = default camera
    fly = Fly(spawn_orientation=(0, np.pi, 0), spawn_pos=(0, 0, 3))
    sim = SingleFlySimulation(fly=fly, cameras=cameras)
    run_time = 0.02

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    rendered_images = []
    info_hist = []
    while sim.curr_time < run_time - 1e-5:
        action = {"joints": fly_init_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        img = None if is_rendering_skipped else sim.render()[0]
        if img is not None:
            rendered_images.append(img)
        info_hist.append(info)
    sim.close()

    assert all([info["flip"] for info in info_hist])

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")
    out_dir = temp_base_dir / "fly_flip_detection"
    if not is_rendering_skipped:
        sim.cameras[0].save_video(out_dir / "flipped.mp4", stabilization_time=0)


def test_fly_not_flipped():
    np.random.seed(0)

    print("is_rendering_skipped: ", is_rendering_skipped)

    cameras = [] if is_rendering_skipped else None  # None = default camera
    fly = Fly(spawn_orientation=(0, 0, 0), spawn_pos=(0, 0, 3))
    sim = SingleFlySimulation(fly=fly, cameras=cameras)
    run_time = 0.05

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    rendered_images = []
    info_hist = []
    while sim.curr_time < run_time - 1e-5:
        action = {"joints": fly_init_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        img = None if is_rendering_skipped else sim.render()[0]
        if img is not None:
            rendered_images.append(img)
        info_hist.append(info)
    sim.close()

    assert all([not info["flip"] for info in info_hist])

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")
    out_dir = temp_base_dir / "fly_flip_detection"
    if not is_rendering_skipped:
        sim.cameras[0].save_video(out_dir / "not_flipped.mp4", stabilization_time=0)
