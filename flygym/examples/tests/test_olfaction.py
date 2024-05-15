from flygym.examples.olfaction import run_simulation
import numpy as np


def test_odor_taxis_example():
    # test the default odor taxis example
    odor_source = np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]])
    target_pos = odor_source[0, :2]
    peak_odor_intensity = np.array([[1, 0], [0, 1], [0, 1]])
    distance_threshold = 2

    obs_hist = run_simulation(
        odor_source,
        peak_odor_intensity,
        spawn_pos=(0, 0, 0.2),
        spawn_orientation=(0, 0, np.pi / 2),
        run_time=5,
        decision_interval=0.05,
        attractive_gain=-500,
        aversive_gain=80,
        attractive_palps_antennae_weights=(1, 9),
        aversive_palps_antennae_weights=(0, 10),
        target_pos=target_pos,
        distance_threshold=distance_threshold,
    )

    last_pos = obs_hist[-1]["fly"][0, :2]
    assert np.linalg.norm(last_pos - target_pos) <= distance_threshold


def test_odor_taxis_full_turn():
    # Test that the fly can make a full turn if the attractive odor source
    # is placed behind it
    odor_source = np.array([[-5, 0, 1.5]])
    target_pos = odor_source[0, :2]
    distance_threshold = 2

    obs_hist = run_simulation(
        odor_source,
        peak_odor_intensity=np.array([[1, 1e-3]]),
        spawn_pos=(0, 0, 0.2),
        spawn_orientation=(0, 0, np.pi / 2),
        run_time=5,
        target_pos=target_pos,
        distance_threshold=distance_threshold,
        video_path="temp.mp4",
    )

    last_pos = obs_hist[-1]["fly"][0, :2]
    assert np.linalg.norm(last_pos - target_pos) <= distance_threshold
