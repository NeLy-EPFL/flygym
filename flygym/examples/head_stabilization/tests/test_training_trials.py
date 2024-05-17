import numpy as np
import pickle
from tempfile import TemporaryDirectory
from pathlib import Path
from flygym.examples.head_stabilization.collect_training_data import run_simulation


def test_simulation():
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        run_simulation(
            gait="tripod",
            terrain="flat",
            sim_duration=0.1,
            enable_rendering=False,
            live_display=False,
            pbar=False,
            output_dir=tmpdir,
        )
        with open(tmpdir / "sim_data.pkl", "rb") as f:
            sim_data = pickle.load(f)

    for info in sim_data["info_hist"]:
        assert "roll" in info
        assert "pitch" in info
        assert np.isfinite(info["roll"])
        assert np.isfinite(info["pitch"])

    roll_ts_flat = np.array([info["roll"] for info in sim_data["info_hist"]])
    pitch_ts_flat = np.array([info["pitch"] for info in sim_data["info_hist"]])

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        run_simulation(
            gait="tripod",
            terrain="blocks",
            sim_duration=0.1,
            enable_rendering=False,
            live_display=False,
            pbar=False,
            output_dir=tmpdir,
        )
        with open(tmpdir / "sim_data.pkl", "rb") as f:
            sim_data = pickle.load(f)
    roll_ts_blocks = np.array([info["roll"] for info in sim_data["info_hist"]])
    pitch_ts_blocks = np.array([info["pitch"] for info in sim_data["info_hist"]])
    # import matplotlib.pyplot as plt
    # plt.plot(roll_ts_flat, label="flat")
    # plt.plot(roll_ts_blocks, label="blocks")
    # plt.plot(pitch_ts_flat, label="flat")
    # plt.plot(pitch_ts_blocks, label="blocks")
    # plt.show()
    # print(np.abs(roll_ts_flat).mean(), np.abs(roll_ts_blocks).mean())
    # print(np.abs(pitch_ts_flat).mean(), np.abs(pitch_ts_blocks).mean())
    assert np.abs(roll_ts_flat).mean() < np.abs(roll_ts_blocks).mean()
    assert np.abs(pitch_ts_flat).mean() < np.abs(pitch_ts_blocks).mean()
