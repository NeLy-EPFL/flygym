import numpy as np
import pickle
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import Dataset
from flygym.examples.head_stabilization.collect_training_data import run_simulation
from flygym.examples.head_stabilization.data import JointAngleScaler, WalkingDataset


def test_scaler_from_data():
    data = np.random.rand(100, 42)
    scaler = JointAngleScaler.from_data(data)
    scaled_data = scaler(data)
    mean = np.mean(scaled_data, axis=0)
    std = np.std(scaled_data, axis=0)
    assert np.allclose(mean, np.zeros(42))
    assert np.allclose(std, np.ones(42))
    assert np.allclose(scaler.mean, np.mean(data, axis=0))
    assert np.allclose(scaler.std, np.std(data, axis=0))


def test_scaler_from_params():
    scaler = JointAngleScaler.from_params(mean=np.ones(42), std=np.ones(42) * 2)
    data = np.random.rand(100, 42)
    scaled_data = scaler(data)
    assert np.allclose(scaled_data, (data - 1) / 2)


def test_dataset():
    with TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir) / "tripod_flat_DEBUG_DEBUG_1.0_1.0"
        tmpdir.mkdir()
        run_simulation(
            gait="tripod",
            terrain="flat",
            spawn_xy=(0, 0),
            dn_drive=(1, 1),
            sim_duration=0.1,
            enable_rendering=False,
            live_display=False,
            pbar=False,
            output_dir=tmpdir,
        )
        with open(tmpdir / "sim_data.pkl", "rb") as f:
            sim_data = pickle.load(f)

        scaler = JointAngleScaler.from_params(mean=np.ones(42), std=np.ones(42) * 2)
        dataset = WalkingDataset(
            sim_data_file=tmpdir / "sim_data.pkl",
            joint_angle_scaler=scaler,
            ignore_first_n=200,
        )
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1000 - 200  # nsteps - ignore_first_n

    roll_pitch_ts_ds = np.array([x["roll_pitch"] for x in dataset])
    joint_angles_ts_ds = np.array([x["joint_angles"] for x in dataset])
    contact_mask_ds = np.array([x["contact_mask"] for x in dataset])
    roll_from_sim = np.array([info["roll"] for info in sim_data["info_hist"]])
    pitch_from_sim = np.array([info["pitch"] for info in sim_data["info_hist"]])
    angles_from_sim = np.array([obs["joints"][0, :] for obs in sim_data["obs_hist"]])
    assert roll_pitch_ts_ds.shape[1] == 2
    assert joint_angles_ts_ds.shape[1] == 42
    assert contact_mask_ds.shape[1] == 6
    assert np.allclose(
        roll_pitch_ts_ds, np.stack([roll_from_sim, pitch_from_sim], axis=1)[200:, :]
    )
    assert np.allclose(joint_angles_ts_ds, (angles_from_sim[200:, :] - 1) / 2)  # scaled
