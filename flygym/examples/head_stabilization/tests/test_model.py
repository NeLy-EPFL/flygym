import numpy as np
import pickle
import torch
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from flygym.util import get_data_path
from flygym.examples.head_stabilization.collect_training_data import run_simulation
from flygym.examples.head_stabilization import JointAngleScaler, WalkingDataset
from flygym.examples.head_stabilization import (
    ThreeLayerMLP,
    HeadStabilizationInferenceWrapper,
)


def test_torch_model():
    r2_scores = {}
    for terrain in ["flat", "blocks"]:
        # Load scaler
        test_data_dir = get_data_path(
            "flygym", "data/trained_models/head_stabilization"
        )
        with open(test_data_dir / "joint_angle_scaler_params.pkl", "rb") as f:
            params = pickle.load(f)
            scaler = JointAngleScaler.from_params(**params)

        # Load model
        model = ThreeLayerMLP.load_from_checkpoint(
            test_data_dir / "all_dofs_model.ckpt",
            map_location=torch.device("cpu"),
        )

        # Simulate data
        with TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir) / "tripod_flat_DEBUG_DEBUG_1.0_1.0"
            tmpdir.mkdir()
            run_simulation(
                gait="tripod",
                terrain=terrain,
                spawn_xy=(0, 0),
                dn_drive=(1, 1),
                sim_duration=0.5,
                enable_rendering=False,
                live_display=False,
                pbar=False,
                output_dir=tmpdir,
            )
            with open(tmpdir / "sim_data.pkl", "rb") as f:
                sim_data = pickle.load(f)
            roll_ts = np.array([info["roll"] for info in sim_data["info_hist"]])
            pitch_ts = np.array([info["pitch"] for info in sim_data["info_hist"]])
            dataset = WalkingDataset(
                sim_data_file=tmpdir / "sim_data.pkl",
                joint_angle_scaler=scaler,
                ignore_first_n=200,
            )
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Check prediction format
        pred_all = []
        for batch in data_loader:
            joint_angles = batch["joint_angles"]
            contact_mask = batch["contact_mask"]
            input_tensor = torch.cat([joint_angles, contact_mask], dim=1)
            pred = model(input_tensor).detach().numpy()
            pred_all.append(pred)
        pred_all = np.concatenate(pred_all, axis=0)
        assert pred_all.shape == (len(roll_ts) - 200, 2)

        # Check prediction accuracy
        # import matplotlib.pyplot as plt
        # plt.plot(pred_all)
        # plt.plot(roll_ts[200:])
        # plt.plot(pitch_ts[200:])
        # plt.show()
        r2_scores[terrain] = {
            "roll": r2_score(roll_ts[200:], pred_all[:, 0]),
            "pitch": r2_score(pitch_ts[200:], pred_all[:, 1]),
        }

    # print(r2_scores)
    # Note that the r2 scores might be lower than reported because the simulations
    # are very short
    assert 0.8 < r2_scores["flat"]["roll"] < 1
    assert 0.8 < r2_scores["flat"]["pitch"] < 1
    assert 0.6 < r2_scores["blocks"]["roll"] < 1
    assert 0.6 < r2_scores["blocks"]["pitch"] < 1
    # Should be better over flat terrain than blocks terrain
    assert r2_scores["flat"]["roll"] > r2_scores["blocks"]["roll"]
    assert r2_scores["flat"]["pitch"] > r2_scores["blocks"]["pitch"]


def test_model_wrapper():
    # Load scaler
    test_data_dir = get_data_path("flygym", "data/trained_models/head_stabilization")
    with open(test_data_dir / "joint_angle_scaler_params.pkl", "rb") as f:
        params = pickle.load(f)
        scaler = JointAngleScaler.from_params(**params)

    # Load torch model
    model = ThreeLayerMLP.load_from_checkpoint(
        test_data_dir / "all_dofs_model.ckpt",
        map_location=torch.device("cpu"),
    )

    # Load wrapper model
    wrapper_model = HeadStabilizationInferenceWrapper(
        model_path=test_data_dir / "all_dofs_model.ckpt",
        scaler_param_path=test_data_dir / "joint_angle_scaler_params.pkl",
        contact_force_thr=(0.5, 1, 3),  # model in test_data was trained with these
    )

    # Simulate data
    with TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir) / "tripod_flat_DEBUG_DEBUG_1.0_1.0"
        tmpdir.mkdir()
        run_simulation(
            gait="tripod",
            terrain="flat",
            spawn_xy=(0, 0),
            dn_drive=(1, 1),
            sim_duration=0.05,
            enable_rendering=False,
            live_display=False,
            pbar=False,
            output_dir=tmpdir,
        )
        with open(tmpdir / "sim_data.pkl", "rb") as f:
            sim_data = pickle.load(f)
        joint_angles_ts = np.array([x["joints"][0, :] for x in sim_data["obs_hist"]])
        cont_forces_ts = np.array([x["contact_forces"] for x in sim_data["obs_hist"]])
        dataset = WalkingDataset(
            sim_data_file=tmpdir / "sim_data.pkl",
            joint_angle_scaler=scaler,
            ignore_first_n=200,
        )
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Predict from torch module
    pred_all_torch = []
    for batch in data_loader:
        joint_angles = batch["joint_angles"]
        contact_mask = batch["contact_mask"]
        input_tensor = torch.cat([joint_angles, contact_mask], dim=1)
        pred = model(input_tensor).detach().numpy()
        pred_all_torch.append(pred)
    pred_all_torch = np.concatenate(pred_all_torch, axis=0)

    # Predict from wrapper
    pred_all_wrapper = []
    for i in range(len(joint_angles_ts)):
        pred = wrapper_model(joint_angles_ts[i, :], cont_forces_ts[i, :])
        pred_all_wrapper.append(pred)
    pred_all_wrapper = np.array(pred_all_wrapper)
    assert np.allclose(pred_all_wrapper[200:, :], pred_all_torch, atol=1e-4)
