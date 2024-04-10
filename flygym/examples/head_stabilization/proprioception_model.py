import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import Dataset
from torchmetrics.regression import R2Score
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Callable


class JointAngleScaler:
    @classmethod
    def from_data(cls, joint_angles: np.ndarray):
        scaler = cls()
        scaler.mean = np.mean(joint_angles, axis=0)
        scaler.std = np.std(joint_angles, axis=0)
        return scaler

    @classmethod
    def from_params(cls, mean: np.ndarray, std: np.ndarray):
        scaler = cls()
        scaler.mean = mean
        scaler.std = std
        return scaler

    def __call__(self, joint_angles: np.ndarray):
        return (joint_angles - self.mean) / self.std


class WalkingDataset(Dataset):
    def __init__(
        self,
        sim_data_file: Path,
        contact_force_thr: float = 3,
        joint_angle_scaler: Optional[Callable] = None,
        ignore_first_n: int = 200,
        joint_mask=None,
    ) -> None:
        super().__init__()
        trial_name = sim_data_file.parent.name
        gait, terrain, subset, _, dn_left, dn_right = trial_name.split("_")
        self.gait = gait
        self.terrain = terrain
        self.subset = subset
        self.dn_drive = f"{dn_left}_{dn_right}"
        self.contact_force_thr = contact_force_thr
        self.joint_angle_scaler = joint_angle_scaler
        self.ignore_first_n = ignore_first_n
        self.joint_mask = joint_mask

        with open(sim_data_file, "rb") as f:
            sim_data = pickle.load(f)

        self.cotains_fly_flip = sim_data["errors"]["fly_flipped"]
        self.contains_physics_error = sim_data["errors"]["physics_error"]

        # Extract the roll and pitch angles
        roll = np.array([info["roll"] for info in sim_data["info_hist"]])
        pitch = np.array([info["pitch"] for info in sim_data["info_hist"]])
        self.roll_pitch_ts = np.stack([roll, pitch], axis=1)
        self.roll_pitch_ts = self.roll_pitch_ts[self.ignore_first_n :, :]

        # Extract joint angles and scale them
        joint_angles_raw = np.array(
            [obs["joints"][0, :] for obs in sim_data["obs_hist"]]
        )
        if self.joint_angle_scaler is None:
            self.joint_angle_scaler = JointAngleScaler.from_data(joint_angles_raw)
        self.joint_angles = self.joint_angle_scaler(joint_angles_raw)
        self.joint_angles = self.joint_angles[self.ignore_first_n :, :]

        # Extract contact forces
        contact_forces = np.array(
            [obs["contact_forces"] for obs in sim_data["obs_hist"]]
        )
        contact_forces = np.linalg.norm(contact_forces, axis=2)  # magnitude
        contact_forces = contact_forces.reshape(-1, 6, 6).sum(axis=2)  # sum per leg
        self.contact_mask = (contact_forces >= self.contact_force_thr).astype(np.int16)
        self.contact_mask = self.contact_mask[self.ignore_first_n :, :]

    def __len__(self):
        return self.roll_pitch_ts.shape[0]

    def __getitem__(self, idx):
        joint_angles = self.joint_angles[idx].astype(np.float32, copy=True)
        if self.joint_mask is not None:
            joint_angles[~self.joint_mask] = 0
        return {
            "roll_pitch": self.roll_pitch_ts[idx].astype(np.float32),
            "joint_angles": joint_angles,
            "contact_mask": self.contact_mask[idx].astype(np.float32),
        }


class ThreeLayerMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        input_size = 42 + 6
        hidden_size = 32
        output_size = 2
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.r2_score = R2Score()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = torch.concat([batch["joint_angles"], batch["contact_mask"]], dim=1)
        y = batch["roll_pitch"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.concat([batch["joint_angles"], batch["contact_mask"]], dim=1)
        y = batch["roll_pitch"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        if y.shape[0] > 1:
            r2_roll = self.r2_score(y_hat[:, 0], y[:, 0])
            r2_pitch = self.r2_score(y_hat[:, 1], y[:, 1])
        else:
            r2_roll, r2_pitch = np.nan, np.nan
        self.log("val_r2_roll", r2_roll)
        self.log("val_r2_pitch", r2_pitch)


class HeadStabilizationInferenceWrapper:
    """
    Wrapper for the head stabilization model to make predictions on
    observations. Whereas data are collected in large tensors during
    training, this class provides a "flat" interface for making predictions
    one observation (i.e., time step) at a time. This is useful for
    deploying the model in closed loop.
    """

    def __init__(
        self,
        model_path: Path,
        scaler_param_path: Path,
        input_size: int = 42 + 6,
        hidden_size: int = 8,
        contact_force_thr: float = 3.0,
    ):
        # Load scaler params
        with open(scaler_param_path, "rb") as f:
            scaler_params = pickle.load(f)
        self.scaler_mean = scaler_params["mean"]
        self.scaler_std = scaler_params["std"]

        # Load model
        self.model = ThreeLayerMLP(
            input_size=input_size, hidden_size=hidden_size, output_size=2
        ).cpu()  # it's not worth moving data to the GPU, just run it on the CPU
        self.model.load_state_dict(torch.load(model_path))
        self.contact_force_thr = contact_force_thr

    def __call__(
        self, leg_joint_angles: np.ndarray, leg_contact_forces: np.ndarray
    ) -> np.ndarray:
        leg_joint_angles = (leg_joint_angles - self.scaler_mean) / self.scaler_std
        leg_contact_forces = np.linalg.norm(leg_contact_forces, axis=1)
        leg_contact_forces = leg_contact_forces.reshape(6, 6).sum(axis=1)
        x = np.concatenate([leg_joint_angles, leg_contact_forces], dtype=np.float32)
        input_tensor = torch.tensor(x[None, :]).cpu()
        output_tensor = self.model(input_tensor)
        return output_tensor.detach().numpy().squeeze()
