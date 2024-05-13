import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import Dataset
from torchmetrics.regression import R2Score
from pathlib import Path
from typing import Tuple, Optional, Callable


class JointAngleScaler:
    """
    A class for standardizing joint angles (i.e., using mean and standard
    deviation.

    Attributes
    ----------
    mean : np.ndarray
        The mean values used for scaling.
    std : np.ndarray
        The standard deviation values used for scaling.
    """

    @classmethod
    def from_data(cls, joint_angles: np.ndarray):
        """
        Create a JointAngleScaler instance from joint angle data. The mean
        and standard deviation values are calculated from the data.

        Parameters
        ----------
        joint_angles : np.ndarray
            The joint angle data. The shape should be (n_samples, n_joints)
            where n_samples is, for example, the length of a time series of
            joint angles.

        Returns
        -------
        JointAngleScaler
            A JointAngleScaler instance.
        """
        scaler = cls()
        scaler.mean = np.mean(joint_angles, axis=0)
        scaler.std = np.std(joint_angles, axis=0)
        return scaler

    @classmethod
    def from_params(cls, mean: np.ndarray, std: np.ndarray):
        """
        Create a JointAngleScaler instance from predetermined mean and
        standard deviation values.

        Parameters
        ----------
        mean : np.ndarray
            The mean values. The shape should be (n_joints,).
        std : np.ndarray
            The standard deviation values. The shape should be (n_joints,).

        Returns
        -------
        JointAngleScaler
            A JointAngleScaler instance.
        """
        scaler = cls()
        scaler.mean = mean
        scaler.std = std
        return scaler

    def __call__(self, joint_angles: np.ndarray):
        """
        Scale the given joint angles.

        Parameters
        ----------
        joint_angles : np.ndarray
            The joint angles to be scaled. The shape should be (n_samples,
            n_joints) where n_samples is, for example, the length of a time
            series of joint angles.

        Returns
        -------
        np.ndarray
            The scaled joint angles.
        """
        return (joint_angles - self.mean) / self.std


class WalkingDataset(Dataset):
    """
    PyTorch Dataset class for walking data.

    Parameters
    ----------
    sim_data_file : Path
        The path to the simulation data file.
    contact_force_thr : Tuple[float, float, float], optional
        The threshold values for contact forces, by default (0.5, 1, 3).
    joint_angle_scaler : Optional[Callable], optional
        A callable object used to scale joint angles, by default None.
    ignore_first_n : int, optional
        The number of initial data points to ignore, by default 200.
    joint_mask : Optional, optional
        A mask to apply on joint angles, by default None.

    Attributes
    ----------
    gait : str
        The type of gait.
    terrain : str
        The type of terrain.
    subset : str
        The subset of the data, i.e., "train" or "test".
    dn_drive : str
        The DN drive used to generate the data.
    contact_force_thr : np.ndarray
        The threshold values for contact forces.
    joint_angle_scaler : Callable
        The callable object used to scale joint angles.
    ignore_first_n : int
        The number of initial data points to ignore.
    joint_mask : Optional
        The mask applied on joint angles. This is used to zero out certain
        DoFs to evaluate which DoFs are likely more important for head
        stabilization.
    contains_fly_flip : bool
        Indicates if the simulation data contains fly flip errors.
    contains_physics_error : bool
        Indicates if the simulation data contains physics errors.
    roll_pitch_ts : np.ndarray
        The optimal roll and pitch correction angles. The shape is
        (n_samples, 2).
    joint_angles : np.ndarray
        The scaled joint angle time series. The shape is (n_samples,
        n_joints).
    contact_mask : np.ndarray
        The contact force mask (i.e., 1 if leg touching the floor, 0
        otherwise). The shape is (n_samples, 6).
    """

    def __init__(
        self,
        sim_data_file: Path,
        contact_force_thr: Tuple[float, float, float] = (0.5, 1, 3),
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
        self.contact_force_thr = np.array([*contact_force_thr, *contact_force_thr])
        self.joint_angle_scaler = joint_angle_scaler
        self.ignore_first_n = ignore_first_n
        self.joint_mask = joint_mask

        with open(sim_data_file, "rb") as f:
            sim_data = pickle.load(f)

        self.contains_fly_flip = sim_data["errors"]["fly_flipped"]
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
    """
    A PyTorch Lightning module for a three-layer MLP that predicts the
    head roll and pitch correction angles based on proprioception and
    tactile information.
    """

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
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. The shape should be (n_samples, 42 + 6)
            where 42 is the number of joint angles and 6 is the number of
            contact masks.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def configure_optimizers(self):
        """Use the Adam optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step of the PyTorch Lightning module."""
        x = torch.concat([batch["joint_angles"], batch["contact_mask"]], dim=1)
        y = batch["roll_pitch"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step of the PyTorch Lightning module."""
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
        contact_force_thr: Tuple[float, float, float] = (0.5, 1, 3),
    ):
        """
        Parameters
        ----------
        model_path : Path
            The path to the trained model.
        scaler_param_path : Path
            The path to the pickle file containing scaler parameters.
        contact_force_thr : Tuple[float, float, float], optional
            The threshold values for contact forces that are used to
            determine the floor contact flags, by default (0.5, 1, 3).
        """
        # Load scaler params
        with open(scaler_param_path, "rb") as f:
            scaler_params = pickle.load(f)
        self.scaler_mean = scaler_params["mean"]
        self.scaler_std = scaler_params["std"]

        # Load model
        # it's not worth moving data to the GPU, just run it on the CPU
        self.model = ThreeLayerMLP.load_from_checkpoint(
            model_path, map_location=torch.device("cpu")
        )
        self.contact_force_thr = np.array([*contact_force_thr, *contact_force_thr])

    def __call__(
        self, joint_angles: np.ndarray, contact_forces: np.ndarray
    ) -> np.ndarray:
        """
        Make a prediction given joint angles and contact forces. This is
        a light wrapper around the model's forward method and works without
        batching.

        Parameters
        ----------
        joint_angles : np.ndarray
            The joint angles. The shape should be (n_joints,).
        contact_forces : np.ndarray
            The contact forces. The shape should be (n_legs, 6).

        Returns
        -------
        np.ndarray
            The predicted roll and pitch angles. The shape is (2,).
        """
        joint_angles = (joint_angles - self.scaler_mean) / self.scaler_std
        contact_forces = np.linalg.norm(contact_forces, axis=1)
        contact_forces = contact_forces.reshape(6, 6).sum(axis=1)
        contact_mask = contact_forces >= self.contact_force_thr
        x = np.concatenate([joint_angles, contact_mask], dtype=np.float32)
        input_tensor = torch.tensor(x[None, :], device=torch.device("cpu"))
        output_tensor = self.model(input_tensor)
        return output_tensor.detach().numpy().squeeze()
