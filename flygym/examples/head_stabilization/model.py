import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics.regression import R2Score
from pathlib import Path


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
        contact_force_thr: tuple[float, float, float] = (0.5, 1, 3),
    ):
        """
        Parameters
        ----------
        model_path : Path
            The path to the trained model.
        scaler_param_path : Path
            The path to the pickle file containing scaler parameters.
        contact_force_thr : tuple[float, float, float], optional
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
            The contact forces. The shape should be (n_legs * n_segments).

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
