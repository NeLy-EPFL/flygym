import pickle
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable


class JointAngleScaler:
    """
    A class for standardizing joint angles (i.e., using mean and standard
    deviation.)

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
    contact_force_thr : tuple[float, float, float], optional
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
        contact_force_thr: tuple[float, float, float] = (0.5, 1, 3),
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
