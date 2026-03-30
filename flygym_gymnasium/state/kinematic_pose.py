import numpy as np
import yaml
from pathlib import Path
from typing import Any, Iterable

from .base import BaseState


class KinematicPose(BaseState):
    def __init__(self, joint_pos: dict[str, float]) -> None:
        """Pose (joint angles) of an animal.

        Parameters
        ----------
        joint_pos : dict[str, float]
            Specifies the joint angles in radian.
        """
        self.joint_pos = joint_pos

    @classmethod
    def from_yaml(cls, pose_file: Path) -> "KinematicPose":
        """Load pose from YAML file.

        Parameters
        ----------
        pose_file : Path
            Path to the YAML file containing joint angles.
        """
        with open(pose_file) as f:
            raw_pose_dict = yaml.safe_load(f)
        joint_pos = {k: np.deg2rad(v) for k, v in raw_pose_dict["joints"].items()}
        return cls(joint_pos)

    def __getitem__(self, key: str) -> Any:
        return self.joint_pos[key]

    def __iter__(self) -> Iterable:
        return iter(self.joint_pos)
