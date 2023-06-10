import numpy as np
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Iterable


class BaseState(ABC):
    """Base class for animal state (eg. pose) representations. Behaves
    like a dictionary."""

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError


class KinematicPose(BaseState):
    def __init__(self, joint_pos: Dict[str, float]) -> None:
        """Pose (joint angles) of an animal.

        Parameters
        ----------
        joint_pos : Dict[str, float]
            Specifies the joint angles in radian.
        """
        self.joint_pos = joint_pos

    @classmethod
    def from_yaml(cls, pose_file: Path) -> None:
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
