import numpy as np
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Iterable


class BaseState(ABC):
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError


class KinematicPose(BaseState):
    def __init__(self, pose_file: Path) -> None:
        self._pose_file = pose_file
        with open(pose_file) as f:
            raw_pose_dict = yaml.safe_load(f)
        self.joint_pos = {k: np.deg2rad(v) for k, v in raw_pose_dict["joints"].items()}

    def __getitem__(self, key: str) -> Any:
        return self.joint_pos[key]

    def __iter__(self) -> Iterable:
        return iter(self.joint_pos)
