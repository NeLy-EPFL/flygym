from pathlib import Path
from os import PathLike
from enum import Enum

import numpy as np
import yaml

from flygym import assets_dir
from flygym.anatomy import AxisOrder, JointDOF, BodySegment, RotationAxis

__all__ = ["KinematicPose", "KinematicPosePreset"]


class KinematicPose:
    """A snapshot of joint angles defining a static fly pose.

    Args:
        path:
            Path to YAML file containing joint angles and metadata. Either this or
            `joint_angles_rad_dict` must be provided, but not both.
        joint_angles_rad_dict:
            Dictionary mapping joint DoF names to angles in radians. Either this or
            `path` must be provided, but not both.
        axis_order:
            The AxisOrder of the provided angles if initializing from
            `joint_angles_rad_dict` (required). If initializing from `path`, this
            attribute must not be specified because the axis order will be loaded from
            the file.
        mirror_left2right:
            If True, mirror left-side joint angles to right-side when not provided.

    Example:

        pose = KinematicPose(path="neutral.yaml", mirror_left2right=True)
        # then use `pose.joint_angles_lookup_rad`
    """

    def __init__(
        self,
        *,
        path: PathLike | None = None,
        joint_angles_rad_dict: dict[str, float] | None = None,
        axis_order: AxisOrder | str | list[RotationAxis | str] | None = None,
        mirror_left2right: bool = True,
    ) -> None:
        if joint_angles_rad_dict is not None and path is None:
            if axis_order is None:
                raise ValueError(
                    "When initializing from `joint_angles_rad_dict`, axis_order must "
                    "also be provided."
                )
            axis_order = AxisOrder(axis_order)
        elif path is not None and joint_angles_rad_dict is None:
            if axis_order is not None:
                raise ValueError(
                    "When initializing from `path`, `axis_order` should not be "
                    "provided because it will be loaded from the pose file."
                )
            joint_angles_rad_dict, axis_order = _load_pose_yaml(path)
        else:
            raise ValueError(
                "Either joint_angles_rad_dict or path must be provided, but not both."
            )

        joint_angles_rad_dict = dict(joint_angles_rad_dict)  # don't mutate caller dict
        if mirror_left2right:
            _mirror_pose_left2right_in_place(joint_angles_rad_dict)

        self.axis_order = axis_order
        self.joint_angles_lookup_rad = joint_angles_rad_dict

    def copy(self) -> "KinematicPose":
        """Return a deep copy of this pose."""
        return KinematicPose(
            joint_angles_rad_dict=self.joint_angles_lookup_rad.copy(),
            axis_order=self.axis_order,
        )


def _load_pose_yaml(path: PathLike) -> tuple[dict[str, float], AxisOrder]:
    with open(path, "r") as f:
        pose_data = yaml.safe_load(f)

    angle_unit = pose_data.get("angle_unit")
    if angle_unit not in ("degree", "radian"):
        raise ValueError("YAML file must contain angle_unit: 'degree' or 'radian'.")

    joint_angles = pose_data.get("joint_angles")
    if not isinstance(joint_angles, dict):
        raise ValueError("YAML file must contain 'joint_angles' mapping.")
    for k, v in joint_angles.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Joint angle for '{k}' must be a number.")

    joint_angles = {k: float(v) for k, v in joint_angles.items()}
    if angle_unit == "degree":
        joint_angles = {k: float(np.deg2rad(v)) for k, v in joint_angles.items()}

    axis_order_raw = pose_data.get("axis_order")
    try:
        axis_order = AxisOrder(axis_order_raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid or missing axis_order: {axis_order_raw}")

    return joint_angles, axis_order


def _mirror_pose_left2right_in_place(joint_angles: dict[str, float]) -> None:
    """
    Mirror left-side to right-side when missing. Mutates dict in place.
    """
    # We must iterate over a snapshot because we may add new keys
    items = list(joint_angles.items())
    for joint_name, angle in items:
        jointdof = JointDOF.from_name(joint_name)
        if jointdof.child.name[0] != "l":
            continue

        mirror_parent = BodySegment(
            ("r" + jointdof.parent.name[1:])
            if jointdof.parent.name[0] == "l"
            else jointdof.parent.name
        )
        mirror_child = BodySegment("r" + jointdof.child.name[1:])
        mirror_jointdof = JointDOF(mirror_parent, mirror_child, jointdof.axis)

        if mirror_jointdof.name not in joint_angles:
            joint_angles[mirror_jointdof.name] = float(angle)


class KinematicPosePreset(Enum):
    """Presets for commonly used fly poses.

    Attributes:
        NEUTRAL: The neutral (resting) pose of the fly.
    """

    NEUTRAL = "neutral"

    def get_dir(self) -> Path:
        match self:
            case KinematicPosePreset.NEUTRAL:
                return assets_dir / "model/pose/neutral/"
            case _:
                raise ValueError(f"Unsupported KinematicPosePreset: {self.value}")

    def get_pose_by_axis_order(
        self, axis_order: AxisOrder, mirror_left2right: bool = True
    ) -> KinematicPose:
        """Load the preset pose for a given axis order.

        Args:
            axis_order: The axis order to use.
            mirror_left2right: If True, mirror left-side angles to the right side.

        Returns:
            The loaded `KinematicPose`.
        """
        pose_dir = self.get_dir()
        pose_path = pose_dir / f"{axis_order.to_str()}.yaml"
        return KinematicPose(path=pose_path, mirror_left2right=mirror_left2right)
