"""FlyBody-specific anatomical definitions.

This module extends the default anatomy types from ``flygym.anatomy``.
"""

from __future__ import annotations

import warnings
from typing import Iterator
from enum import Enum

from flygym.anatomy import (
    BaseRotationAxis,
    BaseAxisOrder,
    RotationAxis,
    AxisOrder,
    AxesSet,
    AnatomicalJoint,
    BaseActuatedDOFPreset,
    BaseJointPreset,
    BaseContactBodiesPreset,
    JointDOF,
    Skeleton,
    SIDES,
    LEGS,
    _chain2joints,
)

from flygym.anatomy import (
    BodySegment,
)

from flygym.utils.math import orderedset

__all__ = [
    "FlyBodyRotationAxis",
    "WingFlyBodyRotationAxis",
    "FlyBodyAxesSet",
    "WingFlyBodyAxesSet",
    "FlyBodyAxisOrder",
    "WingFlyBodyAxisOrder",
    "FlyBodyBodySegment",
    "FlyBodyJointPreset",
    "FlyBodyActuatedDOFPreset",
    "FlyBodyContactBodiesPreset",
    "FlyBodySkeleton",
    "FlyBodyJointDOF",
    "FlyBodyAnatomicalJoint",
]


class FlyBodyRotationAxis(BaseRotationAxis):
    """FlyBody axis convention.

    yaw -> z, pitch -> x, roll -> y.
    """

    PITCH = "pitch"
    P = PITCH
    ROLL = "roll"
    R = ROLL
    YAW = "yaw"
    Y = YAW

    @classmethod
    def _vector_by_axis(cls) -> dict[str, tuple[float, float, float]]:
        return {
            "pitch": (1, 0, 0),
            "roll": (0, 1, 0),
            "yaw": (0, 0, 1),
        }


class WingFlyBodyRotationAxis(BaseRotationAxis):
    """FlyBody wing axis convention.

    Same as FlyBody except pitch/roll are swapped:
    yaw -> z, pitch -> y, roll -> x.
    """

    PITCH = "pitch"
    P = PITCH
    ROLL = "roll"
    R = ROLL
    YAW = "yaw"
    Y = YAW

    @classmethod
    def _vector_by_axis(cls) -> dict[str, tuple[float, float, float]]:
        return {
            "pitch": (0, 1, 0),
            "roll": (1, 0, 0),
            "yaw": (0, 0, 1),
        }


class FlyBodyAxesSet(AxesSet):
    """Set of rotation axes using FlyBody's axis convention."""

    rotation_axis_class = FlyBodyRotationAxis


class WingFlyBodyAxesSet(AxesSet):
    """Set of rotation axes for wings using WingFlyBody's axis convention."""

    rotation_axis_class = WingFlyBodyRotationAxis


class FlyBodyAxisOrder(BaseAxisOrder, Enum):
    """Axis order enum based on FlyBodyRotationAxis."""

    @classmethod
    def _axis_enum_cls(cls):
        return FlyBodyRotationAxis

    PITCH_ROLL_YAW = (
        FlyBodyRotationAxis.PITCH,
        FlyBodyRotationAxis.ROLL,
        FlyBodyRotationAxis.YAW,
    )
    PRY = PITCH_ROLL_YAW
    PITCH_YAW_ROLL = (
        FlyBodyRotationAxis.PITCH,
        FlyBodyRotationAxis.YAW,
        FlyBodyRotationAxis.ROLL,
    )
    PYR = PITCH_YAW_ROLL
    ROLL_PITCH_YAW = (
        FlyBodyRotationAxis.ROLL,
        FlyBodyRotationAxis.PITCH,
        FlyBodyRotationAxis.YAW,
    )
    RPY = ROLL_PITCH_YAW
    ROLL_YAW_PITCH = (
        FlyBodyRotationAxis.ROLL,
        FlyBodyRotationAxis.YAW,
        FlyBodyRotationAxis.PITCH,
    )
    RYP = ROLL_YAW_PITCH
    YAW_PITCH_ROLL = (
        FlyBodyRotationAxis.YAW,
        FlyBodyRotationAxis.PITCH,
        FlyBodyRotationAxis.ROLL,
    )
    YPR = YAW_PITCH_ROLL
    YAW_ROLL_PITCH = (
        FlyBodyRotationAxis.YAW,
        FlyBodyRotationAxis.ROLL,
        FlyBodyRotationAxis.PITCH,
    )
    YRP = YAW_ROLL_PITCH

    DONTCARE = PITCH_ROLL_YAW


class WingFlyBodyAxisOrder(BaseAxisOrder, Enum):
    """Axis order enum based on WingFlyBodyRotationAxis."""

    @classmethod
    def _axis_enum_cls(cls):
        return WingFlyBodyRotationAxis

    PITCH_ROLL_YAW = (
        WingFlyBodyRotationAxis.PITCH,
        WingFlyBodyRotationAxis.ROLL,
        WingFlyBodyRotationAxis.YAW,
    )
    PRY = PITCH_ROLL_YAW
    PITCH_YAW_ROLL = (
        WingFlyBodyRotationAxis.PITCH,
        WingFlyBodyRotationAxis.YAW,
        WingFlyBodyRotationAxis.ROLL,
    )
    PYR = PITCH_YAW_ROLL
    ROLL_PITCH_YAW = (
        WingFlyBodyRotationAxis.ROLL,
        WingFlyBodyRotationAxis.PITCH,
        WingFlyBodyRotationAxis.YAW,
    )
    RPY = ROLL_PITCH_YAW
    ROLL_YAW_PITCH = (
        WingFlyBodyRotationAxis.ROLL,
        WingFlyBodyRotationAxis.YAW,
        WingFlyBodyRotationAxis.PITCH,
    )
    RYP = ROLL_YAW_PITCH
    YAW_PITCH_ROLL = (
        WingFlyBodyRotationAxis.YAW,
        WingFlyBodyRotationAxis.PITCH,
        WingFlyBodyRotationAxis.ROLL,
    )
    YPR = YAW_PITCH_ROLL
    YAW_ROLL_PITCH = (
        WingFlyBodyRotationAxis.YAW,
        WingFlyBodyRotationAxis.ROLL,
        WingFlyBodyRotationAxis.PITCH,
    )
    YRP = YAW_ROLL_PITCH

    DONTCARE = PITCH_ROLL_YAW


FLYBODY_LEG_LINKS: list[str] = [
    "coxa",
    "trochanterfemur",
    "tibia",
    *(f"tarsus{seg}" for seg in "12345"),
]
FLYBODY_PROBOSCIS_LINKS: list[str] = ["rostrum", "haustellum"]
FLYBODY_ABDOMEN_LINKS: list[str] = [f"abdomen{seg}" for seg in "1234567"]
FLYBODY_PASSIVE_TARSAL_LINKS: list[str] = [f"tarsus{seg}" for seg in "2345"]
FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS: list[tuple[str, str]] = [
    ("c_thorax", "c_head"),
    *(_chain2joints("c_head", *(f"c_{lk}" for lk in FLYBODY_PROBOSCIS_LINKS))),
    *(("c_haustellum", f"{s}_labrum") for s in SIDES),
    *(("c_head", f"{s}_antenna") for s in SIDES),
    *(_chain2joints("c_thorax", *(f"c_{lk}" for lk in FLYBODY_ABDOMEN_LINKS))),
    *(("c_thorax", f"{s}_wing") for s in SIDES),
    *(("c_thorax", f"{s}_haltere") for s in SIDES),
    *(
        edge
        for leg in LEGS
        for edge in _chain2joints(
            "c_thorax", *(f"{leg}_{lk}" for lk in FLYBODY_LEG_LINKS)
        )
    ),
]
FLYBODY_PROBOSCIS_LINKS += ["labrum"]
FLYBODY_ALL_SEGMENT_NAMES: list[str] = orderedset(
    [seg for joint in FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS for seg in joint]
)


class FlyBodyBodySegment(BodySegment):
    """FlyBody-specific body segment class."""

    def __post_init__(self):
        if self.name not in FLYBODY_ALL_SEGMENT_NAMES:
            raise ValueError(
                f"Invalid body segment name: {self.name}. "
                f"Must be one of {FLYBODY_ALL_SEGMENT_NAMES}."
            )

    def is_proboscis(self) -> bool:
        """Return True if this segment belongs to the proboscis."""
        return self.link in FLYBODY_PROBOSCIS_LINKS

    def is_eye(self) -> bool:
        """No eyes in flybody model, eyes are part of the head."""
        return False

    def is_antenna(self) -> bool:
        """Return True if this segment belongs to an antenna."""
        return self.link == "antenna"

    def is_leg(self) -> bool:
        """Return True if this segment belongs to a leg."""
        return self.pos in LEGS

    def is_abdomen(self) -> bool:
        """Return True if this segment belongs to the abdomen."""
        return self.link in FLYBODY_ABDOMEN_LINKS


class FlyBodyJointDOF(JointDOF):
    """Joint DOF specific to the flybody model."""

    @classmethod
    def from_name(cls, name: str) -> "FlyBodyJointDOF":
        """Create a FlyBodyJointDOF from a name of the form 'parent-child-axis'."""
        try:
            parent, child, axis = name.split("-")
            # check if child is wing
            bs_child = FlyBodyBodySegment(child)
            return cls(
                parent=FlyBodyBodySegment(parent),
                child=bs_child,
                axis=FlyBodyRotationAxis(axis)
                if not bs_child.is_wing()
                else WingFlyBodyRotationAxis(axis),
            )
        except ValueError:
            raise ValueError(f"Invalid joint DOF name: {name}. ")


class FlyBodyAnatomicalJoint(AnatomicalJoint):
    """Anatomical joint specific to the flybody model."""

    def iter_dofs(self, axis_order: AxisOrder) -> Iterator[FlyBodyJointDOF]:
        """Iterate through the DOFs of this joint in the specified axis order."""
        if self.child.is_wing():
            wing_axis_order = WingFlyBodyAxisOrder(
                [axis.value for axis in axis_order.value]
            )
            for axis in wing_axis_order.value:
                if axis in self.axes:
                    yield FlyBodyJointDOF(
                        parent=self.parent,
                        child=self.child,
                        axis=axis,
                    )
            return

        for axis in axis_order.value:
            if axis in self.axes:
                yield FlyBodyJointDOF(
                    parent=self.parent,
                    child=self.child,
                    axis=axis,
                )


class FlyBodyJointPreset(BaseJointPreset):
    ALL_POSSIBLE = "all_possible"
    ALL_BIOLOGICAL = "all_biological"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_connected_segment_pairs(cls):
        return FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS

    @classmethod
    def _get_passive_tarsal_links(cls):
        return FLYBODY_PASSIVE_TARSAL_LINKS

    @classmethod
    def _get_all_possible_joints(cls) -> list[FlyBodyAnatomicalJoint]:
        all_possible_joints = []
        for parent, child in cls._get_connected_segment_pairs():
            parent_bs = FlyBodyBodySegment(parent)
            child_bs = FlyBodyBodySegment(child)
            if child_bs.is_wing():
                axes = WingFlyBodyAxesSet(WingFlyBodyRotationAxis)
            else:
                axes = FlyBodyAxesSet(FlyBodyRotationAxis)
            all_possible_joints.append(
                FlyBodyAnatomicalJoint(
                    parent=parent_bs,
                    child=child_bs,
                    axes=axes,
                )
            )
        return all_possible_joints

    @classmethod
    def _get_all_biological_joints(cls) -> list[FlyBodyAnatomicalJoint]:
        joints = cls._get_all_possible_joints()
        for joint in joints:
            if joint.child.is_leg():
                match joint.child.link:
                    case "coxa":
                        pass
                    case "trochanterfemur":
                        joint.axes.remove("yaw")
                    case _:
                        joint.axes.remove("roll")
                        joint.axes.remove("yaw")
            if joint.child.is_proboscis():
                match joint.child.link:
                    case "rostrum":
                        joint.axes.remove("yaw")
                        joint.axes.remove("roll")
                    case "haustellum":
                        joint.axes.remove("roll")
                    case "labrum":
                        joint.axes.remove("yaw")
                        joint.axes.remove("roll")
            if joint.child.is_abdomen():
                joint.axes.remove("roll")
            if joint.child.is_haltere():
                joint.axes.remove("yaw")
                joint.axes.remove("roll")
        return joints


class FlyBodyActuatedDOFPreset(BaseActuatedDOFPreset):
    """Presets for which flybody joint DoFs should be actuated."""

    ALL = "all"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_passive_tarsal_links(cls) -> list[str]:
        return FLYBODY_PASSIVE_TARSAL_LINKS


class FlyBodyContactBodiesPreset(BaseContactBodiesPreset):
    """Presets for which flybody segments can collide with the ground."""

    ALL = "all"
    LEGS_THORAX_ABDOMEN_HEAD = "legs_thorax_abdomen_head"
    LEGS_ONLY = "legs_only"
    TIBIA_TARSUS_ONLY = "tibia_tarsus_only"

    @classmethod
    def _get_all_segments(cls):
        return [FlyBodyBodySegment(segname) for segname in FLYBODY_ALL_SEGMENT_NAMES]


class FlyBodySkeleton(Skeleton):
    """Skeleton specific to the flybody model."""

    def __init__(
        self,
        *,
        axis_order: FlyBodyAxisOrder
        | WingFlyBodyAxisOrder
        | AxisOrder
        | list[RotationAxis | FlyBodyRotationAxis | WingFlyBodyRotationAxis | str],
        joint_preset: "FlyBodyJointPreset | str | None" = None,
        anatomical_joints: list[FlyBodyAnatomicalJoint] | None = None,
    ) -> None:
        if not (joint_preset is None) ^ (anatomical_joints is None):
            raise ValueError(
                "Skeleton must be initiated from either joint_preset or "
                "anatomical_joints, but not both."
            )

        if joint_preset is not None:
            anatomical_joints = FlyBodyJointPreset(joint_preset).to_joint_list()
        self.anatomical_joints = anatomical_joints

        self.joint_lookup = {(j.parent, j.child): j for j in anatomical_joints}
        self.body_segments = orderedset(
            [seg for nodes in self.joint_lookup.keys() for seg in nodes]
        )
        if isinstance(axis_order, AxisOrder):
            warnings.warn(
                "Using a generic AxisOrder with FlyBodySkeleton; "
                "converting to FlyBodyAxisOrder."
            )
            axis_order = axis_order.to_list_of_str()
        self.axis_order = FlyBodyAxisOrder(axis_order)

    def iter_jointdofs(
        self,
        root: FlyBodyBodySegment | str = "c_thorax",
    ) -> Iterator[FlyBodyJointDOF]:
        """Iterate through joint DOFs in depth-first order starting from the root."""
        if isinstance(root, str):
            root = FlyBodyBodySegment(root)
        tree = self.get_tree()
        for parent, child in tree.dfs_edges(root):
            anatomical_joint = self.joint_lookup[(parent, child)]
            for jointdof in anatomical_joint.iter_dofs(self.axis_order):
                yield jointdof

    def get_actuated_dofs_from_preset(
        self, preset: FlyBodyActuatedDOFPreset | str
    ) -> list[FlyBodyJointDOF]:
        """Given a flybody preset of actuated DoFs, return an explicit list of joints."""
        if isinstance(preset, BaseActuatedDOFPreset):
            preset = FlyBodyActuatedDOFPreset(preset.value)
        else:
            preset = FlyBodyActuatedDOFPreset(preset)
        return preset.filter(list(self.iter_jointdofs()))
