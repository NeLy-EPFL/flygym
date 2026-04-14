"""Flybody-specific anatomical definitions.

This module extends the default anatomy types from ``flygym.anatomy_base``.
"""

from __future__ import annotations

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
    "FlybodyRotationAxis",
    "WingFlybodyRotationAxis",
    "FlybodyAxesSet",
    "FlybodyAxisOrder",
    "WingFlybodyAxisOrder",
    "FlybodyBodySegment",
    "FlybodyJointPreset",
    "FlybodyActuatedDOFPreset",
    "FlybodyContactBodiesPreset",
    "FlybodySkeleton",
    "FlybodyJointDOF",
]


class FlybodyRotationAxis(BaseRotationAxis):
    """Flybody axis convention.

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


class WingFlybodyRotationAxis(BaseRotationAxis):
    """Flybody wing axis convention.

    Same as Flybody except pitch/roll are swapped:
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


class FlybodyAxesSet(AxesSet):
    """Set of rotation axes using Flybody's axis convention."""

    rotation_axis_class = FlybodyRotationAxis

class WingFlybodyAxesSet(AxesSet):
    """Set of rotation axes for wings using WingFlybody's axis convention."""

    rotation_axis_class = WingFlybodyRotationAxis

class FlybodyAxisOrder(BaseAxisOrder, Enum):
    """Axis order enum based on FlybodyRotationAxis."""

    @classmethod
    def _axis_enum_cls(cls):
        return FlybodyRotationAxis

    PITCH_ROLL_YAW = (
        FlybodyRotationAxis.PITCH,
        FlybodyRotationAxis.ROLL,
        FlybodyRotationAxis.YAW,
    )
    PRY = PITCH_ROLL_YAW
    PITCH_YAW_ROLL = (
        FlybodyRotationAxis.PITCH,
        FlybodyRotationAxis.YAW,
        FlybodyRotationAxis.ROLL,
    )
    PYR = PITCH_YAW_ROLL
    ROLL_PITCH_YAW = (
        FlybodyRotationAxis.ROLL,
        FlybodyRotationAxis.PITCH,
        FlybodyRotationAxis.YAW,
    )
    RPY = ROLL_PITCH_YAW
    ROLL_YAW_PITCH = (
        FlybodyRotationAxis.ROLL,
        FlybodyRotationAxis.YAW,
        FlybodyRotationAxis.PITCH,
    )
    RYP = ROLL_YAW_PITCH
    YAW_PITCH_ROLL = (
        FlybodyRotationAxis.YAW,
        FlybodyRotationAxis.PITCH,
        FlybodyRotationAxis.ROLL,
    )
    YPR = YAW_PITCH_ROLL
    YAW_ROLL_PITCH = (
        FlybodyRotationAxis.YAW,
        FlybodyRotationAxis.ROLL,
        FlybodyRotationAxis.PITCH,
    )
    YRP = YAW_ROLL_PITCH

    DONTCARE = PITCH_ROLL_YAW


class WingFlybodyAxisOrder(BaseAxisOrder, Enum):
    """Axis order enum based on WingFlybodyRotationAxis."""

    @classmethod
    def _axis_enum_cls(cls):
        return WingFlybodyRotationAxis

    PITCH_ROLL_YAW = (
        WingFlybodyRotationAxis.PITCH,
        WingFlybodyRotationAxis.ROLL,
        WingFlybodyRotationAxis.YAW,
    )
    PRY = PITCH_ROLL_YAW
    PITCH_YAW_ROLL = (
        WingFlybodyRotationAxis.PITCH,
        WingFlybodyRotationAxis.YAW,
        WingFlybodyRotationAxis.ROLL,
    )
    PYR = PITCH_YAW_ROLL
    ROLL_PITCH_YAW = (
        WingFlybodyRotationAxis.ROLL,
        WingFlybodyRotationAxis.PITCH,
        WingFlybodyRotationAxis.YAW,
    )
    RPY = ROLL_PITCH_YAW
    ROLL_YAW_PITCH = (
        WingFlybodyRotationAxis.ROLL,
        WingFlybodyRotationAxis.YAW,
        WingFlybodyRotationAxis.PITCH,
    )
    RYP = ROLL_YAW_PITCH
    YAW_PITCH_ROLL = (
        WingFlybodyRotationAxis.YAW,
        WingFlybodyRotationAxis.PITCH,
        WingFlybodyRotationAxis.ROLL,
    )
    YPR = YAW_PITCH_ROLL
    YAW_ROLL_PITCH = (
        WingFlybodyRotationAxis.YAW,
        WingFlybodyRotationAxis.ROLL,
        WingFlybodyRotationAxis.PITCH,
    )
    YRP = YAW_ROLL_PITCH

    DONTCARE = PITCH_ROLL_YAW

FLYBODY_LEG_LINKS: list[str] = [
    "coxa",
    "trochanterfemur",
    "tibia",
    *(f"tarsus{seg}" for seg in "1234"),
    "claw",
]
FLYBODY_PROBOSCIS_LINKS: list[str] = ["rostrum", "haustellum"]
FLYBODY_ABDOMEN_LINKS: list[str] = [f"abdomen{seg}" for seg in "1234567"]
FLYBODY_PASSIVE_TARSAL_LINKS: list[str] = ["claw", *(f"tarsus{seg}" for seg in "234")]
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
        for edge in _chain2joints("c_thorax", *(f"{leg}_{lk}" for lk in FLYBODY_LEG_LINKS))
    ),
]
FLYBODY_PROBOSCIS_LINKS += ["labrum"]
FLYBODY_ALL_SEGMENT_NAMES: list[str] = orderedset(
    [seg for joint in FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS for seg in joint]
)


class FlybodyBodySegment(BodySegment):
    """Flybody-specific body segment class."""

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

    def is_claw(self) -> bool:
        """Return True if this segment is a claw."""
        return self.link == "claw"
    
class FlybodyJointDOF(JointDOF):
    """Joint DOF specific to the flybody model."""

    def from_name(cls, name: str) -> "FlybodyJointDOF":
        """Create a FlybodyJointDOF from a name of the form 'parent_child_axis'."""
        try:
            parent, child, axis = name.split("_")
            # check if child is wing
            bs_child = FlybodyBodySegment(child)
            return cls(
                parent=FlybodyBodySegment(parent),
                child=bs_child,
                axis=FlybodyRotationAxis(axis) if not bs_child.is_wing() else WingFlybodyRotationAxis(axis),
            )
        except ValueError:
            raise ValueError(
                f"Invalid joint DOF name: {name}. "
            )

class FlybodyAnatomicalJoint(AnatomicalJoint):
    """Anatomical joint specific to the flybody model."""
    
    def iter_dofs(self, axis_order: AxisOrder) -> Iterator[FlybodyJointDOF]:
        """Iterate through the DOFs of this joint in the specified axis order."""
        if self.child.is_wing():
            wing_axis_order = WingFlybodyAxisOrder(
                [axis.value for axis in axis_order.value]
            )
            for axis in wing_axis_order.value:
                if axis in self.axes:
                    yield FlybodyJointDOF(
                        parent=self.parent,
                        child=self.child,
                        axis=axis,
                    )
            return

        for axis in axis_order.value:
            if axis in self.axes:
                yield FlybodyJointDOF(
                    parent=self.parent,
                    child=self.child,
                    axis=axis,
                )

class FlybodyJointPreset(BaseJointPreset):
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
    def _get_all_possible_joints(cls) -> list[FlybodyAnatomicalJoint]:
        all_possible_joints = []
        for parent, child in cls._get_connected_segment_pairs():
            parent_bs = FlybodyBodySegment(parent)
            child_bs = FlybodyBodySegment(child)
            if child_bs.is_wing():
                axes = WingFlybodyAxesSet(WingFlybodyRotationAxis)
            else:
                axes = FlybodyAxesSet(FlybodyRotationAxis)
            all_possible_joints.append(
                FlybodyAnatomicalJoint(
                    parent=parent_bs,
                    child=child_bs,
                    axes=axes,
                )
            )
        return all_possible_joints
    
    @classmethod
    def _get_all_biological_joints(cls) -> list[FlybodyAnatomicalJoint]:
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


class FlybodyActuatedDOFPreset(BaseActuatedDOFPreset):
    """Presets for which flybody joint DoFs should be actuated."""

    ALL = "all"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_passive_tarsal_links(cls) -> list[str]:
        return FLYBODY_PASSIVE_TARSAL_LINKS


class FlybodyContactBodiesPreset(BaseContactBodiesPreset):
    """Presets for flybody body segments that should be able to collide with the ground."""

    ALL = "all"
    LEGS_THORAX_ABDOMEN_HEAD = "legs_thorax_abdomen_head"
    LEGS_ONLY = "legs_only"
    TIBIA_TARSUS_ONLY = "tibia_tarsus_only"

    @classmethod
    def _get_all_segments(cls):
        return [FlybodyBodySegment(segname) for segname in FLYBODY_ALL_SEGMENT_NAMES]


class FlybodySkeleton(Skeleton):
    """Skeleton specific to the flybody model."""

    def __init__(
        self,
        *,
        axis_order: FlybodyAxisOrder | WingFlybodyAxisOrder | AxisOrder | list[RotationAxis | FlybodyRotationAxis | WingFlybodyRotationAxis | str],
        joint_preset: "FlybodyJointPreset | str | None" = None,
        anatomical_joints: list[FlybodyAnatomicalJoint] | None = None,
    ) -> None:
        if not (joint_preset is None) ^ (anatomical_joints is None):
            raise ValueError(
                "Skeleton must be initiated from either joint_preset or "
                "anatomical_joints, but not both."
            )

        if joint_preset is not None:
            anatomical_joints = FlybodyJointPreset(joint_preset).to_joint_list()
        self.anatomical_joints = anatomical_joints

        self.joint_lookup = {(j.parent, j.child): j for j in anatomical_joints}
        self.body_segments = orderedset(
            [seg for nodes in self.joint_lookup.keys() for seg in nodes]
        )
        if isinstance(axis_order, AxisOrder):
            axis_order = axis_order.to_list_of_str()
        self.axis_order = FlybodyAxisOrder(axis_order)

    def iter_jointdofs(
        self,
        root: FlybodyBodySegment | str = "c_thorax",
    ) -> Iterator[FlybodyJointDOF]:
        """Iterate through joint DOFs in depth-first order starting from the root."""
        if isinstance(root, str):
            root = FlybodyBodySegment(root)
        tree = self.get_tree()
        for parent, child in tree.dfs_edges(root):
            anatomical_joint = self.joint_lookup[(parent, child)]
            for jointdof in anatomical_joint.iter_dofs(self.axis_order):
                yield jointdof

    def get_actuated_dofs_from_preset(
        self, preset: FlybodyActuatedDOFPreset | str
    ) -> list[FlybodyJointDOF]:
        """Given a flybody preset of actuated DoFs, return an explicit list of joints."""
        if isinstance(preset, BaseActuatedDOFPreset):
            preset = FlybodyActuatedDOFPreset(preset.value)
        else:
            preset = FlybodyActuatedDOFPreset(preset)
        return preset.filter(list(self.iter_jointdofs()))
