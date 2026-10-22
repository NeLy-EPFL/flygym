"""Base anatomical definitions for the default fly model.

This module contains model-agnostic and default-model anatomy definitions.
FlyBody-specific anatomy is defined in
``flygym.flybody.anatomy_flybody``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, Iterator, Iterable

from flygym.utils.math import orderedset, Tree
from flygym.utils.exceptions import FlyGymInternalError

__all__ = [
    "BaseRotationAxis",
    "RotationAxis",
    "RotationAxisLike",
    "BaseAxisOrder",
    "AxesSet",
    "AxesSetLike",
    "AxisOrder",
    "JointPreset",
    "SeparateTrochanterFemurJointPreset",
    "BaseActuatedDOFPreset",
    "ActuatedDOFPreset",
    "ContactBodiesPreset",
    "SeparateTrochanterFemurContactBodiesPreset",
    "BaseContactBodiesPreset",
    "BodySegment",
    "JointDOF",
    "AnatomicalJoint",
    "BaseJointPreset",
    "Skeleton",
    "_chain2joints",
    "SIDES",
    "LEGS",
    "BODY_POSITIONS",
    "LEG_LINKS",
    "ANTENNA_LINKS",
    "PROBOSCIS_LINKS",
    "ABDOMEN_LINKS",
    "PASSIVE_TARSAL_LINKS",
    "ALL_CONNECTED_SEGMENT_PAIRS",
    "ALL_SEGMENT_NAMES",
    "LEG_LINKS_SEPARATE_TROCHANTERFEMUR",
    "ALL_CONNECTED_SEGMENT_PAIRS_SEPARATE_TROCHANTERFEMUR",
    "ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR",
    "separate_trochanterfemur_in_dof_name",
]


class BaseRotationAxis(Enum):
    """Base enum for axis naming and coordinate conversion.

    Child enums must define PITCH / ROLL / YAW members and implement
    `_vector_by_axis`.
    """

    @classmethod
    def _vector_by_axis(cls) -> dict[str, tuple[float, float, float]]:
        raise NotImplementedError

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 1:
            if value.lower() == "p":
                return cls.PITCH
            if value.lower() == "r":
                return cls.ROLL
            if value.lower() == "y":
                return cls.YAW
        return super()._missing_(value)

    def to_vector(self) -> tuple[float, float, float]:
        """Convert rotation axis to a 3D unit vector in XYZ order."""
        vector_by_axis = type(self)._vector_by_axis()
        return vector_by_axis[self.value]

    def to_letter_xyz(self) -> str:
        """Convert rotation axis to its corresponding letter ('x', 'y', or 'z')."""
        return {
            (1, 0, 0): "x",
            (0, 1, 0): "y",
            (0, 0, 1): "z",
        }[self.to_vector()]


class RotationAxis(BaseRotationAxis):
    """Default flygym axis convention.

    pitch -> y, roll -> z, yaw -> x.
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
            "roll": (0, 0, 1),
            "yaw": (1, 0, 0),
        }


RotationAxisLike: TypeAlias = RotationAxis | str


class AxesSet(set[RotationAxis]):
    """Set of rotation axes with automatic RotationAxis conversion. Useful for
    specifying which rotational DoFs are present at an anatomical joint."""

    rotation_axis_class = RotationAxis

    def __init__(self, iterable: Iterable = None, /):
        if iterable is None:
            super().__init__()
        else:
            super().__init__({self.rotation_axis_class(x) for x in iterable})

    def add(self, value, /):
        super().add(self.rotation_axis_class(value))

    def remove(self, value, /):
        super().remove(self.rotation_axis_class(value))


AxesSetLike: TypeAlias = AxesSet | Iterable[RotationAxisLike]


class BaseAxisOrder:
    """Base class for axis-order enums parametrized by an axis enum type."""

    @classmethod
    def _axis_enum_cls(cls):
        return RotationAxis

    @classmethod
    def _missing_(cls, value):
        axis_enum_cls = cls._axis_enum_cls()

        if isinstance(value, axis_enum_cls):
            value = [value]

        if isinstance(value, str) and len((split_values := value.split("_"))) == 3:
            value = split_values

        if isinstance(value, (list, tuple)) and len(value) == 3:
            normalized = [
                axis_enum_cls(x.value) if isinstance(x, Enum) else axis_enum_cls(x)
                for x in value
            ]
            return cls(tuple(normalized))

        return super()._missing_(value)

    def to_letters_xyz(self) -> str:
        """Convert axis order to a permutation of 'x', 'y', and 'z' (as one string)."""
        return "".join(axis.to_letter_xyz() for axis in self.value)

    def to_list_of_str(self) -> list[str]:
        """Convert to a list of axis name strings (e.g. ``['pitch', 'roll', 'yaw']``)."""
        return [axis.value for axis in self.value]

    def to_str(self) -> str:
        """Convert to an underscore-joined axis string (e.g. ``'pitch_roll_yaw'``)."""
        return "_".join(self.to_list_of_str())


class AxisOrder(BaseAxisOrder, Enum):
    """An enum specifying the order by which one-axis DoFs are chained together at
    anatomical joints with multiple DoFs.

    This is important because 3D rotations do not commute under Euler angle
    representations. Keep this consistent with your data (e.g., axis order used for
    inverse kinematics on experimental recordings).

    Special case: sometimes we might not care about the within-joint DoF order (e.g.
    when iterate over the skeleton to configure body segments but not joints). In this
    case, we can use `DONTCARE` (which aliases to `PITCH_ROLL_YAW`) to make our
    intention explicit.
    """

    @classmethod
    def _axis_enum_cls(cls):
        return RotationAxis

    PITCH_ROLL_YAW = (RotationAxis.PITCH, RotationAxis.ROLL, RotationAxis.YAW)
    PRY = PITCH_ROLL_YAW
    PITCH_YAW_ROLL = (RotationAxis.PITCH, RotationAxis.YAW, RotationAxis.ROLL)
    PYR = PITCH_YAW_ROLL
    ROLL_PITCH_YAW = (RotationAxis.ROLL, RotationAxis.PITCH, RotationAxis.YAW)
    RPY = ROLL_PITCH_YAW
    ROLL_YAW_PITCH = (RotationAxis.ROLL, RotationAxis.YAW, RotationAxis.PITCH)
    RYP = ROLL_YAW_PITCH
    YAW_PITCH_ROLL = (RotationAxis.YAW, RotationAxis.PITCH, RotationAxis.ROLL)
    YPR = YAW_PITCH_ROLL
    YAW_ROLL_PITCH = (RotationAxis.YAW, RotationAxis.ROLL, RotationAxis.PITCH)
    YRP = YAW_ROLL_PITCH

    DONTCARE = PITCH_ROLL_YAW


def _chain2joints(*args: str) -> list[tuple[str, str]]:
    """Helper function to convert a sequence of segment names into a list of connected
    parent-child pairs representing anatomical joints."""
    return [(args[i], args[i + 1]) for i in range(len(args) - 1)]


SIDES: list[str] = ["l", "r"]
LEGS: list[str] = [f"{side}{pos}" for side in SIDES for pos in "fmh"]
BODY_POSITIONS: list[str] = ["c", *SIDES, *LEGS]

LEG_LINKS: list[str] = [
    "coxa",
    "trochanterfemur",
    "tibia",
    *(f"tarsus{seg}" for seg in "12345"),
]
ANTENNA_LINKS: list[str] = ["pedicel", "funiculus", "arista"]
PROBOSCIS_LINKS: list[str] = ["rostrum", "haustellum"]
ABDOMEN_LINKS: list[str] = ["abdomen12", *(f"abdomen{seg}" for seg in "3456")]
PASSIVE_TARSAL_LINKS: list[str] = [f"tarsus{seg}" for seg in "2345"]

ALL_CONNECTED_SEGMENT_PAIRS: list[tuple[str, str]] = [
    ("c_thorax", "c_head"),
    *(_chain2joints("c_head", *(f"c_{lk}" for lk in PROBOSCIS_LINKS))),
    *(_chain2joints("c_thorax", *(f"c_{lk}" for lk in ABDOMEN_LINKS))),
    *(("c_head", f"{s}_eye") for s in SIDES),
    *(
        edge
        for s in SIDES
        for edge in _chain2joints("c_head", *(f"{s}_{lk}" for lk in ANTENNA_LINKS))
    ),
    *(("c_thorax", f"{s}_wing") for s in SIDES),
    *(("c_thorax", f"{s}_haltere") for s in SIDES),
    *(
        edge
        for leg in LEGS
        for edge in _chain2joints("c_thorax", *(f"{leg}_{lk}" for lk in LEG_LINKS))
    ),
]
ALL_SEGMENT_NAMES: list[str] = orderedset(
    [seg for joint in ALL_CONNECTED_SEGMENT_PAIRS for seg in joint]
)

# Variant of the default anatomy where the trochanter and the femur are separate
# segments (used by `NeuroMechFlyWithSeparateTrochanterFemur`).
LEG_LINKS_SEPARATE_TROCHANTERFEMUR: list[str] = [
    "coxa",
    "trochanter",
    "femur",
    "tibia",
    *(f"tarsus{seg}" for seg in "12345"),
]
ALL_CONNECTED_SEGMENT_PAIRS_SEPARATE_TROCHANTERFEMUR: list[tuple[str, str]] = [
    *(
        pair
        for pair in ALL_CONNECTED_SEGMENT_PAIRS
        if pair[1].split("_")[0] not in LEGS
    ),
    *(
        edge
        for leg in LEGS
        for edge in _chain2joints(
            "c_thorax", *(f"{leg}_{lk}" for lk in LEG_LINKS_SEPARATE_TROCHANTERFEMUR)
        )
    ),
]
ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR: list[str] = orderedset(
    [
        seg
        for joint in ALL_CONNECTED_SEGMENT_PAIRS_SEPARATE_TROCHANTERFEMUR
        for seg in joint
    ]
)
_VALID_SEGMENT_NAMES = set(ALL_SEGMENT_NAMES) | set(
    ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR
)


def separate_trochanterfemur_in_dof_name(name: str) -> str:
    """Rename a joint DoF of the fused trochanterfemur to its separated equivalent.

    The joint to the coxa now connects to the trochanter, and the joint to the tibia
    now connects to the femur. Since there is no DoF between the trochanter and the
    femur by default, joint angles carry over unchanged. Names that do not involve the
    trochanterfemur are returned as is.

    Example: `"lf_trochanterfemur-lf_tibia-pitch"` becomes `"lf_femur-lf_tibia-pitch"`.
    """
    parent, child, axis = name.split("-")
    if child.endswith("_trochanterfemur"):
        child = child.replace("_trochanterfemur", "_trochanter")
    if parent.endswith("_trochanterfemur"):
        parent = parent.replace("_trochanterfemur", "_femur")
    return f"{parent}-{child}-{axis}"


@dataclass(frozen=True)
class BodySegment:
    """Represents a body segment in the fly anatomy.

    See `flygym.anatomy.ALL_SEGMENT_NAMES` and
    `flygym.anatomy.ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR` for all possible names.

    Attributes:
        name:
            Unique identifier for the body segment following the pattern `{pos}_{link}`.
        pos:
            Body location (e.g., `c` for center segments like `c_thorax`, `lf` for left
            front leg, and `l` for left non-leg segments like `l_eye`).
        link:
            Name of the segment in the kinematic chain (e.g., `tibia`).
    """

    name: str

    def __post_init__(self):
        if self.name not in _VALID_SEGMENT_NAMES:
            raise ValueError(
                f"Invalid body segment name: {self.name}. "
                f"Must be one of {sorted(_VALID_SEGMENT_NAMES)}."
            )

    @property
    def pos(self) -> str:
        """Body position prefix (e.g. ``'lf'``, ``'c'``)."""
        return self.name.split("_")[0]

    @property
    def link(self) -> str:
        """Link name within the kinematic chain (e.g. ``'tibia'``)."""
        return self.name.split("_")[1]

    def is_thorax(self) -> bool:
        """Return True if this segment is the thorax."""
        return self.name == "c_thorax"

    def is_head(self) -> bool:
        """Return True if this segment is the head."""
        return self.name == "c_head"

    def is_proboscis(self) -> bool:
        """Return True if this segment belongs to the proboscis."""
        return self.link in PROBOSCIS_LINKS

    def is_eye(self) -> bool:
        """Return True if this segment is an eye."""
        return self.link == "eye"

    def is_antenna(self) -> bool:
        """Return True if this segment belongs to an antenna."""
        return self.link in ANTENNA_LINKS

    def is_wing(self) -> bool:
        """Return True if this segment is a wing."""
        return self.link == "wing"

    def is_haltere(self) -> bool:
        """Return True if this segment is a haltere."""
        return self.link == "haltere"

    def is_leg(self) -> bool:
        """Return True if this segment belongs to a leg."""
        return self.pos in LEGS

    def is_abdomen(self) -> bool:
        """Return True if this segment belongs to the abdomen."""
        return self.link in ABDOMEN_LINKS

    def is_claw(self) -> bool:
        """Return True if this segment is tarsus5."""
        return self.link == "tarsus5"


@dataclass(frozen=True)
class JointDOF:
    """A single rotational degree of freedom in an anatomical joint."""

    parent: BodySegment
    child: BodySegment
    axis: RotationAxis

    def __post_init__(self):
        if isinstance(self.parent, str):
            self.parent = BodySegment(self.parent)
        if isinstance(self.child, str):
            self.child = BodySegment(self.child)
        if isinstance(self.axis, str):
            self.axis = RotationAxis(self.axis)

    @property
    def name(self) -> str:
        """Unique name following the pattern ``{parent}-{child}-{axis}``."""
        return f"{self.parent.name}-{self.child.name}-{self.axis.value}"

    @classmethod
    def from_name(cls, name: str) -> "JointDOF":
        """Create a JointDOF instance by parsing a name string."""
        try:
            parent_name, child_name, axis_name = name.split("-")
            return cls(
                BodySegment(parent_name),
                BodySegment(child_name),
                RotationAxis(axis_name),
            )
        except Exception as e:
            raise ValueError(f"Invalid JointDOF name: {name}") from e


@dataclass
class AnatomicalJoint:
    """Represents an anatomical joint connecting two body segments."""

    parent: BodySegment
    child: BodySegment
    axes: AxesSet = field(default_factory=lambda: AxesSet(RotationAxis))

    def __post_init__(self):
        if isinstance(self.parent, str):
            self.parent = BodySegment(self.parent)
        if isinstance(self.child, str):
            self.child = BodySegment(self.child)
        if not isinstance(self.axes, AxesSet):
            self.axes = AxesSet(self.axes)

    def iter_dofs(self, axis_order: AxisOrder) -> Iterator[JointDOF]:
        """Iterate through the `JointDOF`s in the specified axis order."""
        for axis in axis_order.value:
            if axis in self.axes:
                yield JointDOF(self.parent, self.child, axis)

    @property
    def name(self) -> str:
        """Unique name for the anatomical joint following the pattern
        `{parent}-{child}`."""
        return f"{self.parent.name}-{self.child.name}"

    def __hash__(self):
        return hash((self.parent, self.child))


class BaseJointPreset(Enum):
    """Base class containing all joint preset logic."""

    @classmethod
    def _get_connected_segment_pairs(cls) -> list[tuple[str, str]]:
        raise NotImplementedError

    @classmethod
    def _get_passive_tarsal_links(cls) -> list[str]:
        raise NotImplementedError

    def to_joint_list(self) -> list[AnatomicalJoint]:
        match self.value:
            case "all_possible":
                return self._get_all_possible_joints()
            case "all_biological":
                return self._get_all_biological_joints()
            case "legs_only":
                return self._get_leg_joints()
            case "legs_active_only":
                return self._get_leg_active_joints()

    @classmethod
    def _get_all_possible_joints(cls) -> list[AnatomicalJoint]:
        return [
            AnatomicalJoint(
                BodySegment(parent), BodySegment(child), AxesSet(RotationAxis)
            )
            for parent, child in cls._get_connected_segment_pairs()
        ]

    @classmethod
    def _get_all_biological_joints(cls) -> list[AnatomicalJoint]:
        joints = cls._get_all_possible_joints()
        for joint in joints:
            if joint.child.is_leg():
                match joint.child.link:
                    case "coxa":
                        pass
                    case "trochanterfemur" | "trochanter":
                        joint.axes.remove("yaw")
                    case "femur":
                        # Trochanter and femur are often considered to be fused in real
                        # flies (at list in NMFv1, Lobato-Rios et al. 2022)
                        joint.axes.clear()
                    case _:
                        joint.axes.remove("roll")
                        joint.axes.remove("yaw")
        return joints

    @classmethod
    def _get_leg_joints(cls) -> list[AnatomicalJoint]:
        return [j for j in cls._get_all_biological_joints() if j.child.is_leg()]

    @classmethod
    def _get_leg_active_joints(cls) -> list[AnatomicalJoint]:
        return [
            j
            for j in cls._get_leg_joints()
            if j.child.link not in cls._get_passive_tarsal_links()
        ]


class JointPreset(BaseJointPreset):
    """Presets for which rotational DoFs are present at which anatomical joints."""

    ALL_POSSIBLE = "all_possible"
    ALL_BIOLOGICAL = "all_biological"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_connected_segment_pairs(cls):
        return ALL_CONNECTED_SEGMENT_PAIRS

    @classmethod
    def _get_passive_tarsal_links(cls):
        return PASSIVE_TARSAL_LINKS


class SeparateTrochanterFemurJointPreset(BaseJointPreset):
    """Same as `JointPreset`, but for the anatomy with separate trochanter and femur
    segments (see `NeuroMechFlyWithSeparateTrochanterFemur`).

    Biological presets add no DoF between the trochanter and the femur.
    """

    ALL_POSSIBLE = "all_possible"
    ALL_BIOLOGICAL = "all_biological"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_connected_segment_pairs(cls):
        return ALL_CONNECTED_SEGMENT_PAIRS_SEPARATE_TROCHANTERFEMUR

    @classmethod
    def _get_passive_tarsal_links(cls):
        return PASSIVE_TARSAL_LINKS


class BaseActuatedDOFPreset(Enum):
    """Base class for presets that select which joint DoFs should be actuated."""

    @classmethod
    def _get_passive_tarsal_links(cls) -> list[str]:
        raise NotImplementedError

    def filter(self, jointdofs: list[JointDOF]) -> list[JointDOF]:
        """Filter given joint DoFs according to the preset."""
        if self == self.__class__.ALL:
            return list(jointdofs)
        if self == self.__class__.LEGS_ONLY:
            return self._get_leg_only(jointdofs)
        if self == self.__class__.LEGS_ACTIVE_ONLY:
            return self._get_leg_active_only(jointdofs)
        raise ValueError(f"Unhandled actuated DoF preset: {self}")

    def _get_leg_only(self, jointdofs: list[JointDOF]) -> list[JointDOF]:
        return [dof for dof in jointdofs if dof.child.is_leg()]

    def _get_leg_active_only(self, jointdofs: list[JointDOF]) -> list[JointDOF]:
        return [
            dof
            for dof in self._get_leg_only(jointdofs)
            if dof.child.link not in self._get_passive_tarsal_links()
        ]


class ActuatedDOFPreset(BaseActuatedDOFPreset):
    """Presets for which joint DoFs present in a skeleton should be actuated."""

    ALL = "all"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    @classmethod
    def _get_passive_tarsal_links(cls) -> list[str]:
        return PASSIVE_TARSAL_LINKS


class BaseContactBodiesPreset(Enum):
    """Base class for body-segment contact presets."""

    @classmethod
    def _get_all_segments(cls) -> list[BodySegment]:
        raise NotImplementedError

    @classmethod
    def _get_legs_thorax_abdomen_segments(cls) -> list[BodySegment]:
        return [
            seg
            for seg in cls._get_all_segments()
            if seg.is_leg() or seg.is_thorax() or seg.is_abdomen() or seg.is_head()
        ]

    @classmethod
    def _get_leg_segments(cls) -> list[BodySegment]:
        return [seg for seg in cls._get_all_segments() if seg.is_leg()]

    @classmethod
    def _get_tibia_tarsus_segments(cls) -> list[BodySegment]:
        return [
            seg
            for seg in cls._get_leg_segments()
            if seg.link == "tibia" or seg.link.startswith("tarsus")
        ]

    def to_body_segments_list(self) -> list[BodySegment]:
        """Return the list of `BodySegment` objects defined by this preset."""
        match self.value:
            case "all":
                return type(self)._get_all_segments()
            case "legs_thorax_abdomen_head":
                return type(self)._get_legs_thorax_abdomen_segments()
            case "legs_only":
                return type(self)._get_leg_segments()
            case "tibia_tarsus_only":
                return type(self)._get_tibia_tarsus_segments()
            case _:
                raise FlyGymInternalError(
                    f"FlyGym internal error: unhandled ContactBodiesPreset {self}"
                )


class ContactBodiesPreset(BaseContactBodiesPreset):
    """Presets for which body segments should be able to collide with the ground."""

    ALL = "all"
    LEGS_THORAX_ABDOMEN_HEAD = "legs_thorax_abdomen_head"
    LEGS_ONLY = "legs_only"
    TIBIA_TARSUS_ONLY = "tibia_tarsus_only"

    @classmethod
    def _get_all_segments(cls) -> list[BodySegment]:
        return [BodySegment(segname) for segname in ALL_SEGMENT_NAMES]


class SeparateTrochanterFemurContactBodiesPreset(BaseContactBodiesPreset):
    """Same as `ContactBodiesPreset`, but for the anatomy with separate trochanter and
    femur segments (see `NeuroMechFlyWithSeparateTrochanterFemur`)."""

    ALL = "all"
    LEGS_THORAX_ABDOMEN_HEAD = "legs_thorax_abdomen_head"
    LEGS_ONLY = "legs_only"
    TIBIA_TARSUS_ONLY = "tibia_tarsus_only"

    @classmethod
    def _get_all_segments(cls) -> list[BodySegment]:
        return [
            BodySegment(segname)
            for segname in ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR
        ]


class Skeleton:
    """Fly skeleton defining joint structure and degrees of freedom.

    Args:
        axis_order: Order in which DoFs are chained at multi-DoF joints.
        joint_preset: Preset defining the anatomical joints. A string is interpreted
            as a `JointPreset`. For other anatomies, pass a member of the matching
            preset enum (e.g. `SeparateTrochanterFemurJointPreset.LEGS_ONLY`).
        anatomical_joints: Explicit list of anatomical joints, as an alternative to
            `joint_preset`.
    """

    def __init__(
        self,
        *,
        axis_order: AxisOrder | list[RotationAxis | str],
        joint_preset: "BaseJointPreset | str | None" = None,
        anatomical_joints: list[AnatomicalJoint] | None = None,
    ) -> None:
        if not (joint_preset is None) ^ (anatomical_joints is None):
            raise ValueError(
                "Skeleton must be initiated from either joint_preset or "
                "anatomical_joints, but not both."
            )

        if joint_preset is not None:
            if not isinstance(joint_preset, BaseJointPreset):
                joint_preset = JointPreset(joint_preset)
            anatomical_joints = joint_preset.to_joint_list()
        self.anatomical_joints = anatomical_joints

        self.joint_lookup = {(j.parent, j.child): j for j in anatomical_joints}
        self.body_segments = orderedset(
            [seg for nodes in self.joint_lookup.keys() for seg in nodes]
        )
        self.axis_order = AxisOrder(axis_order)

    def get_tree(self) -> Tree:
        """Construct a tree data structure representing the skeleton."""
        try:
            tree = Tree(nodes=self.body_segments, edges=self.joint_lookup.keys())
        except ValueError as e:
            raise ValueError("Skeleton is invalid - must be a tree.") from e
        return tree

    def iter_jointdofs(
        self,
        root: BodySegment | str = "c_thorax",
    ) -> Iterator[JointDOF]:
        """Iterate through joint DOFs in depth-first order starting from the root."""
        if isinstance(root, str):
            root = BodySegment(root)
        tree = self.get_tree()
        for parent, child in tree.dfs_edges(root):
            anatomical_joint = self.joint_lookup[(parent, child)]
            for jointdof in anatomical_joint.iter_dofs(self.axis_order):
                yield jointdof

    def get_actuated_dofs_from_preset(
        self, preset: BaseActuatedDOFPreset | str
    ) -> list[JointDOF]:
        """Given a preset of actuated DoFs, return an explicit list of `JointDOF`."""
        if isinstance(preset, BaseActuatedDOFPreset):
            preset = ActuatedDOFPreset(preset.value)
        else:
            preset = ActuatedDOFPreset(preset)
        return preset.filter(list(self.iter_jointdofs()))
