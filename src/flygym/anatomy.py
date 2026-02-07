from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, Iterator, Iterable
from collections.abc import Sequence, Collection

from flygym.utils.math import orderedset, Tree
from flygym.utils.exceptions import FlyGymInternalError


class RotationAxis(Enum):
    PITCH = "pitch"
    P = PITCH
    ROLL = "roll"
    R = ROLL
    YAW = "yaw"
    Y = YAW

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 1:
            if value.lower() == "p":
                return cls.PITCH
            elif value.lower() == "r":
                return cls.ROLL
            elif value.lower() == "y":
                return cls.YAW
        return super()._missing_(value)

    def to_vector(self, mirror: bool = False) -> tuple[float, float, float]:
        one = -1 if mirror else 1
        match self:
            case RotationAxis.PITCH:
                return (0, one, 0)
            case RotationAxis.ROLL:
                return (0, 0, one)
            case RotationAxis.YAW:
                return (one, 0, 0)


RotationAxisLike: TypeAlias = RotationAxis | str


class AxesSet(set[RotationAxis]):
    def __init__(self, iterable: Iterable = None, /):
        if iterable is None:
            super().__init__()
        else:
            super().__init__({RotationAxis(x) for x in iterable})

    def add(self, value, /):
        super().add(RotationAxis(value))

    def remove(self, value, /):
        super().remove(RotationAxis(value))


AxesSetLike: TypeAlias = AxesSet | Iterable[RotationAxisLike]


class AxisOrder(Enum):
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

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len((split_values := value.split("_"))) == 3:
            value = split_values
        if isinstance(value, Sequence) and len(value) == 3:
            try:
                return cls(tuple(RotationAxis(x) for x in value))
            except Exception as e:
                raise e
        return super()._missing_(value)


def _chain2joints(*args: str) -> list[tuple[str, str]]:
    return [(args[i], args[i + 1]) for i in range(len(args) - 1)]


SIDES = ["l", "r"]
LEGS = [f"{side}{pos}" for side in SIDES for pos in "fmh"]
BODY_POSITIONS = ["c", *SIDES, *LEGS]

LEG_LINKS = ["coxa", "trochanterfemur", "tibia", *(f"tarsus{seg}" for seg in "12345")]
ANTENNA_LINKS = ["pedicel", "funiculus", "arista"]
PROBOSCIS_LINKS = ["rostrum", "haustellum"]
ABDOMEN_LINKS = ["abdomen12", *(f"abdomen{seg}" for seg in "3456")]
PASSIVE_TARSAL_LINKS = [f"tarsus{seg}" for seg in "2345"]

ALL_CONNECTED_SEGMENT_PAIRS = [
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
ALL_SEGMENT_NAMES = orderedset(
    [seg for joint in ALL_CONNECTED_SEGMENT_PAIRS for seg in joint]
)


@dataclass(frozen=True)
class BodySegment:
    name: str

    def __post_init__(self):
        if self.name not in ALL_SEGMENT_NAMES:
            raise ValueError(
                f"Invalid body segment name: {self.name}. "
                f"Must be one of {ALL_SEGMENT_NAMES}."
            )

    @property
    def pos(self) -> str:
        return self.name.split("_")[0]

    @property
    def link(self) -> str:
        return self.name.split("_")[1]

    def is_thorax(self):
        return self.name == "c_thorax"

    def is_head(self):
        return self.name == "c_head"

    def is_proboscis(self):
        return self.link in PROBOSCIS_LINKS

    def is_eye(self):
        return self.link == "eye"

    def is_antenna(self):
        return self.link in ANTENNA_LINKS

    def is_wing(self):
        return self.link == "wing"

    def is_haltere(self):
        return self.link == "haltere"

    def is_leg(self):
        return self.pos in LEGS

    def is_abdomen(self):
        return self.link in ABDOMEN_LINKS


@dataclass(frozen=True)
class JointDOF:
    parent: BodySegment
    child: BodySegment
    axis: RotationAxis

    @property
    def name(self):
        return f"{self.parent.name}-{self.child.name}-{self.axis.value}"

    @classmethod
    def from_name(cls, name: str) -> "JointDOF":
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
    parent: BodySegment
    child: BodySegment
    # Rotation axes that exist at this joint. Defaults to all three.
    axes: AxesSet = field(default_factory=lambda: AxesSet(RotationAxis))

    def iter_dofs(
        self,
        axis_order: AxisOrder,
    ) -> Iterator[JointDOF]:
        for axis in axis_order.value:
            if axis in self.axes:
                yield JointDOF(self.parent, self.child, axis)


class JointPreset(Enum):
    ALL_POSSIBLE = "all_possible"
    ALL_BIOLOGICAL = "all_biological"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    def to_joint_list(self) -> list[AnatomicalJoint]:
        match self:
            case JointPreset.ALL_POSSIBLE:
                return self._get_all_possible_joints()
            case JointPreset.ALL_BIOLOGICAL:
                return self._get_all_biological_joints()
            case JointPreset.LEGS_ONLY:
                return self._get_leg_joints()
            case JointPreset.LEGS_ACTIVE_ONLY:
                return self._get_leg_active_joints()

    @staticmethod
    def _get_all_possible_joints() -> list[AnatomicalJoint]:
        return [
            AnatomicalJoint(
                BodySegment(parent), BodySegment(child), AxesSet(RotationAxis)
            )
            for parent, child in ALL_CONNECTED_SEGMENT_PAIRS
        ]

    @staticmethod
    def _get_all_biological_joints() -> list[AnatomicalJoint]:
        joints = JointPreset._get_all_possible_joints()
        for joint in joints:
            if joint.child.is_leg():
                match joint.child.link:
                    case "coxa":
                        # thorax-coxa has all 3 DoFs
                        pass
                    case "trochanterfemur":
                        # thorax-trochanter has pitch and roll
                        joint.axes.remove("yaw")
                    case _:
                        # the rest have only pitch
                        joint.axes.remove("roll")
                        joint.axes.remove("yaw")
        return joints

    @staticmethod
    def _get_leg_joints() -> list[AnatomicalJoint]:
        return [j for j in JointPreset._get_all_biological_joints() if j.child.is_leg()]

    @staticmethod
    def _get_leg_active_joints() -> list[AnatomicalJoint]:
        return [
            j
            for j in JointPreset._get_leg_joints()
            if j.child.link not in PASSIVE_TARSAL_LINKS
        ]


class ActuatedDOFPreset(Enum):
    ALL = "all"
    LEGS_ONLY = "legs_only"
    LEGS_ACTIVE_ONLY = "legs_active_only"

    def filter(self, jointdofs: Collection[JointDOF]) -> list[JointDOF]:
        match self:
            case ActuatedDOFPreset.ALL:
                return list(jointdofs)
            case ActuatedDOFPreset.LEGS_ONLY:
                return self._get_leg_only(jointdofs)
            case ActuatedDOFPreset.LEGS_ACTIVE_ONLY:
                return self._get_leg_active_only(jointdofs)

    def _get_leg_only(self, jointdofs: Collection[JointDOF]) -> list[JointDOF]:
        return [dof for dof in jointdofs if dof.child.is_leg()]

    def _get_leg_active_only(self, jointdofs: Collection[JointDOF]) -> list[JointDOF]:
        return [
            dof
            for dof in self._get_leg_only(jointdofs)
            if dof.child.link not in PASSIVE_TARSAL_LINKS
        ]


class ContactBodiesPreset(Enum):
    ALL = "all"
    LEGS_THORAX_ABDOMEN_HEAD = "legs_thorax_abdomen_head"
    LEGS_ONLY = "legs_only"
    TIBIA_TARSUS_ONLY = "tibia_tarsus_only"

    def to_body_segments_list(self) -> list[BodySegment]:
        match self:
            case ContactBodiesPreset.ALL:
                return ContactBodiesPreset._get_all_segments()
            case ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD:
                return ContactBodiesPreset._get_legs_thorax_abdomen_segments()
            case ContactBodiesPreset.LEGS_ONLY:
                return ContactBodiesPreset._get_leg_segments()
            case ContactBodiesPreset.TIBIA_TARSUS_ONLY:
                return ContactBodiesPreset._get_tibia_tarsus_segments()
            case _:
                raise FlyGymInternalError(
                    f"FlyGym internal error: unhandled ContactBodiesPreset {self}"
                )

    @staticmethod
    def _get_all_segments() -> list[BodySegment]:
        return [BodySegment(segname) for segname in ALL_SEGMENT_NAMES]

    @staticmethod
    def _get_legs_thorax_abdomen_segments() -> list[BodySegment]:
        return [
            seg
            for seg in ContactBodiesPreset._get_all_segments()
            if seg.is_leg() or seg.is_thorax() or seg.is_abdomen() or seg.is_head()
        ]

    @staticmethod
    def _get_leg_segments() -> list[BodySegment]:
        return [seg for seg in ContactBodiesPreset._get_all_segments() if seg.is_leg()]

    @staticmethod
    def _get_tibia_tarsus_segments() -> list[BodySegment]:
        return [
            seg
            for seg in ContactBodiesPreset._get_leg_segments()
            if seg.link == "tibia" or seg.link.startswith("tarsus")
        ]


class Skeleton:
    def __init__(
        self,
        *,
        axis_order: AxisOrder | Sequence[RotationAxis | str],
        joint_preset: "JointPreset | str | None" = None,
        anatomical_joints: list[AnatomicalJoint] | None = None,
    ) -> None:
        if int(joint_preset is None) + int(anatomical_joints is None) != 1:
            raise ValueError(
                "Skeleton must be initiated from either joint_preset or "
                "anatomical_joints, but not both."
            )

        if joint_preset is not None:
            anatomical_joints = JointPreset(joint_preset).to_joint_list()

        self.joint_lookup = {(j.parent, j.child): j for j in anatomical_joints}
        self.body_segments = orderedset(
            [seg for nodes in self.joint_lookup.keys() for seg in nodes]
        )
        self.axis_order = AxisOrder(axis_order)

    def get_tree(self) -> Tree:
        try:
            tree = Tree(nodes=self.body_segments, edges=self.joint_lookup.keys())
        except ValueError as e:
            raise ValueError("Skeleton is invalid - must be a tree.") from e
        return tree

    def iter_jointdofs(
        self,
        root: BodySegment | str = "c_thorax",
    ) -> Iterator[JointDOF]:
        if isinstance(root, str):
            root = BodySegment(root)
        tree = self.get_tree()
        for parent, child in tree.dfs_edges(root):
            anatomical_joint = self.joint_lookup[(parent, child)]
            for jointdof in anatomical_joint.iter_dofs(self.axis_order):
                yield jointdof

    def get_actuated_dofs_from_preset(
        self, preset: ActuatedDOFPreset | str
    ) -> list[JointDOF]:
        preset = ActuatedDOFPreset(preset)
        return preset.filter(list(self.iter_jointdofs()))
