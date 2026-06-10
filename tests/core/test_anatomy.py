"""Unit tests for flygym.anatomy."""

import pytest

from flygym.anatomy import (
    RotationAxis,
    AxesSet,
    AxisOrder,
    BodySegment,
    JointDOF,
    AnatomicalJoint,
    JointPreset,
    ActuatedDOFPreset,
    ContactBodiesPreset,
    Skeleton,
    SIDES,
    LEGS,
    BODY_POSITIONS,
    LEG_LINKS,
    ANTENNA_LINKS,
    PROBOSCIS_LINKS,
    ABDOMEN_LINKS,
    PASSIVE_TARSAL_LINKS,
    ALL_CONNECTED_SEGMENT_PAIRS,
    ALL_SEGMENT_NAMES,
)
from flygym.assets.model.flybody.anatomy_flybody import (
    FlybodyRotationAxis,
    WingFlybodyRotationAxis,
    FlybodyAxesSet,
    WingFlybodyAxesSet,
    FlybodyAxisOrder,
    WingFlybodyAxisOrder,
    FlybodyBodySegment,
    FlybodyJointDOF,
    FlybodyAnatomicalJoint,
    FlybodyJointPreset,
    FlybodyActuatedDOFPreset,
    FlybodyContactBodiesPreset,
    FlybodySkeleton,
    FLYBODY_ALL_SEGMENT_NAMES,
    FLYBODY_LEG_LINKS,
    FLYBODY_PASSIVE_TARSAL_LINKS,
    FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS,
)


# ==============================================================================
# Constants
# ==============================================================================


class TestConstants:
    def test_sides(self):
        assert set(SIDES) == {"l", "r"}

    def test_legs(self):
        assert set(LEGS) == {"lf", "lm", "lh", "rf", "rm", "rh"}

    def test_leg_links(self):
        # coxa, trochanterfemur, tibia, tarsus1-5 = 8 segments
        assert LEG_LINKS[0] == "coxa"
        assert "tibia" in LEG_LINKS
        assert len([lk for lk in LEG_LINKS if lk.startswith("tarsus")]) == 5

    def test_passive_tarsal_links(self):
        # tarsus2-5 are passive
        assert len(PASSIVE_TARSAL_LINKS) == 4
        for lk in PASSIVE_TARSAL_LINKS:
            assert lk.startswith("tarsus")
        assert "tarsus1" not in PASSIVE_TARSAL_LINKS

    def test_all_segment_names_unique(self):
        assert len(ALL_SEGMENT_NAMES) == len(set(ALL_SEGMENT_NAMES))

    def test_all_segment_names_contains_thorax(self):
        assert "c_thorax" in ALL_SEGMENT_NAMES

    def test_all_connected_segment_pairs_are_in_all_segments(self):
        seg_set = set(ALL_SEGMENT_NAMES)
        for parent, child in ALL_CONNECTED_SEGMENT_PAIRS:
            assert parent in seg_set, f"{parent} not in ALL_SEGMENT_NAMES"
            assert child in seg_set, f"{child} not in ALL_SEGMENT_NAMES"

    def test_body_positions_contains_center_and_sides_and_legs(self):
        assert "c" in BODY_POSITIONS
        assert "l" in BODY_POSITIONS
        assert "r" in BODY_POSITIONS
        for leg in LEGS:
            assert leg in BODY_POSITIONS


# ==============================================================================
# AxesSet
# ==============================================================================


class TestAxesSet:
    def test_construction_from_strings(self):
        axes = AxesSet(["pitch", "roll", "yaw"])
        assert RotationAxis.PITCH in axes
        assert RotationAxis.ROLL in axes
        assert RotationAxis.YAW in axes

    def test_construction_from_enum(self):
        axes = AxesSet([RotationAxis.PITCH, RotationAxis.YAW])
        assert RotationAxis.PITCH in axes
        assert RotationAxis.YAW in axes
        assert RotationAxis.ROLL not in axes

    def test_empty_construction(self):
        axes = AxesSet()
        assert len(axes) == 0

    def test_add_string(self):
        axes = AxesSet()
        axes.add("pitch")
        assert RotationAxis.PITCH in axes

    def test_remove_string(self):
        axes = AxesSet(["pitch", "roll"])
        axes.remove("pitch")
        assert RotationAxis.PITCH not in axes
        assert RotationAxis.ROLL in axes

    def test_construction_from_all_enum_values(self):
        axes = AxesSet(RotationAxis)
        assert len(axes) == 3


# ==============================================================================
# AxisOrder
# ==============================================================================


class TestAxisOrder:
    def test_aliases(self):
        assert AxisOrder.YPR is AxisOrder.YAW_PITCH_ROLL
        assert AxisOrder.PRY is AxisOrder.PITCH_ROLL_YAW
        assert AxisOrder.DONTCARE is AxisOrder.PITCH_ROLL_YAW

    def test_missing_from_string(self):
        ao = AxisOrder("yaw_pitch_roll")
        assert ao is AxisOrder.YAW_PITCH_ROLL

    def test_missing_from_sequence(self):
        ao = AxisOrder(["yaw", "pitch", "roll"])
        assert ao is AxisOrder.YAW_PITCH_ROLL

    def test_missing_from_rotation_axis_sequence(self):
        ao = AxisOrder((RotationAxis.YAW, RotationAxis.PITCH, RotationAxis.ROLL))
        assert ao is AxisOrder.YAW_PITCH_ROLL

    def test_to_letters_xyz(self):
        # YAW=x, PITCH=y, ROLL=z
        assert AxisOrder.YAW_PITCH_ROLL.to_letters_xyz() == "xyz"
        assert AxisOrder.PITCH_ROLL_YAW.to_letters_xyz() == "yzx"
        assert AxisOrder.ROLL_PITCH_YAW.to_letters_xyz() == "zyx"

    def test_all_six_permutations_are_distinct(self):
        all_orders = [
            AxisOrder.PITCH_ROLL_YAW,
            AxisOrder.PITCH_YAW_ROLL,
            AxisOrder.ROLL_PITCH_YAW,
            AxisOrder.ROLL_YAW_PITCH,
            AxisOrder.YAW_PITCH_ROLL,
            AxisOrder.YAW_ROLL_PITCH,
        ]
        assert len(set(ao.to_letters_xyz() for ao in all_orders)) == 6


# ==============================================================================
# BodySegment
# ==============================================================================


class TestBodySegment:
    def test_valid_segment(self):
        seg = BodySegment("c_thorax")
        assert seg.name == "c_thorax"
        assert seg.pos == "c"
        assert seg.link == "thorax"

    def test_left_front_coxa(self):
        seg = BodySegment("lf_coxa")
        assert seg.pos == "lf"
        assert seg.link == "coxa"
        assert seg.is_leg()
        assert not seg.is_thorax()
        assert not seg.is_head()

    def test_thorax_predicates(self):
        seg = BodySegment("c_thorax")
        assert seg.is_thorax()
        assert not seg.is_leg()
        assert not seg.is_head()

    def test_head_predicates(self):
        seg = BodySegment("c_head")
        assert seg.is_head()
        assert not seg.is_thorax()
        assert not seg.is_leg()

    def test_wing_predicates(self):
        seg = BodySegment("l_wing")
        assert seg.is_wing()
        assert not seg.is_leg()

    def test_abdomen_predicates(self):
        seg = BodySegment("c_abdomen12")
        assert seg.is_abdomen()

    def test_antenna_predicates(self):
        seg = BodySegment("l_pedicel")
        assert seg.is_antenna()

    def test_proboscis_predicates(self):
        seg = BodySegment("c_rostrum")
        assert seg.is_proboscis()

    def test_eye_predicates(self):
        seg = BodySegment("l_eye")
        assert seg.is_eye()

    def test_all_legs_classified_as_leg(self):
        for leg in LEGS:
            seg = BodySegment(f"{leg}_coxa")
            assert seg.is_leg(), f"{leg}_coxa should be a leg segment"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid body segment name"):
            BodySegment("invalid_segment")

    def test_frozen_dataclass(self):
        seg = BodySegment("c_thorax")
        with pytest.raises((AttributeError, TypeError)):
            seg.name = "something_else"


# ==============================================================================
# JointDOF
# ==============================================================================


class TestJointDOF:
    def test_name_property(self):
        parent = BodySegment("c_thorax")
        child = BodySegment("lf_coxa")
        dof = JointDOF(parent, child, RotationAxis.YAW)
        assert dof.name == "c_thorax-lf_coxa-yaw"

    def test_from_name_roundtrip(self):
        name = "c_thorax-lf_coxa-pitch"
        dof = JointDOF.from_name(name)
        assert dof.parent.name == "c_thorax"
        assert dof.child.name == "lf_coxa"
        assert dof.axis is RotationAxis.PITCH
        assert dof.name == name

    def test_from_name_all_axes(self):
        for axis_str in ("yaw", "pitch", "roll"):
            dof = JointDOF.from_name(f"c_thorax-lf_coxa-{axis_str}")
            assert dof.axis.value == axis_str

    def test_from_name_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid JointDOF name"):
            JointDOF.from_name("not-a-valid-joint-dof-name-xyz")

    def test_frozen_dataclass(self):
        dof = JointDOF.from_name("c_thorax-lf_coxa-pitch")
        with pytest.raises((AttributeError, TypeError)):
            dof.axis = RotationAxis.YAW


# ==============================================================================
# AnatomicalJoint
# ==============================================================================


class TestAnatomicalJoint:
    def test_iter_dofs_all_three(self):
        joint = AnatomicalJoint(
            BodySegment("c_thorax"),
            BodySegment("lf_coxa"),
            AxesSet([RotationAxis.PITCH, RotationAxis.ROLL, RotationAxis.YAW]),
        )
        dofs = list(joint.iter_dofs(AxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 3
        # check order matches YAW_PITCH_ROLL
        assert dofs[0].axis is RotationAxis.YAW
        assert dofs[1].axis is RotationAxis.PITCH
        assert dofs[2].axis is RotationAxis.ROLL

    def test_iter_dofs_subset(self):
        joint = AnatomicalJoint(
            BodySegment("c_thorax"),
            BodySegment("lf_coxa"),
            AxesSet([RotationAxis.PITCH]),
        )
        dofs = list(joint.iter_dofs(AxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 1
        assert dofs[0].axis is RotationAxis.PITCH

    def test_iter_dofs_empty(self):
        joint = AnatomicalJoint(
            BodySegment("c_thorax"),
            BodySegment("lf_coxa"),
            AxesSet(),
        )
        dofs = list(joint.iter_dofs(AxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 0


# ==============================================================================
# JointPreset
# ==============================================================================


class TestJointPreset:
    def test_all_possible_returns_joints_for_all_pairs(self):
        joints = JointPreset.ALL_POSSIBLE.to_joint_list()
        assert len(joints) == len(ALL_CONNECTED_SEGMENT_PAIRS)
        for joint in joints:
            assert len(joint.axes) == 3

    def test_all_biological_has_fewer_dofs_than_all_possible(self):
        all_possible = JointPreset.ALL_POSSIBLE.to_joint_list()
        biological = JointPreset.ALL_BIOLOGICAL.to_joint_list()
        assert len(biological) == len(all_possible)  # same number of joints...
        total_possible_dofs = sum(len(j.axes) for j in all_possible)
        total_biological_dofs = sum(len(j.axes) for j in biological)
        assert total_biological_dofs < total_possible_dofs  # ...but fewer DoFs

    def test_legs_only_excludes_non_leg_joints(self):
        joints = JointPreset.LEGS_ONLY.to_joint_list()
        for joint in joints:
            assert joint.child.is_leg(), f"{joint.child.name} should be a leg segment"

    def test_legs_active_only_excludes_passive_tarsal(self):
        joints = JointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        for joint in joints:
            assert joint.child.link not in PASSIVE_TARSAL_LINKS, (
                f"{joint.child.link} should not be in legs_active_only"
            )

    def test_legs_active_only_subset_of_legs_only(self):
        legs_only_children = {
            j.child.name for j in JointPreset.LEGS_ONLY.to_joint_list()
        }
        legs_active_children = {
            j.child.name for j in JointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        }
        assert legs_active_children.issubset(legs_only_children)
        assert legs_only_children != legs_active_children  # passive tarsals removed

    def test_from_string(self):
        assert JointPreset("legs_only") is JointPreset.LEGS_ONLY


# ==============================================================================
# ActuatedDOFPreset
# ==============================================================================


class TestActuatedDOFPreset:
    @pytest.fixture
    def all_bio_dofs(self):
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.ALL_BIOLOGICAL,
        )
        return list(skeleton.iter_jointdofs())

    def test_all_returns_everything(self, all_bio_dofs):
        filtered = ActuatedDOFPreset.ALL.filter(all_bio_dofs)
        assert len(filtered) == len(all_bio_dofs)

    def test_legs_only_returns_only_leg_dofs(self, all_bio_dofs):
        filtered = ActuatedDOFPreset.LEGS_ONLY.filter(all_bio_dofs)
        for dof in filtered:
            assert dof.child.is_leg()

    def test_legs_active_only_excludes_passive_tarsals(self, all_bio_dofs):
        filtered = ActuatedDOFPreset.LEGS_ACTIVE_ONLY.filter(all_bio_dofs)
        for dof in filtered:
            assert dof.child.link not in PASSIVE_TARSAL_LINKS

    def test_from_string(self):
        assert ActuatedDOFPreset("legs_only") is ActuatedDOFPreset.LEGS_ONLY


# ==============================================================================
# ContactBodiesPreset
# ==============================================================================


class TestContactBodiesPreset:
    def test_all_returns_all_segments(self):
        segs = ContactBodiesPreset.ALL.to_body_segments_list()
        assert len(segs) == len(ALL_SEGMENT_NAMES)

    def test_legs_only_returns_only_leg_segs(self):
        segs = ContactBodiesPreset.LEGS_ONLY.to_body_segments_list()
        for seg in segs:
            assert seg.is_leg()

    def test_tibia_tarsus_only_is_subset_of_legs_only(self):
        tibia_tarsus = {
            s.name
            for s in ContactBodiesPreset.TIBIA_TARSUS_ONLY.to_body_segments_list()
        }
        legs_only = {
            s.name for s in ContactBodiesPreset.LEGS_ONLY.to_body_segments_list()
        }
        assert tibia_tarsus.issubset(legs_only)

    def test_tibia_tarsus_only_contains_tibia_and_tarsus(self):
        segs = ContactBodiesPreset.TIBIA_TARSUS_ONLY.to_body_segments_list()
        for seg in segs:
            assert seg.link == "tibia" or seg.link.startswith("tarsus")

    def test_legs_thorax_abdomen_head_contains_thorax(self):
        segs = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        seg_names = {s.name for s in segs}
        assert "c_thorax" in seg_names

    def test_from_string(self):
        assert ContactBodiesPreset("legs_only") is ContactBodiesPreset.LEGS_ONLY


class TestFlybodyContactBodiesPreset:
    def test_all_returns_all_segments(self):
        segs = FlybodyContactBodiesPreset.ALL.to_body_segments_list()
        assert len(segs) == len(FLYBODY_ALL_SEGMENT_NAMES)

    def test_legs_only_returns_only_leg_segs(self):
        segs = FlybodyContactBodiesPreset.LEGS_ONLY.to_body_segments_list()
        for seg in segs:
            assert seg.is_leg()

    def test_from_string(self):
        assert (
            FlybodyContactBodiesPreset("legs_only")
            is FlybodyContactBodiesPreset.LEGS_ONLY
        )


# ==============================================================================
# Skeleton
# ==============================================================================


class TestSkeleton:
    def test_construction_from_preset(self):
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        assert skel.axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_construction_from_anatomical_joints(self):
        joints = JointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            anatomical_joints=joints,
        )
        dofs = list(skel.iter_jointdofs())
        assert len(dofs) > 0

    def test_must_provide_exactly_one_of_preset_or_joints(self):
        with pytest.raises(ValueError):
            Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL)  # neither provided

        joints = JointPreset.LEGS_ONLY.to_joint_list()
        with pytest.raises(ValueError):
            Skeleton(
                axis_order=AxisOrder.YAW_PITCH_ROLL,
                joint_preset=JointPreset.LEGS_ONLY,
                anatomical_joints=joints,
            )  # both provided

    def test_iter_jointdofs_returns_joint_dof_objects(self):
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        dofs = list(skel.iter_jointdofs())
        assert all(isinstance(d, JointDOF) for d in dofs)

    def test_iter_jointdofs_respects_axis_order(self):
        """For the coxa (3-DoF ball joint), dofs should be in YAW-PITCH-ROLL order."""
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        dofs = list(skel.iter_jointdofs())
        # The first three dofs for the first leg's thorax-coxa joint should be YPR
        thorax_coxa_dofs = [
            d for d in dofs if d.parent.name == "c_thorax" and d.child.name == "lf_coxa"
        ]
        assert len(thorax_coxa_dofs) == 3
        assert thorax_coxa_dofs[0].axis is RotationAxis.YAW
        assert thorax_coxa_dofs[1].axis is RotationAxis.PITCH
        assert thorax_coxa_dofs[2].axis is RotationAxis.ROLL

    def test_iter_jointdofs_different_axis_orders_give_different_names(self):
        skel_ypr = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        skel_pry = Skeleton(
            axis_order=AxisOrder.PITCH_ROLL_YAW,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        dofs_ypr = [d.name for d in skel_ypr.iter_jointdofs()]
        dofs_pry = [d.name for d in skel_pry.iter_jointdofs()]
        # Same DoFs are present, just in different order
        assert set(dofs_ypr) == set(dofs_pry)
        assert dofs_ypr != dofs_pry

    def test_get_actuated_dofs_from_preset_legs_active_only(self):
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.ALL_BIOLOGICAL,
        )
        dofs = skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        for dof in dofs:
            assert dof.child.is_leg()
            assert dof.child.link not in PASSIVE_TARSAL_LINKS

    def test_get_tree_is_valid(self):
        skel = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.ALL_BIOLOGICAL,
        )
        tree = skel.get_tree()
        assert tree is not None


# ##############################################################################
# Flybody duplicates: same tests as above but for the FlybodyFly anatomy types
# (defined in flygym.assets.model.flybody.anatomy_flybody and used by
# flygym.compose.fly.FlybodyFly).
# ##############################################################################


# ==============================================================================
# Flybody Constants
# ==============================================================================


class TestFlybodyConstants:
    # Note: test_sides and test_legs are NOT duplicated because flybody reuses
    # SIDES and LEGS from the base anatomy module (already covered above).
    # Note: test_body_positions_contains_center_and_sides_and_legs is NOT
    # duplicated because there is no FLYBODY_BODY_POSITIONS — flybody reuses
    # the base BODY_POSITIONS constant.

    def test_flybody_leg_links(self):
        assert FLYBODY_LEG_LINKS[0] == "coxa"
        assert "tibia" in FLYBODY_LEG_LINKS
        assert len([lk for lk in FLYBODY_LEG_LINKS if lk.startswith("tarsus")]) == 5

    def test_flybody_passive_tarsal_links(self):
        assert len(FLYBODY_PASSIVE_TARSAL_LINKS) == 4
        for lk in FLYBODY_PASSIVE_TARSAL_LINKS:
            assert lk.startswith("tarsus")
        assert "tarsus1" not in FLYBODY_PASSIVE_TARSAL_LINKS

    def test_flybody_all_segment_names_unique(self):
        assert len(FLYBODY_ALL_SEGMENT_NAMES) == len(set(FLYBODY_ALL_SEGMENT_NAMES))

    def test_flybody_all_segment_names_contains_thorax(self):
        assert "c_thorax" in FLYBODY_ALL_SEGMENT_NAMES

    def test_flybody_all_connected_segment_pairs_are_in_all_segments(self):
        seg_set = set(FLYBODY_ALL_SEGMENT_NAMES)
        for parent, child in FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS:
            assert parent in seg_set, f"{parent} not in FLYBODY_ALL_SEGMENT_NAMES"
            assert child in seg_set, f"{child} not in FLYBODY_ALL_SEGMENT_NAMES"


# ==============================================================================
# FlybodyAxesSet / WingFlybodyAxesSet
# ==============================================================================


class TestFlybodyAxesSet:
    def test_construction_from_strings(self):
        axes = FlybodyAxesSet(["pitch", "roll", "yaw"])
        assert FlybodyRotationAxis.PITCH in axes
        assert FlybodyRotationAxis.ROLL in axes
        assert FlybodyRotationAxis.YAW in axes

    def test_construction_from_enum(self):
        axes = FlybodyAxesSet([FlybodyRotationAxis.PITCH, FlybodyRotationAxis.YAW])
        assert FlybodyRotationAxis.PITCH in axes
        assert FlybodyRotationAxis.YAW in axes
        assert FlybodyRotationAxis.ROLL not in axes

    def test_empty_construction(self):
        axes = FlybodyAxesSet()
        assert len(axes) == 0

    def test_add_string(self):
        axes = FlybodyAxesSet()
        axes.add("pitch")
        assert FlybodyRotationAxis.PITCH in axes

    def test_remove_string(self):
        axes = FlybodyAxesSet(["pitch", "roll"])
        axes.remove("pitch")
        assert FlybodyRotationAxis.PITCH not in axes
        assert FlybodyRotationAxis.ROLL in axes

    def test_construction_from_all_enum_values(self):
        axes = FlybodyAxesSet(FlybodyRotationAxis)
        assert len(axes) == 3


class TestWingFlybodyAxesSet:
    def test_construction_from_strings(self):
        axes = WingFlybodyAxesSet(["pitch", "roll", "yaw"])
        assert WingFlybodyRotationAxis.PITCH in axes
        assert WingFlybodyRotationAxis.ROLL in axes
        assert WingFlybodyRotationAxis.YAW in axes

    def test_construction_from_enum(self):
        axes = WingFlybodyAxesSet(
            [WingFlybodyRotationAxis.PITCH, WingFlybodyRotationAxis.YAW]
        )
        assert WingFlybodyRotationAxis.PITCH in axes
        assert WingFlybodyRotationAxis.YAW in axes
        assert WingFlybodyRotationAxis.ROLL not in axes

    def test_construction_from_all_enum_values(self):
        axes = WingFlybodyAxesSet(WingFlybodyRotationAxis)
        assert len(axes) == 3


# ==============================================================================
# FlybodyAxisOrder / WingFlybodyAxisOrder
# ==============================================================================


class TestFlybodyAxisOrder:
    def test_aliases(self):
        assert FlybodyAxisOrder.YPR is FlybodyAxisOrder.YAW_PITCH_ROLL
        assert FlybodyAxisOrder.PRY is FlybodyAxisOrder.PITCH_ROLL_YAW
        assert FlybodyAxisOrder.DONTCARE is FlybodyAxisOrder.PITCH_ROLL_YAW

    def test_missing_from_string(self):
        ao = FlybodyAxisOrder("yaw_pitch_roll")
        assert ao is FlybodyAxisOrder.YAW_PITCH_ROLL

    def test_missing_from_sequence(self):
        ao = FlybodyAxisOrder(["yaw", "pitch", "roll"])
        assert ao is FlybodyAxisOrder.YAW_PITCH_ROLL

    def test_missing_from_rotation_axis_sequence(self):
        ao = FlybodyAxisOrder(
            (
                FlybodyRotationAxis.YAW,
                FlybodyRotationAxis.PITCH,
                FlybodyRotationAxis.ROLL,
            )
        )
        assert ao is FlybodyAxisOrder.YAW_PITCH_ROLL

    def test_to_letters_xyz(self):
        # Flybody convention: PITCH=x, ROLL=y, YAW=z
        assert FlybodyAxisOrder.PITCH_ROLL_YAW.to_letters_xyz() == "xyz"
        assert FlybodyAxisOrder.YAW_PITCH_ROLL.to_letters_xyz() == "zxy"
        assert FlybodyAxisOrder.ROLL_PITCH_YAW.to_letters_xyz() == "yxz"

    def test_all_six_permutations_are_distinct(self):
        all_orders = [
            FlybodyAxisOrder.PITCH_ROLL_YAW,
            FlybodyAxisOrder.PITCH_YAW_ROLL,
            FlybodyAxisOrder.ROLL_PITCH_YAW,
            FlybodyAxisOrder.ROLL_YAW_PITCH,
            FlybodyAxisOrder.YAW_PITCH_ROLL,
            FlybodyAxisOrder.YAW_ROLL_PITCH,
        ]
        assert len(set(ao.to_letters_xyz() for ao in all_orders)) == 6


class TestWingFlybodyAxisOrder:
    def test_aliases(self):
        assert WingFlybodyAxisOrder.YPR is WingFlybodyAxisOrder.YAW_PITCH_ROLL
        assert WingFlybodyAxisOrder.PRY is WingFlybodyAxisOrder.PITCH_ROLL_YAW
        assert WingFlybodyAxisOrder.DONTCARE is WingFlybodyAxisOrder.PITCH_ROLL_YAW

    def test_to_letters_xyz(self):
        # WingFlybody convention: PITCH=y, ROLL=x, YAW=z
        assert WingFlybodyAxisOrder.PITCH_ROLL_YAW.to_letters_xyz() == "yxz"
        assert WingFlybodyAxisOrder.YAW_PITCH_ROLL.to_letters_xyz() == "zyx"
        assert WingFlybodyAxisOrder.ROLL_PITCH_YAW.to_letters_xyz() == "xyz"

    def test_all_six_permutations_are_distinct(self):
        all_orders = [
            WingFlybodyAxisOrder.PITCH_ROLL_YAW,
            WingFlybodyAxisOrder.PITCH_YAW_ROLL,
            WingFlybodyAxisOrder.ROLL_PITCH_YAW,
            WingFlybodyAxisOrder.ROLL_YAW_PITCH,
            WingFlybodyAxisOrder.YAW_PITCH_ROLL,
            WingFlybodyAxisOrder.YAW_ROLL_PITCH,
        ]
        assert len(set(ao.to_letters_xyz() for ao in all_orders)) == 6


# ==============================================================================
# FlybodyBodySegment
# ==============================================================================


class TestFlybodyBodySegment:
    # Note: test_eye_predicates is NOT duplicated — the flybody model has no
    # separate eye segments (FlybodyBodySegment.is_eye always returns False,
    # and "l_eye"/"r_eye" are not in FLYBODY_ALL_SEGMENT_NAMES).

    def test_valid_segment(self):
        seg = FlybodyBodySegment("c_thorax")
        assert seg.name == "c_thorax"
        assert seg.pos == "c"
        assert seg.link == "thorax"

    def test_left_front_coxa(self):
        seg = FlybodyBodySegment("lf_coxa")
        assert seg.pos == "lf"
        assert seg.link == "coxa"
        assert seg.is_leg()
        assert not seg.is_thorax()
        assert not seg.is_head()

    def test_thorax_predicates(self):
        seg = FlybodyBodySegment("c_thorax")
        assert seg.is_thorax()
        assert not seg.is_leg()
        assert not seg.is_head()

    def test_head_predicates(self):
        seg = FlybodyBodySegment("c_head")
        assert seg.is_head()
        assert not seg.is_thorax()
        assert not seg.is_leg()

    def test_wing_predicates(self):
        seg = FlybodyBodySegment("l_wing")
        assert seg.is_wing()
        assert not seg.is_leg()

    def test_abdomen_predicates(self):
        # In the flybody model, abdomen segments are c_abdomen1..c_abdomen7.
        seg = FlybodyBodySegment("c_abdomen1")
        assert seg.is_abdomen()

    def test_antenna_predicates(self):
        # In the flybody model, each antenna is a single segment named
        # "{side}_antenna" (no pedicel/funiculus/arista split).
        seg = FlybodyBodySegment("l_antenna")
        assert seg.is_antenna()

    def test_proboscis_predicates(self):
        seg = FlybodyBodySegment("c_rostrum")
        assert seg.is_proboscis()

    def test_labrum_is_proboscis(self):
        # Flybody-specific: labrum is part of the proboscis chain.
        seg = FlybodyBodySegment("l_labrum")
        assert seg.is_proboscis()

    def test_all_legs_classified_as_leg(self):
        for leg in LEGS:
            seg = FlybodyBodySegment(f"{leg}_coxa")
            assert seg.is_leg(), f"{leg}_coxa should be a leg segment"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid body segment name"):
            FlybodyBodySegment("invalid_segment")

    def test_frozen_dataclass(self):
        seg = FlybodyBodySegment("c_thorax")
        with pytest.raises((AttributeError, TypeError)):
            seg.name = "something_else"


# ==============================================================================
# FlybodyJointDOF
# ==============================================================================


class TestFlybodyJointDOF:
    # Note: from_name tests are NOT duplicated. FlybodyJointDOF.from_name is
    # currently broken — it is defined as a regular method (missing the
    # @classmethod decorator) and splits the name on "_" instead of "-",
    # which conflicts with segment names like "c_thorax" that contain "_".
    # Adding round-trip tests here would fail. Once from_name is fixed, the
    # equivalent of TestJointDOF.test_from_name_* should be added.

    def test_name_property(self):
        parent = FlybodyBodySegment("c_thorax")
        child = FlybodyBodySegment("lf_coxa")
        dof = FlybodyJointDOF(parent, child, FlybodyRotationAxis.YAW)
        assert dof.name == "c_thorax-lf_coxa-yaw"

    def test_frozen_dataclass(self):
        dof = FlybodyJointDOF(
            FlybodyBodySegment("c_thorax"),
            FlybodyBodySegment("lf_coxa"),
            FlybodyRotationAxis.PITCH,
        )
        with pytest.raises((AttributeError, TypeError)):
            dof.axis = FlybodyRotationAxis.YAW


# ==============================================================================
# FlybodyAnatomicalJoint
# ==============================================================================


class TestFlybodyAnatomicalJoint:
    def test_iter_dofs_all_three(self):
        joint = FlybodyAnatomicalJoint(
            FlybodyBodySegment("c_thorax"),
            FlybodyBodySegment("lf_coxa"),
            FlybodyAxesSet(
                [
                    FlybodyRotationAxis.PITCH,
                    FlybodyRotationAxis.ROLL,
                    FlybodyRotationAxis.YAW,
                ]
            ),
        )
        dofs = list(joint.iter_dofs(FlybodyAxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 3
        assert dofs[0].axis is FlybodyRotationAxis.YAW
        assert dofs[1].axis is FlybodyRotationAxis.PITCH
        assert dofs[2].axis is FlybodyRotationAxis.ROLL

    def test_iter_dofs_subset(self):
        joint = FlybodyAnatomicalJoint(
            FlybodyBodySegment("c_thorax"),
            FlybodyBodySegment("lf_coxa"),
            FlybodyAxesSet([FlybodyRotationAxis.PITCH]),
        )
        dofs = list(joint.iter_dofs(FlybodyAxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 1
        assert dofs[0].axis is FlybodyRotationAxis.PITCH

    def test_iter_dofs_empty(self):
        joint = FlybodyAnatomicalJoint(
            FlybodyBodySegment("c_thorax"),
            FlybodyBodySegment("lf_coxa"),
            FlybodyAxesSet(),
        )
        dofs = list(joint.iter_dofs(FlybodyAxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 0

    def test_iter_dofs_wing_uses_wing_axis_order(self):
        # Flybody-specific: wing joints internally use WingFlybodyAxisOrder so
        # axes resolve to WingFlybodyRotationAxis (not FlybodyRotationAxis).
        joint = FlybodyAnatomicalJoint(
            FlybodyBodySegment("c_thorax"),
            FlybodyBodySegment("l_wing"),
            WingFlybodyAxesSet(WingFlybodyRotationAxis),
        )
        dofs = list(joint.iter_dofs(FlybodyAxisOrder.YAW_PITCH_ROLL))
        assert len(dofs) == 3
        for dof in dofs:
            assert isinstance(dof.axis, WingFlybodyRotationAxis)


# ==============================================================================
# FlybodyJointPreset
# ==============================================================================


class TestFlybodyJointPreset:
    def test_all_possible_returns_joints_for_all_pairs(self):
        joints = FlybodyJointPreset.ALL_POSSIBLE.to_joint_list()
        assert len(joints) == len(FLYBODY_ALL_CONNECTED_SEGMENT_PAIRS)
        for joint in joints:
            assert len(joint.axes) == 3

    def test_all_biological_has_fewer_dofs_than_all_possible(self):
        all_possible = FlybodyJointPreset.ALL_POSSIBLE.to_joint_list()
        biological = FlybodyJointPreset.ALL_BIOLOGICAL.to_joint_list()
        assert len(biological) == len(all_possible)
        total_possible_dofs = sum(len(j.axes) for j in all_possible)
        total_biological_dofs = sum(len(j.axes) for j in biological)
        assert total_biological_dofs < total_possible_dofs

    def test_legs_only_excludes_non_leg_joints(self):
        joints = FlybodyJointPreset.LEGS_ONLY.to_joint_list()
        for joint in joints:
            assert joint.child.is_leg(), f"{joint.child.name} should be a leg segment"

    def test_legs_active_only_excludes_passive_tarsal(self):
        joints = FlybodyJointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        for joint in joints:
            assert joint.child.link not in FLYBODY_PASSIVE_TARSAL_LINKS, (
                f"{joint.child.link} should not be in legs_active_only"
            )

    def test_legs_active_only_subset_of_legs_only(self):
        legs_only_children = {
            j.child.name for j in FlybodyJointPreset.LEGS_ONLY.to_joint_list()
        }
        legs_active_children = {
            j.child.name for j in FlybodyJointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        }
        assert legs_active_children.issubset(legs_only_children)
        assert legs_only_children != legs_active_children

    def test_from_string(self):
        assert FlybodyJointPreset("legs_only") is FlybodyJointPreset.LEGS_ONLY


# ==============================================================================
# FlybodyActuatedDOFPreset
# ==============================================================================


class TestFlybodyActuatedDOFPreset:
    @pytest.fixture
    def all_bio_dofs(self):
        skeleton = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.ALL_BIOLOGICAL,
        )
        return list(skeleton.iter_jointdofs())

    def test_all_returns_everything(self, all_bio_dofs):
        filtered = FlybodyActuatedDOFPreset.ALL.filter(all_bio_dofs)
        assert len(filtered) == len(all_bio_dofs)

    def test_legs_only_returns_only_leg_dofs(self, all_bio_dofs):
        filtered = FlybodyActuatedDOFPreset.LEGS_ONLY.filter(all_bio_dofs)
        for dof in filtered:
            assert dof.child.is_leg()

    def test_legs_active_only_excludes_passive_tarsals(self, all_bio_dofs):
        filtered = FlybodyActuatedDOFPreset.LEGS_ACTIVE_ONLY.filter(all_bio_dofs)
        for dof in filtered:
            assert dof.child.link not in FLYBODY_PASSIVE_TARSAL_LINKS

    def test_from_string(self):
        assert (
            FlybodyActuatedDOFPreset("legs_only") is FlybodyActuatedDOFPreset.LEGS_ONLY
        )


# ==============================================================================
# FlybodyContactBodiesPreset
# ==============================================================================
#
# Note: a minimal TestFlybodyContactBodiesPreset class already exists higher in
# this file (covering ALL, LEGS_ONLY, and from_string). The class below extends
# it with the remaining tests mirroring TestContactBodiesPreset.


class TestFlybodyContactBodiesPresetExtras:
    def test_tibia_tarsus_only_is_subset_of_legs_only(self):
        tibia_tarsus = {
            s.name
            for s in FlybodyContactBodiesPreset.TIBIA_TARSUS_ONLY.to_body_segments_list()
        }
        legs_only = {
            s.name for s in FlybodyContactBodiesPreset.LEGS_ONLY.to_body_segments_list()
        }
        assert tibia_tarsus.issubset(legs_only)

    def test_tibia_tarsus_only_contains_tibia_and_tarsus(self):
        segs = FlybodyContactBodiesPreset.TIBIA_TARSUS_ONLY.to_body_segments_list()
        for seg in segs:
            assert seg.link == "tibia" or seg.link.startswith("tarsus")

    def test_legs_thorax_abdomen_head_contains_thorax(self):
        segs = (
            FlybodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        )
        seg_names = {s.name for s in segs}
        assert "c_thorax" in seg_names


# ==============================================================================
# FlybodySkeleton
# ==============================================================================


class TestFlybodySkeleton:
    def test_construction_from_preset(self):
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.LEGS_ONLY,
        )
        assert skel.axis_order is FlybodyAxisOrder.YAW_PITCH_ROLL

    def test_construction_from_anatomical_joints(self):
        joints = FlybodyJointPreset.LEGS_ACTIVE_ONLY.to_joint_list()
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            anatomical_joints=joints,
        )
        dofs = list(skel.iter_jointdofs())
        assert len(dofs) > 0

    def test_must_provide_exactly_one_of_preset_or_joints(self):
        with pytest.raises(ValueError):
            FlybodySkeleton(axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL)  # neither

        joints = FlybodyJointPreset.LEGS_ONLY.to_joint_list()
        with pytest.raises(ValueError):
            FlybodySkeleton(
                axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
                joint_preset=FlybodyJointPreset.LEGS_ONLY,
                anatomical_joints=joints,
            )  # both

    def test_iter_jointdofs_returns_joint_dof_objects(self):
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.LEGS_ONLY,
        )
        dofs = list(skel.iter_jointdofs())
        assert all(isinstance(d, FlybodyJointDOF) for d in dofs)

    def test_iter_jointdofs_respects_axis_order(self):
        """For the coxa (3-DoF ball joint), dofs should be in YAW-PITCH-ROLL order."""
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.LEGS_ONLY,
        )
        dofs = list(skel.iter_jointdofs())
        thorax_coxa_dofs = [
            d for d in dofs if d.parent.name == "c_thorax" and d.child.name == "lf_coxa"
        ]
        assert len(thorax_coxa_dofs) == 3
        assert thorax_coxa_dofs[0].axis is FlybodyRotationAxis.YAW
        assert thorax_coxa_dofs[1].axis is FlybodyRotationAxis.PITCH
        assert thorax_coxa_dofs[2].axis is FlybodyRotationAxis.ROLL

    def test_iter_jointdofs_different_axis_orders_give_different_names(self):
        skel_ypr = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.LEGS_ONLY,
        )
        skel_pry = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.PITCH_ROLL_YAW,
            joint_preset=FlybodyJointPreset.LEGS_ONLY,
        )
        dofs_ypr = [d.name for d in skel_ypr.iter_jointdofs()]
        dofs_pry = [d.name for d in skel_pry.iter_jointdofs()]
        assert set(dofs_ypr) == set(dofs_pry)
        assert dofs_ypr != dofs_pry

    def test_get_actuated_dofs_from_preset_legs_active_only(self):
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.ALL_BIOLOGICAL,
        )
        dofs = skel.get_actuated_dofs_from_preset(FlybodyActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        for dof in dofs:
            assert dof.child.is_leg()
            assert dof.child.link not in FLYBODY_PASSIVE_TARSAL_LINKS

    def test_get_tree_is_valid(self):
        skel = FlybodySkeleton(
            axis_order=FlybodyAxisOrder.YAW_PITCH_ROLL,
            joint_preset=FlybodyJointPreset.ALL_BIOLOGICAL,
        )
        tree = skel.get_tree()
        assert tree is not None
