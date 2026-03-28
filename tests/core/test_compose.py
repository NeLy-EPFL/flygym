"""Integration tests for flygym.compose (Fly, World)."""

import pytest
import mujoco
import numpy as np

import flygym
from flygym.anatomy import (
    AxisOrder,
    JointPreset,
    ActuatedDOFPreset,
    ContactBodiesPreset,
    Skeleton,
    LEGS,
)
from flygym.compose.fly import Fly, ActuatorType
from flygym.compose.world import FlatGroundWorld, TetheredWorld
from flygym.compose.pose import KinematicPose
from flygym.utils.math import Rotation3D


# ==============================================================================
# Fly construction
# ==============================================================================


class TestFlyConstruction:
    def test_fly_default_name(self):
        fly = Fly()
        assert fly.name == "nmf"

    def test_fly_custom_name(self):
        fly = Fly(name="myfly")
        assert fly.name == "myfly"

    def test_body_segments_populated(self):
        fly = Fly()
        # All body segments should have a corresponding MJCF body and geom
        assert len(fly.bodyseg_to_mjcfbody) > 0
        assert len(fly.bodyseg_to_mjcfgeom) > 0
        assert set(fly.bodyseg_to_mjcfbody.keys()) == set(fly.bodyseg_to_mjcfgeom.keys())

    def test_all_segment_names_present(self):
        from flygym.anatomy import ALL_SEGMENT_NAMES
        fly = Fly()
        body_names = {seg.name for seg in fly.bodyseg_to_mjcfbody}
        assert set(ALL_SEGMENT_NAMES) == body_names

    def test_no_joints_before_add_joints(self):
        fly = Fly()
        assert fly.skeleton is None
        assert len(fly.jointdof_to_mjcfjoint) == 0

    def test_no_actuators_before_add_actuators(self):
        fly = Fly()
        for ty in ActuatorType:
            assert len(fly.jointdof_to_mjcfactuator_by_type[ty]) == 0

    def test_compile_before_joints(self):
        """Should be able to compile a bare fly (no joints/actuators)."""
        fly = Fly()
        mj_model, mj_data = fly.compile()
        assert mj_model is not None
        assert mj_data is not None


class TestFlyAddJoints:
    def test_skeleton_set_after_add_joints(self, fly_with_joints):
        assert fly_with_joints.skeleton is not None

    def test_joints_populated_for_legs_only(self, fly_with_joints):
        for dof in fly_with_joints.jointdof_to_mjcfjoint:
            assert dof.child.is_leg(), f"{dof.child.name} should be a leg segment"

    def test_number_of_joints(self, fly_with_joints, skeleton_ypr):
        expected = list(skeleton_ypr.iter_jointdofs())
        assert len(fly_with_joints.jointdof_to_mjcfjoint) == len(expected)

    def test_neutral_angles_stored(self, fly_with_joints):
        assert len(fly_with_joints.jointdof_to_neutralangle) > 0

    def test_add_joints_with_no_neutral_pose_defaults_to_zero(self, skeleton_ypr):
        fly = Fly(name="noposefly")
        fly.add_joints(skeleton_ypr, neutral_pose=None)
        for angle in fly.jointdof_to_neutralangle.values():
            assert angle == 0.0

    def test_add_joints_invalid_neutral_pose_raises(self, skeleton_ypr):
        fly = Fly(name="badposefly")
        with pytest.raises(ValueError, match="KinematicPose"):
            fly.add_joints(skeleton_ypr, neutral_pose={"not": "a_pose"})


class TestFlyAddActuators:
    def test_position_actuators_populated(self, fly_with_joints):
        pos_actuators = fly_with_joints.jointdof_to_mjcfactuator_by_type[
            ActuatorType.POSITION
        ]
        assert len(pos_actuators) > 0

    def test_actuator_count_matches_active_dofs(self, fly_with_joints, skeleton_ypr):
        active_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        )
        pos_actuators = fly_with_joints.jointdof_to_mjcfactuator_by_type[
            ActuatorType.POSITION
        ]
        assert len(pos_actuators) == len(active_dofs)

    def test_get_actuated_jointdofs_order_matches_add_actuators(
        self, fly_with_joints, skeleton_ypr
    ):
        active_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        )
        returned_order = list(
            fly_with_joints.get_actuated_jointdofs_order(ActuatorType.POSITION)
        )
        assert len(returned_order) == len(active_dofs)


class TestFlyAddLegAdhesion:
    def test_adhesion_actuators_added(self, fly_with_adhesion):
        assert len(fly_with_adhesion.leg_to_adhesionactuator) == 6

    def test_adhesion_for_each_leg(self, fly_with_adhesion):
        for leg in LEGS:
            assert leg in fly_with_adhesion.leg_to_adhesionactuator

    def test_add_leg_adhesion_twice_raises(self, fly_with_adhesion):
        with pytest.raises(ValueError, match="already been added"):
            fly_with_adhesion.add_leg_adhesion()


class TestFlyCompile:
    def test_compile_produces_mujoco_model(self, fly_with_joints):
        mj_model, mj_data = fly_with_joints.compile()
        assert isinstance(mj_model, mujoco.MjModel)
        assert isinstance(mj_data, mujoco.MjData)

    def test_compiled_model_has_correct_n_joints(self, fly_with_joints, skeleton_ypr):
        mj_model, _ = fly_with_joints.compile()
        expected_dofs = len(list(skeleton_ypr.iter_jointdofs()))
        # Each hinge joint contributes 1 DoF; the freejoints are absent (no world yet)
        # nv should equal n_leg_dofs (no freejoint here, standalone fly)
        assert mj_model.nv == expected_dofs

    def test_get_bodysegs_order_length(self, fly_with_joints):
        from flygym.anatomy import ALL_SEGMENT_NAMES
        order = list(fly_with_joints.get_bodysegs_order())
        assert len(order) == len(ALL_SEGMENT_NAMES)

    def test_get_jointdofs_order_length(self, fly_with_joints, skeleton_ypr):
        expected = list(skeleton_ypr.iter_jointdofs())
        order = list(fly_with_joints.get_jointdofs_order())
        assert len(order) == len(expected)

    def test_get_legs_order(self, fly_with_joints):
        assert fly_with_joints.get_legs_order() == LEGS


# ==============================================================================
# FlatGroundWorld
# ==============================================================================


class TestFlatGroundWorld:
    def test_construction(self):
        world = FlatGroundWorld()
        assert world is not None
        assert len(world.fly_lookup) == 0

    def test_custom_name(self):
        world = FlatGroundWorld(name="myworld")
        assert world.mjcf_root.model == "myworld"

    def test_add_fly_registers_in_lookup(self, flat_world_with_fly, fly_with_joints):
        assert fly_with_joints.name in flat_world_with_fly.fly_lookup

    def test_duplicate_fly_name_raises(self):
        # Use fresh Fly instances and TetheredWorld (no ground-contact sensors)
        # so bare flies compile cleanly. The name-uniqueness check is world-agnostic.
        fly_a = Fly(name="dupfly")
        fly_b = Fly(name="dupfly")
        world = TetheredWorld(name="dupworld")
        world.add_fly(fly_a, spawn_position=[0, 0, 1.5],
                      spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]))
        with pytest.raises(ValueError, match="already exists"):
            world.add_fly(fly_b, spawn_position=[1, 0, 1.5],
                          spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]))

    def test_spawn_rotation_must_be_quat(self):
        # Use a fresh, unattached fly so dm_control doesn't error first
        fly = Fly(name="quatfly")
        world = TetheredWorld(name="rotworld")
        with pytest.raises(ValueError, match="quaternion"):
            world.add_fly(
                fly,
                spawn_position=[0, 0, 1.5],
                spawn_rotation=Rotation3D("euler", [0, 0, 0]),
            )

    def test_compile_returns_mujoco_model(self, flat_world_with_fly):
        mj_model, mj_data = flat_world_with_fly.compile()
        assert isinstance(mj_model, mujoco.MjModel)
        assert isinstance(mj_data, mujoco.MjData)

    def test_compiled_model_has_freejoint(self, flat_world_with_fly):
        """Fly should be attached with a 6-DoF free joint (7 qpos, 6 qvel)."""
        mj_model, _ = flat_world_with_fly.compile()
        # Free joint: 7 qpos (xyz + quat) + n_leg_dofs
        assert mj_model.nq > 7
        assert mj_model.nv > 6

    def test_contact_sensors_populated(self, flat_world_with_fly):
        assert flat_world_with_fly.legpos_to_groundcontactsensors_by_fly is not None
        fly_name = list(flat_world_with_fly.fly_lookup.keys())[0]
        sensors = flat_world_with_fly.legpos_to_groundcontactsensors_by_fly[fly_name]
        assert len(sensors) == 6  # one per leg

    def test_world_dof_neutral_states_set(self, flat_world_with_fly):
        assert len(flat_world_with_fly.world_dof_neutral_states) > 0


class TestTetheredWorld:
    def test_construction(self):
        world = TetheredWorld()
        assert world is not None

    def test_add_fly_registers_in_lookup(self, tethered_world_with_fly):
        assert "tethered_fly" in tethered_world_with_fly.fly_lookup

    def test_compile_returns_mujoco_model(self, tethered_world_with_fly):
        mj_model, mj_data = tethered_world_with_fly.compile()
        assert isinstance(mj_model, mujoco.MjModel)
        assert isinstance(mj_data, mujoco.MjData)
