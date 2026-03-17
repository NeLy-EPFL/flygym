"""Integration tests for flygym.simulation (Simulation)."""

import pytest
import numpy as np

from flygym.anatomy import AxisOrder, JointPreset, ActuatedDOFPreset, Skeleton, LEGS
from flygym.compose.fly import Fly, ActuatorType
from flygym.compose.world import TetheredWorld, FlatGroundWorld
from flygym.compose.pose import KinematicPose
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D
from flygym.simulation import Simulation
import flygym


# ==============================================================================
# Simulation construction
# ==============================================================================


class TestSimulationConstruction:
    def test_construction_succeeds(self, simulation):
        assert simulation is not None

    def test_world_empty_raises(self):
        world = TetheredWorld(name="emptyworld")
        with pytest.raises(ValueError, match="at least one fly"):
            Simulation(world)

    def test_time_starts_at_zero_after_reset(self, simulation):
        simulation.reset()
        assert simulation.time == pytest.approx(0.0)

    def test_mj_model_accessible(self, simulation):
        import mujoco
        assert isinstance(simulation.mj_model, mujoco.MjModel)

    def test_mj_data_accessible(self, simulation):
        import mujoco
        assert isinstance(simulation.mj_data, mujoco.MjData)


# ==============================================================================
# Step and time
# ==============================================================================


class TestSimulationStep:
    def test_step_advances_time(self, simulation):
        simulation.reset()
        dt = simulation.mj_model.opt.timestep
        simulation.step()
        assert simulation.time == pytest.approx(dt, rel=1e-6)

    def test_multiple_steps(self, simulation):
        simulation.reset()
        dt = simulation.mj_model.opt.timestep
        n = 10
        for _ in range(n):
            simulation.step()
        assert simulation.time == pytest.approx(n * dt, rel=1e-6)

    def test_reset_resets_time(self, simulation):
        simulation.reset()
        for _ in range(5):
            simulation.step()
        simulation.reset()
        assert simulation.time == pytest.approx(0.0)


# ==============================================================================
# get_joint_angles
# ==============================================================================


class TestGetJointAngles:
    def test_returns_array(self, simulation, fly_with_adhesion, skeleton_ypr):
        simulation.reset()
        angles = simulation.get_joint_angles(fly_with_adhesion.name)
        assert isinstance(angles, np.ndarray)

    def test_correct_length(self, simulation, fly_with_adhesion, skeleton_ypr):
        simulation.reset()
        angles = simulation.get_joint_angles(fly_with_adhesion.name)
        expected_n = len(list(skeleton_ypr.iter_jointdofs()))
        assert len(angles) == expected_n

    def test_at_neutral_angles_close_to_neutral_pose(
        self, simulation, fly_with_adhesion, neutral_pose, skeleton_ypr
    ):
        """After reset the joint angles should roughly match the neutral pose."""
        simulation.reset()
        angles = simulation.get_joint_angles(fly_with_adhesion.name)
        neutral_lookup = neutral_pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        dof_order = list(skeleton_ypr.iter_jointdofs())
        for i, dof in enumerate(dof_order):
            if dof.name in neutral_lookup:
                expected = neutral_lookup[dof.name]
                assert angles[i] == pytest.approx(expected, abs=0.2), (
                    f"Joint {dof.name}: expected ~{expected:.3f} rad, got {angles[i]:.3f}"
                )


# ==============================================================================
# get_joint_velocities
# ==============================================================================


class TestGetJointVelocities:
    def test_returns_array(self, simulation, fly_with_adhesion):
        simulation.reset()
        vels = simulation.get_joint_velocities(fly_with_adhesion.name)
        assert isinstance(vels, np.ndarray)

    def test_correct_length(self, simulation, fly_with_adhesion, skeleton_ypr):
        simulation.reset()
        vels = simulation.get_joint_velocities(fly_with_adhesion.name)
        expected_n = len(list(skeleton_ypr.iter_jointdofs()))
        assert len(vels) == expected_n

    def test_velocities_near_zero_at_reset(self, simulation, fly_with_adhesion):
        simulation.reset()
        vels = simulation.get_joint_velocities(fly_with_adhesion.name)
        np.testing.assert_allclose(vels, 0.0, atol=1e-8)


# ==============================================================================
# get_body_positions
# ==============================================================================


class TestGetBodyPositions:
    def test_returns_2d_array(self, simulation, fly_with_adhesion):
        simulation.reset()
        pos = simulation.get_body_positions(fly_with_adhesion.name)
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_correct_number_of_bodies(self, simulation, fly_with_adhesion):
        from flygym.anatomy import ALL_SEGMENT_NAMES
        simulation.reset()
        pos = simulation.get_body_positions(fly_with_adhesion.name)
        assert pos.shape[0] == len(ALL_SEGMENT_NAMES)


# ==============================================================================
# get_body_rotations
# ==============================================================================


class TestGetBodyRotations:
    def test_returns_2d_array(self, simulation, fly_with_adhesion):
        simulation.reset()
        rots = simulation.get_body_rotations(fly_with_adhesion.name)
        assert rots.ndim == 2
        assert rots.shape[1] == 4  # quaternions

    def test_correct_number_of_bodies(self, simulation, fly_with_adhesion):
        from flygym.anatomy import ALL_SEGMENT_NAMES
        simulation.reset()
        rots = simulation.get_body_rotations(fly_with_adhesion.name)
        assert rots.shape[0] == len(ALL_SEGMENT_NAMES)

    def test_quaternions_are_unit(self, simulation, fly_with_adhesion):
        # xquat is a derived quantity - only valid after at least one step/forward pass
        simulation.reset()
        simulation.step()
        rots = simulation.get_body_rotations(fly_with_adhesion.name)
        norms = np.linalg.norm(rots, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ==============================================================================
# set_actuator_inputs / get_actuator_forces
# ==============================================================================


class TestActuatorIO:
    def test_set_and_get_actuator_forces(self, simulation, fly_with_adhesion, skeleton_ypr):
        simulation.reset()
        n_actuators = len(
            skeleton_ypr.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        )
        target_inputs = np.zeros(n_actuators)
        simulation.set_actuator_inputs(
            fly_with_adhesion.name, ActuatorType.POSITION, target_inputs
        )
        simulation.step()
        forces = simulation.get_actuator_forces(
            fly_with_adhesion.name, ActuatorType.POSITION
        )
        assert isinstance(forces, np.ndarray)
        assert len(forces) == n_actuators

    def test_set_actuator_inputs_wrong_length_raises(
        self, simulation, fly_with_adhesion, skeleton_ypr
    ):
        simulation.reset()
        n_actuators = len(
            skeleton_ypr.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        )
        bad_inputs = np.zeros(n_actuators + 5)
        with pytest.raises(ValueError):
            simulation.set_actuator_inputs(
                fly_with_adhesion.name, ActuatorType.POSITION, bad_inputs
            )


# ==============================================================================
# set_leg_adhesion_states
# ==============================================================================


class TestLegAdhesion:
    def test_set_all_adhesion_on(self, simulation, fly_with_adhesion):
        simulation.reset()
        adhesion_on = np.ones(6, dtype=bool)
        # Should not raise
        simulation.set_leg_adhesion_states(fly_with_adhesion.name, adhesion_on)
        simulation.step()

    def test_set_all_adhesion_off(self, simulation, fly_with_adhesion):
        simulation.reset()
        adhesion_off = np.zeros(6, dtype=bool)
        simulation.set_leg_adhesion_states(fly_with_adhesion.name, adhesion_off)
        simulation.step()

    def test_set_adhesion_wrong_length_raises(self, simulation, fly_with_adhesion):
        simulation.reset()
        with pytest.raises(ValueError):
            simulation.set_leg_adhesion_states(
                fly_with_adhesion.name, np.ones(5, dtype=bool)
            )


# ==============================================================================
# Ground contact info (requires FlatGroundWorld)
# ==============================================================================


class TestGroundContactInfo:
    @pytest.fixture(scope="class")
    def flat_sim(self, flat_world_with_fly, fly_with_joints):
        sim = Simulation(flat_world_with_fly)
        sim.reset()
        return sim

    def test_returns_six_tuples(self, flat_sim, fly_with_joints):
        flat_sim.reset()
        contact_active, forces, torques, positions, normals, tangents = (
            flat_sim.get_ground_contact_info(fly_with_joints.name)
        )
        assert len(contact_active) == 6
        assert forces.shape == (6, 3)
        assert torques.shape == (6, 3)
        assert positions.shape == (6, 3)
        assert normals.shape == (6, 3)
        assert tangents.shape == (6, 3)


# ==============================================================================
# warmup
# ==============================================================================


class TestWarmup:
    def test_warmup_advances_time(self, simulation):
        simulation.reset()
        warmup_duration = 0.001  # 1 ms
        simulation.warmup(duration_s=warmup_duration)
        assert simulation.time > 0.0

    def test_warmup_zero_duration_does_not_change_time(self, simulation):
        simulation.reset()
        simulation.warmup(duration_s=0.0)
        assert simulation.time == pytest.approx(0.0)
