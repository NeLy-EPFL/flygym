"""Shared pytest fixtures for flygym integration tests.

Fixtures that require building the full MuJoCo fly model (which loads mesh files)
are scoped to the module level so the expensive setup is done only once per test file.
"""

import pytest

from flygym.anatomy import AxisOrder, JointPreset, ActuatedDOFPreset, Skeleton
from flygym.compose.fly import Fly, ActuatorType
from flygym.compose.world import FlatGroundWorld, TetheredWorld
from flygym.compose.pose import KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym.simulation import Simulation


# ---------------------------------------------------------------------------
# Shared pose / skeleton (no I/O side-effects)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neutral_pose():
    """Neutral standing pose in YAW_PITCH_ROLL order."""
    return KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)


@pytest.fixture(scope="module")
def skeleton_ypr():
    """Legs-only skeleton with YAW_PITCH_ROLL axis order."""
    return Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )


# ---------------------------------------------------------------------------
# Fly fixtures (each attached to at most one world per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fly_with_joints(neutral_pose, skeleton_ypr):
    """Fly with joints and position actuators. Used by compose tests."""
    fly = Fly(name="test_fly")
    fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
    actuated_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50,
    )
    return fly


@pytest.fixture(scope="module")
def fly_with_adhesion(neutral_pose, skeleton_ypr):
    """Fly with joints, position actuators, and leg adhesion. Used by simulation tests."""
    fly = Fly(name="sim_fly")
    fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
    actuated_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50,
    )
    fly.add_leg_adhesion(gain=1.0)
    return fly


# ---------------------------------------------------------------------------
# World fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat_world_with_fly(fly_with_joints):
    """FlatGroundWorld with fly_with_joints at the origin."""
    world = FlatGroundWorld()
    world.add_fly(
        fly_with_joints,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    return world


@pytest.fixture(scope="module")
def tethered_world_with_fly(neutral_pose, skeleton_ypr):
    """TetheredWorld with a standalone fly (used by compose tests)."""
    fly = Fly(name="tethered_fly")
    fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
    actuated_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50,
    )
    world = TetheredWorld()
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    return world


@pytest.fixture(scope="module")
def tethered_world_for_sim(fly_with_adhesion):
    """TetheredWorld with fly_with_adhesion. Backed by Simulation fixture."""
    world = TetheredWorld(name="sim_tethered_world")
    world.add_fly(
        fly_with_adhesion,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    return world


# ---------------------------------------------------------------------------
# Simulation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulation(tethered_world_for_sim):
    """A ready-to-use Simulation backed by a TetheredWorld with an adhesion fly."""
    sim = Simulation(tethered_world_for_sim)
    sim.reset()
    return sim
