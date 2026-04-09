"""Shared fixtures for the warp test suite.

All fixtures that build a GPU simulation are module-scoped: constructing a
GPUSimulation triggers CUDA kernel JIT compilation, which takes ~10 s on the
first run (subsequent runs use the cache).  Module scope ensures compilation
happens at most once per test file.
"""

import warnings
import pytest

from flygym.anatomy import (
    Skeleton,
    JointPreset,
    ActuatedDOFPreset,
    AxisOrder,
    AnatomicalJoint,
    BodySegment,
)
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym.warp import GPUSimulation


def build_gpu_sim(
    n_worlds: int = 4, fly_name: str = "warp_fly", add_joint_sites: bool = False
) -> tuple:
    """Create a minimal GPUSimulation and return ``(sim, fly, cam)``.

    The fly has legs-only joints, position actuators, leg adhesion, and one
    tracking camera.  The world is a flat-ground world.
    """
    fly = Fly(name=fly_name)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    fly.add_joints(skeleton, neutral_pose=pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        kp=50,
        neutral_input=pose,
    )
    if add_joint_sites:
        fly.add_joint_sites(
            [
                AnatomicalJoint(
                    BodySegment("c_thorax"), BodySegment("lf_coxa")
                ),
                AnatomicalJoint(
                    BodySegment("c_thorax"), BodySegment("rf_coxa")
                ),
            ]
        )
    fly.add_leg_adhesion()
    cam = fly.add_tracking_camera()

    world = FlatGroundWorld()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = GPUSimulation(world, n_worlds=n_worlds)

    return sim, fly, cam


@pytest.fixture(scope="module")
def gpu_sim_factory():
    """Fixture that yields the build_gpu_sim factory function.

    Tests needing a custom configuration (n_worlds, fly_name) can call
    ``gpu_sim_factory(n_worlds=..., fly_name=...)`` directly.
    """
    return build_gpu_sim
