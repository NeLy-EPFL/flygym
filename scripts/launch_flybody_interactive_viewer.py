import warnings

from flygym.flybody.anatomy_flybody import (
    FlyBodyJointPreset,
    FlyBodyAxisOrder,
    FlyBodySkeleton,
    FlyBodyContactBodiesPreset,
    FlyBodyActuatedDOFPreset,
)

from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.compose.fly import FlyBody
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D

warnings.warn(
    "FlyBody support is experimental. The API may change in future releases, "
    "and not all NeuroMechFly features are currently supported.",
    stacklevel=1,
)

joint_preset = FlyBodyJointPreset.ALL_BIOLOGICAL
axis_order = FlyBodyAxisOrder.YAW_ROLL_PITCH
actuated_dofs = FlyBodyActuatedDOFPreset.ALL
actuator_type = ActuatorType.POSITION
neutral_pose = KinematicPosePreset.FLYBODY_NEUTRAL
bodysegs_with_ground_contact = FlyBodyContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
spawn_position = (0, 0, 0.8)  # xyz in mm
spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))  # wxyz in quaternion
run_async = False  # might need to change to True if launched from a notebook


def main():
    fly = FlyBody()

    skeleton = FlyBodySkeleton(joint_preset=joint_preset, axis_order=axis_order)
    fly.add_joints(skeleton, neutral_pose)

    actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type,
    )
    fly.add_tendons()
    fly.add_tendon_actuators()

    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position,
        spawn_rotation,
        bodysegs_with_ground_contact=bodysegs_with_ground_contact,
        add_ground_contact_sensors=True,
    )

    # Compile model and get data container.
    mj_model, mj_data = world.compile()

    # Launch interactive viewer.
    launch_interactive_viewer(mj_model, mj_data, run_async=run_async)


if __name__ == "__main__":
    main()
