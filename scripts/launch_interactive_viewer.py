import mujoco
import tyro

from flygym.anatomy import Skeleton, ActuatedDOFPreset
from flygym.compose import Fly, FlatGroundWorld
from flygym.rendering import launch_interactive_viewer


def launch_viewer(
    joint_preset: str = "all_biological",
    axis_order: str = "roll_yaw_pitch",
    actuated_joints: str = "legs_active_only",
    actuator_type: str = "position",
    position_actuator_kp: float = 50.0,
    spawn_height: float = 3.0,
    bodysegs_with_ground_contact="legs_thorax_abdomen_head",
    run_async: bool = False,
):
    skeleton = Skeleton(joint_preset=joint_preset, axis_order=axis_order)

    fly = Fly()
    fly.add_joints(skeleton)

    actuated_dofs = ActuatedDOFPreset(actuated_joints).find_actuated_dofs_in(skeleton)
    if actuator_type == "position":
        fly.add_actuators(
            actuated_dofs, actuator_type=actuator_type, kp=position_actuator_kp
        )
    else:
        raise NotImplementedError(
            f"Actuator type '{actuator_type}' not implemented in this viewer."
        )

    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=(0, 0, spawn_height),
        bodysegs_with_ground_contact=bodysegs_with_ground_contact,
    )

    # Compile model and get data container
    mj_model, mj_data = world.compile()

    launch_interactive_viewer(mj_model, mj_data, run_async=run_async)


if __name__ == "__main__":
    tyro.cli(launch_viewer)
