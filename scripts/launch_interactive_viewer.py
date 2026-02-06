from pathlib import Path

import mujoco
import numpy as np

from flygym import assets_dir
from flygym.anatomy import (
    Skeleton,
    JointPreset,
    AxisOrder,
    ActuatedDOFPreset,
    ContactBodiesPreset,
)
from flygym.compose import Fly, ActuatorType, PoseDict, FlatGroundWorld
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D

joint_preset = JointPreset.ALL_BIOLOGICAL
axis_order = AxisOrder.ROLL_YAW_PITCH
actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
actuator_type = ActuatorType.POSITION
actuator_params = {"kp": 50.0}
neutral_pose_file = assets_dir / "kinematics/neutral_pose.yaml"
spawn_height = 1.0
bodysegs_with_ground_contact = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
run_async = False


def get_actuator_input_to_maintain_pose(
    fly: Fly, mj_model: mujoco.MjModel, target_pose: PoseDict
) -> np.ndarray:
    actuator_lookup = fly.jointdof_to_mjcfactuator_by_type[ActuatorType.POSITION]
    ctrl_input = np.zeros(mj_model.nu)
    for jointdof, actuator in actuator_lookup.items():
        internal_actuatorid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.full_identifier
        )
        ctrl_input[internal_actuatorid] = target_pose.get(jointdof.name, 0.0)
    return ctrl_input


def main():
    fly = Fly()

    skeleton = Skeleton(joint_preset=joint_preset, axis_order=axis_order)
    neutral_pose = PoseDict(file_path=neutral_pose_file)
    fly.add_joints(skeleton, neutral_pose)
    actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(actuated_dofs_list, actuator_type, **actuator_params)

    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=(0, 0, spawn_height),
        spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)),
        bodysegs_with_ground_contact=bodysegs_with_ground_contact,
    )

    # Compile model and get data container
    mj_model, mj_data = world.compile()

    # Tell position actuators to go to the neutral pose at the start of the simulation
    ctrl_input = get_actuator_input_to_maintain_pose(fly, mj_model, neutral_pose)
    mj_data.ctrl[:] = ctrl_input

    world.save_xml_with_assets(Path("~/my_world_model/").expanduser())
    launch_interactive_viewer(mj_model, mj_data, run_async=run_async)


if __name__ == "__main__":
    main()
