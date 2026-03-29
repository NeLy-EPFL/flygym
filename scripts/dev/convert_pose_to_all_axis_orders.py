import shutil
from pathlib import Path

import yaml
import mujoco as mj
import numpy as np

from flygym import assets_dir
from flygym.anatomy import (
    Skeleton,
    JointPreset,
    JointDOF,
    AxisOrder,
    ActuatedDOFPreset,
    ContactBodiesPreset,
)
from flygym.compose import Fly, ActuatorType, KinematicPose, FlatGroundWorld
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D
from flygym.utils.pose_conversion import convert_pose_axis_order

joint_preset = JointPreset.ALL_BIOLOGICAL
actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
actuator_type = ActuatorType.POSITION
actuator_position_gain = 50.0
spawn_position = (0, 0, 0.8)  # xyz in mm
spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))  # wxyz in quaternion
bodysegs_with_ground_contact = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD


def launch_viewer(neutral_pose: KinematicPose):
    fly = Fly()

    skeleton = Skeleton(joint_preset=joint_preset, axis_order=neutral_pose.axis_order)
    fly.add_joints(skeleton, neutral_pose)

    actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type,
        neutral_input=neutral_pose,
        kp=actuator_position_gain,
        ctrlrange=(-3.14, 3.14),
    )

    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position,
        spawn_rotation,
        bodysegs_with_ground_contact=bodysegs_with_ground_contact,
    )

    # Compile model and get data container
    mj_model, mj_data = world.compile()

    # Launch interactive viewer
    launch_interactive_viewer(mj_model, mj_data)

    # Get final joint angles after viewer is closed
    final_joint_angles_rad_dict = {}
    for jid in range(mj_model.njnt):
        joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, jid)
        qposadr = mj_model.jnt_qposadr[jid]
        final_joint_angles_rad_dict[joint_name] = mj_data.qpos[qposadr]

    return final_joint_angles_rad_dict


def write_pose_yaml(pose: KinematicPose, output_path: Path):
    # Sort joint angles by joint name for consistency
    joint_angles_rad_dict = dict(sorted(pose.joint_angles_lookup_rad.items()))

    # Remove all joints on the right side since they can be mirrored from the left side
    joint_angles_rad_dict = {
        k: v
        for k, v in joint_angles_rad_dict.items()
        if JointDOF.from_name(k).child.pos[0] != "r"
    }

    # Convert to degrees for better readability in YAML file
    # Also round to the nearest integer and remove zeros
    MIN_ANGLE_RAD = np.deg2rad(0.5)
    joint_angles_deg_dict = {
        k: int(np.round(np.rad2deg(v)))
        for k, v in joint_angles_rad_dict.items()
        if abs(v) >= MIN_ANGLE_RAD
    }

    output_data = {
        "angle_unit": "degree",
        "axis_order": pose.axis_order.to_list_of_str(),
        "joint_angles": joint_angles_deg_dict,
    }

    with open(output_path, "w") as f:
        yaml.dump(output_data, f)


if __name__ == "__main__":
    pose_dir = neutral_pose_file = assets_dir / "model/pose/"
    all_manually_specified_files = pose_dir.glob("_manual_specs/*.yaml")

    for path in all_manually_specified_files:
        pose_name = path.stem
        output_dir = pose_dir / pose_name
        output_dir.mkdir(exist_ok=True)

        manual_pose = KinematicPose(path=path)

        output_path = output_dir / f"{manual_pose.axis_order.to_str()}.yaml"
        shutil.copyfile(path, output_path)

        for target_axis_order in AxisOrder:
            if target_axis_order == manual_pose.axis_order:
                continue

            fitted_pose = convert_pose_axis_order(manual_pose, target_axis_order)
            launch_viewer(fitted_pose)
            output_path = output_dir / f"{target_axis_order.to_str()}.yaml"
            write_pose_yaml(fitted_pose, output_path)
