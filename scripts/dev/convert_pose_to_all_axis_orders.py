import shutil
from pathlib import Path

import yaml
import mujoco
import numpy as np
import scipy.optimize
from loguru import logger

from flygym import assets_dir
from flygym.anatomy import (
    Skeleton,
    JointPreset,
    JointDOF,
    RotationAxis,
    AxisOrder,
    ActuatedDOFPreset,
    ContactBodiesPreset,
)
from flygym.compose import Fly, ActuatorType, KinematicPose, FlatGroundWorld
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D

joint_preset = JointPreset.ALL_BIOLOGICAL
actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
actuator_type = ActuatorType.POSITION
actuator_position_gain = 50.0
spawn_position = (0, 0, 0.8)  # xyz in mm
spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))  # wxyz in quaternion
bodysegs_with_ground_contact = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD


def fit_pose_to_skeleton(
    goal_pose: KinematicPose,
    target_skeleton: Skeleton,
    fitting_pos_weight: float = 1.0,
    fitting_rot_weight: float = 1.0,
    n_iters: int = 100,
) -> KinematicPose:
    if goal_pose.axis_order == target_skeleton.axis_order:
        return goal_pose.copy()

    ref_fly = Fly(name="__temp_ref_fly")

    ref_skeleton = Skeleton(
        axis_order=goal_pose.axis_order,
        anatomical_joints=target_skeleton.anatomical_joints,
    )

    ref_jointdof_to_angle = {}
    for jointdof in ref_skeleton.iter_jointdofs(ref_fly.root_segment):
        angle_rad = goal_pose.joint_angles_lookup_rad.get(jointdof.name, 0.0)
        ref_jointdof_to_angle[jointdof] = angle_rad

        # Flip axis direction for right side's roll and yaw so that axes are defined
        # symmetrically (e.g., positive roll is always "outward").
        vec = np.array(jointdof.axis.to_vector())
        if jointdof.child.pos[0] == "r" and jointdof.axis != RotationAxis.PITCH:
            vec = -vec

        child_body = ref_fly.bodyseg_to_mjcfbody[jointdof.child]
        ref_fly.jointdof_to_mjcfjoint[jointdof] = child_body.add(
            "joint", name=jointdof.name, type="hinge", axis=vec
        )

    ref_mj_model, ref_mj_data = ref_fly.compile()
    ref_qpos = np.zeros(ref_mj_model.nq)
    for jointdof, angle in ref_jointdof_to_angle.items():
        mjcfjoint = ref_fly.jointdof_to_mjcfjoint[jointdof]
        internal_jointid = mujoco.mj_name2id(
            ref_mj_model, mujoco.mjtObj.mjOBJ_JOINT, mjcfjoint.full_identifier
        )
        qposadr = ref_mj_model.jnt_qposadr[internal_jointid]
        ref_qpos[qposadr] = angle

    ref_mj_data.qpos[:] = ref_qpos
    mujoco.mj_kinematics(ref_mj_model, ref_mj_data)
    ref_xpos = ref_mj_data.xpos.copy()
    ref_xquat = ref_mj_data.xquat.copy()

    target_fly = Fly(name="__temp_target_fly")

    for jointdof in target_skeleton.iter_jointdofs(target_fly.root_segment):
        # Flip axis direction for right side's roll and yaw so that axes are defined
        # symmetrically (e.g., positive roll is always "outward").
        vec = np.array(jointdof.axis.to_vector())
        if jointdof.child.pos[0] == "r" and jointdof.axis != RotationAxis.PITCH:
            vec = -vec

        child_body = target_fly.bodyseg_to_mjcfbody[jointdof.child]
        target_fly.jointdof_to_mjcfjoint[jointdof] = child_body.add(
            "joint", name=jointdof.name, type="hinge", axis=vec
        )

    target_mj_model, target_mj_data = target_fly.compile()

    # Check if the two models have the same body/geom ordering
    if ref_mj_model.nbody != target_mj_model.nbody:
        raise ValueError("Reference and target models have different number of bodies.")
    for internal_bodyname in range(ref_mj_model.nbody):
        ref_bodyname = mujoco.mj_id2name(
            ref_mj_model, mujoco.mjtObj.mjOBJ_BODY, internal_bodyname
        )
        target_bodyname = mujoco.mj_id2name(
            target_mj_model, mujoco.mjtObj.mjOBJ_BODY, internal_bodyname
        )
        if ref_bodyname != target_bodyname:
            raise ValueError(
                "Inconsistent body ordering between reference and target models."
            )
    if ref_mj_model.ngeom != target_mj_model.ngeom:
        raise ValueError("Reference and target models have different number of geoms.")
    for internal_geomname in range(ref_mj_model.ngeom):
        ref_geomname = mujoco.mj_id2name(
            ref_mj_model, mujoco.mjtObj.mjOBJ_GEOM, internal_geomname
        )
        target_geomname = mujoco.mj_id2name(
            target_mj_model, mujoco.mjtObj.mjOBJ_GEOM, internal_geomname
        )
        if ref_geomname != target_geomname:
            raise ValueError(
                "Inconsistent geom ordering between reference and target models."
            )

    _cost_hist = []

    def objective(fitted_qpos):
        target_mj_data.qpos[:] = fitted_qpos
        mujoco.mj_kinematics(target_mj_model, target_mj_data)
        fitted_xpos = target_mj_data.xpos.copy()
        fitted_xquat = target_mj_data.xquat.copy()

        cost = 0
        for bodyid in range(ref_mj_model.nbody):
            dpos = fitted_xpos[bodyid] - ref_xpos[bodyid]
            pos_cost = np.dot(dpos, dpos) ** 2
            cost += fitting_pos_weight * pos_cost

            q1 = ref_xquat[bodyid] / np.linalg.norm(ref_xquat[bodyid])
            q2 = fitted_xquat[bodyid] / np.linalg.norm(fitted_xquat[bodyid])
            dquat = np.dot(q1, q2)
            dquat = np.abs(dquat)  # quaternion difference invariant to double cover
            dquat = np.clip(dquat, -1.0, 1.0)  # avoid numerical issues
            rot_cost = 1 - dquat**2  # equiv to sin^2(theta/2)
            cost += fitting_rot_weight * rot_cost

        _cost_hist.append(cost)
        return cost

    joint_limits_lower = target_mj_model.jnt_range[:, 0].copy()
    joint_limits_upper = target_mj_model.jnt_range[:, 1].copy()
    for i in range(target_mj_model.njnt):
        if not target_mj_model.jnt_limited[i]:
            joint_limits_lower[i] = -np.inf
            joint_limits_upper[i] = np.inf
    joint_limits = list(zip(joint_limits_lower, joint_limits_upper))

    initial_qpos = np.zeros(target_mj_model.nq)
    logger.info(f"Optimizing pose fit to skeleton (n_iters={n_iters})...")
    result = scipy.optimize.minimize(
        objective,
        initial_qpos,
        method="L-BFGS-B",
        bounds=joint_limits,
        options={"maxiter": n_iters, "ftol": 1e-6, "gtol": 1e-6},
    )
    solved_qpos = result.x
    final_cost = result.fun
    logger.info(
        f"Pose fitting optimization finished with final cost {final_cost:.6f} "
        f"(initial cost { _cost_hist[0]:.6f})"
    )

    fitted_joint_angles_rad_dict = {}
    for internal_jointid in range(target_mj_model.njnt):
        qposadr = target_mj_model.jnt_qposadr[internal_jointid]
        solved_angle = solved_qpos[qposadr]
        joint_name = mujoco.mj_id2name(
            target_mj_model, mujoco.mjtObj.mjOBJ_JOINT, internal_jointid
        )
        jointdof = JointDOF.from_name(joint_name)
        if jointdof.child.pos[0] != "r":
            fitted_joint_angles_rad_dict[joint_name] = solved_angle

    fitted_pose = KinematicPose(
        joint_angles_rad_dict=fitted_joint_angles_rad_dict,
        axis_order=target_skeleton.axis_order,
        mirror_left2right=True,
    )
    return fitted_pose


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
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qposadr = mj_model.jnt_qposadr[jid]
        final_joint_angles_rad_dict[joint_name] = mj_data.qpos[qposadr]

    return final_joint_angles_rad_dict


def write_pose_yaml(joint_angles_rad_dict, axis_order, output_path):
    # Sort joint angles by joint name for consistency
    joint_angles_rad_dict = dict(sorted(joint_angles_rad_dict.items()))

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
        "axis_order": axis_order.to_list_of_str(),
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

        for axis_order in AxisOrder:
            if axis_order == manual_pose.axis_order:
                continue

            target_skeleton = Skeleton(joint_preset=joint_preset, axis_order=axis_order)
            fitted_pose = fit_pose_to_skeleton(manual_pose, target_skeleton)
            launch_viewer(fitted_pose)
            output_path = output_dir / f"{axis_order.to_str()}.yaml"
            write_pose_yaml(
                fitted_pose.joint_angles_lookup_rad, axis_order, output_path
            )
