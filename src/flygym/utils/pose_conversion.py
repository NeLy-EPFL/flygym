import mujoco as mj
import numpy as np
import scipy.optimize
from loguru import logger

from flygym.anatomy import Skeleton, JointPreset, JointDOF, AxisOrder
from flygym.compose import Fly, KinematicPose


def get_body_names(mj_model: mj.MjModel):
    """Return a list of body names in the order they appear in the model."""
    return [
        mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, bid)
        for bid in range(mj_model.nbody)
    ]


def get_xpos0_xquat0(
    mj_model: mj.MjModel, mj_data: mj.MjData
) -> tuple[np.ndarray, np.ndarray]:
    """Return body positions and quaternions at keyframe 0.

    Resets to keyframe 0 and runs forward kinematics.

    Returns:
        Tuple of ``(xpos, xquat)`` arrays, each shape ``(n_bodies, 3)`` and
        ``(n_bodies, 4)`` respectively.
    """
    mj.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj.mj_kinematics(mj_model, mj_data)
    xpos = mj_data.xpos.copy()
    xquat = mj_data.xquat.copy()
    return xpos, xquat


def fit_qpos_to_xpos_xquat(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    target_xpos: np.ndarray,
    target_xquat: np.ndarray,
    fitting_pos_weight: float = 1.0,
    fitting_rot_weight: float = 1.0,
    max_iters: int = 100,
) -> np.ndarray:
    """Fit joint angles (qpos) to match target body positions and orientations.

    Uses L-BFGS-B optimization to minimize a weighted sum of position and rotation
    errors across all bodies.

    Args:
        mj_model: Compiled MuJoCo model.
        mj_data: Associated MuJoCo data (modified in place during optimization).
        target_xpos: Target global body positions, shape ``(n_bodies, 3)``.
        target_xquat: Target global body quaternions, shape ``(n_bodies, 4)``.
        fitting_pos_weight: Weight applied to positional error.
        fitting_rot_weight: Weight applied to rotational error.
        max_iters: Maximum number of optimizer iterations.

    Returns:
        Optimized qpos array, shape ``(nq,)``.
    """
    _cost_hist = []

    def objective(qpos0):
        mj_data.qpos[:] = qpos0
        mj.mj_kinematics(mj_model, mj_data)
        fitted_xpos = mj_data.xpos.copy()
        fitted_xquat = mj_data.xquat.copy()

        cost = 0
        for bodyid in range(mj_model.nbody):
            dpos = fitted_xpos[bodyid] - target_xpos[bodyid]
            pos_cost = np.dot(dpos, dpos) ** 2
            cost += fitting_pos_weight * pos_cost

            q1 = target_xquat[bodyid] / np.linalg.norm(target_xquat[bodyid])
            q2 = fitted_xquat[bodyid] / np.linalg.norm(fitted_xquat[bodyid])
            dquat = np.dot(q1, q2)
            dquat = np.abs(dquat)  # quaternion difference invariant to double cover
            dquat = np.clip(dquat, -1.0, 1.0)  # avoid numerical issues
            rot_cost = 1 - dquat**2  # equiv to sin^2(theta/2)
            cost += fitting_rot_weight * rot_cost

        _cost_hist.append(cost)
        return cost

    # Get joint limits
    joint_limits_lower = mj_model.jnt_range[:, 0].copy()
    joint_limits_upper = mj_model.jnt_range[:, 1].copy()
    for i in range(mj_model.njnt):
        if not mj_model.jnt_limited[i]:
            joint_limits_lower[i] = -np.inf
            joint_limits_upper[i] = np.inf
    joint_limits = list(zip(joint_limits_lower, joint_limits_upper))

    initial_qpos = np.zeros(mj_model.nq)

    logger.info(f"Optimizing pose fit to skeleton (max_iters={max_iters})...")
    result = scipy.optimize.minimize(
        objective,
        initial_qpos,
        method="L-BFGS-B",
        bounds=joint_limits,
        options={"maxiter": max_iters, "ftol": 1e-6, "gtol": 1e-6},
    )
    solved_qpos = result.x
    final_cost = result.fun
    logger.info(
        f"Pose fitting optimization finished with final cost {final_cost:.6f} "
        f"(initial cost { _cost_hist[0]:.6f})"
    )

    return solved_qpos


def qpos_to_kinematic_pose(
    mj_model: mj.MjModel, qpos: np.ndarray, axis_order: AxisOrder
) -> KinematicPose:
    """Convert a qpos array to a `KinematicPose` using left-side joints only.

    Right-side angles are populated by mirroring.

    Args:
        mj_model: Compiled MuJoCo model.
        qpos: Joint position vector, shape ``(nq,)``.
        axis_order: The axis order used for the model's joints.

    Returns:
        A `KinematicPose` with left-side angles and mirroring enabled.
    """
    fitted_joint_angles_rad_dict = {}
    for internal_jointid in range(mj_model.njnt):
        qposadr = mj_model.jnt_qposadr[internal_jointid]
        solved_angle = qpos[qposadr]
        joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, internal_jointid)
        jointdof = JointDOF.from_name(joint_name)
        if jointdof.child.pos[0] != "r":
            fitted_joint_angles_rad_dict[joint_name] = solved_angle

    return KinematicPose(
        joint_angles_rad_dict=fitted_joint_angles_rad_dict,
        axis_order=axis_order,
        mirror_left2right=True,
    )


def convert_pose_axis_order(
    pose: KinematicPose,
    target_axis_order: AxisOrder,
    joint_preset: JointPreset = JointPreset.ALL_BIOLOGICAL,
    ref_fly_kwargs: dict = dict(),
    fitted_fly_kwargs: dict = dict(),
) -> KinematicPose:
    """Convert a `KinematicPose` to a different joint axis order.

    Builds two fly models (one per axis order), fits the target-order model's joint
    angles to match the reference model's body poses using IK, and returns the
    converted pose.

    Args:
        pose: Input pose to convert.
        target_axis_order: Axis order for the output pose.
        joint_preset: Joint preset to use for both models.
        ref_fly_kwargs: Extra kwargs passed to the reference `Fly`.
        fitted_fly_kwargs: Extra kwargs passed to the target `Fly`.

    Returns:
        A `KinematicPose` in the target axis order.
    """
    ref_fly = Fly(**ref_fly_kwargs)
    ref_skeleton = Skeleton(axis_order=pose.axis_order, joint_preset=joint_preset)
    ref_fly.add_joints(ref_skeleton, neutral_pose=pose)
    ref_mj_model, ref_mj_data = ref_fly.compile()

    fitted_fly = Fly(**fitted_fly_kwargs)
    fitted_skeleton = Skeleton(axis_order=target_axis_order, joint_preset=joint_preset)
    fitted_fly.add_joints(fitted_skeleton, neutral_pose=pose)
    fitted_mj_model, fitted_mj_data = fitted_fly.compile()

    if get_body_names(ref_mj_model) != get_body_names(fitted_mj_model):
        raise RuntimeError("Fly models have different body names.")

    ref_xpos, ref_xquat = get_xpos0_xquat0(ref_mj_model, ref_mj_data)
    fitted_qpos0 = fit_qpos_to_xpos_xquat(
        mj_model=fitted_mj_model,
        mj_data=fitted_mj_data,
        target_xpos=ref_xpos,
        target_xquat=ref_xquat,
    )

    fitted_pose = qpos_to_kinematic_pose(
        mj_model=fitted_mj_model, qpos=fitted_qpos0, axis_order=target_axis_order
    )
    return fitted_pose
