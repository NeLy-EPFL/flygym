"""Native MuJoCo inverse kinematics: fit joint angles to keypoint positions.

Given tracked body keypoints (2D or 3D, e.g. from video pose tracking),
`flygym.ik` fits joint angles that reproduce the pose by minimizing weighted
squared distance across all requested keypoints -- not just leg tips --
similar to the external [SeqIKPy](https://nely-epfl.github.io/sequential-inverse-kinematics/)
package, but implemented natively with MuJoCo's own kinematics and Jacobian
machinery instead of `ikpy`.

Example:

    from flygym.anatomy import AxisOrder, JointPreset, Skeleton
    from flygym.compose import NeuroMechFly
    from flygym.ik import KeypointSet, fit_qpos_to_keypoints

    fly = NeuroMechFly()
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly.add_joints(skeleton)
    mj_model, mj_data = fly.compile()

    # (leg, parent_link, child_link) triples, e.g. from MotionSnippet.keypoints
    triples = [("lf", "thorax", "coxa"), ("lf", "coxa", "trochanterfemur"), ...]
    keypoints = KeypointSet.from_keypoint_triples(triples, mj_model=mj_model)
    keypoints.set_weight("lf-thorax-coxa", 0.1)  # down-weight a hard-to-track keypoint

    result = fit_qpos_to_keypoints(mj_model, mj_data, keypoints, target_positions)
"""

from .keypoints import KeypointTarget, KeypointSet, estimate_terminal_offset
from .seqikpy_defaults import seqikpy_joint_bounds, seqikpy_initial_guess_qpos
from .solve import IKResult, fit_qpos_to_keypoints, fit_qpos_trajectory_to_keypoints
from .visualize import render_ik_convergence_video

__all__ = [
    "KeypointTarget",
    "KeypointSet",
    "estimate_terminal_offset",
    "seqikpy_joint_bounds",
    "seqikpy_initial_guess_qpos",
    "IKResult",
    "fit_qpos_to_keypoints",
    "fit_qpos_trajectory_to_keypoints",
    "render_ik_convergence_video",
]
