"""Generate the `flygym.ik` single-frame convergence demo video.

Builds a `NeuroMechFly` with its genuine neutral-pose keyframe as the fit's
starting pose, targets one frame of the bundled Spotlight motion-capture
clip, and renders `render_ik_convergence_video`'s output to
`outputs/ik_demo/convergence_demo.mp4`.
"""

import warnings
from pathlib import Path

import mujoco as mj
import numpy as np
from loguru import logger

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose import NeuroMechFly
from flygym.compose.pose import KinematicPosePreset
from flygym.ik import KeypointSet, render_ik_convergence_video
from flygym_demo.spotlight_data.preprocessing import MotionSnippet

OUTPUT_PATH = Path(__file__).parents[2] / "outputs" / "ik_demo" / "convergence_demo.mp4"
TARGET_FRAME = 100


def _qpos_from_snippet_frame(mj_model, jointdofs, snippet, frame_idx):
    qpos = np.zeros(mj_model.nq)
    for dof in jointdofs:
        leg_idx = snippet.legs.index(dof.child.pos)
        dof_idx = snippet.dofs_per_leg.index(
            (dof.parent.link, dof.child.link, dof.axis.value)
        )
        joint_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_JOINT, dof.name)
        qpos[mj_model.jnt_qposadr[joint_id]] = snippet.joint_angles[
            frame_idx, leg_idx, dof_idx
        ]
    return qpos


def _fk_points(mj_model, mj_data, qpos, body_ids, local_offsets):
    mj_data.qpos[:] = qpos
    mj.mj_kinematics(mj_model, mj_data)
    xpos = mj_data.xpos[body_ids]
    xmat = mj_data.xmat[body_ids].reshape(-1, 3, 3)
    return xpos + np.einsum("nij,nj->ni", xmat, local_offsets)


def main():
    # Neutral keyframe (genuine biological standing pose) becomes the fit's
    # starting point via `render_ik_convergence_video`'s `initial_qpos=None`.
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.YAW_PITCH_ROLL
    )
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly = NeuroMechFly(name="ik_demo_fly")
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    fly.colorize()
    camera = fly.add_tracking_camera()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mj_model, mj_data = fly.compile()

    snippet = MotionSnippet()
    keypoints = KeypointSet.from_keypoint_triples(snippet.keypoints, mj_model=mj_model)
    body_ids = keypoints.resolve_body_ids(mj_model)
    local_offsets = keypoints.local_offsets()

    jointdofs = [
        dof
        for dof in skeleton.iter_jointdofs()
        if dof.child.link in {"coxa", "trochanterfemur", "tibia", "tarsus1"}
    ]
    target_qpos = _qpos_from_snippet_frame(mj_model, jointdofs, snippet, TARGET_FRAME)
    target_positions = _fk_points(
        mj_model, mj_data, target_qpos, body_ids, local_offsets
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = render_ik_convergence_video(
        mj_model,
        mj_data,
        keypoints,
        target_positions,
        str(OUTPUT_PATH),
        camera,
        initial_qpos=None,
        max_iters=20,
        fps=2,
        hold_final_frames=10,
    )
    logger.info(f"Converged in {result.n_iters} iterations, cost={result.cost:.4f}")
    logger.info(f"Video written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
