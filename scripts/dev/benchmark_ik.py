"""Benchmark CPU vs. Warp (GPU) throughput for flygym.ik trajectory fitting.

Fits a range of frame counts from the bundled Spotlight motion-capture clip
with both backends and reports frames/second.
"""

import time
import warnings

import mujoco as mj
import numpy as np
from loguru import logger

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose import NeuroMechFly
from flygym.ik import KeypointSet, fit_qpos_trajectory_to_keypoints
from flygym_demo.spotlight_data.preprocessing import MotionSnippet


def _build_model():
    all_leg_joints = JointPreset.LEGS_ONLY.to_joint_list()
    active_links = {"coxa", "trochanterfemur", "tibia", "tarsus1"}
    filtered = [j for j in all_leg_joints if j.child.link in active_links]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=filtered)
    fly = NeuroMechFly(name="benchmark_ik")
    fly.add_joints(skeleton)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mj_model, mj_data = fly.compile()
    return mj_model, mj_data, skeleton


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


if __name__ == "__main__":
    logger.remove()  # quiet per-frame/per-fit INFO logs during timing

    mj_model, mj_data, skeleton = _build_model()
    jointdofs = list(skeleton.iter_jointdofs())
    snippet = MotionSnippet()
    keypoints = KeypointSet.from_keypoint_triples(snippet.keypoints, mj_model=mj_model)

    initial_qpos = _qpos_from_snippet_frame(mj_model, jointdofs, snippet, 0)

    frame_counts = [10, 30, 100, 300, 600]
    print(f"{'n_frames':>10} {'backend':>8} {'time_s':>10} {'frames/s':>10}")
    for n_frames in frame_counts:
        targets = snippet.rawpred_egoxyz[:n_frames].astype(np.float32)

        t0 = time.perf_counter()
        fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets,
            mj_data=mj_data,
            initial_qpos=initial_qpos,
            backend="cpu",
            max_iters=50,
        )
        cpu_time = time.perf_counter() - t0
        print(
            f"{n_frames:>10} {'cpu':>8} {cpu_time:>10.3f} {n_frames / cpu_time:>10.1f}"
        )

        # First warp call includes one-time kernel JIT compilation; run once to
        # warm the cache before timing.
        fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets[:1],
            initial_qpos=initial_qpos,
            backend="warp",
            max_iters=30,
        )
        t0 = time.perf_counter()
        fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets,
            initial_qpos=initial_qpos,
            backend="warp",
            max_iters=30,
        )
        warp_time = time.perf_counter() - t0
        print(
            f"{n_frames:>10} {'warp':>8} {warp_time:>10.3f} {n_frames / warp_time:>10.1f}"
        )
