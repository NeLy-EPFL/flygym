"""Tests for the GPU-batched IK backend (flygym.ik.warp_solve)."""

import warnings

import mujoco as mj
import numpy as np
import pytest

# These tests require the optional warp (GPU) extra; tag them so they can be
# excluded with ``-m "not warp"``, and skip the whole module if warp is absent.
pytestmark = pytest.mark.warp
wp = pytest.importorskip("warp")

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose import NeuroMechFly
from flygym.ik import (
    KeypointSet,
    fit_qpos_to_keypoints,
    fit_qpos_trajectory_to_keypoints,
    seqikpy_joint_bounds,
)
from flygym.ik.solve import _keypoint_world_points


@pytest.fixture(scope="module")
def active_dofs_model():
    """A compiled NeuroMechFly restricted to the 7 actuated DOFs/leg (coxa,
    trochanterfemur, tibia, tarsus1) -- matches the fixture of the same name
    in tests/core/test_ik.py, kept as its own copy since warp tests are
    collected as a separate module."""
    all_leg_joints = JointPreset.LEGS_ONLY.to_joint_list()
    active_links = {"coxa", "trochanterfemur", "tibia", "tarsus1"}
    filtered = [j for j in all_leg_joints if j.child.link in active_links]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=filtered)
    fly = NeuroMechFly(name="warp_ik_fly")
    fly.add_joints(skeleton)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mj_model, mj_data = fly.compile()
    return mj_model, mj_data


def _random_qpos_within_limits(mj_model, rng):
    qpos = np.zeros(mj_model.nq)
    for joint_id in range(mj_model.njnt):
        qposadr = mj_model.jnt_qposadr[joint_id]
        if mj_model.jnt_limited[joint_id]:
            lower, upper = mj_model.jnt_range[joint_id]
            qpos[qposadr] = rng.uniform(lower, upper)
        else:
            qpos[qposadr] = rng.uniform(-0.3, 0.3)
    return qpos


_ALL_LEGS_TRIPLES = [
    (leg, parent, child)
    for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]
    for parent, child in [
        ("thorax", "coxa"),
        ("coxa", "trochanterfemur"),
        ("trochanterfemur", "tibia"),
        ("tibia", "tarsus1"),
    ]
]


class TestFitQposTrajectoryToKeypointsWarp:
    def test_recovers_known_qpos_noiseless(self, active_dofs_model):
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(0)
        n_frames = 8
        qpos_true = np.stack(
            [_random_qpos_within_limits(mj_model, rng) for _ in range(n_frames)]
        )
        targets = np.empty((n_frames, len(body_ids), 3))
        for t in range(n_frames):
            mj_data.qpos[:] = qpos_true[t]
            mj.mj_kinematics(mj_model, mj_data)
            targets[t] = _keypoint_world_points(mj_data, body_ids, local_offsets)

        qpos_fit = fit_qpos_trajectory_to_keypoints(
            mj_model, keypoints, targets, backend="warp", max_iters=40
        )
        assert qpos_fit.shape == (n_frames, mj_model.nq)

        for t in range(n_frames):
            mj_data.qpos[:] = qpos_fit[t]
            mj.mj_kinematics(mj_model, mj_data)
            fitted = _keypoint_world_points(mj_data, body_ids, local_offsets)
            np.testing.assert_allclose(fitted, targets[t], atol=1e-3)

    def test_matches_cpu_backend_on_the_same_problem(self, active_dofs_model):
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(1)
        qpos_true = _random_qpos_within_limits(mj_model, rng)
        mj_data.qpos[:] = qpos_true
        mj.mj_kinematics(mj_model, mj_data)
        target = _keypoint_world_points(mj_data, body_ids, local_offsets)

        cpu_result = fit_qpos_to_keypoints(
            mj_model, mj_data, keypoints, target, initial_qpos=np.zeros(mj_model.nq)
        )
        warp_qpos = fit_qpos_trajectory_to_keypoints(
            mj_model, keypoints, target[None], backend="warp", max_iters=40
        )[0]

        mj_data.qpos[:] = cpu_result.qpos
        mj.mj_kinematics(mj_model, mj_data)
        cpu_fitted = _keypoint_world_points(mj_data, body_ids, local_offsets)
        mj_data.qpos[:] = warp_qpos
        mj.mj_kinematics(mj_model, mj_data)
        warp_fitted = _keypoint_world_points(mj_data, body_ids, local_offsets)

        np.testing.assert_allclose(cpu_fitted, target, atol=1e-3)
        np.testing.assert_allclose(warp_fitted, target, atol=1e-3)

    def test_respects_joint_limits(self, active_dofs_model):
        mj_model, _ = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        n_frames = 4
        far_targets = np.tile(
            [100.0, 100.0, 100.0], (n_frames, len(keypoints.targets), 1)
        )
        bounds = seqikpy_joint_bounds(mj_model)
        qpos_fit = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            far_targets,
            backend="warp",
            bounds=bounds,
            max_iters=20,
        )
        lower, upper = bounds
        assert np.all(qpos_fit >= lower - 1e-4)
        assert np.all(qpos_fit <= upper + 1e-4)

    def test_2d_projection_axis_drop(self, active_dofs_model):
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(2)
        qpos_true = _random_qpos_within_limits(mj_model, rng)
        mj_data.qpos[:] = qpos_true
        mj.mj_kinematics(mj_model, mj_data)
        target_3d = _keypoint_world_points(mj_data, body_ids, local_offsets)
        target_2d = target_3d[:, (0, 1)]

        qpos_fit = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            target_2d[None],
            backend="warp",
            projection_axes=(0, 1),
            max_iters=40,
        )[0]
        mj_data.qpos[:] = qpos_fit
        mj.mj_kinematics(mj_model, mj_data)
        fitted = _keypoint_world_points(mj_data, body_ids, local_offsets)
        np.testing.assert_allclose(fitted[:, (0, 1)], target_2d, atol=1e-3)

    def test_invalid_backend_raises(self, active_dofs_model):
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        targets = np.zeros((1, len(keypoints.targets), 3))
        with pytest.raises(ValueError, match="backend"):
            fit_qpos_trajectory_to_keypoints(
                mj_model, keypoints, targets, mj_data=mj_data, backend="nonexistent"
            )

    def test_batch_size_chunking_matches_single_batch(self, active_dofs_model):
        """Splitting into chunks (batch_size < n_frames) should fit each
        frame's keypoints just as well as solving everything as one batch --
        compared in keypoint-position space (what IK actually guarantees),
        not raw qpos, since a chain with redundant DOFs can have more than
        one qpos reproducing the same keypoint positions; tiny floating-point
        differences between batch sizes are enough to tip a frame into a
        different (but equally valid) one."""
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(3)
        n_frames = 10
        qpos_true = np.stack(
            [_random_qpos_within_limits(mj_model, rng) for _ in range(n_frames)]
        )
        targets = np.empty((n_frames, len(body_ids), 3), dtype=np.float32)
        for t in range(n_frames):
            mj_data.qpos[:] = qpos_true[t]
            mj.mj_kinematics(mj_model, mj_data)
            targets[t] = _keypoint_world_points(mj_data, body_ids, local_offsets)

        single_batch = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets,
            backend="warp",
            warp_batch_size=None,
            max_iters=40,
        )
        chunked = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets,
            backend="warp",
            warp_batch_size=3,
            max_iters=40,
        )
        assert chunked.shape == single_batch.shape == (n_frames, mj_model.nq)

        for qpos_fit in (single_batch, chunked):
            for t in range(n_frames):
                mj_data.qpos[:] = qpos_fit[t]
                mj.mj_kinematics(mj_model, mj_data)
                fitted = _keypoint_world_points(mj_data, body_ids, local_offsets)
                np.testing.assert_allclose(fitted, targets[t], atol=1e-3)

    def test_graph_capture_matches_uncaptured(self, active_dofs_model):
        mj_model, mj_data = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES)
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(4)
        qpos_true = _random_qpos_within_limits(mj_model, rng)
        mj_data.qpos[:] = qpos_true
        mj.mj_kinematics(mj_model, mj_data)
        target = _keypoint_world_points(mj_data, body_ids, local_offsets)[None]

        captured = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            target,
            backend="warp",
            warp_use_graph_capture=True,
            max_iters=40,
        )
        uncaptured = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            target,
            backend="warp",
            warp_use_graph_capture=False,
            max_iters=40,
        )
        np.testing.assert_allclose(captured, uncaptured, atol=1e-4)
