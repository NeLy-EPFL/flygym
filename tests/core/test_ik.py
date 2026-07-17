"""Unit and end-to-end tests for flygym.ik (keypoint-based inverse kinematics)."""

import mujoco as mj
import numpy as np
import pytest

from flygym.anatomy import (
    PASSIVE_TARSAL_LINKS,
    AxisOrder,
    BodySegment,
    JointDOF,
    JointPreset,
    Skeleton,
)
from flygym.compose import NeuroMechFly
from flygym.compose.pose import KinematicPosePreset
from flygym.ik import (
    KeypointSet,
    KeypointTarget,
    estimate_terminal_offset,
    fit_qpos_to_keypoints,
    fit_qpos_trajectory_to_keypoints,
    seqikpy_joint_bounds,
    seqikpy_initial_guess_qpos,
)
from flygym_demo.spotlight_data.preprocessing import MotionSnippet


# ==============================================================================
# Fixtures and helpers
# ==============================================================================


@pytest.fixture(scope="module")
def legs_only_model():
    """A compiled NeuroMechFly with all biologically-plausible leg joints."""
    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY
    )
    fly = NeuroMechFly(name="ik_test_fly")
    fly.add_joints(skeleton, neutral_pose=pose)
    return fly.compile()


@pytest.fixture(scope="module")
def active_dofs_model():
    """A compiled NeuroMechFly restricted to the 7 actuated DOFs/leg used by
    the bundled experimental recordings (coxa, trochanterfemur, tibia,
    tarsus1) -- tarsus2-5 are left unjointed, i.e. rigidly fused to tarsus1.
    """
    all_leg_joints = JointPreset.LEGS_ONLY.to_joint_list()
    active_links = {"coxa", "trochanterfemur", "tibia", "tarsus1"}
    filtered = [j for j in all_leg_joints if j.child.link in active_links]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=filtered)
    fly = NeuroMechFly(name="ik_active_fly")
    fly.add_joints(skeleton)
    mj_model, mj_data = fly.compile()
    return mj_model, mj_data, skeleton


@pytest.fixture(scope="module")
def snippet():
    """The bundled experimental motion-capture clip used across tutorials."""
    return MotionSnippet()


def _fk_points(mj_model, mj_data, qpos, body_ids, local_offsets):
    """Forward-kinematics keypoint positions for a given qpos."""
    mj_data.qpos[:] = qpos
    mj.mj_kinematics(mj_model, mj_data)
    xpos = mj_data.xpos[body_ids]
    xmat = mj_data.xmat[body_ids].reshape(-1, 3, 3)
    return xpos + np.einsum("nij,nj->ni", xmat, local_offsets)


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


def _qpos_from_snippet_frame(mj_model, jointdofs, snippet, frame_idx):
    """Build a qpos vector directly from one frame of a MotionSnippet's raw
    joint angles (bypassing the smoothing/interpolation done by
    `MotionSnippet.get_joint_angles`, since exact per-frame values are
    wanted here)."""
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


_LEFT_FRONT_LEG_TRIPLES = [
    ("lf", "thorax", "coxa"),
    ("lf", "coxa", "trochanterfemur"),
    ("lf", "trochanterfemur", "tibia"),
    ("lf", "tibia", "tarsus1"),
    ("lf", "tarsus5", None),
]

_ALL_LEGS_TRIPLES = [
    (leg, parent, child)
    for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]
    for parent, child in [
        ("thorax", "coxa"),
        ("coxa", "trochanterfemur"),
        ("trochanterfemur", "tibia"),
        ("tibia", "tarsus1"),
        ("tarsus5", None),
    ]
]


# ==============================================================================
# KeypointTarget
# ==============================================================================


class TestKeypointTarget:
    def test_from_anatomical_names_interior_joint(self):
        target = KeypointTarget.from_anatomical_names("lf", "thorax", "coxa")
        assert target.body.name == "lf_coxa"
        np.testing.assert_array_equal(target.local_offset, np.zeros(3))
        assert target.name == "lf-thorax-coxa"

    def test_from_anatomical_names_terminal_requires_offset_or_model(self):
        with pytest.raises(ValueError, match="local_offset"):
            KeypointTarget.from_anatomical_names("lf", "tarsus5", None)

    def test_from_anatomical_names_terminal_with_explicit_offset(self):
        offset = np.array([0.0, 0.0, 0.05])
        target = KeypointTarget.from_anatomical_names(
            "lf", "tarsus5", None, local_offset=offset
        )
        assert target.body.name == "lf_tarsus5"
        np.testing.assert_array_equal(target.local_offset, offset)
        assert target.name == "lf-tarsus5-tip"

    def test_from_anatomical_names_terminal_with_model(self, legs_only_model):
        mj_model, _ = legs_only_model
        target = KeypointTarget.from_anatomical_names(
            "lf", "tarsus5", None, mj_model=mj_model
        )
        assert target.local_offset.shape == (3,)
        assert np.linalg.norm(target.local_offset) > 0

    def test_invalid_body_name_raises(self):
        with pytest.raises(ValueError):
            KeypointTarget.from_anatomical_names("xx", "thorax", "coxa")

    def test_local_offset_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            KeypointTarget(body=BodySegment("lf_coxa"), local_offset=np.zeros(2))


class TestEstimateTerminalOffset:
    def test_offset_is_nonzero_for_claw(self, legs_only_model):
        mj_model, _ = legs_only_model
        offset = estimate_terminal_offset(mj_model, BodySegment("lf_tarsus5"))
        assert np.linalg.norm(offset) > 0

    def test_offset_shape(self, legs_only_model):
        mj_model, _ = legs_only_model
        offset = estimate_terminal_offset(mj_model, BodySegment("rh_tarsus5"))
        assert offset.shape == (3,)

    def test_offset_is_more_distal_than_body_origin(self, legs_only_model):
        """The claw tip should be farther from the thorax than the tarsus5
        body's own origin (which sits at the tarsus4-tarsus5 joint, i.e. the
        *proximal* end) -- regression test for a bug where the offset was
        computed in the geom's local frame but applied as if it were already
        in the body's frame, silently pointing it in the wrong direction
        whenever the geom has a non-identity pos/quat relative to its body."""
        mj_model, mj_data = legs_only_model
        body = BodySegment("lf_tarsus5")
        offset = estimate_terminal_offset(mj_model, body)

        thorax_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, "c_thorax")
        tarsus5_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, body.name)
        mj.mj_kinematics(mj_model, mj_data)
        thorax_pos = mj_data.xpos[thorax_id]
        tarsus5_origin = mj_data.xpos[tarsus5_id]
        tarsus5_mat = mj_data.xmat[tarsus5_id].reshape(3, 3)
        tip = tarsus5_origin + tarsus5_mat @ offset

        assert np.linalg.norm(tip - thorax_pos) > np.linalg.norm(
            tarsus5_origin - thorax_pos
        )

    def test_offset_matches_independently_computed_farthest_body_frame_vertex(
        self, legs_only_model
    ):
        """Regression test for a bug where mesh vertices were ranked by
        distance in the *geom's* local frame (before transforming into the
        body frame), rather than the body frame. Since `geom_pos` for a leaf
        segment is typically comparable in magnitude to the segment's own
        length, that bug picks a vertex near the *proximal* joint instead of
        the distal tip -- this independently re-derives the correct answer
        (transform all vertices to the body frame, then rank) and checks
        `estimate_terminal_offset` agrees, rather than just checking the
        result points in a plausible general direction."""
        mj_model, _ = legs_only_model
        body = BodySegment("lf_tarsus5")
        body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, body.name)
        geom_id = mj_model.body_geomadr[body_id]
        mesh_id = mj_model.geom_dataid[geom_id]
        vert_start = mj_model.mesh_vertadr[mesh_id]
        vert_count = mj_model.mesh_vertnum[mesh_id]
        verts_geom_frame = mj_model.mesh_vert[
            vert_start : vert_start + vert_count
        ].astype(float)
        geom_rotmat = np.empty(9)
        mj.mju_quat2Mat(geom_rotmat, mj_model.geom_quat[geom_id])
        verts_body_frame = (
            verts_geom_frame @ geom_rotmat.reshape(3, 3).T + mj_model.geom_pos[geom_id]
        )
        expected = verts_body_frame[np.argmax(np.linalg.norm(verts_body_frame, axis=1))]

        offset = estimate_terminal_offset(mj_model, body)
        np.testing.assert_allclose(offset, expected)


# ==============================================================================
# seqikpy_defaults
# ==============================================================================


class TestSeqikpyDefaults:
    def test_bounds_same_for_left_and_right(self, active_dofs_model):
        """flygym mirrors the right-side axis internally, so a SeqIKPy DOF
        type's bounds should apply identically to both sides."""
        mj_model, _, _ = active_dofs_model
        lower, upper = seqikpy_joint_bounds(mj_model)
        for parent, child, axis in [
            ("thorax", "coxa", "roll"),
            ("coxa", "trochanterfemur", "pitch"),
            ("tibia", "tarsus1", "pitch"),
        ]:
            left_id = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, f"c_thorax-lf_{child}-{axis}"
            )
            right_id = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, f"c_thorax-rf_{child}-{axis}"
            )
            if left_id < 0:
                # Non-thorax-rooted joints (CTr, TiTa) have a different parent name.
                left_id = mj.mj_name2id(
                    mj_model, mj.mjtObj.mjOBJ_JOINT, f"lf_{parent}-lf_{child}-{axis}"
                )
                right_id = mj.mj_name2id(
                    mj_model, mj.mjtObj.mjOBJ_JOINT, f"rf_{parent}-rf_{child}-{axis}"
                )
            left_adr = mj_model.jnt_qposadr[left_id]
            right_adr = mj_model.jnt_qposadr[right_id]
            assert lower[left_adr] == pytest.approx(lower[right_adr])
            assert upper[left_adr] == pytest.approx(upper[right_adr])

    def test_bounds_finite_for_modeled_dofs(self, active_dofs_model):
        """Every DOF in the DOF-restricted model is one SeqIKPy models, so
        none should be left at the +/-inf default."""
        mj_model, _, _ = active_dofs_model
        lower, upper = seqikpy_joint_bounds(mj_model)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower < upper)

    def test_bounds_unbounded_for_unmodeled_dofs(self, legs_only_model):
        """legs_only_model includes tarsus2-5 joints, which SeqIKPy doesn't
        model -- those should be left unbounded."""
        mj_model, _ = legs_only_model
        lower, upper = seqikpy_joint_bounds(mj_model)
        joint_id = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_JOINT, "lf_tarsus1-lf_tarsus2-pitch"
        )
        qposadr = mj_model.jnt_qposadr[joint_id]
        assert lower[qposadr] == -np.inf
        assert upper[qposadr] == np.inf

    def test_initial_guess_qpos_within_bounds(self, active_dofs_model):
        mj_model, _, _ = active_dofs_model
        lower, upper = seqikpy_joint_bounds(mj_model)
        initial_guess = seqikpy_initial_guess_qpos(mj_model)
        assert np.all(initial_guess >= lower)
        assert np.all(initial_guess <= upper)

    def test_initial_guess_qpos_zero_for_unmodeled_dofs(self, legs_only_model):
        mj_model, _ = legs_only_model
        initial_guess = seqikpy_initial_guess_qpos(mj_model)
        joint_id = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_JOINT, "lf_tarsus1-lf_tarsus2-pitch"
        )
        assert initial_guess[mj_model.jnt_qposadr[joint_id]] == 0.0

    def test_fit_with_seqikpy_bounds_and_initial_guess(self, active_dofs_model):
        """End-to-end: fitting with SeqIKPy's bounds/initial guess together
        should converge to a pose that both matches the targets and respects
        the bounds."""
        mj_model, mj_data, _ = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(
            _ALL_LEGS_TRIPLES, mj_model=mj_model
        )
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        bounds = seqikpy_joint_bounds(mj_model)
        initial_guess = seqikpy_initial_guess_qpos(mj_model)
        lower, upper = bounds
        rng = np.random.default_rng(0)
        qpos_true = lower + rng.uniform(0.3, 0.7, size=lower.shape) * (upper - lower)

        targets = _fk_points(mj_model, mj_data, qpos_true, body_ids, local_offsets)
        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            keypoints,
            targets,
            initial_qpos=initial_guess,
            bounds=bounds,
        )
        assert result.success
        fitted = _fk_points(mj_model, mj_data, result.qpos, body_ids, local_offsets)
        np.testing.assert_allclose(fitted, targets, atol=1e-3)
        assert np.all(result.qpos >= lower - 1e-6)
        assert np.all(result.qpos <= upper + 1e-6)


# ==============================================================================
# KeypointSet
# ==============================================================================


class TestKeypointSet:
    def _front_leg_targets(self):
        return [
            KeypointTarget.from_anatomical_names(*triple)
            for triple in _LEFT_FRONT_LEG_TRIPLES[:4]
        ]

    def test_default_weights_are_one(self):
        keypoints = KeypointSet(targets=self._front_leg_targets())
        np.testing.assert_array_equal(keypoints.weights, np.ones(4))

    def test_set_weight_updates_in_place(self):
        keypoints = KeypointSet(targets=self._front_leg_targets())
        keypoints.set_weight("lf-thorax-coxa", 0.1)
        assert keypoints.weights[0] == pytest.approx(0.1)

    def test_set_weight_unknown_name_raises(self):
        keypoints = KeypointSet(targets=self._front_leg_targets())
        with pytest.raises(ValueError, match="No keypoint named"):
            keypoints.set_weight("nonexistent", 1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            KeypointSet(
                targets=self._front_leg_targets(),
                weights=np.array([-1.0, 1.0, 1.0, 1.0]),
            )

    def test_resolve_body_ids_matches_model(self, legs_only_model):
        mj_model, _ = legs_only_model
        keypoints = KeypointSet(targets=self._front_leg_targets())
        body_ids = keypoints.resolve_body_ids(mj_model)
        expected = [
            mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, name)
            for name in ["lf_coxa", "lf_trochanterfemur", "lf_tibia", "lf_tarsus1"]
        ]
        np.testing.assert_array_equal(body_ids, expected)

    def test_from_keypoint_triples(self, legs_only_model):
        mj_model, _ = legs_only_model
        keypoints = KeypointSet.from_keypoint_triples(
            _LEFT_FRONT_LEG_TRIPLES, mj_model=mj_model
        )
        assert len(keypoints.targets) == 5
        assert keypoints.targets[-1].name == "lf-tarsus5-tip"


# ==============================================================================
# fit_qpos_to_keypoints (synthetic self-consistency)
# ==============================================================================


class TestFitQposToKeypoints:
    @pytest.fixture
    def all_leg_keypoints(self, legs_only_model):
        mj_model, _ = legs_only_model
        return KeypointSet.from_keypoint_triples(_ALL_LEGS_TRIPLES, mj_model=mj_model)

    def test_recovers_known_qpos_noiseless(self, legs_only_model, all_leg_keypoints):
        mj_model, mj_data = legs_only_model
        rng = np.random.default_rng(0)
        qpos_true = _random_qpos_within_limits(mj_model, rng)

        body_ids = all_leg_keypoints.resolve_body_ids(mj_model)
        local_offsets = all_leg_keypoints.local_offsets()
        targets = _fk_points(mj_model, mj_data, qpos_true, body_ids, local_offsets)

        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            targets,
            initial_qpos=np.zeros(mj_model.nq),
        )
        assert result.success
        fitted = _fk_points(mj_model, mj_data, result.qpos, body_ids, local_offsets)
        np.testing.assert_allclose(fitted, targets, atol=1e-4)

    def test_respects_joint_limits(self, legs_only_model, all_leg_keypoints):
        mj_model, mj_data = legs_only_model
        body_ids = all_leg_keypoints.resolve_body_ids(mj_model)
        far_targets = np.tile([100.0, 100.0, 100.0], (len(body_ids), 1))

        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            far_targets,
            initial_qpos=np.zeros(mj_model.nq),
            max_iters=50,
        )

        limited = mj_model.jnt_limited.astype(bool)
        qposadr = mj_model.jnt_qposadr[limited]
        lower, upper = mj_model.jnt_range[limited, 0], mj_model.jnt_range[limited, 1]
        assert np.all(result.qpos[qposadr] >= lower - 1e-8)
        assert np.all(result.qpos[qposadr] <= upper + 1e-8)

    def test_passive_tarsal_joints_stay_within_default_bound(
        self, legs_only_model, all_leg_keypoints
    ):
        """The four inter-tarsal joints per leg (tarsus1-tarsus2 through
        tarsus4-tarsus5) have no `jnt_range` set on the compiled model, but
        `_joint_bounds` should still clamp them to +/-10 degrees by default
        (they have no actuator of their own and are otherwise free to take
        on an arbitrary bend) -- unlike tibia-tarsus1, which is unbounded."""
        mj_model, mj_data = legs_only_model
        body_ids = all_leg_keypoints.resolve_body_ids(mj_model)
        far_targets = np.tile([100.0, 100.0, 100.0], (len(body_ids), 1))

        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            far_targets,
            initial_qpos=np.zeros(mj_model.nq),
            max_iters=50,
        )

        for joint_id in range(mj_model.njnt):
            joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            try:
                dof = JointDOF.from_name(joint_name)
            except ValueError:
                continue
            if dof.child.link not in PASSIVE_TARSAL_LINKS:
                continue
            qposadr = mj_model.jnt_qposadr[joint_id]
            assert -np.radians(10.0) - 1e-8 <= result.qpos[qposadr]
            assert result.qpos[qposadr] <= np.radians(10.0) + 1e-8

    def test_2d_projection_axis_drop(self, legs_only_model, all_leg_keypoints):
        mj_model, mj_data = legs_only_model
        rng = np.random.default_rng(1)
        qpos_true = _random_qpos_within_limits(mj_model, rng)
        body_ids = all_leg_keypoints.resolve_body_ids(mj_model)
        local_offsets = all_leg_keypoints.local_offsets()
        targets_3d = _fk_points(mj_model, mj_data, qpos_true, body_ids, local_offsets)
        targets_2d = targets_3d[:, (0, 1)]

        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            targets_2d,
            initial_qpos=np.zeros(mj_model.nq),
            projection_axes=(0, 1),
        )
        fitted = _fk_points(mj_model, mj_data, result.qpos, body_ids, local_offsets)
        np.testing.assert_allclose(fitted[:, (0, 1)], targets_2d, atol=1e-4)

    def test_weighting_deemphasizes_keypoint(self, legs_only_model, all_leg_keypoints):
        mj_model, mj_data = legs_only_model
        rng = np.random.default_rng(2)
        qpos_true = _random_qpos_within_limits(mj_model, rng)
        body_ids = all_leg_keypoints.resolve_body_ids(mj_model)
        local_offsets = all_leg_keypoints.local_offsets()
        targets = _fk_points(
            mj_model, mj_data, qpos_true, body_ids, local_offsets
        ).copy()
        corrupted_name = all_leg_keypoints.targets[0].name
        targets[0] += np.array([5.0, 0.0, 0.0])

        def other_keypoint_error(qpos):
            fitted = _fk_points(mj_model, mj_data, qpos, body_ids, local_offsets)
            return np.linalg.norm(fitted[1:] - targets[1:], axis=1).max()

        all_leg_keypoints.set_weight(corrupted_name, 1.0)
        result_full_weight = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            targets,
            initial_qpos=np.zeros(mj_model.nq),
        )
        all_leg_keypoints.set_weight(corrupted_name, 1e-6)
        result_down_weighted = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            all_leg_keypoints,
            targets,
            initial_qpos=np.zeros(mj_model.nq),
        )
        all_leg_keypoints.set_weight(corrupted_name, 1.0)  # restore

        assert other_keypoint_error(result_down_weighted.qpos) < other_keypoint_error(
            result_full_weight.qpos
        )


# ==============================================================================
# fit_qpos_trajectory_to_keypoints
# ==============================================================================


class TestFitQposTrajectoryToKeypoints:
    def test_output_shape(self, legs_only_model):
        mj_model, mj_data = legs_only_model
        keypoints = KeypointSet.from_keypoint_triples(
            _LEFT_FRONT_LEG_TRIPLES, mj_model=mj_model
        )
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()
        rng = np.random.default_rng(3)
        n_frames = 4
        targets = np.stack(
            [
                _fk_points(
                    mj_model,
                    mj_data,
                    _random_qpos_within_limits(mj_model, rng),
                    body_ids,
                    local_offsets,
                )
                for _ in range(n_frames)
            ]
        )
        qpos_traj = fit_qpos_trajectory_to_keypoints(
            mj_model, keypoints, targets, mj_data=mj_data
        )
        assert qpos_traj.shape == (n_frames, mj_model.nq)

    def test_warm_starting_keeps_trajectory_smooth(self, active_dofs_model):
        """A trajectory fit via `fit_qpos_trajectory_to_keypoints` (which
        warm-starts each frame from the previous frame's solution) should
        track a slowly-varying target smoothly, with small qpos changes
        between consecutive frames -- rather than jumping to a different,
        equally-valid configuration from frame to frame, which is the
        failure mode warm-starting avoids.

        Uses the DOF-restricted model (rather than `legs_only_model`, which
        includes tarsus2-4 joints with no keypoint of their own to constrain
        them) so the fit is fully determined and this isn't confounded by
        genuine kinematic redundancy."""
        mj_model, mj_data, _ = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(
            _LEFT_FRONT_LEG_TRIPLES, mj_model=mj_model
        )
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        rng = np.random.default_rng(4)
        base_qpos = _random_qpos_within_limits(mj_model, rng)
        n_frames, step = 10, 0.005
        targets = np.stack(
            [
                _fk_points(
                    mj_model, mj_data, base_qpos + step * t, body_ids, local_offsets
                )
                for t in range(n_frames)
            ]
        )

        qpos_trajectory = fit_qpos_trajectory_to_keypoints(
            mj_model, keypoints, targets, mj_data=mj_data, initial_qpos=base_qpos
        )
        max_consecutive_change = np.abs(np.diff(qpos_trajectory, axis=0)).max()
        assert max_consecutive_change < 0.05


# ==============================================================================
# End-to-end: bundled experimental recording (MotionSnippet)
# ==============================================================================


class TestMotionSnippetEndToEnd:
    """End-to-end tests using the bundled Spotlight motion-capture clip."""

    def test_keypoints_from_snippet_metadata(self, active_dofs_model, snippet):
        mj_model, _, _ = active_dofs_model
        keypoints = KeypointSet.from_keypoint_triples(
            snippet.keypoints, mj_model=mj_model
        )
        assert len(keypoints.targets) == 30
        assert len(snippet.legs) == 6

    def test_fk_matches_bundled_ground_truth_up_to_constant_offset(
        self, active_dofs_model, snippet
    ):
        """flygym's own forward kinematics on the ground-truth joint angles
        should reproduce the bundled `fwdkin_egoxyz` keypoints up to a fixed
        rigid translation between the recording's ego-frame and this
        compiled model's world frame (the two conventions don't share an
        origin). The offset, once estimated from one frame, should still
        explain a held-out frame to within a small residual."""
        mj_model, mj_data, skeleton = active_dofs_model
        jointdofs = list(skeleton.iter_jointdofs())
        keypoints = KeypointSet.from_keypoint_triples(
            snippet.keypoints, mj_model=mj_model
        )
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        def fk_keypoints(frame_idx):
            qpos = _qpos_from_snippet_frame(mj_model, jointdofs, snippet, frame_idx)
            return _fk_points(mj_model, mj_data, qpos, body_ids, local_offsets)

        calibration_frame, held_out_frame = 0, 300
        offset = (
            fk_keypoints(calibration_frame) - snippet.fwdkin_egoxyz[calibration_frame]
        ).mean(axis=0)
        held_out_diff = fk_keypoints(held_out_frame) - (
            snippet.fwdkin_egoxyz[held_out_frame] + offset
        )
        assert np.linalg.norm(held_out_diff, axis=1).max() < 0.5

    def test_fit_trajectory_recovers_joint_angles_with_good_initialization(
        self, active_dofs_model, snippet
    ):
        """Fits a short range of raw tracked keypoints and checks that the
        recovered per-DOF joint angles correlate well with the ground truth,
        when warm-started from the true qpos of the first frame (as a
        realistic prior/calibration pose would provide). Without a
        reasonable starting point, position-only IK over a multi-DOF chain
        can converge to an alternate, equally-valid-looking configuration
        (e.g. an elbow-up vs. elbow-down solution) -- expected behavior for
        local optimization on sparse keypoints, not a defect, but it means
        this fit is not guaranteed to recover the ground truth from an
        arbitrary cold start."""
        mj_model, mj_data, skeleton = active_dofs_model
        jointdofs = list(skeleton.iter_jointdofs())
        keypoints = KeypointSet.from_keypoint_triples(
            snippet.keypoints, mj_model=mj_model
        )
        body_ids = keypoints.resolve_body_ids(mj_model)
        local_offsets = keypoints.local_offsets()

        def fk_keypoints(frame_idx):
            qpos = _qpos_from_snippet_frame(mj_model, jointdofs, snippet, frame_idx)
            return _fk_points(mj_model, mj_data, qpos, body_ids, local_offsets)

        offset = (fk_keypoints(0) - snippet.fwdkin_egoxyz[0]).mean(axis=0)

        start, n_frames = 100, 30
        targets = snippet.rawpred_egoxyz[start : start + n_frames] + offset
        initial_qpos = _qpos_from_snippet_frame(mj_model, jointdofs, snippet, start)

        qpos_trajectory = fit_qpos_trajectory_to_keypoints(
            mj_model,
            keypoints,
            targets,
            mj_data=mj_data,
            initial_qpos=initial_qpos,
            max_iters=50,
        )
        ground_truth_qpos = np.stack(
            [
                _qpos_from_snippet_frame(mj_model, jointdofs, snippet, frame_idx)
                for frame_idx in range(start, start + n_frames)
            ]
        )

        correlations = [
            np.corrcoef(qpos_trajectory[:, j], ground_truth_qpos[:, j])[0, 1]
            for j in range(mj_model.nq)
            if np.std(ground_truth_qpos[:, j]) > 1e-6
        ]
        assert np.mean(correlations) > 0.7
