"""Integration tests for flygym_examples.spotlight_data.MotionSnippet."""

import pytest
import numpy as np

from flygym.anatomy import AxisOrder, JointPreset, ActuatedDOFPreset, Skeleton
from flygym.compose.fly import Fly, ActuatorType
from flygym.compose.pose import KinematicPosePreset
from flygym_examples.spotlight_data.preprocessing import MotionSnippet


@pytest.fixture(scope="module")
def snippet():
    return MotionSnippet()


@pytest.fixture(scope="module")
def fly_for_snippet():
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly = Fly(name="snippet_fly")
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50,
    )
    return fly


# ==============================================================================
# MotionSnippet construction
# ==============================================================================


class TestMotionSnippetConstruction:
    def test_loads_default_data(self, snippet):
        assert snippet is not None

    def test_joint_angles_shape(self, snippet):
        # (n_frames, 6_legs, 7_dofs_per_leg)
        assert snippet.joint_angles.ndim == 3
        assert snippet.joint_angles.shape[1] == 6
        assert snippet.joint_angles.shape[2] == 7

    def test_legs_list(self, snippet):
        assert len(snippet.legs) == 6
        left_legs = [leg for leg in snippet.legs if leg.startswith("l")]
        right_legs = [leg for leg in snippet.legs if leg.startswith("r")]
        assert len(left_legs) == 3
        assert len(right_legs) == 3

    def test_dofs_per_leg(self, snippet):
        assert len(snippet.dofs_per_leg) == 7
        # Each DoF is a (parent_link, child_link, axis) tuple
        for dof in snippet.dofs_per_leg:
            assert len(dof) == 3

    def test_data_fps_positive(self, snippet):
        assert snippet.data_fps > 0

    def test_fwdkin_and_rawpred_shapes_match(self, snippet):
        assert snippet.fwdkin_egoxyz.shape == snippet.rawpred_egoxyz.shape

    def test_coordinate_conversion_applied(self):
        """With angles_global2anatomical=True (default), right-leg roll/yaw should
        be sign-flipped vs. the raw global-frame data."""
        raw = MotionSnippet(angles_global2anatomical=False)
        converted = MotionSnippet(angles_global2anatomical=True)

        right_leg_indices = [
            i for i, leg in enumerate(raw.legs) if leg.startswith("r")
        ]
        mirror_dof_indices = [
            i
            for i, (_, _, axis) in enumerate(raw.dofs_per_leg)
            if axis in ("roll", "yaw")
        ]

        for leg_idx in right_leg_indices:
            for dof_idx in mirror_dof_indices:
                np.testing.assert_array_almost_equal(
                    converted.joint_angles[:, leg_idx, dof_idx],
                    -raw.joint_angles[:, leg_idx, dof_idx],
                    decimal=10,
                )


# ==============================================================================
# get_joint_angles
# ==============================================================================


class TestMotionSnippetGetJointAngles:
    def test_output_shape(self, snippet, fly_for_snippet):
        sim_timestep = 1e-3  # 1 kHz (fast for testing)
        dof_order = list(fly_for_snippet.get_actuated_jointdofs_order(ActuatorType.POSITION))
        angles = snippet.get_joint_angles(
            output_timestep=sim_timestep,
            output_dof_order=dof_order,
        )
        n_frames = snippet.joint_angles.shape[0]
        expected_steps = int((n_frames / snippet.data_fps) / sim_timestep)
        assert angles.ndim == 2
        assert angles.shape[0] == expected_steps
        assert angles.shape[1] == len(dof_order)

    def test_output_is_finite(self, snippet, fly_for_snippet):
        dof_order = list(fly_for_snippet.get_actuated_jointdofs_order(ActuatorType.POSITION))
        angles = snippet.get_joint_angles(
            output_timestep=1e-3,
            output_dof_order=dof_order,
        )
        assert np.all(np.isfinite(angles))

    def test_angles_roughly_in_radians(self, snippet, fly_for_snippet):
        """Biological leg angles should be within ±π radians."""
        dof_order = list(fly_for_snippet.get_actuated_jointdofs_order(ActuatorType.POSITION))
        angles = snippet.get_joint_angles(
            output_timestep=1e-3,
            output_dof_order=dof_order,
        )
        assert np.all(np.abs(angles) < np.pi + 0.1)

    def test_resampling_preserves_duration(self, snippet, fly_for_snippet):
        """Output at two different timesteps should cover approximately the same duration."""
        dof_order = list(fly_for_snippet.get_actuated_jointdofs_order(ActuatorType.POSITION))
        dt_fast = 1e-3
        dt_slow = 5e-3
        angles_fast = snippet.get_joint_angles(dt_fast, dof_order)
        angles_slow = snippet.get_joint_angles(dt_slow, dof_order)
        duration_fast = angles_fast.shape[0] * dt_fast
        duration_slow = angles_slow.shape[0] * dt_slow
        assert duration_fast == pytest.approx(duration_slow, rel=dt_slow)

    def test_smoothing_reduces_high_frequency_noise(self, snippet, fly_for_snippet):
        """Heavily smoothed output should have lower mean absolute first-difference
        than a lightly smoothed output (at the same timestep)."""
        dof_order = list(fly_for_snippet.get_actuated_jointdofs_order(ActuatorType.POSITION))
        dt = 1 / snippet.data_fps  # output at native recording rate

        angles_heavy = snippet.get_joint_angles(
            output_timestep=dt,
            output_dof_order=dof_order,
            sgfilter_window_sec=0.1,  # very wide window → heavy smoothing
        )
        # Light smoothing: minimum valid window for polyorder=3 is 5 frames (odd, > 3)
        angles_light = snippet.get_joint_angles(
            output_timestep=dt,
            output_dof_order=dof_order,
            sgfilter_window_sec=5 / snippet.data_fps,  # 5-frame window → minimal
        )
        diff_heavy = np.mean(np.abs(np.diff(angles_heavy, axis=0)))
        diff_light = np.mean(np.abs(np.diff(angles_light, axis=0)))
        assert diff_heavy <= diff_light
