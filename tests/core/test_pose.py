"""Unit tests for flygym.compose.pose (KinematicPose)."""

import math
import pytest
import numpy as np

import flygym
from flygym.anatomy import AxisOrder, JointDOF, RotationAxis
from flygym.compose.pose import KinematicPose


# Minimal valid joint-angle dict: one 3-DoF ball joint (thorax->lf_coxa)
_BALL_JOINT_ANGLES_YPR = {
    "c_thorax-lf_coxa-yaw": 0.1,
    "c_thorax-lf_coxa-pitch": 0.2,
    "c_thorax-lf_coxa-roll": 0.3,
}

# One 1-DoF hinge joint (lf_coxa -> lf_trochanterfemur, pitch only)
_HINGE_ANGLES = {
    "lf_coxa-lf_trochanterfemur-pitch": -1.5,
}


# ==============================================================================
# Construction validation
# ==============================================================================


class TestKinematicPoseConstruction:
    def test_from_dict_basic(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        assert pose is not None

    def test_from_dict_requires_axis_order(self):
        with pytest.raises(ValueError, match="axis_order"):
            KinematicPose(joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR)

    def test_from_path_does_not_accept_axis_order(self):
        neutral_yaml = flygym.assets_dir / "model/pose/neutral.yaml"
        with pytest.raises(ValueError, match="axis_order"):
            KinematicPose(path=neutral_yaml, axis_order=AxisOrder.YAW_PITCH_ROLL)

    def test_both_path_and_dict_raises(self):
        neutral_yaml = flygym.assets_dir / "model/pose/neutral.yaml"
        with pytest.raises(ValueError):
            KinematicPose(
                path=neutral_yaml,
                joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            )

    def test_neither_path_nor_dict_raises(self):
        with pytest.raises(ValueError):
            KinematicPose()

    def test_from_yaml(self):
        neutral_yaml = flygym.assets_dir / "model/pose/neutral.yaml"
        pose = KinematicPose(path=neutral_yaml)
        assert pose is not None

    def test_axis_order_stored(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        assert pose._native_axis_order is AxisOrder.YAW_PITCH_ROLL


# ==============================================================================
# get_angles_lookup - same axis order (identity path)
# ==============================================================================


class TestGetAnglesLookupSameOrder:
    def test_returns_native_angles_unchanged(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        assert result["c_thorax-lf_coxa-yaw"] == pytest.approx(0.1)
        assert result["c_thorax-lf_coxa-pitch"] == pytest.approx(0.2)
        assert result["c_thorax-lf_coxa-roll"] == pytest.approx(0.3)

    def test_degrees_conversion(self):
        pose = KinematicPose(
            joint_angles_rad_dict={"c_thorax-lf_coxa-pitch": math.pi / 2},
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result_rad = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL, degrees=False)
        result_deg = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL, degrees=True)
        assert result_rad["c_thorax-lf_coxa-pitch"] == pytest.approx(math.pi / 2)
        assert result_deg["c_thorax-lf_coxa-pitch"] == pytest.approx(90.0)

    def test_1dof_hinge_unchanged(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_HINGE_ANGLES,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result = pose.get_angles_lookup(AxisOrder.PITCH_ROLL_YAW)
        # 1-DoF joints are returned unchanged regardless of output axis order
        assert result["lf_coxa-lf_trochanterfemur-pitch"] == pytest.approx(-1.5)


# ==============================================================================
# get_angles_lookup - axis order conversion (3-DoF)
# ==============================================================================


class TestGetAnglesLookup3DOFConversion:
    def test_round_trip_is_approximately_identity(self):
        """Convert YPR -> PRY -> YPR; the result should be close to the original."""
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        # Convert to a different order
        intermediate = pose.get_angles_lookup(AxisOrder.PITCH_ROLL_YAW)
        # Build a new pose from the converted angles and convert back
        pose2 = KinematicPose(
            joint_angles_rad_dict=intermediate,
            axis_order=AxisOrder.PITCH_ROLL_YAW,
            mirror_left2right=False,
        )
        back = pose2.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)

        for key, original_angle in _BALL_JOINT_ANGLES_YPR.items():
            assert back[key] == pytest.approx(original_angle, abs=1e-6), (
                f"Round-trip failed for {key}: expected {original_angle}, got {back[key]}"
            )

    def test_different_axis_orders_produce_different_names_for_same_joint(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result = pose.get_angles_lookup(AxisOrder.PITCH_ROLL_YAW)
        # Same DoF names should still be present (name encodes axis, not order)
        assert "c_thorax-lf_coxa-yaw" in result
        assert "c_thorax-lf_coxa-pitch" in result
        assert "c_thorax-lf_coxa-roll" in result

    def test_zero_rotation_converts_to_zero(self):
        zero_angles = {
            "c_thorax-lf_coxa-yaw": 0.0,
            "c_thorax-lf_coxa-pitch": 0.0,
            "c_thorax-lf_coxa-roll": 0.0,
        }
        pose = KinematicPose(
            joint_angles_rad_dict=zero_angles,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result = pose.get_angles_lookup(AxisOrder.PITCH_ROLL_YAW)
        for key in zero_angles:
            assert result[key] == pytest.approx(0.0)


# ==============================================================================
# Mirroring left-to-right
# ==============================================================================


class TestMirroring:
    def test_right_side_filled_from_left(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=True,
        )
        result = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        # Right-side counterpart should be present
        assert "c_thorax-rf_coxa-yaw" in result
        assert "c_thorax-rf_coxa-pitch" in result
        assert "c_thorax-rf_coxa-roll" in result

    def test_right_side_angles_equal_left_side(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=True,
        )
        result = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        assert result["c_thorax-rf_coxa-yaw"] == pytest.approx(
            result["c_thorax-lf_coxa-yaw"]
        )
        assert result["c_thorax-rf_coxa-pitch"] == pytest.approx(
            result["c_thorax-lf_coxa-pitch"]
        )
        assert result["c_thorax-rf_coxa-roll"] == pytest.approx(
            result["c_thorax-lf_coxa-roll"]
        )

    def test_no_mirroring_leaves_right_side_absent(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        result = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        assert "c_thorax-rf_coxa-yaw" not in result

    def test_explicit_right_side_not_overwritten_by_mirroring(self):
        angles = {
            **_BALL_JOINT_ANGLES_YPR,
            "c_thorax-rf_coxa-yaw": 9.9,  # explicit right value
        }
        pose = KinematicPose(
            joint_angles_rad_dict=angles,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=True,
        )
        result = pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        # Explicit right-side value should be preserved
        assert result["c_thorax-rf_coxa-yaw"] == pytest.approx(9.9)


# ==============================================================================
# Loading from the bundled neutral.yaml
# ==============================================================================


class TestNeutralYamlLoading:
    @pytest.fixture(scope="class")
    def neutral_pose(self):
        return KinematicPose(path=flygym.assets_dir / "model/pose/neutral.yaml")

    def test_has_joint_angles(self, neutral_pose):
        assert len(neutral_pose._native_joint_angles_rad_dict) > 0

    def test_axis_order_is_yaw_pitch_roll(self, neutral_pose):
        assert neutral_pose._native_axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_left_leg_joints_present(self, neutral_pose):
        angles = neutral_pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        # Thorax-coxa yaw should be in there for the left front leg
        assert any("lf_coxa" in k for k in angles)

    def test_right_leg_joints_filled_by_mirroring(self, neutral_pose):
        angles = neutral_pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        # With default mirror_left2right=True, right-side joints must be present
        assert any("rf_coxa" in k for k in angles)

    def test_angles_in_radians(self, neutral_pose):
        angles = neutral_pose.get_angles_lookup(AxisOrder.YAW_PITCH_ROLL)
        for angle in angles.values():
            # Angles should be well within ±π (no accidental degrees left)
            assert abs(angle) < math.pi + 0.1
