"""Unit tests for flygym.compose.pose (KinematicPose, KinematicPosePreset)."""

import math
import pytest

import flygym
from flygym.anatomy import AxisOrder, RotationAxis
from flygym.compose.pose import KinematicPose, KinematicPosePreset


# Minimal valid joint-angle dicts (YAW_PITCH_ROLL order)
_BALL_JOINT_ANGLES_YPR = {
    "c_thorax-lf_coxa-yaw": 0.1,
    "c_thorax-lf_coxa-pitch": 0.2,
    "c_thorax-lf_coxa-roll": 0.3,
}

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
        neutral_yaml = flygym.assets_dir / "model/pose/neutral/yaw_pitch_roll.yaml"
        with pytest.raises(ValueError, match="axis_order"):
            KinematicPose(path=neutral_yaml, axis_order=AxisOrder.YAW_PITCH_ROLL)

    def test_both_path_and_dict_raises(self):
        neutral_yaml = flygym.assets_dir / "model/pose/neutral/yaw_pitch_roll.yaml"
        with pytest.raises(ValueError):
            KinematicPose(
                path=neutral_yaml,
                joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            )

    def test_neither_path_nor_dict_raises(self):
        with pytest.raises(ValueError):
            KinematicPose()

    def test_from_yaml(self):
        neutral_yaml = flygym.assets_dir / "model/pose/neutral/yaw_pitch_roll.yaml"
        pose = KinematicPose(path=neutral_yaml)
        assert pose is not None

    def test_axis_order_stored(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        assert pose.axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_joint_angles_accessible(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        assert pose.joint_angles_lookup_rad["c_thorax-lf_coxa-yaw"] == pytest.approx(0.1)
        assert pose.joint_angles_lookup_rad["c_thorax-lf_coxa-pitch"] == pytest.approx(0.2)
        assert pose.joint_angles_lookup_rad["c_thorax-lf_coxa-roll"] == pytest.approx(0.3)

    def test_hinge_joint_stored(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_HINGE_ANGLES,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        assert pose.joint_angles_lookup_rad["lf_coxa-lf_trochanterfemur-pitch"] == pytest.approx(-1.5)

    def test_from_yaml_loads_angles(self):
        ypr_yaml = flygym.assets_dir / "model/pose/neutral/yaw_pitch_roll.yaml"
        pose = KinematicPose(path=ypr_yaml)
        assert len(pose.joint_angles_lookup_rad) > 0

    def test_from_yaml_loads_axis_order(self):
        ypr_yaml = flygym.assets_dir / "model/pose/neutral/yaw_pitch_roll.yaml"
        pose = KinematicPose(path=ypr_yaml)
        assert pose.axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_pry_yaml_axis_order(self):
        pry_yaml = flygym.assets_dir / "model/pose/neutral/pitch_roll_yaw.yaml"
        pose = KinematicPose(path=pry_yaml)
        assert pose.axis_order is AxisOrder.PITCH_ROLL_YAW


# ==============================================================================
# copy()
# ==============================================================================


class TestCopy:
    def test_copy_returns_new_instance(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        copy = pose.copy()
        assert copy is not pose

    def test_copy_has_same_angles(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        copy = pose.copy()
        assert copy.joint_angles_lookup_rad == pose.joint_angles_lookup_rad

    def test_copy_has_same_axis_order(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        copy = pose.copy()
        assert copy.axis_order is pose.axis_order

    def test_copy_is_independent(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR.copy(),
            axis_order=AxisOrder.YAW_PITCH_ROLL,
        )
        copy = pose.copy()
        copy.joint_angles_lookup_rad["c_thorax-lf_coxa-yaw"] = 999.0
        assert pose.joint_angles_lookup_rad["c_thorax-lf_coxa-yaw"] == pytest.approx(0.1)


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
        assert "c_thorax-rf_coxa-yaw" in pose.joint_angles_lookup_rad
        assert "c_thorax-rf_coxa-pitch" in pose.joint_angles_lookup_rad
        assert "c_thorax-rf_coxa-roll" in pose.joint_angles_lookup_rad

    def test_right_side_angles_equal_left_side(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=True,
        )
        angles = pose.joint_angles_lookup_rad
        assert angles["c_thorax-rf_coxa-yaw"] == pytest.approx(angles["c_thorax-lf_coxa-yaw"])
        assert angles["c_thorax-rf_coxa-pitch"] == pytest.approx(angles["c_thorax-lf_coxa-pitch"])
        assert angles["c_thorax-rf_coxa-roll"] == pytest.approx(angles["c_thorax-lf_coxa-roll"])

    def test_no_mirroring_leaves_right_side_absent(self):
        pose = KinematicPose(
            joint_angles_rad_dict=_BALL_JOINT_ANGLES_YPR,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=False,
        )
        assert "c_thorax-rf_coxa-yaw" not in pose.joint_angles_lookup_rad

    def test_explicit_right_side_not_overwritten_by_mirroring(self):
        angles = {
            **_BALL_JOINT_ANGLES_YPR,
            "c_thorax-rf_coxa-yaw": 9.9,
        }
        pose = KinematicPose(
            joint_angles_rad_dict=angles,
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            mirror_left2right=True,
        )
        assert pose.joint_angles_lookup_rad["c_thorax-rf_coxa-yaw"] == pytest.approx(9.9)


# ==============================================================================
# KinematicPosePreset
# ==============================================================================


class TestKinematicPosePreset:
    def test_neutral_preset_exists(self):
        assert KinematicPosePreset.NEUTRAL is not None

    def test_get_pose_by_axis_order_ypr(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        assert pose.axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_get_pose_by_axis_order_pry(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.PITCH_ROLL_YAW)
        assert pose.axis_order is AxisOrder.PITCH_ROLL_YAW

    def test_all_six_axis_orders_loadable(self):
        for axis_order in AxisOrder:
            pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(axis_order)
            assert pose.axis_order is axis_order
            assert len(pose.joint_angles_lookup_rad) > 0

    def test_different_axis_orders_share_leg_joints(self):
        """All axis-order variants of the same preset should cover the same leg joints."""
        pose_ypr = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        pose_pry = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.PITCH_ROLL_YAW)
        leg_keys_ypr = {k for k in pose_ypr.joint_angles_lookup_rad if "_coxa" in k or "_tibia" in k}
        leg_keys_pry = {k for k in pose_pry.joint_angles_lookup_rad if "_coxa" in k or "_tibia" in k}
        assert leg_keys_ypr == leg_keys_pry

    def test_neutral_pose_has_left_leg_joints(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        assert any("lf_coxa" in k for k in pose.joint_angles_lookup_rad)

    def test_neutral_pose_has_right_leg_joints(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        assert any("rf_coxa" in k for k in pose.joint_angles_lookup_rad)

    def test_neutral_pose_angles_in_radians(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        for angle in pose.joint_angles_lookup_rad.values():
            assert abs(angle) < math.pi + 0.1

    def test_get_dir_returns_existing_directory(self):
        d = KinematicPosePreset.NEUTRAL.get_dir()
        assert d.is_dir()

    def test_mirror_left2right_false(self):
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL, mirror_left2right=False
        )
        # Without mirroring, right-side joints may still appear if explicitly in the file
        # but the count should be less than or equal to the mirrored version
        pose_mirrored = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL, mirror_left2right=True
        )
        assert len(pose.joint_angles_lookup_rad) <= len(pose_mirrored.joint_angles_lookup_rad)
