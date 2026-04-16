"""Unit tests for flygym.utils (math, exceptions, mjcf, profiling, video, pose_conversion)."""

import pathlib
import pytest
import numpy as np

from flygym.utils.math import Tree, orderedset, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError


# ==============================================================================
# orderedset
# ==============================================================================


class TestOrderedset:
    def test_deduplication(self):
        result = orderedset([1, 2, 2, 3, 1])
        assert result == [1, 2, 3]

    def test_preserves_order(self):
        result = orderedset([3, 1, 2, 1, 3])
        assert result == [3, 1, 2]

    def test_empty(self):
        assert orderedset([]) == []

    def test_strings(self):
        result = orderedset(["b", "a", "b", "c", "a"])
        assert result == ["b", "a", "c"]

    def test_single_element(self):
        assert orderedset([42]) == [42]


# ==============================================================================
# Tree
# ==============================================================================


class TestTree:
    def test_valid_tree(self):
        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (1, 3), (3, 4)]
        tree = Tree(nodes, edges)
        assert tree is not None

    def test_dfs_edges_visits_all_nodes(self):
        nodes = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("a", "c"), ("c", "d")]
        tree = Tree(nodes, edges)
        visited = set()
        for parent, child in tree.dfs_edges("a"):
            visited.add(parent)
            visited.add(child)
        assert visited == set(nodes)

    def test_dfs_edges_order(self):
        # Simple chain: root -> a -> b
        nodes = ["root", "a", "b"]
        edges = [("root", "a"), ("a", "b")]
        tree = Tree(nodes, edges)
        dfs = list(tree.dfs_edges("root"))
        # First edge must start from root
        assert dfs[0][0] == "root"

    def test_single_node_tree(self):
        tree = Tree([1], [])
        assert list(tree.dfs_edges(1)) == []

    def test_cycle_raises(self):
        nodes = [1, 2, 3]
        edges = [(1, 2), (2, 3), (3, 1)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_disconnected_raises(self):
        nodes = [1, 2, 3, 4]
        edges = [(1, 2), (3, 4)]  # Two components
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_parallel_edges_raises(self):
        nodes = [1, 2]
        edges = [(1, 2), (1, 2)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_self_loop_raises(self):
        nodes = [1, 2]
        edges = [(1, 2), (1, 1)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_duplicate_nodes_raises(self):
        nodes = [1, 1, 2]
        edges = [(1, 2)]
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_edge_with_nonexistent_node_raises(self):
        nodes = [1, 2]
        edges = [(1, 99)]  # 99 not in nodes
        with pytest.raises(ValueError):
            Tree(nodes, edges)

    def test_invalid_root_raises(self):
        tree = Tree([1, 2], [(1, 2)])
        with pytest.raises(ValueError):
            list(tree.dfs_edges(99))


# ==============================================================================
# Rotation3D
# ==============================================================================


class TestRotation3D:
    def test_quat_valid(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        assert rot.format == "quat"
        assert tuple(rot.values) == (1, 0, 0, 0)

    def test_euler_valid(self):
        rot = Rotation3D("euler", (0.0, 0.0, 0.0))
        assert rot.format == "euler"

    def test_axisangle_valid(self):
        rot = Rotation3D("axisangle", (0, 0, 1, 1))
        assert rot.format == "axisangle"

    def test_xyaxes_valid(self):
        rot = Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0))
        assert rot.format == "xyaxes"

    def test_zaxis_valid(self):
        rot = Rotation3D("zaxis", (0, 0, 1))
        assert rot.format == "zaxis"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            Rotation3D("badformat", (1, 0, 0, 0))

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError):
            Rotation3D("quat", (1, 0, 0))  # quat needs 4, got 3

    def test_as_kwargs_quat(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        kwargs = rot.as_kwargs()
        assert kwargs == {"quat": (1, 0, 0, 0)}

    def test_as_kwargs_euler(self):
        rot = Rotation3D("euler", (0.1, 0.2, 0.3))
        kwargs = rot.as_kwargs()
        assert kwargs == {"euler": (0.1, 0.2, 0.3)}

    def test_frozen_dataclass(self):
        rot = Rotation3D("quat", (1, 0, 0, 0))
        with pytest.raises((AttributeError, TypeError)):
            rot.format = "euler"

    def test_non_number_values_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Rotation3D("euler", ("a", "b", "c"))


# ==============================================================================
# FlyGymInternalError
# ==============================================================================


class TestFlyGymInternalError:
    def test_is_exception(self):
        err = FlyGymInternalError("test")
        assert isinstance(err, Exception)

    def test_message(self):
        msg = "something went wrong internally"
        err = FlyGymInternalError(msg)
        assert str(err) == msg

    def test_can_be_raised(self):
        with pytest.raises(FlyGymInternalError):
            raise FlyGymInternalError("boom")


# ==============================================================================
# Rotation3D.as_kwargs – all formats
# ==============================================================================


class TestRotation3DAsKwargs:
    def test_axisangle(self):
        rot = Rotation3D("axisangle", (0, 0, 1, 1))
        assert rot.as_kwargs() == {"axisangle": (0, 0, 1, 1)}

    def test_xyaxes(self):
        rot = Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0))
        assert rot.as_kwargs() == {"xyaxes": (1, 0, 0, 0, 1, 0)}

    def test_zaxis(self):
        rot = Rotation3D("zaxis", (0, 0, 1))
        assert rot.as_kwargs() == {"zaxis": (0, 0, 1)}

    def test_returns_dict_with_single_key(self):
        for fmt, vals in [
            ("quat", (1, 0, 0, 0)),
            ("euler", (0.1, 0.2, 0.3)),
            ("axisangle", (0, 0, 1, 1)),
            ("xyaxes", (1, 0, 0, 0, 1, 0)),
            ("zaxis", (0, 1, 0)),
        ]:
            rot = Rotation3D(fmt, vals)
            kw = rot.as_kwargs()
            assert len(kw) == 1
            assert fmt in kw


# ==============================================================================
# utils.mjcf: set_params_recursive / set_mujoco_globals
# ==============================================================================


class TestSetParamsRecursive:
    def test_sets_attribute_on_root(self):
        """set_params_recursive can set a direct attribute on the MJCF root."""
        import dm_control.mjcf as mjcf

        root = mjcf.RootElement()
        # 'timestep' lives under root.option
        from flygym.utils.mjcf import set_params_recursive

        set_params_recursive(root, {"option": {"timestep": 0.001}})
        assert root.option.timestep == pytest.approx(0.001)

    def test_sets_nested_attribute(self):
        """set_params_recursive traverses nested child elements."""
        import dm_control.mjcf as mjcf
        from flygym.utils.mjcf import set_params_recursive

        root = mjcf.RootElement()
        set_params_recursive(root, {"option": {"gravity": [0, 0, -9.81]}})
        assert root.option.gravity[2] == pytest.approx(-9.81)

    def test_non_dict_child_raises(self):
        """Providing a non-dict value for a child element key should raise ValueError."""
        import dm_control.mjcf as mjcf
        from flygym.utils.mjcf import set_params_recursive

        root = mjcf.RootElement()
        with pytest.raises(ValueError, match="Expected dict"):
            set_params_recursive(root, {"option": "not_a_dict"})

    def test_unknown_key_is_silently_ignored(self):
        """Keys that exist neither as attributes nor children are silently skipped."""
        import dm_control.mjcf as mjcf
        from flygym.utils.mjcf import set_params_recursive

        root = mjcf.RootElement()
        # Should not raise even though 'nonexistent_key' is unknown
        set_params_recursive(root, {"nonexistent_key": 42})


class TestSetMujocoGlobals:
    def test_applies_yaml_settings(self, tmp_path):
        """set_mujoco_globals reads a YAML file and applies the settings."""
        import dm_control.mjcf as mjcf
        from flygym.utils.mjcf import set_mujoco_globals

        yaml_content = "option:\n  timestep: 0.0005\n"
        yaml_path = tmp_path / "globals.yaml"
        yaml_path.write_text(yaml_content)

        root = mjcf.RootElement()
        set_mujoco_globals(root, yaml_path)
        assert root.option.timestep == pytest.approx(0.0005)

    def test_missing_yaml_raises(self, tmp_path):
        """set_mujoco_globals raises when the YAML file does not exist."""
        import dm_control.mjcf as mjcf
        from flygym.utils.mjcf import set_mujoco_globals

        root = mjcf.RootElement()
        with pytest.raises(FileNotFoundError):
            set_mujoco_globals(root, tmp_path / "nonexistent.yaml")


# ==============================================================================
# utils.profiling: print_perf_report / print_perf_report_parallel
# ==============================================================================


class TestPrintPerfReport:
    def test_outputs_performance_table(self, capsys):
        from flygym.utils.profiling import print_perf_report

        print_perf_report(
            total_physics_time_ns=1_000_000,
            total_render_time_ns=500_000,
            n_steps=100,
            n_frames_rendered=10,
            timestep=0.001,
        )
        captured = capsys.readouterr()
        assert "PERFORMANCE" in captured.out
        assert "Physics" in captured.out

    def test_no_frames_rendered_shows_note(self, capsys):
        from flygym.utils.profiling import print_perf_report

        print_perf_report(
            total_physics_time_ns=1_000_000,
            total_render_time_ns=0,
            n_steps=100,
            n_frames_rendered=0,
            timestep=0.001,
        )
        captured = capsys.readouterr()
        assert "No frames" in captured.out

    def test_zero_steps_raises(self):
        from flygym.utils.profiling import print_perf_report

        with pytest.raises(ValueError, match="n_steps"):
            print_perf_report(
                total_physics_time_ns=1_000_000,
                total_render_time_ns=0,
                n_steps=0,
                n_frames_rendered=0,
                timestep=0.001,
            )

    def test_rendered_frames_note_contains_count(self, capsys):
        from flygym.utils.profiling import print_perf_report

        print_perf_report(
            total_physics_time_ns=2_000_000,
            total_render_time_ns=500_000,
            n_steps=200,
            n_frames_rendered=5,
            timestep=0.001,
        )
        captured = capsys.readouterr()
        assert "5" in captured.out


class TestPrintPerfReportParallel:
    def test_outputs_performance_table(self, capsys):
        from flygym.utils.profiling import print_perf_report_parallel

        print_perf_report_parallel(
            total_physics_time_ns=1_000_000,
            total_render_time_ns=500_000,
            n_steps=100,
            n_frames_rendered=10,
            timestep=0.001,
            n_worlds=4,
            n_worlds_rendered=2,
        )
        captured = capsys.readouterr()
        assert "PERFORMANCE" in captured.out
        assert "Physics" in captured.out

    def test_zero_steps_raises(self):
        from flygym.utils.profiling import print_perf_report_parallel

        with pytest.raises(ValueError, match="n_steps"):
            print_perf_report_parallel(
                total_physics_time_ns=1_000_000,
                total_render_time_ns=0,
                n_steps=0,
                n_frames_rendered=0,
                timestep=0.001,
                n_worlds=2,
                n_worlds_rendered=0,
            )

    def test_parallel_columns_present(self, capsys):
        from flygym.utils.profiling import print_perf_report_parallel

        print_perf_report_parallel(
            total_physics_time_ns=1_000_000,
            total_render_time_ns=0,
            n_steps=100,
            n_frames_rendered=0,
            timestep=0.001,
            n_worlds=8,
            n_worlds_rendered=0,
        )
        captured = capsys.readouterr()
        assert "parallelized" in captured.out.lower()


# ==============================================================================
# utils.video: write_video_from_frames
# ==============================================================================


class TestWriteVideoFromFrames:
    def test_creates_file(self, tmp_path):
        from flygym.utils.video import write_video_from_frames

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        out = tmp_path / "output.mp4"
        write_video_from_frames(out, frames, fps=30)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        from flygym.utils.video import write_video_from_frames

        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
        out = tmp_path / "subdir" / "nested" / "video.mp4"
        write_video_from_frames(out, frames, fps=10)
        assert out.exists()

    def test_non_multiple_of_16_is_resized(self, tmp_path):
        """write_video_from_frames silently pads frame dimensions to multiples of 16."""
        from flygym.utils.video import write_video_from_frames

        # 30x50 is NOT a multiple of 16 → should be padded and written without error
        frames = [np.zeros((30, 50, 3), dtype=np.uint8) for _ in range(3)]
        out = tmp_path / "padded.mp4"
        write_video_from_frames(out, frames, fps=10)
        assert out.exists()

    def test_multiple_of_16_written_unchanged(self, tmp_path):
        """Frames that are already multiples of 16 should be written without resizing."""
        from flygym.utils.video import write_video_from_frames

        frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(3)]
        out = tmp_path / "no_pad.mp4"
        write_video_from_frames(out, frames, fps=10)
        assert out.exists()


# ==============================================================================
# utils.pose_conversion: get_body_names / get_xpos0_xquat0 / qpos_to_kinematic_pose
# ==============================================================================


@pytest.fixture(scope="module")
def compiled_fly_model():
    """A minimal compiled MuJoCo model from a single fly (no world)."""
    from flygym.anatomy import AxisOrder, JointPreset, Skeleton
    from flygym.compose.fly import Fly
    from flygym.compose.pose import KinematicPosePreset

    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    fly = Fly(name="pc_fly")
    fly.add_joints(skeleton, neutral_pose=pose)
    mj_model, mj_data = fly.compile()
    return mj_model, mj_data


class TestGetBodyNames:
    def test_returns_list(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_body_names

        mj_model, _ = compiled_fly_model
        names = get_body_names(mj_model)
        assert isinstance(names, list)

    def test_length_matches_nbody(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_body_names

        mj_model, _ = compiled_fly_model
        names = get_body_names(mj_model)
        assert len(names) == mj_model.nbody

    def test_contains_known_body(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_body_names

        mj_model, _ = compiled_fly_model
        names = get_body_names(mj_model)
        # The world body is always present at index 0
        assert names[0] == "world"


class TestGetXpos0Xquat0:
    def test_returns_tuple_of_arrays(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_xpos0_xquat0

        mj_model, mj_data = compiled_fly_model
        xpos, xquat = get_xpos0_xquat0(mj_model, mj_data)
        assert isinstance(xpos, np.ndarray)
        assert isinstance(xquat, np.ndarray)

    def test_shapes(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_xpos0_xquat0

        mj_model, mj_data = compiled_fly_model
        xpos, xquat = get_xpos0_xquat0(mj_model, mj_data)
        assert xpos.shape == (mj_model.nbody, 3)
        assert xquat.shape == (mj_model.nbody, 4)

    def test_quaternions_are_unit(self, compiled_fly_model):
        from flygym.utils.pose_conversion import get_xpos0_xquat0

        mj_model, mj_data = compiled_fly_model
        _, xquat = get_xpos0_xquat0(mj_model, mj_data)
        # Skip the world body (index 0, which has zero quat)
        norms = np.linalg.norm(xquat[1:], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestQposToKinematicPose:
    def test_returns_kinematic_pose(self, compiled_fly_model):
        from flygym.anatomy import AxisOrder
        from flygym.compose.pose import KinematicPose
        from flygym.utils.pose_conversion import qpos_to_kinematic_pose

        mj_model, _ = compiled_fly_model
        qpos = np.zeros(mj_model.nq)
        pose = qpos_to_kinematic_pose(mj_model, qpos, AxisOrder.YAW_PITCH_ROLL)
        assert isinstance(pose, KinematicPose)

    def test_axis_order_preserved(self, compiled_fly_model):
        from flygym.anatomy import AxisOrder
        from flygym.utils.pose_conversion import qpos_to_kinematic_pose

        mj_model, _ = compiled_fly_model
        qpos = np.zeros(mj_model.nq)
        pose = qpos_to_kinematic_pose(mj_model, qpos, AxisOrder.YAW_PITCH_ROLL)
        assert pose.axis_order is AxisOrder.YAW_PITCH_ROLL

    def test_right_side_mirrored_from_left(self, compiled_fly_model):
        """qpos_to_kinematic_pose mirrors left joints to the right side."""
        from flygym.anatomy import AxisOrder
        from flygym.utils.pose_conversion import qpos_to_kinematic_pose

        mj_model, _ = compiled_fly_model
        qpos = np.zeros(mj_model.nq)
        pose = qpos_to_kinematic_pose(mj_model, qpos, AxisOrder.YAW_PITCH_ROLL)
        keys = pose.joint_angles_lookup_rad.keys()
        right_keys = [k for k in keys if "r_" in k or k.startswith("r")]
        # Right-side joints should be present (mirroring was applied)
        assert len(right_keys) > 0
