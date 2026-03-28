"""Tests for flygym.rendering (Renderer class).

MuJoCo's renderer requires an OpenGL context.  All tests that need an actual
mujoco.Renderer are guarded by patching it with a lightweight mock so that
the CI can run without a display.  Tests that only exercise pure-Python logic
(path resolution, camera-spec normalisation) work without any mock.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose.fly import Fly
from flygym.compose.pose import KinematicPosePreset
from flygym.compose.world import TetheredWorld
from flygym.utils.math import Rotation3D
from flygym.rendering import Renderer


# ==============================================================================
# Shared fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def compiled_model_with_camera():
    """TetheredWorld with one fly that has a tracking camera."""
    pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    fly = Fly(name="render_fly")
    fly.add_joints(skeleton, neutral_pose=pose)
    fly.add_tracking_camera(name="trackcam")
    world = TetheredWorld(name="render_world")
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    mj_model, mj_data = world.compile()
    return mj_model, mj_data, fly


@pytest.fixture(scope="module")
def cam_name(compiled_model_with_camera):
    """Full identifier of the tracking camera after world attachment."""
    _, _, fly = compiled_model_with_camera
    # dm_control prefixes names with the model name when attaching
    return fly.cameraname_to_mjcfcamera["trackcam"].full_identifier


def _make_mock_mj_renderer(frame_shape=(64, 64, 3)):
    """Return a mock that mimics the mujoco.Renderer interface."""
    mock = MagicMock()
    mock.render.return_value = np.zeros(frame_shape, dtype=np.uint8)
    return mock


@pytest.fixture(scope="module")
def renderer(compiled_model_with_camera, cam_name):
    """A Renderer with a mocked mujoco.Renderer backend."""
    mj_model, _, _ = compiled_model_with_camera
    mock_backend = _make_mock_mj_renderer()
    with patch("mujoco.Renderer", return_value=mock_backend):
        r = Renderer(mj_model, cam_name, camera_res=(64, 64))
    yield r
    # close() calls self.mj_renderer.close(), which is a no-op on the mock
    r.close()


# ==============================================================================
# Construction
# ==============================================================================


class TestRendererConstruction:
    def test_constructs_with_valid_camera(self, compiled_model_with_camera, cam_name):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(mj_model, cam_name, camera_res=(64, 64))
        r.close()
        assert r is not None

    def test_buffer_frames_true_initializes_frames_dict(
        self, compiled_model_with_camera, cam_name
    ):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(mj_model, cam_name, camera_res=(64, 64), buffer_frames=True)
        r.close()
        assert r.frames is not None
        assert cam_name in r.frames
        assert isinstance(r.frames[cam_name], list)

    def test_buffer_frames_false_frames_is_none(
        self, compiled_model_with_camera, cam_name
    ):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(mj_model, cam_name, camera_res=(64, 64), buffer_frames=False)
        r.close()
        assert r.frames is None

    def test_invalid_camera_raises(self, compiled_model_with_camera):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            with pytest.raises(ValueError, match="not found"):
                Renderer(mj_model, "nonexistent_cam", camera_res=(64, 64))

    def test_duplicate_camera_raises(self, compiled_model_with_camera, cam_name):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            with pytest.raises(ValueError, match="Duplicate"):
                Renderer(mj_model, [cam_name, cam_name], camera_res=(64, 64))

    def test_camera_res_stored(self, compiled_model_with_camera, cam_name):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(mj_model, cam_name, camera_res=(48, 96))
        r.close()
        assert r.camera_res == (48, 96)

    def test_playback_speed_and_fps_stored(self, compiled_model_with_camera, cam_name):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(
                mj_model, cam_name, camera_res=(64, 64), playback_speed=0.5, output_fps=30
            )
        r.close()
        assert r.playback_speed == 0.5
        assert r.output_fps == 30


# ==============================================================================
# render_as_needed
# ==============================================================================


class TestRenderAsNeeded:
    def test_first_call_renders(self, renderer, compiled_model_with_camera):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        result = renderer.render_as_needed(mj_data)
        assert result is True

    def test_second_call_too_soon_does_not_render(
        self, renderer, compiled_model_with_camera
    ):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        # First call renders; second call immediately after should not render
        renderer.render_as_needed(mj_data)
        result = renderer.render_as_needed(mj_data)
        assert result is False

    def test_buffers_frame_when_rendering(self, renderer, compiled_model_with_camera):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        renderer.render_as_needed(mj_data)
        cam_name_key = list(renderer.frames.keys())[0]
        assert len(renderer.frames[cam_name_key]) == 1

    def test_frame_has_correct_shape(self, renderer, compiled_model_with_camera):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        renderer.render_as_needed(mj_data)
        cam_name_key = list(renderer.frames.keys())[0]
        frame = renderer.frames[cam_name_key][0]
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB


# ==============================================================================
# reset
# ==============================================================================


class TestRendererReset:
    def test_reset_clears_frames(self, renderer, compiled_model_with_camera):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        renderer.render_as_needed(mj_data)
        assert any(len(v) > 0 for v in renderer.frames.values())
        renderer.reset()
        assert all(len(v) == 0 for v in renderer.frames.values())

    def test_reset_restarts_render_timer(self, renderer, compiled_model_with_camera):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        # After first render, immediately calling again should not render
        renderer.render_as_needed(mj_data)
        no_render = renderer.render_as_needed(mj_data)
        assert no_render is False
        # After reset, the timer is reset so the next call should render again
        renderer.reset()
        renders = renderer.render_as_needed(mj_data)
        assert renders is True


# ==============================================================================
# Context manager
# ==============================================================================


class TestRendererContextManager:
    def test_context_manager_returns_self(self, compiled_model_with_camera, cam_name):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            with Renderer(mj_model, cam_name, camera_res=(64, 64)) as r:
                assert r is not None

    def test_context_manager_calls_close_on_exit(
        self, compiled_model_with_camera, cam_name
    ):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            with Renderer(mj_model, cam_name, camera_res=(64, 64)) as r:
                pass
        mock_backend.close.assert_called_once()

    def test_context_manager_does_not_suppress_exceptions(
        self, compiled_model_with_camera, cam_name
    ):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            with pytest.raises(RuntimeError, match="intentional"):
                with Renderer(mj_model, cam_name, camera_res=(64, 64)):
                    raise RuntimeError("intentional")


# ==============================================================================
# _normalize_camera_spec  (pure Python logic)
# ==============================================================================


class TestNormalizeCameraSpec:
    def test_none_returns_all_cameras(self, renderer):
        result = renderer._normalize_camera_spec(None)
        assert set(result) == set(renderer._cameras_names2id.keys())

    def test_string_camera_returns_list(self, renderer, cam_name):
        # cam_name is the full identifier used when constructing the renderer
        result = renderer._normalize_camera_spec(cam_name)
        assert result == [cam_name]

    def test_sequence_of_cameras(self, renderer, cam_name):
        result = renderer._normalize_camera_spec([cam_name])
        assert result == [cam_name]

    def test_invalid_camera_string_raises(self, renderer):
        with pytest.raises(ValueError, match="not available"):
            renderer._normalize_camera_spec("does_not_exist")

    def test_invalid_type_raises(self, renderer):
        with pytest.raises(ValueError, match="Invalid camera spec type"):
            renderer._normalize_camera_spec(42)


# ==============================================================================
# _resolve_output_paths  (pure Python logic)
# ==============================================================================


class TestResolveOutputPaths:
    def test_single_camera_path_is_file(self, renderer, cam_name, tmp_path):
        out = tmp_path / "vid.mp4"
        result = renderer._resolve_output_paths(out)
        assert result == {cam_name: out}

    def test_dict_mapping_resolves_correctly(self, renderer, cam_name, tmp_path):
        out = tmp_path / "vid.mp4"
        result = renderer._resolve_output_paths({cam_name: out})
        assert result == {cam_name: out}

    def test_dict_with_unknown_camera_raises(self, renderer, tmp_path):
        out = tmp_path / "vid.mp4"
        with pytest.raises(ValueError, match="not available"):
            renderer._resolve_output_paths({"nonexistent": out})


# ==============================================================================
# save_video
# ==============================================================================


class TestSaveVideo:
    def test_save_video_creates_file(
        self, renderer, compiled_model_with_camera, tmp_path
    ):
        _, mj_data, _ = compiled_model_with_camera
        renderer.reset()
        # Render a few frames
        for _ in range(3):
            renderer.render_as_needed(mj_data)
            # Advance the simulated time manually so we get multiple renders
            renderer._last_render_time_sec -= renderer._secs_between_renders

        out = tmp_path / "output.mp4"
        renderer.save_video(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_video_no_frames_raises(
        self, compiled_model_with_camera, cam_name, tmp_path
    ):
        mj_model, _, _ = compiled_model_with_camera
        mock_backend = _make_mock_mj_renderer()
        with patch("mujoco.Renderer", return_value=mock_backend):
            r = Renderer(mj_model, cam_name, camera_res=(64, 64))
        # Don't render anything; save_video should raise
        with pytest.raises(RuntimeError, match="No frames"):
            r.save_video(tmp_path / "out.mp4")
        r.close()
