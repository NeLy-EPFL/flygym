"""Tests for flygym.warp.rendering (WarpCPURenderer, modify_world_for_batch_rendering).
"""

import warnings
import pytest
import numpy as np

from flygym.anatomy import Skeleton, JointPreset, AxisOrder
from flygym.compose import Fly, FlatGroundWorld, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym.warp import WarpCPURenderer
from flygym.warp.rendering import modify_world_for_batch_rendering


# ---------------------------------------------------------------------------
# Module-scoped fixture: GPU simulation + WarpCPURenderer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def render_bundle(gpu_sim_factory):
    """GPUSimulation with 2 worlds and a WarpCPURenderer attached."""
    sim, fly, cam = gpu_sim_factory(n_worlds=2, fly_name="render_fly")
    sim.reset()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        renderer = sim.set_renderer(
            cam,
            camera_res=(64, 64),
            playback_speed=0.001,  # tiny interval: 1/(10/0.001) = 0.0001 s ≈ 1 step
            output_fps=10,
            worlds=[0, 1],
            use_gpu_batch_rendering=False,
            buffer_frames=True,
        )
    yield sim, fly, cam, renderer


# ==============================================================================
# WarpCPURenderer construction
# ==============================================================================


class TestWarpCPURendererConstruction:
    def test_type(self, render_bundle):
        _, _, _, renderer = render_bundle
        assert isinstance(renderer, WarpCPURenderer)

    def test_world_ids(self, render_bundle):
        _, _, _, renderer = render_bundle
        assert renderer.world_ids == [0, 1]

    def test_enabled_cam_names_not_empty(self, render_bundle):
        _, _, _, renderer = render_bundle
        assert len(renderer.enabled_cam_names) == 1

    def test_camera_res_stored(self, render_bundle):
        _, _, _, renderer = render_bundle
        assert renderer.camera_res == (64, 64)

    def test_buffer_frames_initialized_empty(self, render_bundle):
        _, _, _, renderer = render_bundle
        renderer.reset()
        assert renderer._frames == []

    def test_empty_worlds_raises(self, gpu_sim_factory):
        """Providing an empty worlds list should raise ValueError."""
        sim, fly, cam = gpu_sim_factory(n_worlds=2, fly_name="err_render_fly")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="world"):
                sim.set_renderer(
                    cam,
                    camera_res=(64, 64),
                    worlds=[],
                    use_gpu_batch_rendering=False,
                )


# ==============================================================================
# render_as_needed
# ==============================================================================


def _advance_past_render_interval(sim, renderer):
    """Step just past the renderer's render interval (derived from its own settings)."""
    steps_needed = int(renderer._secs_between_renders / sim.mj_model.opt.timestep) + 2
    for _ in range(steps_needed):
        sim.step()


class TestRenderAsNeeded:

    def test_returns_true_after_enough_time(self, render_bundle):
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        did_render = renderer.render_as_needed(sim.mjw_data)
        assert did_render is True

    def test_returns_false_too_soon(self, render_bundle):
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        renderer.render_as_needed(sim.mjw_data)  # first render

        # Immediately call again — not enough time has elapsed
        did_render = renderer.render_as_needed(sim.mjw_data)
        assert did_render is False

    def test_frame_buffered_after_render(self, render_bundle):
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        renderer.render_as_needed(sim.mjw_data)
        assert len(renderer._frames) == 1

    def test_buffered_frame_has_correct_world_and_cam_dims(self, render_bundle):
        """Stored frame array has leading dims (n_worlds_rendered, n_cams)."""
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        renderer.render_as_needed(sim.mjw_data)
        frame_array = renderer._frames[0]
        assert frame_array.shape[0] == len(renderer.world_ids)
        assert frame_array.shape[1] == len(renderer.enabled_cam_names)


# ==============================================================================
# Reset
# ==============================================================================


class TestWarpCPURendererReset:
    def test_reset_clears_frames(self, render_bundle):
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        renderer.render_as_needed(sim.mjw_data)
        assert len(renderer._frames) > 0

        renderer.reset()
        assert renderer._frames == []

    def test_reset_restarts_render_timer(self, render_bundle):
        """After reset the first render_as_needed at time 0 should fire immediately."""
        sim, fly, cam, renderer = render_bundle
        sim.reset()
        renderer.reset()
        # At time ≈ 0 the render fires because _last_render_time_sec = -inf
        did_render = renderer.render_as_needed(sim.mjw_data)
        assert did_render is True


# ==============================================================================
# save_video
# ==============================================================================


class TestSaveVideo:
    def _record_a_frame(self, sim, renderer):
        sim.reset()
        renderer.reset()
        _advance_past_render_interval(sim, renderer)
        renderer.render_as_needed(sim.mjw_data)

    def test_save_video_creates_file(self, render_bundle, tmp_path):
        sim, fly, cam, renderer = render_bundle
        self._record_a_frame(sim, renderer)
        out = tmp_path / "world0.mp4"
        renderer.save_video(world_id=0, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_video_no_frames_raises(self, render_bundle, tmp_path):
        _, _, _, renderer = render_bundle
        renderer.reset()
        with pytest.raises(RuntimeError, match="No frames"):
            renderer.save_video(world_id=0, output_path=tmp_path / "out.mp4")

    def test_save_video_invalid_world_raises(self, render_bundle, tmp_path):
        """Requesting a world that was not rendered should raise."""
        sim, fly, cam, renderer = render_bundle
        self._record_a_frame(sim, renderer)
        with pytest.raises((ValueError, RuntimeError)):
            renderer.save_video(world_id=99, output_path=tmp_path / "err.mp4")

    def test_save_video_with_camera_path_dict(self, render_bundle, tmp_path):
        """save_video accepts a dict mapping camera name to output path."""
        sim, fly, cam, renderer = render_bundle
        self._record_a_frame(sim, renderer)
        cam_name = renderer.enabled_cam_names[0]
        paths = {cam_name: tmp_path / "cam0.mp4"}
        renderer.save_video(world_id=0, output_path=paths)
        assert (tmp_path / "cam0.mp4").exists()


# ==============================================================================
# Subworld rendering
# ==============================================================================


class TestSubworldRendering:
    def test_only_specified_worlds_in_world_ids(self, gpu_sim_factory):
        """Renderer with worlds=[1] should only expose world 1 in world_ids."""
        sim, fly, cam = gpu_sim_factory(n_worlds=4, fly_name="sub_fly")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = sim.set_renderer(
                cam,
                camera_res=(64, 64),
                worlds=[1],
                use_gpu_batch_rendering=False,
            )
        assert renderer.world_ids == [1]

    def test_default_worlds_is_all_worlds(self, gpu_sim_factory):
        """When worlds=None all n_worlds worlds should be rendered."""
        sim, fly, cam = gpu_sim_factory(n_worlds=3, fly_name="all_worlds_fly")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = sim.set_renderer(
                cam,
                camera_res=(64, 64),
                worlds=None,
                use_gpu_batch_rendering=False,
            )
        assert renderer.world_ids == [0, 1, 2]


# ==============================================================================
# modify_world_for_batch_rendering
# ==============================================================================


class TestModifyWorldForBatchRendering:
    def _plain_world_with_fly(self, fly_name: str):
        fly = Fly(name=fly_name)
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        )
        fly.add_joints(skeleton, neutral_pose=pose)
        world = FlatGroundWorld()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))
        return fly, world

    def test_returns_bool(self):
        fly, world = self._plain_world_with_fly("batch_fly")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = modify_world_for_batch_rendering(world)
        assert isinstance(result, bool)

    def test_strips_textures_from_colorized_fly(self):
        """After modification fly body materials should have no texture."""
        fly = Fly(name="tex_fly")
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        )
        fly.add_joints(skeleton, neutral_pose=pose)
        fly.colorize()
        world = FlatGroundWorld()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modify_world_for_batch_rendering(world)

        for material in world.mjcf_root.asset.find_all("material"):
            if material.full_identifier.startswith(fly.name + "/"):
                assert material.texture is None, (
                    f"Fly material {material.full_identifier!r} still has a texture."
                )

    def test_is_modified_true_for_colorized_fly(self):
        """modify_world_for_batch_rendering should report a modification when
        a colorized fly (with textures) is present."""
        fly = Fly(name="mod_fly")
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        )
        fly.add_joints(skeleton, neutral_pose=pose)
        fly.colorize()
        world = FlatGroundWorld()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = modify_world_for_batch_rendering(world)
        assert result is True
