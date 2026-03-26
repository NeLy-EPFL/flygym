from typing import Any, override
from os import PathLike
from abc import ABC, abstractmethod

import mediapy
import mujoco as mj
import mujoco_warp as mjw
import dm_control.mjcf as mjcf
import warp as wp
import numpy as np
import imageio.v3 as iio

from flygym.rendering import Renderer
from flygym.warp.utils import get_rgb_selected_worlds_and_cameras


class _BaseWarpRenderer(Renderer, ABC):
    @override
    def __init__(
        self,
        mj_model: mj.MjModel,
        cameras: str | mjcf.Element | list[str | mjcf.Element],
        n_worlds_total: int | None = None,
        *,
        worlds: list[int] | None = None,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        **kwargs: Any,
    ):
        # Common setup using MjRenderer
        super().__init__(
            mj_model,
            cameras,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            buffer_frames=buffer_frames,
            scene_option=scene_option,
            **kwargs,
        )

        # Warp-specific setup
        self.mjw_model = mjw.put_model(mj_model)
        self._n_worlds_total = n_worlds_total
        self.camera_res = camera_res
        self.buffer_frames = buffer_frames

        # Figure out which worlds should be rendered
        if worlds is None:
            if n_worlds_total is None:
                raise ValueError(
                    "If 'worlds' is not specified, all worlds are rendered. "
                    "In that case, 'n_worlds_total' must be specified."
                )
            worlds = list(range(n_worlds_total))
        self.world_ids = worlds
        if len(self.world_ids) == 0:
            raise ValueError("At least one valid world must be specified.")

        # Figure out which cameras should be rendered
        if not isinstance(cameras, list):
            cameras = [cameras]
        self.enabled_cam_names = []
        for cam in cameras if isinstance(cameras, list) else [cameras]:
            _, cam_name = self._resolve_camera_id_and_name(cam)
            self.enabled_cam_names.append(cam_name)
        if len(self.enabled_cam_names) == 0:
            raise ValueError("At least one valid camera must be specified.")

        # Set up parameters for automatically determining when to render frames
        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)
        self._last_render_time_sec = -np.inf

        # Buffer to store rendered images
        n_worlds_to_render = len(self.world_ids)
        n_cams_to_render = len(self.enabled_cam_names)
        self._buf_dim_per_frame = (n_worlds_to_render, n_cams_to_render, *camera_res)
        if buffer_frames:
            self._frames: list[np.ndarray | wp.array] = []
        else:
            self._frames = None

        self._render_setup_impl(**kwargs)

    @override
    def render_as_needed(self, mjw_data: mjw.Data) -> bool:
        curr_time = mjw_data.time.numpy()[0]  # assume all worlds have the same time
        if curr_time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = curr_time
            rendered_images = self._render_impl(mjw_data)
            if self.buffer_frames:
                self._frames.append(rendered_images)
            return True
        else:
            return False

    @override
    def reset(self):
        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self._frames = []

    @override
    def show_in_notebook(
        self,
        world_id: int,
        camera: str | mjcf.Element | list[str | mjcf.Element] | None = None,
        **kwargs,
    ):
        """Display recorded frames in a Jupyter notebook.

        Args:
            world_id: Which parallel world to display frames for
            camera: Camera(s) to display. If None, displays all enabled cameras.
            **kwargs: Additional arguments passed to mediapy.show_video
        """
        camera_names = self._normalize_camera_spec(camera)

        for cam_name in camera_names:
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            frames = self._fetch_frames_to_cpu(world_id, cam_id)
            title = f"world {world_id}, camera {cam_name}"
            mediapy.show_video(frames, fps=self.output_fps, title=title, **kwargs)

    @override
    def save_video(
        self,
        world_id: int,
        output_path: dict[str | mjcf.Element, PathLike] | PathLike,
        **kwargs,
    ) -> None:
        """Save recorded frames as video files.

        Args:
            world_id: Which parallel world to save frames for
            output_path: Either a dict mapping camera specs to file paths, or:
                - If single camera: a file path to save to
                - If multiple cameras: a directory path to save all videos to
            **kwargs: Additional arguments passed to imageio.imwrite
        """
        path_by_camera = self._resolve_output_paths(output_path)

        for cam_name, path in path_by_camera.items():
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            frames = self._fetch_frames_to_cpu(world_id, cam_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, frames, fps=self.output_fps, codec="libx264", **kwargs)

    def _fetch_frames_to_cpu(self, world_id: int, cam_id: int) -> list[np.ndarray]:
        if not self.buffer_frames:
            raise RuntimeError(
                "Frame buffering was disabled for this renderer, so recorded frames "
                "are not available for saving or display."
            )

        if len(self._frames) == 0:
            raise RuntimeError("No frames have been recorded yet.")

        if world_id not in self.world_ids:
            raise ValueError(
                f"world_id {world_id} was not among the rendered worlds: "
                f"{self.world_ids}"
            )
        world_id_among_rendered = self.world_ids.index(world_id)

        cam_name = self._cameras_id2name.get(cam_id, None)
        if cam_name is None:
            raise ValueError(f"Camera ID '{cam_id}' not found.")
        if cam_name not in self.enabled_cam_names:
            raise ValueError(
                f"Camera '{cam_name}' (ID {cam_id}) was not among the rendered cameras."
            )
        cam_id_among_rendered = self.enabled_cam_names.index(cam_name)

        return self._fetch_frames_to_cpu_impl(
            world_id_among_rendered, cam_id_among_rendered
        )

    @abstractmethod
    def _render_setup_impl(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def _render_impl(self, mjw_data: mjw.Data) -> np.ndarray | wp.array:
        pass

    @abstractmethod
    def _fetch_frames_to_cpu_impl(
        self, world_id_among_rendered: int, cam_id_among_rendered: int
    ) -> list[np.ndarray]:
        pass


class WarpGPUBatchRenderer(_BaseWarpRenderer):
    """GPU-side renderer using MJWarp's GPU batch rendering functionality."""

    def _render_setup_impl(self, **kwargs: Any) -> None:
        if not self._is_scene_option_default(self.scene_option):
            raise RuntimeError(
                "Custom scene options are not supported with WarpGPUBatchRenderer "
                "because it is not implemented in MJWarp batch rendering."
            )

        self._world_ids_gpu = wp.array(self.world_ids, dtype=wp.int32)
        self._enabled_cam_ids_gpu = wp.array(
            [self._cameras_names2id[n] for n in self.enabled_cam_names], dtype=wp.int32
        )
        cam_mask = [
            self._cameras_id2name[cid] in self.enabled_cam_names
            for cid in range(self.mj_model.ncam)
        ]

        # Create batch rendering context
        self._rendering_context = mjw.create_render_context(
            mjm=self.mj_model,
            nworld=self._n_worlds_total,
            cam_active=cam_mask,
            cam_res=self.camera_res[::-1],  # MJWarp expects (W, H); we use (H, W)
            **kwargs,
        )

        # Remove normal MjRenderer inherited from CPU Renderer
        self.scene_option = None
        self.mj_renderer = None

    def _render_impl(self, mjw_data: mjw.Data) -> bool:
        mjw.refit_bvh(self.mjw_model, mjw_data, self._rendering_context)
        mjw.render(self.mjw_model, mjw_data, self._rendering_context)
        rgb_out = wp.zeros(self._buf_dim_per_frame, dtype=wp.vec3f)
        get_rgb_selected_worlds_and_cameras(
            self._rendering_context,
            self._world_ids_gpu,
            self._enabled_cam_ids_gpu,
            rgb_out,
        )
        return rgb_out

    def _fetch_frames_to_cpu_impl(
        self, world_id_among_rendered: int, cam_id_among_rendered: int
    ) -> list[np.ndarray]:
        frames = []
        for frame_buffer in self._frames:
            frame = frame_buffer[world_id_among_rendered, cam_id_among_rendered, :, :]
            frame = (frame * 255.0).numpy().astype(np.uint8)
            frames.append(frame)
        return frames

    @override
    def close(self):
        return  # nothing to do since we are not using a mj.Renderer context

    @staticmethod
    def _is_scene_option_default(scene_option: mj.MjvOption) -> bool:
        default_option = mj.MjvOption()
        mj.mjv_defaultOption(default_option)
        return scene_option == default_option


class WarpCPURenderer(_BaseWarpRenderer):
    """CPU-side renderer for multi-world MJWarp simulation."""

    def _render_setup_impl(self, **kwargs: Any) -> None:
        self._mj_data_buffer = mj.MjData(self.mj_model)
        # Nothing else to do - just use mjRenderer inherited from CPU Renderer

    def _render_impl(self, mjw_data: mjw.Data) -> bool:
        rendered_images = np.zeros((*self._buf_dim_per_frame, 3), dtype=np.uint8)

        for world_id in self.world_ids:
            wid_among_rendered = self.world_ids.index(world_id)

            # Copy data into CPU MjData struct
            mj.mj_resetData(self.mj_model, self._mj_data_buffer)
            mjw.get_data_into(self._mj_data_buffer, self.mj_model, mjw_data, world_id)

            # Render each enabled camera and store frames
            for cam_name, internal_cam_id in self._cameras_names2id.items():
                cid_among_rendered = self.enabled_cam_names.index(cam_name)

                self.mj_renderer.update_scene(
                    self._mj_data_buffer, internal_cam_id, self.scene_option
                )
                frame = self.mj_renderer.render()

                if self.buffer_frames:
                    rendered_images[wid_among_rendered, cid_among_rendered] = frame

        return rendered_images

    def _fetch_frames_to_cpu_impl(
        self, world_id_among_rendered: int, cam_id_among_rendered: int
    ) -> list[np.ndarray]:
        frames = []
        for rendered_images in self._frames:
            frame = rendered_images[world_id_among_rendered, cam_id_among_rendered, ...]
            frames.append(frame)
        return frames
