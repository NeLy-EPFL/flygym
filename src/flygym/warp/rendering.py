from pathlib import Path
from typing import Any, override
from os import PathLike

import mediapy
import mujoco as mj
import mujoco_warp as mjw
import dm_control.mjcf as mjcf
import warp as wp
import numpy as np
import imageio.v3 as iio
from jaxtyping import Float

from flygym.rendering import Renderer
from flygym.warp.utils import get_rgb_selected_worlds_and_cameras


class GPURenderer(Renderer):
    """GPU-side renderer using mujoco_warp.

    Args:
        TODO
    """

    @override
    def __init__(
        self,
        mj_model: mj.MjModel,
        n_worlds: int,
        cameras: str | mjcf.Element | list[str | mjcf.Element],
        worlds: list[int] | None,
        camera_res: tuple[int, int],
        playback_speed: float,
        output_fps: int,
        buffer_frames: bool,
        **kwargs: Any,
    ):
        self.mj_model = mj_model
        self.camera_res = camera_res
        self.buffer_frames = buffer_frames

        # Figure out which worlds should be rendered
        self.world_ids, self._world_mask, self._world_ids_gpu, self._world_mask_gpu = (
            self._get_world_ids_and_mask(worlds, n_worlds)
        )
        if len(self.world_ids) == 0:
            raise ValueError("At least one valid world must be specified.")

        # Figure out which cameras should be rendered
        self.cam_ids, self._cam_mask, self._cam_ids_gpu, self._cam_mask_gpu = (
            self._get_camera_ids_and_mask(cameras)
        )
        if len(self.cam_ids) == 0:
            raise ValueError("At least one valid camera must be specified.")

        # Create batch rendering context
        self._rendering_context = mjw.create_render_context(
            mjm=mj_model,
            nworld=n_worlds,
            cam_active=self._cam_mask.tolist(),
            cam_res=camera_res[::-1],  # MJWarp expects (W, H); we use (H, W)
            **kwargs,
        )

        # Set up parameters for automatically determining when to render frames
        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)
        self._last_render_time_sec = -np.inf

        # Buffer to store rendered images
        self._buf_dim_per_frame = (len(self.world_ids), len(self.cam_ids), *camera_res)
        if buffer_frames:
            self._frames: list[wp.Array] = []
        else:
            self._frames = None

    @override
    def render_as_needed(self, mjw_model: mjw.Model, mjw_data: mjw.Data) -> bool:
        curr_time = mjw_data.time.numpy()[0]  # assume all worlds have the same time
        if curr_time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = curr_time
            mjw.refit_bvh(mjw_model, mjw_data, self._rendering_context)
            mjw.render(mjw_model, mjw_data, self._rendering_context)
            rgb_out = wp.zeros(self._buf_dim_per_frame, dtype=wp.vec3f)
            get_rgb_selected_worlds_and_cameras(
                self._rendering_context,
                self._world_ids_gpu,
                self._cam_ids_gpu,
                rgb_out,
            )
            if self.buffer_frames:
                self._frames.append(rgb_out)
            return True
        else:
            return False

    @override
    def reset(self):
        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self._frames = []

    @override
    def close(self):
        return  # nothing to do since we are not using a mj.Renderer context

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

        if cam_id not in self.cam_ids:
            raise ValueError(
                f"cam_id '{cam_id}' was not among the rendered cameras: "
                f"{self.cam_ids}"
            )
        cam_id_among_rendered = self.cam_ids.index(cam_id)

        frames = []
        for frame_buffer in self._frames:
            frame = frame_buffer[world_id_among_rendered, cam_id_among_rendered, :, :]
            frame = (frame * 255.0).numpy().astype(np.uint8)
            frames.append(frame)
        return frames

    @override
    def _normalize_camera_spec(
        self,
        camera: str | mjcf.Element | list[str | mjcf.Element] | None,
    ) -> list[str]:
        """Convert various camera specifications to a list of camera names.

        Args:
            camera: Camera specification (single, list, or None for all enabled)

        Returns:
            List of camera names

        Raises:
            ValueError: If camera spec is invalid or refers to disabled cameras
        """
        if camera is None:
            # Return all enabled cameras
            camera_names = []
            for cam_id, is_enabled in enumerate(self._cam_mask):
                if is_enabled:
                    cam_name = mj.mj_id2name(
                        self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_id
                    )
                    camera_names.append(cam_name)
            return camera_names
        elif isinstance(camera, (str, mjcf.Element)):
            _, cam_name = self._resolve_camera_id_and_name(camera)
            camera_names = [cam_name]
        elif isinstance(camera, list):
            camera_names = [self._resolve_camera_id_and_name(c)[1] for c in camera]
        else:
            raise ValueError(
                f"Invalid camera spec type: {type(camera)}. Must be str, "
                "mjcf.Element, list of these, or None."
            )

        # Validate all cameras are enabled
        for cam_name in camera_names:
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            if not self._cam_mask[cam_id]:
                raise ValueError(
                    f"Camera '{cam_name}' (id {cam_id}) was not enabled when "
                    f"the renderer was created."
                )

        return camera_names

    @override
    def _resolve_output_paths(
        self,
        output_path: dict[str | mjcf.Element, PathLike] | PathLike,
    ) -> dict[str, Path]:
        """Convert output_path specification to dict mapping camera names to Paths.

        Args:
            output_path: Either a dict mapping cameras to paths, or a single path

        Returns:
            Dict mapping camera name to output Path

        Raises:
            ValueError: If output_path format is invalid
        """
        if isinstance(output_path, dict):
            # Explicit mapping provided
            result = {}
            for cam_spec, path in output_path.items():
                _, cam_name = self._resolve_camera_id_and_name(cam_spec)
                cam_id, _ = self._resolve_camera_id_and_name(cam_name)
                if not self._cam_mask[cam_id]:
                    raise ValueError(
                        f"Camera '{cam_name}' (id {cam_id}) in output_path was not "
                        f"enabled when the renderer was created."
                    )
                result[cam_name] = Path(path)
            return result

        # Single path provided - interpret based on number of enabled cameras
        path = Path(output_path)

        # Get list of enabled camera names
        enabled_cameras = []
        for cam_id, is_enabled in enumerate(self._cam_mask):
            if is_enabled:
                cam_name = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
                enabled_cameras.append(cam_name)

        if len(enabled_cameras) == 1:
            # Single camera: interpret path as file
            return {enabled_cameras[0]: path}
        else:
            # Multiple cameras: interpret path as directory
            return {
                cam_name: path / f"{cam_name.replace('/', '_')}.mp4"
                for cam_name in enabled_cameras
            }

    @staticmethod
    def _get_world_ids_and_mask(worlds: list[int] | None, n_worlds: int) -> list[int]:
        if worlds is not None:
            world_indices_cpu = list(worlds)
        else:
            world_indices_cpu = list(range(n_worlds))

        worlds_mask_cpu = np.array(
            [w in world_indices_cpu for w in range(n_worlds)], dtype=np.int32
        )

        worlds_indices_gpu = wp.array(world_indices_cpu, dtype=wp.int32)
        worlds_mask_gpu = wp.array(worlds_mask_cpu, dtype=wp.int32)

        return world_indices_cpu, worlds_mask_cpu, worlds_indices_gpu, worlds_mask_gpu

    def _get_camera_ids_and_mask(self, camera):
        if not isinstance(camera, list):
            camera = [camera]
        cams_indices_cpu = [self._resolve_camera_id_and_name(c)[0] for c in camera]
        cams_mask_cpu = np.array(
            [c in cams_indices_cpu for c in range(self.mj_model.ncam)],
            dtype=np.int32,
        )
        cams_indices_gpu = wp.array(cams_indices_cpu, dtype=wp.int32)
        cams_mask_gpu = wp.array(cams_mask_cpu, dtype=wp.int32)
        return cams_indices_cpu, cams_mask_cpu, cams_indices_gpu, cams_mask_gpu
