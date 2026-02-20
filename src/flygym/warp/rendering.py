from pathlib import Path
from typing import Any, override
from collections.abc import Sequence
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


class GPURenderer(Renderer):
    """GPU-side renderer using mujoco_warp.

    See https://github.com/google-deepmind/mujoco_warp/pull/1113.

    Args:
        todo
    """

    @override
    def __init__(
        self,
        mjw_model: mjw.Model,
        mjw_data: mjw.Data,
        mj_model: mj.MjModel,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        **kwargs: Any,
    ):
        self.mjw_model = mjw_model
        self.mj_model = mj_model
        self.camera_res = camera_res

        enabled_cam_ids = (
            [self._resolve_camera_id_and_name(spec)[0] for spec in camera]
            if isinstance(camera, Sequence)
            else [self._resolve_camera_id_and_name(camera)[0]]
        )
        if not enabled_cam_ids:
            raise ValueError("At least one valid camera must be specified.")
        self._cam_enable_mask = [
            intid in enabled_cam_ids for intid in range(mj_model.ncam)
        ]
        self._rendering_context = mjw.create_render_context(
            mjm=mj_model,
            nworld=mjw_data.nworld,
            cam_res=camera_res[::-1],  # mjw render context expects (W, H)!
            cam_active=self._cam_enable_mask,
            **kwargs,
        )

        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)

        self._last_render_time_sec = -np.inf
        self._rgb_data_and_adr: list[tuple[wp.Array, wp.Array]] = []

    @override
    def render_as_needed(self, mjw_data: mjw.Data) -> bool:
        curr_time = mjw_data.time.numpy()[0]  # assume all worlds have the same time
        if curr_time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = curr_time
            mjw.refit_bvh(self.mjw_model, mjw_data, self._rendering_context)
            mjw.render(self.mjw_model, mjw_data, self._rendering_context)
            rgb_data = wp.clone(self._rendering_context.rgb_data)
            rgb_adr = wp.clone(self._rendering_context.rgb_adr)
            self._rgb_data_and_adr.append((rgb_data, rgb_adr))
            return True
        else:
            return False

    @override
    def reset(self):
        self._last_render_time_sec = -np.inf
        self._rgb_data_and_adr.clear()

    @override
    def close(self):
        return  # nothing to do since we are not using a mj.Renderer context

    @override
    def show_in_notebook(
        self,
        world_id: int,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element] | None = None,
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
            cam_id_among_enabled = sum(self._cam_enable_mask[:cam_id])
            frames = self._decode_frames(world_id, cam_id_among_enabled)

            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")

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
            cam_id_among_enabled = sum(self._cam_enable_mask[:cam_id])
            frames = self._decode_frames(world_id, cam_id_among_enabled)

            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")

            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, frames, fps=self.output_fps, codec="libx264", **kwargs)
    
    def _decode_frames(
        self, world_id: int, cam_id_among_enabled: int
    ) -> Float[np.ndarray, "nframes nrows ncols 3"]:
        n_pxs = self.camera_res[0] * self.camera_res[1]
        frames = []
        for rgb_data_arr, rgb_adr_arr in self._rgb_data_and_adr:
            my_rgb_adr = rgb_adr_arr.numpy()[cam_id_among_enabled]
            # RGB images are packed as 0xAARRGGBB in uint32 during rendering,
            # so we need to unpack them
            rgb_packed = rgb_data_arr.numpy()[world_id, my_rgb_adr : my_rgb_adr + n_pxs]
            rgb_packed = rgb_packed.reshape(self.camera_res)
            rgb_image = np.dstack(
                [
                    ((rgb_packed >> 16) & 0xFF).astype(np.uint8),  # R
                    ((rgb_packed >> 8) & 0xFF).astype(np.uint8),  # G
                    (rgb_packed & 0xFF).astype(np.uint8),  # B
                ]
            )
            frames.append(rgb_image)

        return np.stack(frames, axis=0)

    @override
    def _normalize_camera_spec(
        self,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element] | None,
    ) -> list[str]:
        """Convert various camera specifications to a list of camera names.

        Args:
            camera: Camera specification (single, sequence, or None for all enabled)

        Returns:
            List of camera names

        Raises:
            ValueError: If camera spec is invalid or refers to disabled cameras
        """
        if camera is None:
            # Return all enabled cameras
            camera_names = []
            for cam_id, enabled in enumerate(self._cam_enable_mask):
                if enabled:
                    cam_name = mj.mj_id2name(
                        self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_id
                    )
                    camera_names.append(cam_name)
            return camera_names
        elif isinstance(camera, (str, mjcf.Element)):
            _, cam_name = self._resolve_camera_id_and_name(camera)
            camera_names = [cam_name]
        elif isinstance(camera, Sequence):
            camera_names = [self._resolve_camera_id_and_name(c)[1] for c in camera]
        else:
            raise ValueError(
                f"Invalid camera spec type: {type(camera)}. Must be str, "
                "mjcf.Element, Sequence of these, or None."
            )

        # Validate all cameras are enabled
        for cam_name in camera_names:
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            if not self._cam_enable_mask[cam_id]:
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
                if not self._cam_enable_mask[cam_id]:
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
        for cam_id, enabled in enumerate(self._cam_enable_mask):
            if enabled:
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
