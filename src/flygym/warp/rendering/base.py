from typing import Any, override
from os import PathLike
from abc import ABC, abstractmethod
from enum import Enum

import mediapy
import mujoco as mj
import mujoco_warp as mjw
import warp as wp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flygym.rendering.live_rendering import Renderer
from flygym.utils.video import write_video_from_frames
from flygym.utils.plot import find_font_path

__all__ = ["RendererType", "_BaseWarpRenderer"]


class RendererType(Enum):
    CPU = "cpu"
    GPU_BATCH = "gpu_batch"
    RECORDED_TRAJECTORY = "recorded_trajectory"


class _BaseWarpRenderer(Renderer, ABC):
    @override
    def __init__(
        self,
        mj_model: mj.MjModel,
        cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
        sim_timestep: float,
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
        self.curr_time = 0.0
        self.sim_timestep = sim_timestep

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
        if self.curr_time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = self.curr_time
            rendered_images = self._render_impl(mjw_data)
            if self.buffer_frames:
                self._frames.append(rendered_images)
            rendered = True
        else:
            rendered = False
        self.curr_time += self.sim_timestep
        return rendered

    @override
    def reset(self):
        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self._frames = []

    @override
    def show_in_notebook(
        self,
        world_id: int,
        camera: str | mj.MjsCamera | list[str | mj.MjsCamera] | None = None,
        scale: float | None = None,
        **kwargs,
    ):
        """Display recorded frames in a Jupyter notebook.

        Args:
            world_id: Which parallel world to display frames for
            camera: Camera(s) to display. If None, displays all enabled cameras.
            scale: Optional factor by which to rescale frames before display.
            **kwargs: Additional arguments passed to mediapy.show_video
        """
        camera_names = self._normalize_camera_spec(camera)

        for cam_name in camera_names:
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            if isinstance(world_id, int):
                frames = self._fetch_frames_to_cpu_oneworld(world_id, cam_id, scale)
            else:
                frames = self._fetch_frames_to_cpu_multipleworlds(
                    world_id, cam_id, scale
                )
            title = f"world {world_id}, camera {cam_name}"
            mediapy.show_video(frames, fps=self.output_fps, title=title, **kwargs)

    @override
    def save_video(
        self,
        world_id: int | list[int],
        output_path: dict[str | mj.MjsCamera, PathLike] | PathLike,
        scale: float | None = None,
        **kwargs,
    ) -> None:
        """Save recorded frames as video files.

        Args:
            world_id: Which parallel world to save frames for
            output_path: Either a dict mapping camera specs to file paths, or:
                - If single camera: a file path to save to
                - If multiple cameras: a directory path to save all videos to
            scale: Optional factor by which to rescale frames before saving.
            **kwargs: Additional arguments passed to imageio.imwrite
        """
        path_by_camera = self._resolve_output_paths(output_path)

        for cam_name, path in path_by_camera.items():
            cam_id, _ = self._resolve_camera_id_and_name(cam_name)
            if isinstance(world_id, int):
                frames = self._fetch_frames_to_cpu_oneworld(world_id, cam_id, scale)
            else:
                frames = self._fetch_frames_to_cpu_multipleworlds(
                    world_id, cam_id, scale
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            write_video_from_frames(
                path, frames, fps=self.output_fps, codec="libx264", **kwargs
            )

    def _fetch_frames_to_cpu_oneworld(
        self, world_id: int, cam_id: int, scale: float | None = None
    ) -> list[np.ndarray]:
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

        frames = self._fetch_frames_to_cpu_impl(
            world_id_among_rendered, cam_id_among_rendered
        )
        if scale is not None:
            render_res = tuple(int(x * scale) for x in self.camera_res)
            for i, frame in enumerate(frames):
                pil_frame = Image.fromarray(frame)
                pil_frame_resized = pil_frame.resize(
                    render_res[::-1],  # PIL expects (W, H); we use (H, W)
                    resample=Image.Resampling.LANCZOS,
                )
                frame_resized = np.array(pil_frame_resized)
                frames[i] = frame_resized
        return frames

    def _fetch_frames_to_cpu_multipleworlds(
        self, world_ids: list[int], cam_id: int, scale: float | None
    ) -> dict[int, list[np.ndarray]]:
        # Set up canvas for displaying frames from multiple worlds in a grid
        n_worlds = len(world_ids)
        n_rows = int(np.ceil(np.sqrt(n_worlds)))
        n_cols = int(np.ceil(n_worlds / n_rows))
        # If scale is unspecified, make the output resolution roughly matches the
        # resolution of a single world/camera - this avoids creating an excessively
        # large canvas when there are many worlds.
        if scale is None:
            scale = 1 / n_cols
        render_res = tuple(int(x * scale) for x in self.camera_res)
        canvas_shape = (render_res[0] * n_rows, render_res[1] * n_cols)
        n_frames = len(self._frames)
        merged_frames = [
            np.zeros((*canvas_shape, 3), dtype=np.uint8) for _ in range(n_frames)
        ]

        # Set up font for overlaying world IDs
        FONT_FAMILY = "Arial"
        FONT_SIZE_RATIO_OF_HEIGHT = 0.07
        MIN_FONT_SIZE = 7
        font_path = find_font_path(FONT_FAMILY)
        font_size = max(MIN_FONT_SIZE, int(render_res[0] * FONT_SIZE_RATIO_OF_HEIGHT))
        font = ImageFont.truetype(font_path, font_size)

        # Fetch frames for each world and paste them onto the canvas
        for i, wid in enumerate(world_ids):
            row = i // n_cols
            col = i % n_cols
            row_slice = slice(row * render_res[0], (row + 1) * render_res[0])
            col_slice = slice(col * render_res[1], (col + 1) * render_res[1])

            world_frames = self._fetch_frames_to_cpu_oneworld(wid, cam_id, scale)
            assert len(world_frames) == n_frames, "inconsistent frame counts"
            for j, world_frame in enumerate(world_frames):
                # Overlay world ID text
                pil_frame = Image.fromarray(world_frame)
                draw = ImageDraw.Draw(pil_frame)
                text = f"World {wid}"
                text_pos = (0.03 * render_res[1], 0.02 * render_res[0])  # (x, y)
                draw.text(text_pos, text, font=font, fill=(255, 255, 255))
                world_frame = np.array(pil_frame)
                # Paste onto canvas
                merged_frames[j][row_slice, col_slice] = world_frame

        return merged_frames

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
