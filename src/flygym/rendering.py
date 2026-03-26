from multiprocessing import Process
from pathlib import Path
from typing import Any
from os import PathLike
from collections.abc import Sequence

import mujoco as mj
import mujoco.viewer as mjviewer
import dm_control.mjcf as mjcf
import mediapy
import imageio.v3 as iio
import numpy as np

__all__ = ["Renderer", "launch_interactive_viewer", "preview_model"]


class Renderer:
    def __init__(
        self,
        mj_model: mj.MjModel,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        **kwargs: Any,
    ):
        self.mj_model = mj_model
        self.camera_res = camera_res
        nrows, ncols = camera_res
        self.buffer_frames = buffer_frames
        self.mj_renderer = mj.Renderer(mj_model, nrows, ncols, **kwargs)

        if scene_option is None:
            self.scene_option = mj.MjvOption()
        else:
            self.scene_option = scene_option
        mj.mjv_defaultOption(self.scene_option)  # this sets default scene options

        self._cameras_names2id = {}
        for spec in camera if isinstance(camera, Sequence) else [camera]:
            cam_id, cam_name = self._resolve_camera_id_and_name(spec)
            if cam_id == -1:
                raise ValueError(f"Camera {spec} not found in the model.")
            if cam_name in self._cameras_names2id:
                raise ValueError(f"Duplicate camera name detected: {cam_name}.")
            self._cameras_names2id[cam_name] = cam_id
        if len(self._cameras_names2id) == 0:
            raise ValueError("At least one valid camera must be specified.")

        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)

        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self.frames = {cam_name: [] for cam_name in self._cameras_names2id}
        else:
            self.frames = None

    def render_as_needed(self, mj_data: mj.MjData) -> bool:
        if mj_data.time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = mj_data.time
            for cam_name, internal_cam_id in self._cameras_names2id.items():
                self.mj_renderer.update_scene(
                    mj_data, internal_cam_id, self.scene_option
                )
                frame = self.mj_renderer.render()
                if self.buffer_frames:
                    self.frames[cam_name].append(frame)
            return True
        else:
            return False

    def reset(self):
        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self.frames = {cam_name: [] for cam_name in self._cameras_names2id}

    def close(self):
        self.mj_renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def show_in_notebook(
        self,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element] | None = None,
        **kwargs,
    ) -> None:
        """Display recorded frames in a Jupyter notebook.

        Args:
            camera: Camera(s) to display. If None, displays all cameras.
            **kwargs: Additional arguments passed to mediapy.show_video
        """
        camera_names = self._normalize_camera_spec(camera)

        for cam_name in camera_names:
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")
            mediapy.show_video(frames, fps=self.output_fps, title=cam_name, **kwargs)

    def save_video(
        self,
        output_path: dict[str | mjcf.Element, PathLike] | PathLike,
        **kwargs,
    ) -> None:
        """Save recorded frames as video files.

        Args:
            output_path: Either a dict mapping camera specs to file paths, or:
                - If single camera: a file path to save to
                - If multiple cameras: a directory path to save all videos to
            **kwargs: Additional arguments passed to imageio.imwrite
        """
        path_by_camera = self._resolve_output_paths(output_path)

        for cam_name, path in path_by_camera.items():
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")

            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                path,
                frames,
                fps=self.output_fps,
                codec="libx264",
                quality=8,
                **kwargs,
            )

    def _normalize_camera_spec(
        self,
        camera: str | mjcf.Element | Sequence[str | mjcf.Element] | None,
    ) -> list[str]:
        """Convert various camera specifications to a list of camera names.

        Args:
            camera: Camera specification (single, sequence, or None for all)

        Returns:
            List of camera names

        Raises:
            ValueError: If camera spec is invalid or refers to unavailable cameras
        """
        if camera is None:
            return list(self._cameras_names2id.keys())
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

        # Validate all cameras are available
        for cam_name in camera_names:
            if cam_name not in self._cameras_names2id:
                raise ValueError(
                    f"Camera '{cam_name}' is not available in this renderer. "
                    f"Available cameras: {list(self._cameras_names2id.keys())}"
                )

        return camera_names

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
                if cam_name not in self._cameras_names2id:
                    raise ValueError(
                        f"Camera '{cam_name}' in output_path is not available. "
                        f"Available cameras: {list(self._cameras_names2id.keys())}"
                    )
                result[cam_name] = Path(path)
            return result

        # Single path provided - interpret based on number of cameras
        path = Path(output_path)
        available_cameras = list(self._cameras_names2id.keys())

        if len(available_cameras) == 1:
            # Single camera: interpret path as file
            return {available_cameras[0]: path}
        else:
            # Multiple cameras: interpret path as directory
            return {
                cam_name: path / f"{cam_name.replace('/', '_')}.mp4"
                for cam_name in available_cameras
            }

    def _resolve_camera_id_and_name(
        self, camera: str | mjcf.Element, /
    ) -> tuple[int, str]:
        """Convert a camera specification to (internal_id, camera_name)."""
        if isinstance(camera, str):
            cam_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, camera)
            return cam_id, camera
        elif isinstance(camera, mjcf.Element):
            cam_name = camera.full_identifier
            cam_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
            return cam_id, cam_name
        else:
            raise ValueError(
                f"Invalid camera spec: {camera}. Must be one of str or mjcf.Element."
            )


def launch_interactive_viewer(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    run_async: bool = False,
    init_keyframe: str | None = "neutral",
) -> None:
    """Launch MuJoCo's built-in interactive viewer. If `run_async` is True, the viewer
    will be launched in a separate process and this function will return immediately.
    It should be set to True when launching from a Jupyter notebook."""

    if init_keyframe is not None:
        key_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, init_keyframe)
        mj.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    if run_async:
        p = Process(target=mj.viewer.launch, args=(mj_model, mj_data))
        p.start()
        # Don't join!
    else:
        mjviewer.launch(mj_model, mj_data)


def preview_model(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    camera: mjcf.Element | str,
    *,
    init_keyframe: str | None = "neutral",
    duration: float = 0.1,
    camera_res: tuple[int, int] = (240, 320),
    playback_speed=0.1,
    output_fps: int = 25,
    show_in_notebook: bool = False,
    output_path: PathLike | None = None,
    **kwargs,
):
    if init_keyframe is not None:
        key_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, init_keyframe)
        mj.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    n_steps = int(duration / mj_model.opt.timestep)

    with Renderer(
        mj_model,
        camera,
        camera_res=camera_res,
        playback_speed=playback_speed,
        output_fps=output_fps,
        **kwargs,
    ) as renderer:
        for step in range(n_steps):
            mj.mj_step(mj_model, mj_data)
            renderer.render_as_needed(mj_data)

        if show_in_notebook:
            renderer.show_in_notebook()
        if output_path:
            renderer.save_video(output_path)
