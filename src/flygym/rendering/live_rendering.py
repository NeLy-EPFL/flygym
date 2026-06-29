"""Live (real-time) MuJoCo rendering: the `Renderer` and viewer helpers.

For recording kinematic trajectories instead of rasterizing frames, and replaying
them to video, see `flygym.rendering.recorded_trajectory`.
"""

import warnings
from multiprocessing import Process
from pathlib import Path
from typing import Any
from os import PathLike

import mujoco as mj
import mujoco.viewer as mjviewer
import mediapy
import imageio.v3 as iio
import numpy as np


__all__ = ["Renderer", "launch_interactive_viewer", "preview_model"]


class Renderer:
    """Renders MuJoCo scenes to video frames.

    Args:
        mj_model: Compiled MuJoCo model.
        cameras: Camera(s) to render. Can be a camera name, MJCF camera element,
            or a sequence of either.
        camera_res: ``(height, width)`` in pixels.
        playback_speed: Video playback speed relative to real time.
        output_fps: Output video frame rate.
        buffer_frames: If True, store rendered frames on the renderer.
        scene_option: MuJoCo scene options. Uses defaults if None.
        render_rgb: If True, render RGB frames (stored in ``self.frames``).
        render_depth: If True, render depth maps (raw float arrays in
            ``self.depth_frames``). Not saved/shown as video.
        render_segmentation: If True, render segmentation masks (raw int arrays in
            ``self.segmentation_frames``). Not saved/shown as video.
        **kwargs: Passed to ``mujoco.Renderer``.

    ``render_rgb``, ``render_depth``, and ``render_segmentation`` are independent;
    enable any combination. At least one must be True.

    Attributes:
        frames: Dict mapping camera name to list of RGB frames (uint8), or None if
            ``render_rgb`` is False. Only populated when ``buffer_frames=True``.
        depth_frames: Like ``frames`` but float depth maps; None unless
            ``render_depth``.
        segmentation_frames: Like ``frames`` but int32 segmentation object ids
            (-1 = background); None unless ``render_segmentation``.
    """

    def __init__(
        self,
        mj_model: mj.MjModel,
        cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        render_rgb: bool = True,
        render_depth: bool = False,
        render_segmentation: bool = False,
        **kwargs: Any,
    ):
        self.mj_model = mj_model
        self.camera_res = camera_res
        nrows, ncols = camera_res
        self.buffer_frames = buffer_frames

        self.mj_renderer = self._build_mj_renderer(mj_model, nrows, ncols, **kwargs)
        # RGB / depth / segmentation are independent and may be enabled in any
        # combination. Each is produced by its own render() pass (mujoco renders
        # one output type per call), so we don't enable a mode here -- the mode is
        # toggled per pass in render_as_needed and left in the default RGB state.
        self.render_rgb = render_rgb
        self.render_depth = render_depth
        self.render_segmentation = render_segmentation
        if not (render_rgb or render_depth or render_segmentation):
            raise ValueError(
                "At least one of render_rgb, render_depth, or "
                "render_segmentation must be True."
            )

        if scene_option is None:
            self.scene_option = mj.MjvOption()
        else:
            self.scene_option = scene_option
        mj.mjv_defaultOption(self.scene_option)  # this sets default scene options

        self._cameras_names2id = {}
        for spec in cameras if isinstance(cameras, list) else [cameras]:
            cam_id, cam_name = self._resolve_camera_id_and_name(spec)
            if cam_id == -1:
                raise ValueError(f"Camera {spec} not found in the model.")
            if cam_name in self._cameras_names2id:
                raise ValueError(f"Duplicate camera name detected: {cam_name}.")
            self._cameras_names2id[cam_name] = cam_id
        if len(self._cameras_names2id) == 0:
            raise ValueError("At least one valid camera must be specified.")
        self._cameras_id2name = {v: k for k, v in self._cameras_names2id.items()}

        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)

        self._last_render_time_sec = -np.inf
        buffer = self.buffer_frames
        self.frames = self._new_frame_buffer() if buffer and render_rgb else None
        self.depth_frames = (
            self._new_frame_buffer() if buffer and render_depth else None
        )
        self.segmentation_frames = (
            self._new_frame_buffer() if buffer and render_segmentation else None
        )

        # Avoid floating point issues when comparing times
        self.rendering_rounding_tolerance = mj_model.opt.timestep * 0.5

    def _build_mj_renderer(
        self, mj_model: mj.MjModel, nrows: int, ncols: int, **kwargs: Any
    ) -> mj.Renderer:
        """Construct the underlying ``mujoco.Renderer``.

        Factored out so subclasses that never rasterize (e.g. `TrajectoryRecorder`)
        can skip allocating a GL/EGL context by overriding this to return None.
        """
        return mj.Renderer(mj_model, nrows, ncols, **kwargs)

    def _new_frame_buffer(self) -> dict[str, list]:
        return {cam_name: [] for cam_name in self._cameras_names2id}

    def _due_for_render(self, time: float) -> bool:
        """Whether enough time has elapsed since the last render for the next one."""
        min_next_render_time = (
            self._last_render_time_sec
            + self._secs_between_renders
            - self.rendering_rounding_tolerance
        )
        return time >= min_next_render_time

    def get_camera_matrix(
        self, camera: str | mj.MjsCamera, mj_data: mj.MjData, mj_model: mj.MjModel
    ) -> np.ndarray:
        """Get the 3x4 camera projection matrix from the current MjData.

        The returned matrix maps homogeneous world coordinates to homogeneous
        image (pixel) coordinates, following the standard MuJoCo
        ``image @ focal @ rotation @ translation`` composition.
        """
        internal_cam_id, _ = self._resolve_camera_id_and_name(camera)
        # update the scene to get the latest camera position and orientation
        self.mj_renderer.update_scene(mj_data, internal_cam_id, self.scene_option)
        pos = mj_data.cam_xpos[internal_cam_id]
        rot = mj_data.cam_xmat[internal_cam_id].reshape(3, 3)
        fov = mj_model.cam_fovy[internal_cam_id]
        height, width = self.camera_res

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (width - 1) / 2.0
        image[1, 2] = (height - 1) / 2.0

        return image @ focal @ rotation @ translation

    def render_as_needed(self, mj_data: mj.MjData) -> bool:
        """Render frames for all cameras if enough time has elapsed.

        Args:
            mj_data: Current MuJoCo data.

        Returns:
            True if frames were rendered, False otherwise.
        """
        if not self._due_for_render(mj_data.time):
            return False

        self._last_render_time_sec = float(mj_data.time)
        for cam_name, internal_cam_id in self._cameras_names2id.items():
            self.mj_renderer.update_scene(mj_data, internal_cam_id, self.scene_option)
            # One render() pass per enabled output. mujoco renders a single output
            # type per call, so toggle the mode for each and always restore the
            # default (RGB) state afterwards.
            if self.render_rgb:
                rgb = self.mj_renderer.render()
                if self.buffer_frames:
                    self.frames[cam_name].append(rgb)
            if self.render_depth:
                self.mj_renderer.enable_depth_rendering()
                depth = self.mj_renderer.render()
                self.mj_renderer.disable_depth_rendering()
                if self.buffer_frames:
                    self.depth_frames[cam_name].append(depth)
            if self.render_segmentation:
                self.mj_renderer.enable_segmentation_rendering()
                # MuJoCo segmentation returns int32 (H, W, 2): channel 0 is the
                # object id, channel 1 the object type, with -1 marking background.
                # Keep channel 0 only (assumes objects are geoms, so the id is
                # unambiguous), copied to a standalone int32 array so the unused
                # second channel isn't retained.
                seg = self.mj_renderer.render()[:, :, 0].copy()
                self.mj_renderer.disable_segmentation_rendering()
                if self.buffer_frames:
                    self.segmentation_frames[cam_name].append(seg)
        return True

    def reset(self) -> None:
        """Clear buffered frames and reset the render timer."""
        self._last_render_time_sec = -np.inf
        if not self.buffer_frames:
            return
        if self.render_rgb:
            self.frames = self._new_frame_buffer()
        if self.render_depth:
            self.depth_frames = self._new_frame_buffer()
        if self.render_segmentation:
            self.segmentation_frames = self._new_frame_buffer()

    def close(self) -> None:
        """Release the underlying MuJoCo renderer resources."""
        self.mj_renderer.close()

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        self.close()
        return False  # don't suppress exceptions

    def show_in_notebook(
        self,
        camera: str | mj.MjsCamera | list[str | mj.MjsCamera] | None = None,
        **kwargs: Any,
    ) -> None:
        """Display recorded frames in a Jupyter notebook.

        Only RGB frames are displayed. Depth/segmentation renders are kept as raw
        arrays in ``self.depth_frames`` / ``self.segmentation_frames``.

        Args:
            camera: Camera(s) to display. If None, displays all cameras.
            **kwargs: Additional arguments passed to mediapy.show_video
        """
        if not self._warn_unless_rgb("displayed"):
            return
        camera_names = self._normalize_camera_spec(camera)

        for cam_name in camera_names:
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")
            mediapy.show_video(frames, fps=self.output_fps, title=cam_name, **kwargs)

    def save_video(
        self,
        output_path: dict[str | mj.MjsCamera, PathLike] | PathLike,
        **kwargs: Any,
    ) -> None:
        """Save recorded frames as video files.

        Only RGB frames are saved. Depth/segmentation renders are kept as raw
        arrays in ``self.depth_frames`` / ``self.segmentation_frames``.

        Args:
            output_path: Either a dict mapping camera specs to file paths, or:
                - If single camera: a file path to save to
                - If multiple cameras: a directory path to save all videos to
            **kwargs: Additional arguments passed to imageio.imwrite
        """
        if not self._warn_unless_rgb("saved"):
            return
        path_by_camera = self._resolve_output_paths(output_path)

        for cam_name, path in path_by_camera.items():
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")

            path.parent.mkdir(parents=True, exist_ok=True)

            iio.imwrite(
                path, frames, fps=self.output_fps, codec="libx264", quality=8, **kwargs
            )

    def _warn_unless_rgb(self, verb: str) -> bool:
        """Return True if RGB frames exist to save/show; warn and return False if not.

        Only RGB frames can be encoded as video; depth (float) and segmentation
        (int) are kept as raw arrays rather than lossily cast to uint8. Warn when
        such buffers exist so they are not silently expected in the video.
        """
        if not self.render_rgb:
            warnings.warn(
                f"No RGB frames to be {verb} as video (render_rgb=False). Depth/"
                "segmentation frames are raw arrays in `Renderer.depth_frames` / "
                "`Renderer.segmentation_frames`."
            )
            return False
        if self.render_depth or self.render_segmentation:
            warnings.warn(
                f"Only RGB frames are {verb} as video; depth/segmentation frames "
                "are kept as raw arrays in `Renderer.depth_frames` / "
                "`Renderer.segmentation_frames`."
            )
        return True

    def _normalize_camera_spec(
        self,
        camera: str | mj.MjsCamera | list[str | mj.MjsCamera] | None,
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
        elif isinstance(camera, (str, mj.MjsCamera)):
            _, cam_name = self._resolve_camera_id_and_name(camera)
            camera_names = [cam_name]
        elif isinstance(camera, list):
            camera_names = [self._resolve_camera_id_and_name(c)[1] for c in camera]
        else:
            raise ValueError(
                f"Invalid camera spec type: {type(camera)}. Must be str, "
                "mj.MjsCamera, list of these, or None."
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
        output_path: dict[str | mj.MjsCamera, PathLike] | PathLike,
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
        self, camera: str | mj.MjsCamera, /
    ) -> tuple[int, str]:
        """Convert a camera specification to (internal_id, camera_name)."""
        if isinstance(camera, str):
            cam_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, camera)
            return cam_id, camera
        elif isinstance(camera, mj.MjsCamera):
            cam_name = camera.name
            cam_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
            return cam_id, cam_name
        else:
            raise ValueError(
                f"Invalid camera spec: {camera}. Must be one of str or mj.MjsCamera."
            )


def launch_interactive_viewer(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    run_async: bool = False,
    init_keyframe: str | None = "neutral",
) -> None:
    """Launch MuJoCo's built-in interactive viewer.

    Args:
        mj_model: Compiled MuJoCo model.
        mj_data: MuJoCo data.
        run_async: If True, launch the viewer in a separate process and return
            immediately. Use this when calling from a Jupyter notebook.
        init_keyframe: Keyframe name to reset to before launching. Uses the current
            state if None.
    """

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
    camera: mj.MjsCamera | str,
    *,
    init_keyframe: str | None = "neutral",
    duration: float = 0.1,
    camera_res: tuple[int, int] = (240, 320),
    playback_speed: float = 0.1,
    output_fps: int = 25,
    show_in_notebook: bool = False,
    output_path: PathLike | None = None,
    **kwargs: Any,
) -> None:
    """Run a short simulation and render a preview.

    Args:
        mj_model: Compiled MuJoCo model.
        mj_data: MuJoCo data.
        camera: Camera name or MJCF element to use for rendering.
        init_keyframe: Keyframe name to reset to before rendering. Uses the current
            state if None.
        duration: Duration to simulate in seconds.
        camera_res: ``(height, width)`` in pixels.
        playback_speed: Video playback speed relative to real time.
        output_fps: Output video frame rate.
        show_in_notebook: If True, display the video in a Jupyter notebook.
        output_path: Path to save the video. Not saved if None.
        **kwargs: Passed to `Renderer`.
    """
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
