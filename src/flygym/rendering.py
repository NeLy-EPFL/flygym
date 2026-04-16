from multiprocessing import Process
from pathlib import Path
from typing import Any
from os import PathLike

import mujoco as mj
import mujoco.viewer as mjviewer
import dm_control.mjcf as mjcf
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
        buffer_frames: If True, store frames in ``self.frames``.
        scene_option: MuJoCo scene options. Uses defaults if None.
        **kwargs: Passed to ``mujoco.Renderer``.

    Attributes:
        frames: Dict mapping camera name to list of rendered frames.
            Only populated when ``buffer_frames=True``.
    """

    def __init__(
        self,
        mj_model: mj.MjModel,
        cameras: str | mjcf.Element | list[str | mjcf.Element],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
        scene_option: mj.MjvOption | None = None,
        render_depth: bool = False,
        render_segmentation: bool = False,
        **kwargs: Any,
    ):
        self.mj_model = mj_model
        self.camera_res = camera_res
        nrows, ncols = camera_res
        self.buffer_frames = buffer_frames
        
        self.mj_renderer = mj.Renderer(mj_model, nrows, ncols, **kwargs)
        self.render_depth = render_depth
        if self.render_depth:
            self.mj_renderer.enable_depth_rendering()
        self.render_segmentation = render_segmentation
        if self.render_segmentation:
            self.mj_renderer.enable_segmentation_rendering()

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
        if self.buffer_frames:
            self.frames = {cam_name: [] for cam_name in self._cameras_names2id}
        else:
            self.frames = None    

        
    def get_xmat_for_camera(self, camera: str | mjcf.Element, mj_data: mj.MjData, mj_model: mj.MjModel) -> np.ndarray:
        """Get the camera's orientation matrix (xmat) from the current MJData."""
        internal_cam_id, _ = self._resolve_camera_id_and_name(camera)
        # update the scene to get the latest camera position and orientation
        self.mj_renderer.update_scene(mj_data, internal_cam_id, self.scene_option)
        pos = mj_data.cam_xpos[internal_cam_id]
        rot = mj_data.cam_xmat[internal_cam_id].reshape(3, 3)
        fov = mj_model.cam_fovy[internal_cam_id]
        width, height = self.camera_res


        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos
        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot
        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0
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
        if mj_data.time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = mj_data.time
            for cam_name, internal_cam_id in self._cameras_names2id.items():
                self.mj_renderer.update_scene(
                    mj_data, internal_cam_id, self.scene_option
                )
                frame = self.mj_renderer.render()
                if self.buffer_frames:
                    if self.render_segmentation:
                        # segmentation renders 2 channels one is body the other is fly vs background
                        frame = frame[:, :, 0]
                        assert np.all(frame<=255), "More than 255 bodies not supported"
                        frame = frame.astype(np.uint8)
                    self.frames[cam_name].append(frame)
            return True
        else:
            return False

    def reset(self) -> None:
        """Clear buffered frames and reset the render timer."""
        self._last_render_time_sec = -np.inf
        if self.buffer_frames:
            self.frames = {cam_name: [] for cam_name in self._cameras_names2id}

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
        camera: str | mjcf.Element | list[str | mjcf.Element] | None = None,
        **kwargs: Any,
    ) -> None:
        """Display recorded frames in a Jupyter notebook.

        Args:
            camera: Camera(s) to display. If None, displays all cameras.
            **kwargs: Additional arguments passed to mediapy.show_video
        """
        camera_names = self._normalize_camera_spec(camera)

        for cam_name in camera_names:
            if self.render_depth and not self.frames[cam_name][0].dtype == np.uint8:
                self._depth_frames_2uint8(cam_name)
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frames recorded yet for camera '{cam_name}'.")
            mediapy.show_video(frames, fps=self.output_fps, title=cam_name, **kwargs)

    def save_video(
        self,
        output_path: dict[str | mjcf.Element, PathLike] | PathLike,
        **kwargs: Any,
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
            if self.render_depth and not self.frames[cam_name][0].dtype == np.uint8:
                self._depth_frames_2uint8(cam_name)
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
                **kwargs
            )
    
    def _depth_frames_2uint8(self, cam_name: str) -> None:
        """
        Convert depth frames from uint32 to uint8. 
        This is necessary because depth frames are rendered as 32-bit floats, but
        we want to save them as 8-bit videos.
        We will find the max value (background) and scale from the second biggest value to the min value (0-255)
        """

        max_val = np.max(self.frames[cam_name][0]) # first frame should contain background
        no_max_frame = self.frames[cam_name][0][self.frames[cam_name][0] != max_val] # not perfect if suddenly becomes much more distant in next frames more than 0.2
        max_no_max_val = np.max(no_max_frame)
        min_val = np.min(self.frames[cam_name])
        print(f"Depth frame scaling for camera '{cam_name}': max={max_val}, second_max={max_no_max_val}, min={min_val}")
        for i in range(len(self.frames[cam_name])):
            frame = self.frames[cam_name][i]
            frame_norm = np.clip((frame - min_val) / (max_no_max_val + 0.2 - min_val), 0, 1)
            self.frames[cam_name][i] = (frame_norm*255).astype(np.uint8)
    

    def _normalize_camera_spec(
        self,
        camera: str | mjcf.Element | list[str | mjcf.Element] | None,
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
        elif isinstance(camera, list):
            camera_names = [self._resolve_camera_id_and_name(c)[1] for c in camera]
        else:
            raise ValueError(
                f"Invalid camera spec type: {type(camera)}. Must be str, "
                "mjcf.Element, list of these, or None."
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
                        f"Cqamera '{cam_name}' in output_path is not available. "
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
    camera: mjcf.Element | str,
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

