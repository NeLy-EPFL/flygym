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

__all__ = ["Renderer", "CameraSpec", "launch_interactive_viewer", "preview_model"]

CameraSpec = str | int | mj.MjvCamera | mjcf.Element


class Renderer:
    def __init__(
        self,
        mj_model: mj.MjModel,
        camera: CameraSpec | Sequence[CameraSpec],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        **kwargs: Any,
    ):
        self.mj_model = mj_model
        self._cameras_intern_id_lookup = self._resolve_camera_spec(camera)
        self.camera_res = camera_res

        nrows, ncols = camera_res
        self.mj_renderer = mj.Renderer(mj_model, nrows, ncols, **kwargs)

        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self._secs_between_renders = 1 / (output_fps / playback_speed)

        self._last_render_time_sec = -np.inf
        self.frames = {cam_name: [] for cam_name in self._cameras_intern_id_lookup}

    def _resolve_camera_spec(
        self, spec: CameraSpec | Sequence[CameraSpec]
    ) -> dict[str, mj.MjvCamera]:

        def resolve_single_spec(s: CameraSpec) -> tuple[str, mj.MjvCamera]:
            if isinstance(s, mj.MjvCamera):
                internal_cam_id = s.fixedcamid
                full_id = mj.mj_id2name(
                    self.mj_model, mj.mjtObj.mjOBJ_CAMERA, internal_cam_id
                )
                return internal_cam_id, full_id
            elif isinstance(s, str):
                internal_cam_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_CAMERA, s
                )
                return internal_cam_id, s
            elif isinstance(s, mjcf.Element):
                full_id = s.full_identifier
                internal_cam_id = mj.mj_name2id(
                    self.mj_model, mj.mjtObj.mjOBJ_CAMERA, full_id
                )
                return internal_cam_id, full_id
            elif isinstance(s, int):
                full_id = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_CAMERA, s)
                return s, full_id
            else:
                raise ValueError(
                    f"Invalid camera spec: {s}. Must be one of str, int, "
                    "mujoco.MjvCamera, or mjcf.Element."
                )

        resolved = {}
        for s in spec if isinstance(spec, Sequence) else [spec]:
            internal_cam_id, full_id = resolve_single_spec(s)
            resolved[full_id] = internal_cam_id
        return resolved

    def render_as_needed(self, mj_data: mj.MjData) -> bool:
        if mj_data.time >= self._last_render_time_sec + self._secs_between_renders:
            self._last_render_time_sec = mj_data.time
            for cam_name, internal_cam_id in self._cameras_intern_id_lookup.items():
                self.mj_renderer.update_scene(mj_data, internal_cam_id)
                frame = self.mj_renderer.render()
                self.frames[cam_name].append(frame)
            return True
        else:
            return False

    def close(self):
        self.mj_renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def show_in_notebook(
        self, camera: CameraSpec | Sequence[CameraSpec] | None = None, **kwargs
    ) -> None:
        if camera is None:
            camera_names = list(self._cameras_intern_id_lookup.keys())
        else:
            camera_names = list(self._resolve_camera_spec(camera).keys())

        for cam_name in camera_names:
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frame recorded yet for camera {cam_name}.")
            mediapy.show_video(frames, fps=self.output_fps, title=cam_name, **kwargs)

    def save_video(
        self,
        output_path: dict[CameraSpec, PathLike] | PathLike,
        **kwargs,
    ) -> None:
        if isinstance(output_path, str | Path):
            output_path = Path(output_path)
            if len(self._cameras_intern_id_lookup) == 1:
                cam_name = list(self._cameras_intern_id_lookup.keys())[0]
                path_by_camera = {cam_name: output_path}
            else:
                if not output_path.exists():
                    output_path.mkdir(parents=True)
                elif not output_path.is_dir():
                    raise ValueError(
                        "There are >1 cameras and `output_path` is given as a single "
                        "path. In this case, `output_path` is treated as a directory "
                        "and videos from all cameras will be saved under it. However, "
                        f"the provided path {output_path} already exists and is not a "
                        "directory."
                    )
                path_by_camera = {
                    cam_name: output_path / f"{cam_name.replace('/', '-')}.mp4"
                    for cam_name in self._cameras_intern_id_lookup
                }
        elif isinstance(output_path, dict):
            camera_names = list(self._resolve_camera_spec(list(output_path)))
            path_by_camera = {
                cam_name: path
                for cam_name, path in zip(camera_names, output_path.values())
            }
        else:
            raise ValueError(
                f"Invalid type of `output_path`: {type(output_path)}. Must be one of "
                f"{str(Path)}, {dict}, or {str(Path)}."
            )

        for cam_name, path in path_by_camera.items():
            frames = self.frames[cam_name]
            if len(frames) == 0:
                raise RuntimeError(f"No frame recorded yet for camera {cam_name}.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, frames, fps=self.output_fps, codec="libx264", **kwargs)

    def reset(self):
        self.frames = {cam_name: [] for cam_name in self._cameras_intern_id_lookup}


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
