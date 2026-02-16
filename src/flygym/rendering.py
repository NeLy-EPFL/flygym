from multiprocessing import Process
from pathlib import Path
from os import PathLike
from typing import Any

import mujoco
import mujoco.viewer
import dm_control.mjcf as mjcf
import mediapy
import imageio.v3 as iio
import numpy as np

__all__ = ["Renderer", "launch_interactive_viewer", "preview_model"]


class Renderer:
    """Manages rendering of MuJoCo simulations to video frames.

    Automatically determines when to capture frames based on the specified output FPS
    and playback speed, so that the user can call `render_as_needed` inside their
    simulation loop without worrying about timing.

    Args:
        mj_model:
            Compiled MuJoCo model to render.
        height:
            Frame height in pixels.
        width:
            Frame width in pixels.
        playback_speed:
            Playback speed multiplier (1.0 = real-time).
        output_fps:
            FPS of the output video. This takes the playback speed into account: for
            example, if we want a 25-FPS video at 0.5x speed, frames will be rendered 50
            times per simulated second (managed internally by this class).
        camera:
            Camera specification (name, ID, or MjvCamera object).
        **kwargs:
            Additional arguments passed to mujoco.Renderer.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        height: int = 240,
        width: int = 320,
        playback_speed: float = 0.2,
        output_fps: int = 25,
        camera: str | int | mujoco.MjvCamera | mjcf.Element = -1,
        **kwargs: Any,
    ):
        self.mj_renderer = mujoco.Renderer(mj_model, height, width, **kwargs)
        self.playback_speed = playback_speed
        self.output_fps = output_fps
        self.frame_capture_interval_sec = 1 / (output_fps / playback_speed)
        if isinstance(camera, mjcf.Element) and camera.tag == "camera":
            camera = camera.full_identifier
        self.camera = camera
        self.last_render_time_sec = -np.inf
        self.frames = []

    def render_as_needed(
        self,
        mj_data,
        scene_option: mujoco.MjvOption | None = None,
    ) -> np.ndarray | None:
        """Render frame if enough time has elapsed since last render. Returns the
        rendered frame as a numpy array, or None if no frame was rendered. The
        `scene_option` argument is forwarded to `mujoco.Renderer.update_scene()`."""
        curr_time = mj_data.time
        if curr_time >= self.last_render_time_sec + self.frame_capture_interval_sec:
            self.last_render_time_sec = curr_time
            self.mj_renderer.update_scene(mj_data, self.camera, scene_option)
            frame = self.mj_renderer.render()
            self.frames.append(frame)
            return frame
        else:
            return None

    def close(self):
        self.mj_renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def show_in_notebook(self, **kwargs) -> str | None:
        if len(self.frames) == 0:
            raise RuntimeError("No frames have been rendered yet.")

        return mediapy.show_video(self.frames, fps=self.output_fps, **kwargs)

    def save_video(self, output_path: PathLike, **kwargs) -> None:
        if len(self.frames) == 0:
            raise RuntimeError("No frames have been rendered yet.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            output_path, self.frames, fps=self.output_fps, codec="libx264", **kwargs
        )

    def reset(self):
        self.frames = []


def launch_interactive_viewer(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    run_async: bool = False,
    init_keyframe: str | None = "neutral",
) -> None:
    """Launch MuJoCo's built-in interactive viewer. If `run_async` is True, the viewer
    will be launched in a separate process and this function will return immediately.
    It should be set to True when launching from a Jupyter notebook."""

    if init_keyframe is not None:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, init_keyframe)
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    if run_async:
        p = Process(target=mujoco.viewer.launch, args=(mj_model, mj_data))
        p.start()
        # Don't join!
    else:
        mujoco.viewer.launch(mj_model, mj_data)


def preview_model(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    camera: mjcf.Element | str,
    init_keyframe: str | None = "neutral",
    duration: float = 0.1,
    playback_speed=0.1,
    output_fps: int = 25,
    show_in_notebook: bool = False,
    output_path: PathLike | None = None,
    **kwargs,
):
    if init_keyframe is not None:
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, init_keyframe)
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    n_steps = int(duration / mj_model.opt.timestep)

    with Renderer(
        mj_model,
        playback_speed=playback_speed,
        output_fps=output_fps,
        camera=camera,
        **kwargs,
    ) as renderer:
        for step in range(n_steps):
            mujoco.mj_step(mj_model, mj_data)
            renderer.render_as_needed(mj_data)

        if show_in_notebook:
            renderer.show_in_notebook()
        if output_path:
            renderer.save_video(output_path)
