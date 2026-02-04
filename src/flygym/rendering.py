from multiprocessing import Process
from pathlib import Path
from os import PathLike
from typing import Callable

import mujoco
import mujoco.viewer
import mediapy
import imageio.v3 as iio
import numpy as np


class Renderer:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        height: int = 240,
        width: int = 320,
        play_speed: float = 0.2,
        out_fps: int = 25,
        camera: str | int | mujoco.MjvCamera = -1,
        frame_capture_callback: Callable[[np.ndarray], None] | None = None,
        buffer_frames: bool = True,
        **kwargs,
    ):
        self.mj_renderer = mujoco.Renderer(mj_model, height, width, **kwargs)
        self.play_speed = play_speed
        self.out_fps = out_fps
        self.frame_capture_interval_sec = 1 / (out_fps / play_speed)
        self.camera = camera
        self.last_render_time_sec = -np.inf
        self.frame_capture_callback = frame_capture_callback
        self.frames = [] if buffer_frames else None

    def render_as_needed(
        self,
        mj_data,
        camera_override: str | None = None,
        scene_option: mujoco.MjvOption | None = None,
    ) -> np.ndarray | None:
        curr_time = mj_data.time
        if curr_time >= self.last_render_time_sec + self.frame_capture_interval_sec:
            self.last_render_time_sec = curr_time
            camera = camera_override if camera_override else self.camera
            self.mj_renderer.update_scene(mj_data, camera, scene_option)
            frame = self.mj_renderer.render()
            if self.frame_capture_callback:
                self.frame_capture_callback(frame)
            if self.frames is not None:
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
        self._raise_on_bad_buffer()
        return mediapy.show_video(self.frames, fps=self.out_fps, **kwargs)

    def save_video(self, output_path: PathLike, **kwargs) -> None:
        self._raise_on_bad_buffer()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            output_path, self.frames, fps=self.out_fps, codec="libx264", **kwargs
        )

    def _raise_on_bad_buffer(self):
        if self.frames is None:
            raise ValueError(
                "Cannot save video or show it in notebook without buffering frames. "
                "Set `buffer_frames=True` when initializing the renderer."
            )
        if len(self.frames) == 0:
            raise RuntimeError("No frames have been rendered yet.")


def launch_interactive_viewer(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, run_async: bool = False
) -> None | Renderer:
    if run_async:
        # Run MuJoCo's built-in interactive viewer asynchronously.
        # This bypasses synchronization issues when launched from jupyter.
        # The solution shipped by MuJoCo is to use mujoco.viewer.launch_passive, but
        # this only works with a special Python interpreter `mjpython` on macOS.
        p = Process(target=mujoco.viewer.launch, args=(mj_model, mj_data))
        p.start()
        # Don't join!
    else:
        mujoco.viewer.launch(mj_model, mj_data)
