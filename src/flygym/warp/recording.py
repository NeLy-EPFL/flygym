"""Record kinematic trajectories on the GPU and render them with MJWarp batch rendering.

See `flygym.recording` for an overview. Trajectories recorded here are ordinary
`flygym.recording.RecordedTrajectory` objects and can be rendered on either backend.
"""

from os import PathLike

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp

from flygym.recording import (
    RecordedTrajectory,
    as_trajectory_list,
    check_trajectories_match_model,
    resolve_camera_ids,
    save_trajectory_videos,
)
from flygym.rendering import FrameClock
from flygym.warp.rendering import update_render_geometry_gpu
from flygym.warp.utils import get_rgb_selected_worlds_and_cameras

__all__ = ["WarpTrajectoryRecorder", "render_trajectories_gpu"]


class WarpTrajectoryRecorder:
    """Records the kinematic state of selected worlds of a GPU simulation.

    The GPU counterpart of `flygym.recording.TrajectoryRecorder`. Usually created via
    `GPUSimulation.set_trajectory_recorder`. At each frame, the state of all worlds is
    copied to host memory, and the selected worlds are kept.

    Args:
        mj_model: Compiled MuJoCo model.
        n_worlds: Total number of parallel worlds in the simulation.
        worlds: Indices of the worlds to record. Defaults to all worlds.
        playback_speed: Video playback speed relative to real time.
        output_fps: Output video frame rate.
    """

    def __init__(
        self,
        mj_model: mj.MjModel,
        n_worlds: int,
        *,
        worlds: list[int] | None = None,
        playback_speed: float = 0.2,
        output_fps: int = 25,
    ):
        self.world_ids = list(range(n_worlds)) if worlds is None else list(worlds)
        if len(self.world_ids) == 0:
            raise ValueError("At least one world must be recorded.")
        self._clock = FrameClock(playback_speed, output_fps, mj_model.opt.timestep)
        self._has_mocap = mj_model.nmocap > 0
        self.reset()

    def record_as_needed(self, mjw_data: mjw.Data) -> bool:
        """Record the current state of the selected worlds if a frame is due.

        Returns:
            True if a frame was recorded, False otherwise.
        """
        curr_time = mjw_data.time.numpy()[0]  # assume all worlds have the same time
        if not self._clock.tick(curr_time):
            return False
        self._time.append(curr_time)
        self._qpos.append(mjw_data.qpos.numpy()[self.world_ids])
        if self._has_mocap:
            self._mocap_pos.append(mjw_data.mocap_pos.numpy()[self.world_ids])
            self._mocap_quat.append(mjw_data.mocap_quat.numpy()[self.world_ids])
        return True

    def reset(self) -> None:
        """Discard recorded frames and reset the frame clock."""
        self._clock.reset()
        self._time: list[float] = []
        self._qpos: list[np.ndarray] = []
        self._mocap_pos: list[np.ndarray] = []
        self._mocap_quat: list[np.ndarray] = []

    @property
    def trajectories(self) -> list[RecordedTrajectory]:
        """The trajectories recorded so far, one per selected world."""
        if len(self._qpos) == 0:
            raise RuntimeError("No frames have been recorded yet.")
        time = np.array(self._time)
        # Stacked arrays have shape (n_frames, n_selected_worlds, ...)
        qpos = np.stack(self._qpos)
        if self._has_mocap:
            mocap_pos = np.stack(self._mocap_pos)
            mocap_quat = np.stack(self._mocap_quat)
        return [
            RecordedTrajectory(
                time=time,
                qpos=qpos[:, i],
                output_fps=self._clock.output_fps,
                playback_speed=self._clock.playback_speed,
                world_id=world_id,
                mocap_pos=mocap_pos[:, i] if self._has_mocap else None,
                mocap_quat=mocap_quat[:, i] if self._has_mocap else None,
            )
            for i, world_id in enumerate(self.world_ids)
        ]


def render_trajectories_gpu(
    mj_model: mj.MjModel,
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
    cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
    output_path: PathLike | None = None,
    *,
    camera_res: tuple[int, int] = (240, 320),
    worlds_per_batch: int = 32,
) -> list[dict[str, np.ndarray]]:
    """Render recorded trajectories using MJWarp's GPU batch rendering.

    The GPU counterpart of `flygym.recording.render_trajectories`. All frames of all
    trajectories are rendered in batches of `worlds_per_batch` parallel worlds, so the
    batch size is independent of how many worlds were simulated.

    `mj_model` must be compatible with GPU batch rendering, i.e., compiled from a world
    modified by `flygym.warp.rendering.modify_world_for_batch_rendering`. The model of
    a `GPUSimulation` with a renderer set up with `use_gpu_batch_rendering=True` already
    satisfies this.

    Args:
        mj_model: Compiled, batch-rendering-compatible model with the same `qpos`
            layout as the model the trajectories were recorded with.
        trajectories: One trajectory or a list of them.
        cameras: Camera(s) to render.
        output_path: If given, also save the frames as videos (see
            `flygym.recording.save_trajectory_videos`).
        camera_res: `(height, width)` in pixels.
        worlds_per_batch: Number of frames rendered in parallel. Larger values are
            faster but use more GPU memory.

    Returns:
        A list, parallel to `trajectories`, of `{camera_name: frames}` dicts, where
        `frames` is a uint8 array of shape `(n_frames, height, width, 3)`.
    """
    trajectories = as_trajectory_list(trajectories)
    check_trajectories_match_model(mj_model, trajectories)
    camera_ids = resolve_camera_ids(mj_model, cameras)
    has_mocap = mj_model.nmocap > 0

    # Flatten all (trajectory, frame) pairs into one list of frames to render
    frame_refs = [
        (traj_idx, frame_idx)
        for traj_idx, traj in enumerate(trajectories)
        for frame_idx in range(traj.n_frames)
    ]
    batch_size = min(worlds_per_batch, len(frame_refs))

    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(mj_model, mj.MjData(mj_model), nworld=batch_size)
    render_context = mjw.create_render_context(
        mjm=mj_model,
        nworld=batch_size,
        cam_active=[cid in camera_ids.values() for cid in range(mj_model.ncam)],
        cam_res=camera_res[::-1],  # MJWarp expects (W, H); we use (H, W)
    )
    world_ids_gpu = wp.array(list(range(batch_size)), dtype=wp.int32)
    cam_ids_gpu = wp.array(list(camera_ids.values()), dtype=wp.int32)
    rgb_out = wp.zeros((batch_size, len(camera_ids), *camera_res), dtype=wp.vec3f)

    frames = [
        {cam: np.zeros((traj.n_frames, *camera_res, 3), np.uint8) for cam in camera_ids}
        for traj in trajectories
    ]
    for batch_start in range(0, len(frame_refs), batch_size):
        batch = frame_refs[batch_start : batch_start + batch_size]
        # In the last batch, unused worlds keep stale states and are simply ignored
        qpos = mjw_data.qpos.numpy()
        for i, (traj_idx, frame_idx) in enumerate(batch):
            qpos[i] = trajectories[traj_idx].qpos[frame_idx]
        mjw_data.qpos.assign(qpos)
        if has_mocap:
            mocap_pos = mjw_data.mocap_pos.numpy()
            mocap_quat = mjw_data.mocap_quat.numpy()
            for i, (traj_idx, frame_idx) in enumerate(batch):
                mocap_pos[i] = trajectories[traj_idx].mocap_pos[frame_idx]
                mocap_quat[i] = trajectories[traj_idx].mocap_quat[frame_idx]
            mjw_data.mocap_pos.assign(mocap_pos)
            mjw_data.mocap_quat.assign(mocap_quat)

        update_render_geometry_gpu(mjw_model, mjw_data)
        mjw.refit_bvh(mjw_model, mjw_data, render_context)
        mjw.render(mjw_model, mjw_data, render_context)
        get_rgb_selected_worlds_and_cameras(
            render_context, world_ids_gpu, cam_ids_gpu, rgb_out
        )
        # Shape: (batch_size, n_cams, height, width, 3)
        rgb = (rgb_out.numpy() * 255).astype(np.uint8)
        for i, (traj_idx, frame_idx) in enumerate(batch):
            for cam_idx, cam_name in enumerate(camera_ids):
                frames[traj_idx][cam_name][frame_idx] = rgb[i, cam_idx]

    if output_path is not None:
        save_trajectory_videos(trajectories, frames, output_path)
    return frames
