"""Record kinematic trajectories during simulation and render them to video later.

Rendering frames during simulation does not scale to many parallel worlds: buffering
`(n_worlds, n_cams, height, width, 3)` RGB tensors quickly becomes infeasible. Instead,
a trajectory recorder stores only the generalized coordinates (`qpos`, plus mocap poses
if the model has any) at the video frame rate. The resulting `RecordedTrajectory` can
be saved, sub-selected, and rendered to video afterwards with `render_trajectories`
(CPU) or `flygym.warp.recording.render_trajectories_gpu` (GPU).

A trajectory does not store the model or any rendering settings. To render it, pass a
compiled `mujoco.MjModel` with the same `qpos` layout as the one it was recorded with.
Persist the model separately if needed (e.g. with
`BaseCompositionElement.save_xml_with_assets`). Material, texture, and lighting edits
(such as `flygym.warp.rendering.modify_world_for_batch_rendering`) do not change the
`qpos` layout, so a trajectory remains compatible with the edited model.

Trajectories recorded on CPU and GPU have the same format, and either can be rendered
on either backend.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import mujoco as mj
import numpy as np

from flygym.rendering import FrameClock, update_render_geometry
from flygym.utils.video import write_video_from_frames

__all__ = ["RecordedTrajectory", "TrajectoryRecorder", "render_trajectories"]


@dataclass
class RecordedTrajectory:
    """Kinematic trajectory of a single world, sampled at the video frame rate.

    Attributes:
        time: `(n_frames,)` simulation time of each frame, in seconds.
        qpos: `(n_frames, nq)` generalized coordinates of each frame.
        output_fps: Frame rate at which the frames should be played back.
        playback_speed: Playback speed relative to real time.
        world_id: Index of the parallel world this trajectory came from (always 0 for
            CPU simulations).
        mocap_pos: `(n_frames, nmocap, 3)` mocap body positions, or None if the model
            has no mocap bodies.
        mocap_quat: `(n_frames, nmocap, 4)` mocap body quaternions, or None if the
            model has no mocap bodies.
    """

    time: np.ndarray
    qpos: np.ndarray
    output_fps: int
    playback_speed: float
    world_id: int = 0
    mocap_pos: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.qpos.shape[0]

    def save(self, path: PathLike) -> None:
        """Save the trajectory to a `.npz` file. Load it back with `from_file`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "time": self.time,
            "qpos": self.qpos,
            "output_fps": np.asarray(self.output_fps),
            "playback_speed": np.asarray(self.playback_speed),
            "world_id": np.asarray(self.world_id),
        }
        if self.mocap_pos is not None:
            arrays["mocap_pos"] = self.mocap_pos
            arrays["mocap_quat"] = self.mocap_quat
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_file(cls, path: PathLike) -> "RecordedTrajectory":
        """Load a trajectory from a `.npz` file written by `save`."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                time=data["time"],
                qpos=data["qpos"],
                output_fps=int(data["output_fps"]),
                playback_speed=float(data["playback_speed"]),
                world_id=int(data["world_id"]),
                mocap_pos=data["mocap_pos"] if "mocap_pos" in data else None,
                mocap_quat=data["mocap_quat"] if "mocap_quat" in data else None,
            )


class TrajectoryRecorder:
    """Records the kinematic state of a CPU simulation at the video frame rate.

    Frames are sampled at the same timepoints as a `flygym.rendering.Renderer` with the
    same `playback_speed` and `output_fps` would render them. Usually created via
    `Simulation.set_trajectory_recorder`.

    Args:
        mj_model: Compiled MuJoCo model.
        playback_speed: Video playback speed relative to real time.
        output_fps: Output video frame rate.
    """

    def __init__(
        self, mj_model: mj.MjModel, *, playback_speed: float = 0.2, output_fps: int = 25
    ):
        self._clock = FrameClock(playback_speed, output_fps, mj_model.opt.timestep)
        self._has_mocap = mj_model.nmocap > 0
        self.reset()

    def record_as_needed(self, mj_data: mj.MjData) -> bool:
        """Record the current state if a frame is due.

        Returns:
            True if a frame was recorded, False otherwise.
        """
        if not self._clock.tick(mj_data.time):
            return False
        self._time.append(mj_data.time)
        self._qpos.append(mj_data.qpos.copy())
        if self._has_mocap:
            self._mocap_pos.append(mj_data.mocap_pos.copy())
            self._mocap_quat.append(mj_data.mocap_quat.copy())
        return True

    def reset(self) -> None:
        """Discard recorded frames and reset the frame clock."""
        self._clock.reset()
        self._time: list[float] = []
        self._qpos: list[np.ndarray] = []
        self._mocap_pos: list[np.ndarray] = []
        self._mocap_quat: list[np.ndarray] = []

    @property
    def trajectory(self) -> RecordedTrajectory:
        """The trajectory recorded so far."""
        if len(self._qpos) == 0:
            raise RuntimeError("No frames have been recorded yet.")
        return RecordedTrajectory(
            time=np.array(self._time),
            qpos=np.array(self._qpos),
            output_fps=self._clock.output_fps,
            playback_speed=self._clock.playback_speed,
            mocap_pos=np.array(self._mocap_pos) if self._has_mocap else None,
            mocap_quat=np.array(self._mocap_quat) if self._has_mocap else None,
        )


def render_trajectories(
    mj_model: mj.MjModel,
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
    cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
    output_path: PathLike | None = None,
    *,
    camera_res: tuple[int, int] = (240, 320),
    scene_option: mj.MjvOption | None = None,
) -> list[dict[str, np.ndarray]]:
    """Render recorded trajectories on the CPU.

    For GPU batch rendering, see `flygym.warp.recording.render_trajectories_gpu`.

    Args:
        mj_model: Compiled model with the same `qpos` layout as the model the
            trajectories were recorded with.
        trajectories: One trajectory or a list of them.
        cameras: Camera(s) to render.
        output_path: If given, also save the frames as videos (see
            `save_trajectory_videos`).
        camera_res: `(height, width)` in pixels.
        scene_option: MuJoCo scene options. Uses defaults if None.

    Returns:
        A list, parallel to `trajectories`, of `{camera_name: frames}` dicts, where
        `frames` is a uint8 array of shape `(n_frames, height, width, 3)`.
    """
    trajectories = as_trajectory_list(trajectories)
    check_trajectories_match_model(mj_model, trajectories)
    camera_ids = resolve_camera_ids(mj_model, cameras)
    if scene_option is None:
        scene_option = mj.MjvOption()

    mj_data = mj.MjData(mj_model)
    frames = []
    with mj.Renderer(mj_model, *camera_res) as renderer:
        for traj in trajectories:
            frames_by_cam = {cam_name: [] for cam_name in camera_ids}
            for i in range(traj.n_frames):
                mj_data.qpos[:] = traj.qpos[i]
                if traj.mocap_pos is not None:
                    mj_data.mocap_pos[:] = traj.mocap_pos[i]
                    mj_data.mocap_quat[:] = traj.mocap_quat[i]
                update_render_geometry(mj_model, mj_data)
                for cam_name, cam_id in camera_ids.items():
                    renderer.update_scene(mj_data, cam_id, scene_option)
                    frames_by_cam[cam_name].append(renderer.render())
            frames.append({k: np.stack(v) for k, v in frames_by_cam.items()})

    if output_path is not None:
        save_trajectory_videos(trajectories, frames, output_path)
    return frames


def save_trajectory_videos(
    trajectories: list[RecordedTrajectory],
    frames: list[dict[str, np.ndarray]],
    output_path: PathLike,
) -> None:
    """Save frames rendered from trajectories as videos.

    If there is a single trajectory and a single camera and `output_path` ends with
    `.mp4`, the video is written to `output_path`. Otherwise, `output_path` is a
    directory: `{output_path}/{camera}.mp4` for a single trajectory, or
    `{output_path}/traj{i:04d}_world{world_id}/{camera}.mp4` for multiple ones.

    Args:
        trajectories: The rendered trajectories, used for the frame rate and the
            world IDs.
        frames: Rendered frames, as returned by `render_trajectories` or
            `flygym.warp.recording.render_trajectories_gpu`.
        output_path: Output video file or directory (see above).
    """
    output_path = Path(output_path)
    single_file = (
        len(frames) == 1 and len(frames[0]) == 1 and output_path.suffix == ".mp4"
    )
    for i, (traj, frames_by_cam) in enumerate(zip(trajectories, frames)):
        for cam_name, cam_frames in frames_by_cam.items():
            if single_file:
                path = output_path
            else:
                traj_dir = output_path
                if len(trajectories) > 1:
                    traj_dir = output_path / f"traj{i:04d}_world{traj.world_id}"
                path = traj_dir / f"{cam_name.replace('/', '_')}.mp4"
            write_video_from_frames(
                path, list(cam_frames), fps=traj.output_fps, codec="libx264"
            )


def check_trajectories_match_model(
    mj_model: mj.MjModel, trajectories: list[RecordedTrajectory]
) -> None:
    """Raise a `ValueError` if a trajectory's state layout does not match the model."""
    for traj in trajectories:
        nmocap = 0 if traj.mocap_pos is None else traj.mocap_pos.shape[1]
        if traj.qpos.shape[1] != mj_model.nq or nmocap != mj_model.nmocap:
            raise ValueError(
                f"Trajectory of world {traj.world_id} (nq={traj.qpos.shape[1]}, "
                f"nmocap={nmocap}) does not match the model (nq={mj_model.nq}, "
                f"nmocap={mj_model.nmocap}). Was it recorded with a different model?"
            )


def resolve_camera_ids(
    mj_model: mj.MjModel, cameras: str | mj.MjsCamera | list[str | mj.MjsCamera]
) -> dict[str, int]:
    """Map camera specs to `{camera_name: internal_camera_id}`."""
    if not isinstance(cameras, list):
        cameras = [cameras]
    camera_ids = {}
    for cam in cameras:
        cam_name = cam.name if isinstance(cam, mj.MjsCamera) else cam
        cam_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{cam_name}' not found in the model.")
        camera_ids[cam_name] = cam_id
    if len(camera_ids) == 0:
        raise ValueError("At least one camera must be specified.")
    return camera_ids


def as_trajectory_list(
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
) -> list[RecordedTrajectory]:
    """Wrap a single trajectory in a list; raise if no trajectories are given."""
    if isinstance(trajectories, RecordedTrajectory):
        trajectories = [trajectories]
    if len(trajectories) == 0:
        raise ValueError("No trajectories given.")
    return list(trajectories)
