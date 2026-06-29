"""Record minimal kinematic state during simulation and replay it as video (CPU).

This decouples how many worlds are *simulated* in parallel from how many are
*rendered*. During simulation a `TrajectoryRecorder` stores only the generalized
coordinates (``qpos``, plus mocap poses if used) at the render cadence, instead of
rasterizing frames. The result is a backend-agnostic `RecordedTrajectory`, which can
also be produced on GPU (`flygym.warp.rendering.WarpTrajectoryRecorder`) and replayed
on either backend (`render_trajectories` here, or
`flygym.warp.rendering.render_trajectories_gpu`).

A trajectory stores kinematic state only -- never a model. To replay one you pass a
compiled `mujoco.MjModel` yourself. Persist the model separately (e.g. with
`BaseCompositionElement.save_xml_with_assets`, a self-contained ``model.xml`` plus
bundled meshes) and recompile it when you need it. The recorded ``qpos`` layout is
preserved across an XML round trip and across `modify_world_for_batch_rendering`
(material/texture/light edits only), so one trajectory replays against the saved model
on either backend.

At replay time we set ``qpos`` and run *position-only* kinematics (``mj_kinematics`` +
``mj_camlight``) -- not a full ``mj_forward`` -- which deterministically regenerates
every geom/site/camera/light transform the renderer reads. See issue #296.

Note on faithfulness: a replayed frame shows geometry *consistent with the recorded
qpos*. A frame rendered live during simulation instead shows geometry from the
forward-kinematics pass at the *start* of the last step (one integration step stale --
a standard MuJoCo quirk), so replay and live render differ by at most one timestep of
motion; the replay is the qpos-faithful one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from os import PathLike

import mujoco as mj
import numpy as np

from flygym.rendering.live_rendering import Renderer
from flygym.utils.video import write_video_from_frames


__all__ = [
    "RecordedTrajectory",
    "TrajectoryRecorder",
    "save_trajectories",
    "load_trajectories",
    "render_trajectories",
]


@dataclass
class RecordedTrajectory:
    """A single world's recorded kinematic trajectory.

    Holds the per-frame generalized coordinates needed to re-render a world, plus the
    metadata describing how it should be rendered. It carries no trace of which backend
    produced it nor of the model it came from, so a trajectory recorded on CPU
    (`TrajectoryRecorder`) and one recorded on GPU
    (`flygym.warp.rendering.WarpTrajectoryRecorder`) are interchangeable inputs to
    `render_trajectories` / `flygym.warp.rendering.render_trajectories_gpu`, given a
    compatible compiled model.

    Attributes:
        qpos: ``(n_frames, nq)`` generalized coordinates, one row per recorded frame.
        output_fps: Frame rate the frames were sampled at / should be encoded at.
        playback_speed: Playback speed relative to real time (metadata only).
        camera_names: Cameras to render by default at replay time.
        camera_res: ``(height, width)`` in pixels.
        world_id: Index of the world this trajectory came from (0 for CPU).
        mocap_pos: ``(n_frames, nmocap, 3)`` mocap positions, or None if the model
            has no mocap bodies.
        mocap_quat: ``(n_frames, nmocap, 4)`` mocap quaternions, or None.
    """

    qpos: np.ndarray
    output_fps: int
    playback_speed: float
    camera_names: list[str]
    camera_res: tuple[int, int]
    world_id: int = 0
    mocap_pos: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def has_mocap(self) -> bool:
        return self.mocap_pos is not None

    def save(self, path: PathLike) -> None:
        """Save this trajectory to a single self-describing ``.npz`` file.

        Both the per-frame state and the replay metadata are stored, so the file can be
        read back with `load` without any side information.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "qpos": self.qpos,
            "output_fps": np.asarray(self.output_fps),
            "playback_speed": np.asarray(self.playback_speed),
            "camera_names": np.asarray(self.camera_names, dtype=np.str_),
            "camera_res": np.asarray(self.camera_res),
            "world_id": np.asarray(self.world_id),
        }
        if self.has_mocap:
            arrays["mocap_pos"] = self.mocap_pos
            arrays["mocap_quat"] = self.mocap_quat
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: PathLike) -> "RecordedTrajectory":
        """Load a trajectory from a ``.npz`` file written by `save`."""
        with np.load(path, allow_pickle=False) as data:
            has_mocap = "mocap_pos" in data
            return cls(
                qpos=data["qpos"],
                output_fps=int(data["output_fps"]),
                playback_speed=float(data["playback_speed"]),
                camera_names=[str(c) for c in data["camera_names"]],
                camera_res=tuple(int(x) for x in data["camera_res"]),
                world_id=int(data["world_id"]),
                mocap_pos=data["mocap_pos"] if has_mocap else None,
                mocap_quat=data["mocap_quat"] if has_mocap else None,
            )


class TrajectoryRecorder(Renderer):
    """Records ``qpos`` (and mocap poses) instead of rasterizing frames.

    A drop-in replacement for `Renderer` on the CPU single-world `Simulation`: it
    reuses the same render cadence so recorded frames land at exactly the timepoints
    that would have been rendered, but stores only generalized coordinates. The result
    is exposed as `recorded_trajectory` and can be replayed to video later by
    `render_trajectories` (you supply the compiled model).

    Args:
        mj_model: Compiled MuJoCo model.
        cameras: Camera(s) to record as the default render cameras. Recording itself
            does not depend on the cameras; they are stored as replay metadata.
        camera_res: ``(height, width)`` in pixels (replay metadata).
        playback_speed: Video playback speed relative to real time.
        output_fps: Output video frame rate (also sets the recording cadence).

    Attributes:
        recorded_trajectory: The `RecordedTrajectory` accumulated so far.
    """

    def __init__(
        self,
        mj_model: mj.MjModel,
        cameras: str | mj.MjsCamera | list[str | mj.MjsCamera],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        scene_option: mj.MjvOption | None = None,
        **kwargs: Any,
    ):
        # buffer_frames=False so the base class allocates no pixel buffers; we keep
        # our own qpos buffer instead.
        super().__init__(
            mj_model,
            cameras,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            buffer_frames=False,
            scene_option=scene_option,
            **kwargs,
        )
        self._nq = mj_model.nq
        self._nmocap = mj_model.nmocap
        self._qpos_buf: list[np.ndarray] = []
        self._mocap_pos_buf: list[np.ndarray] = []
        self._mocap_quat_buf: list[np.ndarray] = []

    def _build_mj_renderer(
        self, mj_model: mj.MjModel, nrows: int, ncols: int, **kwargs: Any
    ) -> None:
        # No rasterization: skip the GL/EGL context allocation entirely.
        return None

    def render_as_needed(self, mj_data: mj.MjData) -> bool:
        """Record the current state if enough simulation time has elapsed.

        Returns:
            True if a frame of state was recorded, False otherwise.
        """
        if not self._due_for_render(mj_data.time):
            return False
        self._last_render_time_sec = float(mj_data.time)
        self._qpos_buf.append(mj_data.qpos.copy())
        if self._nmocap > 0:
            self._mocap_pos_buf.append(mj_data.mocap_pos.copy())
            self._mocap_quat_buf.append(mj_data.mocap_quat.copy())
        return True

    def reset(self) -> None:
        """Clear the recorded state and reset the render timer."""
        self._last_render_time_sec = -np.inf
        self._qpos_buf = []
        self._mocap_pos_buf = []
        self._mocap_quat_buf = []

    @property
    def recorded_trajectory(self) -> RecordedTrajectory:
        """The trajectory recorded so far (one world)."""
        if len(self._qpos_buf) == 0:
            raise RuntimeError("No frames have been recorded yet.")
        return RecordedTrajectory(
            qpos=np.asarray(self._qpos_buf),
            output_fps=self.output_fps,
            playback_speed=self.playback_speed,
            camera_names=list(self._cameras_names2id.keys()),
            camera_res=self.camera_res,
            world_id=0,
            mocap_pos=np.asarray(self._mocap_pos_buf) if self._nmocap > 0 else None,
            mocap_quat=np.asarray(self._mocap_quat_buf) if self._nmocap > 0 else None,
        )

    def close(self) -> None:
        """No-op: no renderer resources are held."""
        return

    def show_in_notebook(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "TrajectoryRecorder records state, not frames. Save it with "
            "save_trajectories and replay with render_trajectories."
        )

    def save_video(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "TrajectoryRecorder records state, not frames. Save it with "
            "save_trajectories and replay with render_trajectories."
        )


def save_trajectories(
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
    output_dir: PathLike,
) -> None:
    """Save trajectories as individual ``.npz`` files in a folder.

    Writes one ``traj_XXXX.npz`` per trajectory (see `RecordedTrajectory.save`); each
    file is self-describing. The model is intentionally *not* saved here -- persist it
    yourself (e.g. ``world.save_xml_with_assets(...)``) and recompile it at replay time.

    Args:
        trajectories: One trajectory or a list of them.
        output_dir: Destination folder (created if needed).
    """
    trajectories = _as_trajectory_list(trajectories)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        traj.save(output_dir / f"traj_{i:04d}.npz")


def load_trajectories(source: PathLike) -> list[RecordedTrajectory]:
    """Load all trajectories from a folder written by `save_trajectories`.

    Returns the trajectories only; supply the compiled model yourself at replay time.
    """
    source = Path(source)
    files = sorted(source.glob("traj_*.npz"))
    if len(files) == 0:
        raise ValueError(f"No trajectory files (traj_*.npz) found in {source}.")
    return [RecordedTrajectory.load(f) for f in files]


def _as_trajectory_list(
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
) -> list[RecordedTrajectory]:
    if isinstance(trajectories, RecordedTrajectory):
        return [trajectories]
    trajectories = list(trajectories)
    if len(trajectories) == 0:
        raise ValueError("No trajectories given.")
    return trajectories


def _check_model_compatible(
    mj_model: mj.MjModel, trajectories: list[RecordedTrajectory]
) -> None:
    """Raise if a trajectory's ``qpos``/mocap layout does not fit ``mj_model``.

    A cheap, round-trip-robust guard (unlike a compiled-model hash, the ``qpos`` layout
    survives an XML round trip and the batch-render edits): it catches replaying against
    a structurally different model before any rendering work.
    """
    for i, traj in enumerate(trajectories):
        if traj.qpos.shape[1] != mj_model.nq:
            raise ValueError(
                f"Trajectory {i} (world {traj.world_id}) has nq={traj.qpos.shape[1]} "
                f"but the model has nq={mj_model.nq}; it was recorded against a "
                "different model."
            )
        if traj.has_mocap and traj.mocap_pos.shape[1] != mj_model.nmocap:
            raise ValueError(
                f"Trajectory {i} (world {traj.world_id}) has "
                f"nmocap={traj.mocap_pos.shape[1]} but the model has "
                f"nmocap={mj_model.nmocap}; it was recorded against a different model."
            )


def _resolve_render_output_paths(
    trajectories: list[RecordedTrajectory],
    camera_names: list[str],
    output_path: PathLike,
) -> list[dict[str, Path]]:
    """Map each (trajectory, camera) to an output mp4 path.

    Returns a list (parallel to ``trajectories``) of ``{camera_name: path}`` dicts.

    A single trajectory with a single camera and an ``output_path`` ending in
    ``.mp4`` is written directly to that file. Otherwise ``output_path`` is treated
    as a directory: each trajectory gets a ``world_{id}`` subfolder (suffixed with its
    list index to avoid collisions when world ids repeat), with one ``{camera}.mp4``
    per camera.
    """
    output_path = Path(output_path)
    single = len(trajectories) == 1 and len(camera_names) == 1

    paths: list[dict[str, Path]] = []
    for i, traj in enumerate(trajectories):
        if single and output_path.suffix == ".mp4":
            paths.append({camera_names[0]: output_path})
            continue
        traj_dir = (
            output_path
            if len(trajectories) == 1
            else output_path / f"world_{traj.world_id}_idx{i:04d}"
        )
        paths.append(
            {cam: traj_dir / f"{cam.replace('/', '_')}.mp4" for cam in camera_names}
        )
    return paths


def _resolve_cameras(
    cameras: str | list[str] | None, trajectories: list[RecordedTrajectory]
) -> list[str]:
    """Camera names to render: the explicit argument, or the recorded set if None.

    When falling back to the recorded set, all trajectories must agree -- the single
    CPU renderer (and the GPU batch render context) is built once for a shared set.
    """
    if cameras is not None:
        return [cameras] if isinstance(cameras, str) else list(cameras)
    names = list(trajectories[0].camera_names)
    for traj in trajectories[1:]:
        if list(traj.camera_names) != names:
            raise ValueError(
                "Trajectories were recorded with different cameras; pass an explicit "
                "`cameras` argument to choose a common set."
            )
    return names


def _render_trajectory_frames(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    renderer: mj.Renderer,
    traj: RecordedTrajectory,
    camera_ids: dict[str, int],
    scene_option: mj.MjvOption | None,
) -> dict[str, list[np.ndarray]]:
    """Re-render every frame of one trajectory for the requested cameras.

    Sets ``qpos`` (and mocap) per frame, runs position-only kinematics, and
    rasterizes each camera. Returns ``{camera_name: [frame, ...]}``.
    """
    frames: dict[str, list[np.ndarray]] = {cam: [] for cam in camera_ids}
    for f in range(traj.n_frames):
        mj_data.qpos[:] = traj.qpos[f]
        if traj.has_mocap:
            mj_data.mocap_pos[:] = traj.mocap_pos[f]
            mj_data.mocap_quat[:] = traj.mocap_quat[f]
        # Position-only kinematics regenerate all geom/site/camera/light transforms
        # that the renderer reads -- no need for velocities/forces/contacts.
        mj.mj_kinematics(mj_model, mj_data)
        mj.mj_camlight(mj_model, mj_data)
        for cam_name, cam_id in camera_ids.items():
            renderer.update_scene(mj_data, cam_id, scene_option)
            frames[cam_name].append(renderer.render())
    return frames


def render_trajectories(
    mj_model: mj.MjModel,
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
    output_path: PathLike,
    *,
    cameras: str | list[str] | None = None,
    scene_option: mj.MjvOption | None = None,
) -> None:
    """Replay recorded trajectories to video on the CPU.

    For GPU batch replay, use `flygym.warp.rendering.render_trajectories_gpu`.

    Args:
        mj_model: Compiled model the trajectories were recorded against (load it from
            wherever you persisted it, e.g. a folder written by
            `BaseCompositionElement.save_xml_with_assets`). Its ``qpos`` layout must
            match the trajectories.
        trajectories: One trajectory or a list of them (e.g. from `load_trajectories`).
        output_path: Where to write videos. See `_resolve_render_output_paths` for the
            file/directory layout.
        cameras: Camera name(s) to render. Defaults to each trajectory's recorded
            ``camera_names``.
        scene_option: MuJoCo scene options applied at render time.
    """
    trajectories = _as_trajectory_list(trajectories)
    _check_model_compatible(mj_model, trajectories)
    camera_names = _resolve_cameras(cameras, trajectories)
    camera_ids = {
        c: mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_CAMERA, c) for c in camera_names
    }
    out_paths = _resolve_render_output_paths(trajectories, camera_names, output_path)

    height, width = trajectories[0].camera_res
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height, width)
    try:
        for traj, traj_out in zip(trajectories, out_paths):
            frames = _render_trajectory_frames(
                mj_model, mj_data, renderer, traj, camera_ids, scene_option
            )
            for cam, cam_frames in frames.items():
                write_video_from_frames(
                    traj_out[cam], cam_frames, fps=traj.output_fps, codec="libx264"
                )
    finally:
        renderer.close()
