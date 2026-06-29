"""Record qpos trajectories on GPU and replay them via MuJoCo-Warp batch rendering."""

from typing import Any, override
from os import PathLike

import mujoco as mj
import mujoco_warp as mjw
import warp as wp
import numpy as np

from flygym.rendering.recorded_trajectory import (
    RecordedTrajectory,
    _as_trajectory_list,
    _check_model_compatible,
    _resolve_cameras,
    _resolve_render_output_paths,
)
from flygym.warp.rendering.live_rendering import _BaseWarpRenderer
from flygym.warp.utils import get_rgb_selected_worlds_and_cameras
from flygym.utils.video import write_video_from_frames


__all__ = ["WarpTrajectoryRecorder", "render_trajectories_gpu"]


class WarpTrajectoryRecorder(_BaseWarpRenderer):
    """Records ``qpos`` (and mocap poses) per world instead of rasterizing frames.

    The GPU counterpart of `flygym.rendering.TrajectoryRecorder`: a drop-in renderer
    swap for `GPUSimulation` that, at the render cadence, copies the generalized
    coordinates of the selected worlds off the GPU instead of running the batch
    renderer. Each selected world becomes one `RecordedTrajectory`, exposed as
    `recorded_trajectories`, in the same format the CPU recorder produces.
    """

    def _build_mj_renderer(self, mj_model, nrows, ncols, **kwargs):
        # No rasterization: skip the GL/EGL context allocation entirely.
        return None

    def _render_setup_impl(self, **kwargs: Any) -> None:
        self._nq = self.mj_model.nq
        self._nmocap = self.mj_model.nmocap
        # The CPU-side renderer/scene_option inherited from Renderer are unused.
        self.mj_renderer = None
        self.scene_option = None

    def _render_impl(self, mjw_data: mjw.Data) -> tuple:
        # One host transfer per recorded frame: (n_worlds, nq) is tiny next to the
        # (n_worlds, n_cams, H, W, 3) RGB tensor the batch renderer would buffer.
        qpos = mjw_data.qpos.numpy()[self.world_ids].copy()
        if self._nmocap > 0:
            mocap_pos = mjw_data.mocap_pos.numpy()[self.world_ids].copy()
            mocap_quat = mjw_data.mocap_quat.numpy()[self.world_ids].copy()
        else:
            mocap_pos = mocap_quat = None
        return (qpos, mocap_pos, mocap_quat)

    @property
    def recorded_trajectories(self) -> list[RecordedTrajectory]:
        """One `RecordedTrajectory` per recorded world."""
        if not self.buffer_frames:
            raise RuntimeError(
                "Frame buffering was disabled for this recorder, so recorded "
                "trajectories are not available."
            )
        if len(self._frames) == 0:
            raise RuntimeError("No frames have been recorded yet.")

        # self._frames is a list (over time) of (qpos, mocap_pos, mocap_quat) tuples,
        # each batched over the recorded worlds along axis 0.
        qpos_all = np.stack([f[0] for f in self._frames], axis=0)  # (T, n_worlds, nq)
        if self._nmocap > 0:
            mocap_pos_all = np.stack([f[1] for f in self._frames], axis=0)
            mocap_quat_all = np.stack([f[2] for f in self._frames], axis=0)

        trajectories = []
        for w, world_id in enumerate(self.world_ids):
            trajectories.append(
                RecordedTrajectory(
                    qpos=qpos_all[:, w, :],
                    output_fps=self.output_fps,
                    playback_speed=self.playback_speed,
                    camera_names=list(self.enabled_cam_names),
                    camera_res=self.camera_res,
                    world_id=world_id,
                    mocap_pos=mocap_pos_all[:, w] if self._nmocap > 0 else None,
                    mocap_quat=mocap_quat_all[:, w] if self._nmocap > 0 else None,
                )
            )
        return trajectories

    def _fetch_frames_to_cpu_impl(self, world_id_among_rendered, cam_id_among_rendered):
        raise RuntimeError(
            "WarpTrajectoryRecorder records state, not frames. Use "
            "`recorded_trajectories` and replay with render_trajectories_gpu."
        )

    @override
    def save_video(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "WarpTrajectoryRecorder records state, not frames. Save it with "
            "flygym.rendering.save_trajectories and replay with "
            "render_trajectories_gpu / flygym.rendering.render_trajectories."
        )

    @override
    def show_in_notebook(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "WarpTrajectoryRecorder records state, not frames. Save it with "
            "flygym.rendering.save_trajectories and replay with "
            "render_trajectories_gpu / flygym.rendering.render_trajectories."
        )

    @override
    def close(self) -> None:
        return  # no mj.Renderer context held


# Default number of frames staged into one batched GPU render. Tunable via the
# ``worlds_per_batch`` argument; larger uses more GPU memory.
_DEFAULT_WORLDS_PER_BATCH = 32


def render_trajectories_gpu(
    mj_model: mj.MjModel,
    trajectories: RecordedTrajectory | list[RecordedTrajectory],
    output_path: PathLike,
    *,
    cameras: str | list[str] | None = None,
    worlds_per_batch: int | None = None,
) -> None:
    """Replay recorded trajectories to video using MuJoCo-Warp GPU batch rendering.

    The GPU counterpart of `flygym.rendering.render_trajectories`. All ``(trajectory,
    frame)`` pairs are flattened into one work-list and bin-packed into batched
    ``mjw.Data`` renders: each batch stages a block of recorded ``qpos`` rows into the
    parallel worlds, runs position-only kinematics, rasterizes with the same path as
    `WarpGPUBatchRenderer`, then scatters the frames back to their trajectories. This
    decouples the batch size used for *rendering* (``worlds_per_batch``) from the
    number of worlds that were *simulated*.

    ``mj_model`` must already be prepared for GPU batch rendering -- i.e. have textures
    stripped and overhead lights added by
    `flygym.warp.rendering.modify_world_for_batch_rendering` (textured complex meshes
    otherwise corrupt MJWarp memory). The model held by a `GPUSimulation` configured
    with ``use_gpu_batch_rendering=True`` already satisfies this; to replay from a model
    you persisted yourself, call ``modify_world_for_batch_rendering(world)`` and
    recompile before passing it here. Those edits do not change the ``qpos`` layout, so
    the recorded trajectories stay valid.

    Args:
        mj_model: Compiled, batch-render-ready model (see note above). Its ``qpos``
            layout must match the trajectories.
        trajectories: One trajectory or a list of them (e.g. from
            `flygym.rendering.load_trajectories`).
        output_path: Where to write videos (see `_resolve_render_output_paths`).
        cameras: Camera name(s) to render. Defaults to each trajectory's recorded
            ``camera_names``. All trajectories must share the same camera set and
            resolution (the batch render context is built once). Lengths may differ.
        worlds_per_batch: Number of frames staged into one batched render. Defaults to
            a heuristic; larger uses more GPU memory.
    """
    trajectories = _as_trajectory_list(trajectories)
    _check_model_compatible(mj_model, trajectories)
    _render_trajectories_gpu(
        trajectories,
        mj_model,
        output_path,
        cameras=cameras,
        worlds_per_batch=worlds_per_batch,
    )


def _render_trajectories_gpu(
    trajectories: list[RecordedTrajectory],
    mj_model: mj.MjModel,
    output_path: PathLike,
    *,
    cameras: str | list[str] | None,
    worlds_per_batch: int | None,
) -> None:
    camera_names = _resolve_cameras(cameras, trajectories)
    height, width = _validate_shared_camera_res(trajectories)

    cam_ids = [
        mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_CAMERA, name) for name in camera_names
    ]
    for name, cid in zip(camera_names, cam_ids):
        if cid == -1:
            raise ValueError(f"Camera {name!r} not found in the model.")

    # Flatten all (trajectory, frame) pairs into one work-list.
    work = [
        (ti, fi) for ti, traj in enumerate(trajectories) for fi in range(traj.n_frames)
    ]
    if len(work) == 0:
        raise ValueError("Trajectories contain no frames.")

    batch = min(worlds_per_batch or _DEFAULT_WORLDS_PER_BATCH, len(work))
    nmocap = mj_model.nmocap
    nq = mj_model.nq

    # Pre-allocate output, one (n_frames, H, W, 3) array per (trajectory, camera).
    results: list[dict[str, np.ndarray]] = [
        {
            cam: np.zeros((traj.n_frames, height, width, 3), dtype=np.uint8)
            for cam in camera_names
        }
        for traj in trajectories
    ]

    # GPU-side setup: model, a batched Data of size `batch`, and a render context.
    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(mj_model, mj.MjData(mj_model), nworld=batch)
    cam_mask = [cid in cam_ids for cid in range(mj_model.ncam)]
    render_context = mjw.create_render_context(
        mjm=mj_model,
        nworld=batch,
        cam_active=cam_mask,
        cam_res=(width, height),  # MJWarp expects (W, H); we use (H, W)
    )
    world_ids_gpu = wp.array(list(range(batch)), dtype=wp.int32)
    cam_ids_gpu = wp.array(cam_ids, dtype=wp.int32)

    for base in range(0, len(work), batch):
        chunk = work[base : base + batch]
        chunk_len = len(chunk)

        # Stage qpos (and mocap) for this chunk into the parallel worlds. Padding
        # slots (last partial batch) repeat the last real frame and are never read.
        qpos_batch = np.zeros((batch, nq), dtype=np.float32)
        for j, (ti, fi) in enumerate(chunk):
            qpos_batch[j] = trajectories[ti].qpos[fi]
        qpos_batch[chunk_len:] = qpos_batch[chunk_len - 1]
        mjw_data.qpos.assign(qpos_batch)

        if nmocap > 0:
            mocap_pos_batch = np.zeros((batch, nmocap, 3), dtype=np.float32)
            mocap_quat_batch = np.zeros((batch, nmocap, 4), dtype=np.float32)
            for j, (ti, fi) in enumerate(chunk):
                mocap_pos_batch[j] = trajectories[ti].mocap_pos[fi]
                mocap_quat_batch[j] = trajectories[ti].mocap_quat[fi]
            mocap_pos_batch[chunk_len:] = mocap_pos_batch[chunk_len - 1]
            mocap_quat_batch[chunk_len:] = mocap_quat_batch[chunk_len - 1]
            mjw_data.mocap_pos.assign(mocap_pos_batch)
            mjw_data.mocap_quat.assign(mocap_quat_batch)

        # Position-only kinematics regenerate all geom/site/camera/light transforms.
        mjw.kinematics(mjw_model, mjw_data)
        mjw.camlight(mjw_model, mjw_data)

        mjw.refit_bvh(mjw_model, mjw_data, render_context)
        mjw.render(mjw_model, mjw_data, render_context)

        rgb_out = wp.zeros((batch, len(cam_ids), height, width), dtype=wp.vec3f)
        get_rgb_selected_worlds_and_cameras(
            render_context, world_ids_gpu, cam_ids_gpu, rgb_out
        )
        rgb_np = (rgb_out.numpy() * 255.0).astype(np.uint8)  # (batch, ncam, H, W, 3)

        # Scatter rendered frames back to their (trajectory, camera, frame) slot.
        for j, (ti, fi) in enumerate(chunk):
            for ci, cam in enumerate(camera_names):
                results[ti][cam][fi] = rgb_np[j, ci]

    out_paths = _resolve_render_output_paths(trajectories, camera_names, output_path)
    for ti, traj in enumerate(trajectories):
        for cam in camera_names:
            write_video_from_frames(
                out_paths[ti][cam],
                list(results[ti][cam]),
                fps=traj.output_fps,
                codec="libx264",
            )


def _validate_shared_camera_res(
    trajectories: list[RecordedTrajectory],
) -> tuple[int, int]:
    res = tuple(trajectories[0].camera_res)
    for traj in trajectories[1:]:
        if tuple(traj.camera_res) != res:
            raise ValueError(
                "GPU batch rendering requires all trajectories to share one camera "
                f"resolution, but found {res} and {tuple(traj.camera_res)}."
            )
    return res
