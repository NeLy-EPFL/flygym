"""GPU-batched keypoint IK solver using MuJoCo Warp (`mujoco_warp`).

Fits many frames in parallel -- one MuJoCo-Warp "world" per frame -- using a
damped Gauss-Newton (Levenberg-Marquardt) loop with joint-limit projection.
`scipy.optimize.least_squares`'s trust-region method (used by the CPU solver
in `flygym.ik.solve`) has no batched/GPU equivalent, so this is a hand-rolled
solver instead: each step tentatively solves the damped normal equations
`(J^T J + damping * I) delta = -J^T r` per frame, clips the result to the
joint limits (projected Gauss-Newton), and accepts or rejects the step per
frame depending on whether it actually reduced that frame's cost -- adapting
each frame's damping independently (increase after a rejection, decrease
after an acceptance), the standard Levenberg-Marquardt trust-region
heuristic. A fixed damping schedule is unstable here: too little damping
diverges outright for some frames (the keypoint chain has near-singular
directions, e.g. redundant DOFs weakly constrained by sparse keypoints), too
much stalls convergence well short of the optimum.

Performance notes (measured on an RTX 3080 Ti, 12 GB, this model's 42 DOFs /
30 keypoints; see `scripts/dev/benchmark_ik.py`):

- The dominant cost at scale is assembling the batched normal equations
  `J^T J` / `J^T r`. `np.matmul` (which dispatches to a batched BLAS GEMM) is
  ~5-6x faster here than the equivalent `np.einsum` call, which does not take
  the same fast path -- this was by far the largest win found, well above
  either optimization below.
- All per-keypoint GPU work (forward kinematics, the point-gathering kernel,
  `mjw.jac`) is consolidated into single pre-allocated buffers and
  transferred to the host in one `.numpy()` call each, rather than once per
  keypoint; and the Jacobian is only computed once per iteration (a separate,
  cheaper "light" pass without it is used for the trial-step accept/reject
  cost check).
- CUDA graph capture (`use_graph_capture=True`, the default) additionally
  shaves a further ~5-15% by replaying each GPU pass as a single captured
  graph instead of re-dispatching every kernel/`mjw` call from Python each
  iteration -- a real but secondary win once the two optimizations above are
  in place, and one that shrinks further as batch size grows (host-side
  linear algebra dominates at large batch sizes regardless).
- Throughput peaks around **500-1000 frames per batch** (~800-830 frames/s
  fitting all 30 keypoints for 30 iterations) and degrades gradually as batch
  size grows further (~600 frames/s at 8,000; ~460 frames/s at 100,000) --
  `batch_size` defaults to 1000 accordingly. Larger batches aren't wasted
  work, just somewhat less efficient per frame; chunking via `batch_size` is
  about bounding memory, not chasing peak throughput, so the default favors
  throughput and chunks automatically for anything larger.
- The GPU ran out of memory somewhere between 200,000 (succeeded) and
  250,000 frames (failed - the Jacobian buffer alone needs
  `n_keypoints * n_frames * 3 * nv * 4` bytes, e.g. ~4.5 GB at 300,000
  frames) in a single batch on a 12 GB GPU. `batch_size` keeps any single
  batch far below this regardless of total input size.

Requires the `warp` extra (`pip install flygym[warp]`) and an NVIDIA GPU.
"""

import warnings

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp
from loguru import logger

from flygym.ik.keypoints import KeypointSet
from flygym.ik.solve import _joint_bounds

__all__ = ["fit_qpos_trajectory_to_keypoints_warp"]


@wp.kernel
def _all_keypoints_point_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    body_ids: wp.array(dtype=wp.int32),
    local_offsets: wp.array(dtype=wp.vec3),
    out_points: wp.array2d(dtype=wp.vec3),
):
    """Compute all keypoints' world positions in one launch.

    Grid is `(n_keypoints, n_frames)`; `body_ids`/`local_offsets` are indexed
    by keypoint, `xpos`/`xmat` by (frame, body).
    """
    k, world = wp.tid()
    body_id = body_ids[k]
    out_points[k, world] = (
        xpos[world, body_id] + xmat[world, body_id] * local_offsets[k]
    )


class _BatchedGpuPasses:
    """Holds the persistent GPU buffers and (optionally graph-captured) passes
    used each solver iteration, for one fixed `n_frames` batch.

    Two passes are exposed:

    - `light()`: forward kinematics + keypoint positions only. Used for
      cost-only evaluations (e.g. checking whether a trial step should be
      accepted), where the Jacobian isn't needed.
    - `full()`: forward kinematics + keypoint positions + per-keypoint
      Jacobians (via `mjw.jac`, which additionally requires `mjw.com_pos`).
      Used once per iteration, for the pose the next Gauss-Newton step is
      taken from.

    Both write into the same pre-allocated `points` buffer (and `full()`
    additionally into `jacp`); callers should read these immediately via
    `.numpy()` before triggering the other pass, which overwrites them.
    """

    def __init__(
        self,
        mjw_model,
        mjw_data,
        body_ids,
        local_offsets,
        n_frames,
        nv,
        use_graph_capture,
    ):
        self.mjw_model = mjw_model
        self.mjw_data = mjw_data
        n_keypoints = len(body_ids)

        self.body_ids_wp = wp.array(
            np.asarray(body_ids, dtype=np.int32), dtype=wp.int32
        )
        self.offsets_wp = wp.array(local_offsets.astype(np.float32), dtype=wp.vec3)
        # mjw.jac wants one body id per world; every world (frame) uses the same
        # body for a given keypoint, so this is body_ids[k] broadcast over frames.
        self.body_id_wp_by_keypoint = wp.array(
            np.tile(np.asarray(body_ids, dtype=np.int32)[:, None], (1, n_frames)),
            dtype=wp.int32,
        )
        self.points = wp.zeros((n_keypoints, n_frames), dtype=wp.vec3)
        self.jacp = wp.zeros((n_keypoints, n_frames, 3, nv), dtype=wp.float32)
        self.n_keypoints = n_keypoints

        self._light_graph = None
        self._full_graph = None
        if use_graph_capture and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._light_uncaptured()
            self._light_graph = capture.graph
            with wp.ScopedCapture() as capture:
                self._full_uncaptured()
            self._full_graph = capture.graph

    def _gather_points(self):
        wp.launch(
            _all_keypoints_point_kernel,
            dim=(self.n_keypoints, self.points.shape[1]),
            inputs=[
                self.mjw_data.xpos,
                self.mjw_data.xmat,
                self.body_ids_wp,
                self.offsets_wp,
                self.points,
            ],
        )

    def _light_uncaptured(self):
        mjw.kinematics(self.mjw_model, self.mjw_data)
        self._gather_points()

    def _full_uncaptured(self):
        mjw.kinematics(self.mjw_model, self.mjw_data)
        mjw.com_pos(self.mjw_model, self.mjw_data)
        self._gather_points()
        for k in range(self.n_keypoints):
            mjw.jac(
                self.mjw_model,
                self.mjw_data,
                self.jacp[k],
                None,
                self.points[k],
                self.body_id_wp_by_keypoint[k],
            )

    def light(self):
        if self._light_graph is not None:
            wp.capture_launch(self._light_graph)
        else:
            self._light_uncaptured()

    def full(self):
        if self._full_graph is not None:
            wp.capture_launch(self._full_graph)
        else:
            self._full_uncaptured()


def _residual_and_cost(points_np, target_positions, sqrt_weights):
    blocks = [
        sqrt_weights[k] * (points_np[k] - target_positions[:, k, :])
        for k in range(points_np.shape[0])
    ]
    residual = np.concatenate(blocks, axis=1).astype(np.float64)
    cost = 0.5 * np.sum(residual**2, axis=1)
    return residual, cost


def fit_qpos_trajectory_to_keypoints_warp(
    mj_model: mj.MjModel,
    keypoints: KeypointSet,
    target_positions: np.ndarray,
    *,
    initial_qpos: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    projection_axes: tuple[int, int] | None = None,
    max_iters: int = 30,
    initial_damping: float = 1e-2,
    damping_decrease: float = 3.0,
    damping_increase: float = 5.0,
    batch_size: int | None = 1000,
    use_graph_capture: bool = True,
) -> np.ndarray:
    """Fit a trajectory of qpos to keypoint target frames, batched on the GPU.

    Unlike `flygym.ik.solve.fit_qpos_trajectory_to_keypoints`, which fits one
    frame at a time on the CPU and warm-starts each frame from the previous
    one, this fits **all frames simultaneously** as parallel MuJoCo-Warp
    worlds -- there is no warm-starting between frames here, since they are
    solved concurrently rather than sequentially. Every frame starts from the
    same `initial_qpos`.

    Args:
        mj_model: Compiled MuJoCo model. Must have `nv == nq` (see
            `flygym.ik.solve.fit_qpos_to_keypoints`). Mutated in place if
            `mj_model.opt.noslip_iterations > 0` (unsupported by MJWarp; set
            to 0 with a warning).
        keypoints: Keypoint targets and their weights.
        target_positions: Target position for each keypoint in each frame,
            shape `(n_frames, n_keypoints, 3)`, or `(n_frames, n_keypoints,
            2)` if `projection_axes` is given.
        initial_qpos: Initial guess for `qpos`, shared by every frame.
            Defaults to the model's `"neutral"` keyframe if present,
            otherwise zeros.
        bounds: See `flygym.ik.solve.fit_qpos_to_keypoints`.
        projection_axes: See `flygym.ik.solve.fit_qpos_to_keypoints`.
        max_iters: Number of Gauss-Newton iterations to run (fixed -- there is
            no early-exit convergence check, so this always runs exactly this
            many iterations, even for frames that converged earlier).
        initial_damping: Starting Levenberg-Marquardt damping (added to the
            normal equations' diagonal, `J^T J + damping * I`), per frame.
        damping_decrease: Factor to divide a frame's damping by after an
            accepted (cost-reducing) step.
        damping_increase: Factor to multiply a frame's damping by after a
            rejected (cost-increasing) step.
        batch_size: Maximum number of frames solved as one parallel GPU batch
            (one MuJoCo-Warp world per frame). If `target_positions` has more
            frames than this, they are processed sequentially in chunks of
            `batch_size` (each chunk independently parallel), to bound GPU
            memory use -- real recordings can have tens of thousands of
            frames, which may not fit as a single batch. `None` disables
            chunking (always solves all frames as one batch). See
            `scripts/dev/benchmark_ik.py` for throughput vs. batch size.
        use_graph_capture: If True (default), capture each repeated GPU pass
            (forward-kinematics-and-keypoints, and the same plus Jacobians)
            as a CUDA graph and replay it every iteration, instead of
            re-dispatching each kernel/`mjw` call from Python -- a modest
            (~5-15%) but consistent speedup. Silently has no effect if the
            resolved Warp device isn't CUDA (e.g. CPU fallback).

    Returns:
        Solved `qpos` for each frame, shape `(n_frames, nq)`.

    Raises:
        ValueError: If `mj_model` has `nv != nq`, or if `target_positions`
            has the wrong shape.
    """
    if mj_model.nv != mj_model.nq:
        raise ValueError(
            "fit_qpos_trajectory_to_keypoints_warp requires nv == nq (only "
            "hinge/slide joints, no free/ball joints)."
        )
    if mj_model.opt.noslip_iterations > 0:
        warnings.warn(
            "MuJoCo Warp does not support noslip iterations. Setting "
            "mj_model.opt.noslip_iterations to 0 (mutates mj_model in place)."
        )
        mj_model.opt.noslip_iterations = 0

    body_ids = keypoints.resolve_body_ids(mj_model)
    local_offsets = keypoints.local_offsets()
    sqrt_weights = np.sqrt(keypoints.weights)
    n_keypoints = len(keypoints.targets)

    dim = 3 if projection_axes is None else 2
    target_positions = np.asarray(target_positions, dtype=np.float32)
    n_frames_total = target_positions.shape[0]
    expected_shape = (n_frames_total, n_keypoints, dim)
    if target_positions.shape != expected_shape:
        raise ValueError(
            f"target_positions must have shape {expected_shape}, got "
            f"{target_positions.shape}."
        )

    lower, upper = _joint_bounds(mj_model) if bounds is None else bounds
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if initial_qpos is None:
        initial_qpos = (
            mj_model.key_qpos[0].copy() if mj_model.nkey > 0 else np.zeros(mj_model.nq)
        )
    initial_qpos = np.clip(np.asarray(initial_qpos, dtype=np.float32), lower, upper)

    chunk_size = (
        n_frames_total if batch_size is None else min(batch_size, n_frames_total)
    )
    n_chunks = -(-n_frames_total // chunk_size)  # ceil division
    if n_chunks > 1:
        logger.info(
            f"{n_frames_total} frames exceeds batch_size={batch_size}; "
            f"processing in {n_chunks} chunks of up to {chunk_size} frames."
        )

    qpos_out = np.empty((n_frames_total, mj_model.nq), dtype=np.float32)
    for chunk_start in range(0, n_frames_total, chunk_size):
        chunk_targets = target_positions[chunk_start : chunk_start + chunk_size]
        qpos_out[chunk_start : chunk_start + chunk_size] = _fit_one_batch(
            mj_model,
            body_ids,
            local_offsets,
            sqrt_weights,
            n_keypoints,
            chunk_targets,
            initial_qpos,
            lower,
            upper,
            projection_axes,
            max_iters,
            initial_damping,
            damping_decrease,
            damping_increase,
            use_graph_capture,
        )
    return qpos_out


def _fit_one_batch(
    mj_model,
    body_ids,
    local_offsets,
    sqrt_weights,
    n_keypoints,
    target_positions,
    initial_qpos,
    lower,
    upper,
    projection_axes,
    max_iters,
    initial_damping,
    damping_decrease,
    damping_increase,
    use_graph_capture,
):
    n_frames = target_positions.shape[0]
    nv = mj_model.nv

    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(
        mj_model, mj.MjData(mj_model), nworld=n_frames, njmax=1, nconmax=1
    )
    qpos = np.tile(initial_qpos, (n_frames, 1)).astype(np.float32)
    mjw_data.qpos.assign(qpos)

    passes = _BatchedGpuPasses(
        mjw_model, mjw_data, body_ids, local_offsets, n_frames, nv, use_graph_capture
    )

    def points_2d(points_np):
        return points_np if projection_axes is None else points_np[..., projection_axes]

    eye_nv = np.eye(nv, dtype=np.float64)
    damping = np.full(n_frames, initial_damping, dtype=np.float64)

    logger.info(
        f"Fitting {n_frames} frames in parallel on the GPU "
        f"({n_keypoints} keypoints, {max_iters} iterations)..."
    )

    passes.full()
    points_np = points_2d(passes.points.numpy())
    residual, cost = _residual_and_cost(points_np, target_positions, sqrt_weights)

    for _ in range(max_iters):
        jacp_np = passes.jacp.numpy()
        if projection_axes is not None:
            jacp_np = jacp_np[:, :, projection_axes, :]
        jac_blocks = [sqrt_weights[k] * jacp_np[k] for k in range(n_keypoints)]
        jac = np.concatenate(jac_blocks, axis=1).astype(np.float64)

        # np.matmul (unlike the equivalent np.einsum) dispatches to a batched
        # BLAS GEMM here and is ~5-6x faster -- this is the dominant cost at
        # scale, so this is not a stylistic choice.
        jac_t = jac.transpose(0, 2, 1)
        jtj = jac_t @ jac + damping[:, None, None] * eye_nv
        jtr = jac_t @ residual[..., None]
        delta = np.linalg.solve(jtj, -jtr)[..., 0]
        qpos_trial = np.clip(qpos + delta, lower, upper).astype(np.float32)

        mjw_data.qpos.assign(qpos_trial)
        passes.light()
        points_np = points_2d(passes.points.numpy())
        _, cost_trial = _residual_and_cost(points_np, target_positions, sqrt_weights)

        accepted = cost_trial < cost
        qpos = np.where(accepted[:, None], qpos_trial, qpos)
        damping = np.where(
            accepted, damping / damping_decrease, damping * damping_increase
        )

        # mjw_data currently reflects qpos_trial for every frame, but qpos above
        # only actually moved for the accepted subset -- resync GPU state (and
        # points/residual/cost) to the merged qpos before the next iteration's
        # Jacobian evaluation.
        mjw_data.qpos.assign(qpos)
        passes.full()
        points_np = points_2d(passes.points.numpy())
        residual, cost = _residual_and_cost(points_np, target_positions, sqrt_weights)

    return qpos
