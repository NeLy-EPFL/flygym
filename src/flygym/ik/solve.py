"""Keypoint-based inverse kinematics solver."""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco as mj
import numpy as np
import scipy.optimize
from loguru import logger

from flygym.anatomy import PASSIVE_TARSAL_LINKS, JointDOF
from flygym.ik.keypoints import KeypointSet

__all__ = ["IKResult", "fit_qpos_to_keypoints", "fit_qpos_trajectory_to_keypoints"]

# The four inter-tarsal joints per leg (tarsus1-tarsus2 through
# tarsus4-tarsus5, i.e. every leg DOF whose child is in
# `flygym.anatomy.PASSIVE_TARSAL_LINKS`) have no actuator of their own and are
# highly redundant with the tibia-tarsus1 DOF for reaching a given claw
# position -- left unbounded, the solver is free to assign them an arbitrary,
# anatomically meaningless bend. Real tarsal segments flex only slightly
# relative to one another, so they are clamped to this range by default.
_PASSIVE_TARSAL_BOUND_DEG = (-10.0, 10.0)


@dataclass
class IKResult:
    """Result of a single-frame keypoint IK fit.

    Attributes:
        qpos: Solved joint position vector, shape `(nq,)`.
        cost: Final cost, `0.5 * sum(residuals**2)`.
        success: Whether the optimizer reported successful termination.
        n_iters: Number of solver function evaluations used.
    """

    qpos: np.ndarray
    cost: float
    success: bool
    n_iters: int


def _joint_bounds(mj_model: mj.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Per-qpos-element (lower, upper) bounds derived from joint limits.

    Joints with a hard physical limit (`jnt_limited`/`jnt_range`) use that
    range. The passive inter-tarsal joints (see `_PASSIVE_TARSAL_BOUND_DEG`)
    are additionally clamped to +/-10 degrees regardless of `jnt_limited`.
    """
    lower = np.full(mj_model.nq, -np.inf)
    upper = np.full(mj_model.nq, np.inf)
    for joint_id in range(mj_model.njnt):
        if not mj_model.jnt_limited[joint_id]:
            continue
        qposadr = mj_model.jnt_qposadr[joint_id]
        lower[qposadr] = mj_model.jnt_range[joint_id, 0]
        upper[qposadr] = mj_model.jnt_range[joint_id, 1]

    tarsal_lower, tarsal_upper = np.radians(_PASSIVE_TARSAL_BOUND_DEG)
    for joint_id in range(mj_model.njnt):
        joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        try:
            dof = JointDOF.from_name(joint_name)
        except ValueError:
            continue
        if dof.child.link in PASSIVE_TARSAL_LINKS:
            qposadr = mj_model.jnt_qposadr[joint_id]
            lower[qposadr] = tarsal_lower
            upper[qposadr] = tarsal_upper
    return lower, upper


def _keypoint_world_points(
    mj_data: mj.MjData, body_ids: np.ndarray, local_offsets: np.ndarray
) -> np.ndarray:
    """World position of each keypoint, given the model's current state."""
    xpos = mj_data.xpos[body_ids]
    xmat = mj_data.xmat[body_ids].reshape(-1, 3, 3)
    return xpos + np.einsum("nij,nj->ni", xmat, local_offsets)


def fit_qpos_to_keypoints(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    keypoints: KeypointSet,
    target_positions: np.ndarray,
    *,
    initial_qpos: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    projection_axes: tuple[int, int] | None = None,
    max_iters: int = 100,
    ftol: float = 1e-8,
    on_iterate: Callable[[np.ndarray], None] | None = None,
) -> IKResult:
    """Fit joint angles (qpos) to a set of weighted keypoint targets.

    Minimizes the weighted squared distance between each keypoint's world
    position (given the current `qpos`) and its target position, across all
    requested keypoints (not just leg tips), using
    `scipy.optimize.least_squares` with an analytic Jacobian computed via
    `mujoco.mj_jac`.

    Args:
        mj_model: Compiled MuJoCo model. Must have `nv == nq` (i.e. only
            hinge/slide joints, no free/ball joints) -- true for a
            standalone-compiled fly, which has no free joint.
        mj_data: Associated MuJoCo data (modified in place during fitting).
        keypoints: Keypoint targets and their weights.
        target_positions: Target position for each keypoint, in the same
            order as `keypoints.targets`, shape `(n_keypoints, 3)`, or
            `(n_keypoints, 2)` if `projection_axes` is given.
        initial_qpos: Initial guess for `qpos`. Defaults to the model's
            `"neutral"` keyframe if present, otherwise zeros. Matters more
            than it might seem: with sparse keypoints, a multi-DOF chain can
            have more than one locally-optimal configuration (e.g.
            elbow-up vs. elbow-down) that converges to the same keypoint
            positions but very different joint angles. A poor guess can
            converge to the "wrong" (but equally valid) one.
        bounds: Explicit `(lower, upper)` qpos bounds, each shape `(nq,)`, in
            radians. If `None` (the default), bounds are read from
            `mj_model.jnt_range`/`jnt_limited` -- which, unless the fly was
            built with explicit joint `range=` kwargs, are usually
            unbounded (flygym does not set joint limits by default) -- except
            for the passive inter-tarsal joints (tarsus1-tarsus2 through
            tarsus4-tarsus5), which are always clamped to +/-10 degrees since
            they have no actuator of their own and are otherwise free to take
            on an arbitrary, anatomically meaningless bend. Pass
            `flygym.ik.seqikpy_joint_bounds(mj_model)` to use SeqIKPy's
            default leg angle ranges instead.
        projection_axes: If given, a pair of axis indices (e.g. `(0, 1)` for
            xy) to orthographically project fitted 3D keypoint positions onto
            before comparing to `target_positions`, for fitting 2D keypoints.
            If `None`, targets are fit in full 3D.
        max_iters: Maximum number of solver function evaluations.
        ftol: Convergence tolerance passed to `scipy.optimize.least_squares`.
        on_iterate: If given, called with the current `qpos` once per solver
            iteration (technically: each time the Jacobian is evaluated,
            which TRF does once per accepted iterate) -- useful for recording
            the pose trajectory during optimization, e.g. to render a
            convergence video.

    Returns:
        An `IKResult` with the solved `qpos`.

    Raises:
        ValueError: If `mj_model` has `nv != nq`, or if `target_positions`
            has the wrong shape.
    """
    if mj_model.nv != mj_model.nq:
        raise ValueError(
            "fit_qpos_to_keypoints requires nv == nq (only hinge/slide "
            "joints, no free/ball joints) so that mujoco.mj_jac's output can "
            "be used directly as the Jacobian w.r.t. qpos."
        )

    body_ids = keypoints.resolve_body_ids(mj_model)
    local_offsets = keypoints.local_offsets()
    sqrt_weights = np.sqrt(keypoints.weights)

    dim = 3 if projection_axes is None else 2
    target_positions = np.asarray(target_positions, dtype=float)
    expected_shape = (len(keypoints.targets), dim)
    if target_positions.shape != expected_shape:
        raise ValueError(
            f"target_positions must have shape {expected_shape}, got "
            f"{target_positions.shape}."
        )

    lower, upper = _joint_bounds(mj_model) if bounds is None else bounds
    if initial_qpos is None:
        initial_qpos = (
            mj_model.key_qpos[0].copy() if mj_model.nkey > 0 else np.zeros(mj_model.nq)
        )
    initial_qpos = np.clip(np.asarray(initial_qpos, dtype=float), lower, upper)

    def residuals(qpos: np.ndarray) -> np.ndarray:
        mj_data.qpos[:] = qpos
        mj.mj_kinematics(mj_model, mj_data)
        points = _keypoint_world_points(mj_data, body_ids, local_offsets)
        if projection_axes is not None:
            points = points[:, projection_axes]
        return ((points - target_positions) * sqrt_weights[:, None]).reshape(-1)

    def jacobian(qpos: np.ndarray) -> np.ndarray:
        if on_iterate is not None:
            on_iterate(qpos)
        mj_data.qpos[:] = qpos
        mj.mj_kinematics(mj_model, mj_data)
        # mj_jac additionally requires the dof motion subspace (cdof), which
        # mj_kinematics alone does not populate.
        mj.mj_comPos(mj_model, mj_data)
        points = _keypoint_world_points(mj_data, body_ids, local_offsets)
        nv = mj_model.nv
        full_jac = np.empty((len(body_ids), 3, nv))
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        for i, (body_id, point) in enumerate(zip(body_ids, points)):
            mj.mj_jac(mj_model, mj_data, jacp, jacr, point, int(body_id))
            full_jac[i] = jacp
        if projection_axes is not None:
            full_jac = full_jac[:, projection_axes, :]
        full_jac = full_jac * sqrt_weights[:, None, None]
        return full_jac.reshape(-1, nv)

    logger.info(
        f"Fitting qpos to {len(keypoints.targets)} keypoints (max_iters={max_iters})..."
    )
    result = scipy.optimize.least_squares(
        residuals,
        initial_qpos,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_iters,
        ftol=ftol,
        xtol=ftol,
    )
    if on_iterate is not None:
        on_iterate(result.x)  # guarantee the converged qpos is the last iterate
    logger.info(f"IK fit finished with final cost {result.cost:.6g}.")
    return IKResult(
        qpos=result.x,
        cost=result.cost,
        success=result.success,
        n_iters=result.nfev,
    )


def fit_qpos_trajectory_to_keypoints(
    mj_model: mj.MjModel,
    keypoints: KeypointSet,
    target_positions: np.ndarray,
    *,
    mj_data: mj.MjData | None = None,
    initial_qpos: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    projection_axes: tuple[int, int] | None = None,
    max_iters: int = 100,
    ftol: float = 1e-8,
    backend: str = "cpu",
    warp_damping: float = 1e-2,
    warp_batch_size: int | None = 1000,
    warp_use_graph_capture: bool = True,
) -> np.ndarray:
    """Fit a trajectory of qpos to a sequence of keypoint target frames.

    Two backends are available:

    - `backend="cpu"` (default): calls `fit_qpos_to_keypoints` once per
      frame, warm-starting each frame's initial guess from the previous
      frame's solution (falling back to `initial_qpos` for the first frame).
      Frames are solved sequentially, but each one converges quickly since it
      starts near the answer.
    - `backend="warp"`: fits every frame **simultaneously** as parallel
      MuJoCo-Warp worlds on the GPU, using a fixed-iteration-count damped
      Gauss-Newton solver (see `flygym.ik.warp_solve`). There is no
      warm-starting between frames here since they are solved concurrently,
      not sequentially -- every frame starts from the same `initial_qpos`.
      Requires the `warp` extra and an NVIDIA GPU. Much faster for large
      batches of frames (e.g. a full recorded clip); may need more
      iterations than the CPU backend to reach comparable accuracy, since it
      lacks CPU backend's adaptive trust-region convergence check.

    Args:
        mj_model: Compiled MuJoCo model.
        keypoints: Keypoint targets and their weights.
        target_positions: Target positions per frame, shape
            `(n_frames, n_keypoints, 3)`, or `(n_frames, n_keypoints, 2)` if
            `projection_axes` is given.
        mj_data: Associated MuJoCo data (modified in place during fitting).
            Required for `backend="cpu"`; ignored for `backend="warp"` (which
            builds its own scratch `MjData` internally).
        initial_qpos: Initial guess for `qpos`. For `backend="cpu"`, only the
            first frame's guess (later frames are warm-started from the
            previous frame's solution); for `backend="warp"`, every frame's
            guess. Defaults as in `fit_qpos_to_keypoints`.
        bounds: See `fit_qpos_to_keypoints`.
        projection_axes: See `fit_qpos_to_keypoints`.
        max_iters: For `backend="cpu"`, maximum solver function evaluations
            per frame. For `backend="warp"`, the fixed number of Gauss-Newton
            iterations to run for every frame.
        ftol: Convergence tolerance per frame. Only used by `backend="cpu"`.
        backend: `"cpu"` or `"warp"`.
        warp_damping: Levenberg-Marquardt damping. Only used by
            `backend="warp"`; see `flygym.ik.warp_solve`.
        warp_batch_size: Maximum frames solved as one parallel GPU batch;
            larger inputs are chunked sequentially. Only used by
            `backend="warp"`; see `flygym.ik.warp_solve`.
        warp_use_graph_capture: Whether to use CUDA graph capture. Only used
            by `backend="warp"`; see `flygym.ik.warp_solve`.

    Returns:
        Solved `qpos` for each frame, shape `(n_frames, nq)`.

    Raises:
        ValueError: If `backend` is not `"cpu"` or `"warp"`, or if
            `backend="cpu"` and `mj_data` is not given.
    """
    if backend == "warp":
        from flygym.ik.warp_solve import fit_qpos_trajectory_to_keypoints_warp

        return fit_qpos_trajectory_to_keypoints_warp(
            mj_model,
            keypoints,
            target_positions,
            initial_qpos=initial_qpos,
            bounds=bounds,
            projection_axes=projection_axes,
            max_iters=max_iters,
            initial_damping=warp_damping,
            batch_size=warp_batch_size,
            use_graph_capture=warp_use_graph_capture,
        )
    if backend != "cpu":
        raise ValueError(f"Unknown backend {backend!r}; must be 'cpu' or 'warp'.")
    if mj_data is None:
        raise ValueError("mj_data is required for backend='cpu'.")

    target_positions = np.asarray(target_positions, dtype=float)
    n_frames = target_positions.shape[0]
    qpos_trajectory = np.empty((n_frames, mj_model.nq))
    warm_start = initial_qpos
    for t in range(n_frames):
        result = fit_qpos_to_keypoints(
            mj_model,
            mj_data,
            keypoints,
            target_positions[t],
            initial_qpos=warm_start,
            bounds=bounds,
            projection_axes=projection_axes,
            max_iters=max_iters,
            ftol=ftol,
        )
        qpos_trajectory[t] = result.qpos
        warm_start = result.qpos
    return qpos_trajectory
