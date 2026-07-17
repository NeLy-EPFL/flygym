"""Render a video of a single-frame keypoint IK fit converging to its target."""

import mujoco as mj
import numpy as np
from loguru import logger

from flygym.ik.keypoints import KeypointSet
from flygym.ik.solve import IKResult, _keypoint_world_points, fit_qpos_to_keypoints
from flygym.rendering import Renderer
from flygym.utils.video import write_video_from_frames

__all__ = ["render_ik_convergence_video"]


def _resolve_camera_id(mj_model: mj.MjModel, camera: str | mj.MjsCamera) -> int:
    name = camera if isinstance(camera, str) else camera.name
    cam_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"Camera '{name}' not found in the model.")
    return cam_id


def _add_sphere_marker(
    scene: mj.MjvScene, pos: np.ndarray, radius: float, rgba: tuple[float, ...]
) -> None:
    if scene.ngeom >= scene.maxgeom:
        logger.warning("MjvScene is full; skipping a keypoint marker.")
        return
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    scene.ngeom += 1


def render_ik_convergence_video(
    mj_model: mj.MjModel,
    mj_data: mj.MjData,
    keypoints: KeypointSet,
    target_positions: np.ndarray,
    output_path: str,
    camera: str | mj.MjsCamera,
    *,
    initial_qpos: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    max_iters: int = 100,
    ftol: float = 1e-8,
    camera_res: tuple[int, int] = (480, 640),
    fps: int = 10,
    hold_final_frames: int = 10,
    target_marker_radius: float = 0.03,
    target_marker_rgba: tuple[float, float, float, float] = (1.0, 0.1, 0.1, 0.9),
    fitted_marker_radius: float = 0.025,
    fitted_marker_rgba: tuple[float, float, float, float] = (0.1, 1.0, 0.1, 0.9),
) -> IKResult:
    """Fit one frame of keypoints and render a video of the fit converging.

    Runs `fit_qpos_to_keypoints` while recording the pose at each solver
    iteration (via its `on_iterate` callback), then renders one video frame
    per iteration -- starting from whatever pose `initial_qpos` specifies
    (e.g. the previous frame's solution, a neutral pose, or a cold start at
    zero) and ending at the converged fit. Each frame overlays the fixed
    keypoint targets (red spheres) against the current iterate's fitted
    keypoint positions (green spheres), so the video shows the green markers
    moving to meet the red ones as the pose converges.

    Args:
        mj_model: Compiled MuJoCo model.
        mj_data: Associated MuJoCo data (modified in place).
        keypoints: Keypoint targets and their weights.
        target_positions: Target position for each keypoint, shape
            `(n_keypoints, 3)`. Unlike `fit_qpos_to_keypoints`, only 3D
            targets are supported here (`projection_axes` fitting isn't,
            since a 2D target has no third coordinate to place a marker at).
        output_path: Video file path (passed to `imageio.v3.imwrite` via
            `flygym.utils.video.write_video_from_frames`).
        camera: Camera name or `MjsCamera` to render from (e.g. one added via
            `fly.add_tracking_camera()`).
        initial_qpos: Initial guess for `qpos` -- the pose the video starts
            from. E.g. pass the previous frame's solved qpos to visualize a
            warm-started fit, `qpos = 0` (or leave as `None`, the model's own
            `"neutral"` keyframe) for a naturally-posed cold start, or
            `flygym.ik.seqikpy_initial_guess_qpos(mj_model)` to visualize
            what SeqIKPy's own optimizer seed converges from (not a natural
            pose itself -- see that function's docstring).
        bounds: See `fit_qpos_to_keypoints`.
        max_iters: See `fit_qpos_to_keypoints`.
        ftol: See `fit_qpos_to_keypoints`.
        camera_res: `(height, width)` in pixels.
        fps: Output video frame rate. Since one video frame is rendered per
            solver iteration (not per unit of wall-clock or simulated time),
            this just controls playback speed, not accuracy.
        hold_final_frames: Number of extra copies of the final (converged)
            frame appended at the end, so the video doesn't end abruptly.
        target_marker_radius: Sphere radius for target keypoint markers.
        target_marker_rgba: Color for target keypoint markers.
        fitted_marker_radius: Sphere radius for fitted keypoint markers.
        fitted_marker_rgba: Color for fitted keypoint markers.

    Returns:
        The `IKResult` from the underlying fit.
    """
    target_positions = np.asarray(target_positions, dtype=float)
    if target_positions.shape != (len(keypoints.targets), 3):
        raise ValueError(
            f"target_positions must have shape ({len(keypoints.targets)}, 3), "
            f"got {target_positions.shape}."
        )

    qpos_history: list[np.ndarray] = []
    result = fit_qpos_to_keypoints(
        mj_model,
        mj_data,
        keypoints,
        target_positions,
        initial_qpos=initial_qpos,
        bounds=bounds,
        max_iters=max_iters,
        ftol=ftol,
        on_iterate=lambda qpos: qpos_history.append(qpos.copy()),
    )

    body_ids = keypoints.resolve_body_ids(mj_model)
    local_offsets = keypoints.local_offsets()
    cam_id = _resolve_camera_id(mj_model, camera)

    with Renderer(
        mj_model, camera, camera_res=camera_res, buffer_frames=False
    ) as renderer:
        frames = []
        for qpos in qpos_history:
            mj_data.qpos[:] = qpos
            # mj_forward (not just mj_kinematics) is needed here: the renderer
            # reads camera/light poses computed by mj_camlight, which runs as
            # part of mj_forward's pipeline but not bare mj_kinematics.
            mj.mj_forward(mj_model, mj_data)
            fitted_positions = _keypoint_world_points(mj_data, body_ids, local_offsets)

            renderer.mj_renderer.update_scene(mj_data, cam_id, renderer.scene_option)
            scene = renderer.mj_renderer.scene
            for pos in target_positions:
                _add_sphere_marker(scene, pos, target_marker_radius, target_marker_rgba)
            for pos in fitted_positions:
                _add_sphere_marker(scene, pos, fitted_marker_radius, fitted_marker_rgba)
            frames.append(renderer.mj_renderer.render())

        frames.extend([frames[-1]] * hold_final_frames)

    logger.info(f"Writing {len(frames)}-frame IK convergence video to {output_path}...")
    write_video_from_frames(output_path, frames, fps=fps, codec="libx264", quality=8)
    return result
