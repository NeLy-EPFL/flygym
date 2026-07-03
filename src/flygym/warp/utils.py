from typing import Optional

import warp as wp
import mujoco as mj
import mujoco_warp as mjw


@wp.kernel
def wp_gather_indexed_rows_3d(
    src: wp.array3d[float], dst: wp.array3d[float], rows: wp.array[int]
):
    """Gather specific rows (dim 1) from a 3D Warp array into a narrower destination.

    This kernel is to be launched with a 3D launch configuration of
    `(n_worlds, n_rows_narrow, n_cols)`.

    Args:
        src (wp.array of shape (n_worlds, n_rows_wide, n_cols), type float32):
            Source array.
        dst (wp.array of shape (n_worlds, n_rows_narrow, n_cols), type float32):
            Destination array, where n_rows_narrow <= n_rows_wide.
        rows (wp.array of shape (n_rows_narrow,), type int32):
            Array of row indices (dim 1) of `src` to gather into `dst`.
    """
    i, k, j = wp.tid()
    dst[i, k, j] = src[i, rows[k], j]


@wp.kernel
def wp_gather_indexed_rows_vec3f(
    src: wp.array2d[wp.vec3], dst: wp.array3d[float], rows: wp.array[int]
):
    """Gather specific rows from a 2D ``vec3f`` array into a ``(n_worlds, n_rows_narrow, 3)``
    ``float32`` destination.

    This kernel is to be launched with a 2D launch configuration of
    `(n_worlds, n_rows_narrow)`.

    Args:
        src (wp.array of shape (n_worlds, n_rows_wide), type vec3f):
            Source array, e.g. ``mjw_data.xpos``.
        dst (wp.array of shape (n_worlds, n_rows_narrow, 3), type float32):
            Destination array.
        rows (wp.array of shape (n_rows_narrow,), type int32):
            Body indices to gather.
    """
    i, k = wp.tid()
    v = src[i, rows[k]]
    dst[i, k, 0] = v[0]
    dst[i, k, 1] = v[1]
    dst[i, k, 2] = v[2]


@wp.kernel
def wp_gather_indexed_rows_quatf(
    src: wp.array2d[wp.quat], dst: wp.array3d[float], rows: wp.array[int]
):
    """Gather specific rows from a 2D ``quatf`` array into a ``(n_worlds, n_rows_narrow, 4)``
    ``float32`` destination.

    This kernel is to be launched with a 2D launch configuration of
    `(n_worlds, n_rows_narrow)`.

    Args:
        src (wp.array of shape (n_worlds, n_rows_wide), type quatf):
            Source array, e.g. ``mjw_data.xquat``.
        dst (wp.array of shape (n_worlds, n_rows_narrow, 4), type float32):
            Destination array.
        rows (wp.array of shape (n_rows_narrow,), type int32):
            Body indices to gather.
    """
    i, k = wp.tid()
    q = src[i, rows[k]]
    dst[i, k, 0] = q[0]
    dst[i, k, 1] = q[1]
    dst[i, k, 2] = q[2]
    dst[i, k, 3] = q[3]


@wp.kernel
def wp_scatter_indexed_cols_2d(
    src: wp.array2d[float], dst: wp.array2d[float], cols: wp.array[int]
):
    """Scatter a 2D Warp array into specific columns of a wider destination array.

    This kernel is to be launched with a 2D launch configuration of
    `(n_rows, n_cols_narrow)`, where `n_cols_narrow` is the number of columns to copy.

    Args:
        src (wp.array of shape (n_rows, n_cols_narrow), type float32):
            Source array.
        dst (wp.array of shape (n_rows, n_cols_wide), type float32):
            Destination array, where n_cols_wide >= n_cols_narrow.
        cols (wp.array of shape (n_cols_narrow,), type int32):
            Array of column indices of `dst` that `src` will be scattered into.
    """
    i, k = wp.tid()
    dst[i, cols[k]] = src[i, k]


@wp.kernel
def wp_gather_indexed_cols_2d(
    src: wp.array2d[float], dst: wp.array2d[float], cols: wp.array[int]
):
    """Gather specific columns from a 2D Warp array into a narrower destination array.

    This kernel is to be launched with a 2D launch configuration of
    `(n_rows, n_cols_narrow)`, where `n_cols_narrow` is the number of columns to gather.

    Args:
        src (wp.array of shape (n_rows, n_cols_wide), type float32):
            Source array.
        dst (wp.array of shape (n_rows, n_cols_narrow), type float32):
            Destination array, where n_cols_narrow <= n_cols_wide.
        cols (wp.array of shape (n_cols_narrow,), type int32):
            Array of column indices of `src` that will be gathered into `dst`.
    """
    i, k = wp.tid()
    dst[i, k] = src[i, cols[k]]


@wp.kernel
def unpack_rgb_kernel_selected_worlds_and_cameras(
    # In:
    packed: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    worldids_to_render: wp.array[int],
    camids_to_render: wp.array[int],
    # Out:
    rgb_out: wp.array4d[wp.vec3],
):
    """Unpack ABGR uint32 packed pixel data into separate R, G, and B channels."""
    idx_within_worldids, idx_within_camids, pixelid = wp.tid()

    width = rgb_out.shape[3]
    row_idx = pixelid // width
    col_idx = pixelid % width

    rgb_adr_offset = rgb_adr[camids_to_render[idx_within_camids]]
    val = packed[worldids_to_render[idx_within_worldids], rgb_adr_offset + pixelid]
    b = wp.float32(val & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
    g = wp.float32((val >> wp.uint32(8)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
    r = wp.float32((val >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.static(1.0 / 255.0)
    rgb_out[idx_within_worldids, idx_within_camids, row_idx, col_idx] = wp.vec3(r, g, b)


def get_rgb_selected_worlds_and_cameras(
    rc: mjw.RenderContext,
    worldids: wp.array[int],
    camids: wp.array[int],
    rgb_out: wp.array4d[wp.vec3],
):
    """Get the RGB data output from the render context buffers for the selected worlds
    and cameras.

    Args:
        rc:
            The render context on device.
        worldids:
            Indices of the worlds to read RGB data for.
        camids:
            Indices of the cameras to read RGB data for.
        rgb_out:
            The output array to store the RGB data in, with shape
            (len(worldids), len(camids), height, width).
    """
    nworlds_to_render, ncams_to_render, height, width = rgb_out.shape
    if nworlds_to_render != worldids.size:
        raise ValueError(
            f"worldids has {worldids.size} elements, but the rgb_out buffer has "
            f"{nworlds_to_render} elements along the world dimension (dim 0)."
        )
    if ncams_to_render != camids.size:
        raise ValueError(
            f"camids has {camids.size} elements, but the rgb_out buffer has "
            f"{ncams_to_render} elements along the camera dimension (dim 1)."
        )

    wp.launch(
        unpack_rgb_kernel_selected_worlds_and_cameras,
        dim=(nworlds_to_render, ncams_to_render, height * width),
        inputs=[rc.rgb_data, rc.rgb_adr, worldids, camids],
        outputs=[rgb_out],
    )


def check_gpu():
    devices = wp.get_devices()
    gpu_devices = [d for d in devices if d.is_cuda]
    if len(gpu_devices) == 0:
        raise ValueError("NVIDIA GPU required for the flygym.warp module; none found.")
    if len(gpu_devices) > 1:
        print(
            "Multiple NVIDIA GPUs detected; we will only use one. "
            "You can specify which GPU to use by setting the 'CUDA_VISIBLE_DEVICES' "
            "environment variable."
        )


def reset_data_keyframe(
    mj_model: mj.MjModel,
    mjw_model: mjw.Model,
    mjw_data: mjw.Data,
    key: int,
    reset: Optional[wp.array] = None,
):
    """In-place equivalent of ``mj_resetDataKeyframe`` for a batched MJWarp ``Data``.

    This functionality is not provided natively by MuJoCo Warp, so this is a custom
    implementation. Note: this function differs from other Warp utils in that it
    requires both the `mujoco.MjModel` object and the `mujoco_warp.types.Model` object.
    This is because `key_{qpos,qvel,act,mpos,mquat,ctrl}` are not tracked by the
    GPU-side model.

    Args:
        mj_model: CPU-side MuJoCo model holding the keyframe to reset to.
        mjw_model: GPU-side MuJoCo-Warp model instance.
        mjw_data: GPU-side MuJoCo-Warp data instance.
        key: Index of the keyframe (in `mj_model`) to reset to.
        reset: Optional per-world boolean mask, shape `(mjw_data.nworld,)`.
    """
    mjw.reset_data(mjw_model, mjw_data, reset)

    @wp.kernel(module="unique", enable_backward=False)
    def reset_time(
        # From MjModel:
        target_time: float,
        # In:
        reset_in: wp.array[bool],
        # Data out:
        time_out: wp.array[float],
    ):
        worldid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        time_out[worldid] = target_time

    @wp.kernel(module="unique", enable_backward=False)
    def reset_qpos(
        # From MjModel:
        target_qpos: wp.array[float],
        # In:
        reset_in: wp.array[bool],
        # Data out:
        qpos_out: wp.array2d[float],
    ):
        worldid, qid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        qpos_out[worldid, qid] = target_qpos[qid]

    @wp.kernel(module="unique", enable_backward=False)
    def reset_qvel(
        # From MjModel:
        target_qvel: wp.array[float],
        # In:
        reset_in: wp.array[bool],
        # Data out:
        qvel_out: wp.array2d[float],
    ):
        worldid, vid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        qvel_out[worldid, vid] = target_qvel[vid]

    @wp.kernel(module="unique", enable_backward=False)
    def reset_activation(
        # From MjModel:
        target_act: wp.array[float],
        # In:
        reset_in: wp.array[bool],
        # Data out:
        act_out: wp.array2d[float],
    ):
        worldid, aid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        act_out[worldid, aid] = target_act[aid]

    @wp.kernel(module="unique", enable_backward=False)
    def reset_mocap(
        # From MjModel:
        target_mpos: wp.array[wp.vec3],
        target_mquat: wp.array[wp.quat],
        # From mjwarp Model:
        body_mocapid: wp.array[int],
        # In:
        reset_in: wp.array[bool],
        # Data out:
        mocap_pos_out: wp.array2d[wp.vec3],
        mocap_quat_out: wp.array2d[wp.quat],
    ):
        worldid, bodyid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        mocapid = body_mocapid[bodyid]

        if mocapid >= 0:
            mocap_pos_out[worldid, mocapid] = target_mpos[mocapid]
            mocap_quat_out[worldid, mocapid] = target_mquat[mocapid]

    @wp.kernel(module="unique", enable_backward=False)
    def reset_control(
        # From MjModel:
        target_ctrl: wp.array[float],
        # In:
        reset_in: wp.array[bool],
        # Data out:
        ctrl_out: wp.array2d[float],
    ):
        worldid, cid = wp.tid()

        if wp.static(reset is not None):
            if not reset_in[worldid]:
                return

        ctrl_out[worldid, cid] = target_ctrl[cid]

    reset_input = reset or wp.ones(mjw_data.nworld, dtype=bool)

    target_time = mj_model.key_time[key]
    wp.launch(
        reset_time,
        dim=mjw_data.nworld,
        inputs=[target_time, reset_input],
        outputs=[mjw_data.time],
    )

    target_qpos = wp.array(mj_model.key_qpos[key], dtype=float)
    wp.launch(
        reset_qpos,
        dim=(mjw_data.nworld, mjw_model.nq),
        inputs=[target_qpos, reset_input],
        outputs=[mjw_data.qpos],
    )

    target_qvel = wp.array(mj_model.key_qvel[key], dtype=float)
    wp.launch(
        reset_qvel,
        dim=(mjw_data.nworld, mjw_model.nv),
        inputs=[target_qvel, reset_input],
        outputs=[mjw_data.qvel],
    )

    target_act = wp.array(mj_model.key_act[key], dtype=float)
    wp.launch(
        reset_activation,
        dim=(mjw_data.nworld, mjw_model.na),
        inputs=[target_act, reset_input],
        outputs=[mjw_data.act],
    )

    target_mpos = wp.array(mj_model.key_mpos[key], dtype=wp.vec3)
    target_mquat = wp.array(mj_model.key_mquat[key], dtype=wp.quat)
    wp.launch(
        reset_mocap,
        dim=(mjw_data.nworld, mjw_model.nbody),
        inputs=[
            target_mpos,
            target_mquat,
            mjw_model.body_mocapid,
            reset_input,
        ],
        outputs=[mjw_data.mocap_pos, mjw_data.mocap_quat],
    )

    target_ctrl = wp.array(mj_model.key_ctrl[key], dtype=float)
    wp.launch(
        reset_control,
        dim=(mjw_data.nworld, mjw_model.nu),
        inputs=[target_ctrl, reset_input],
        outputs=[mjw_data.ctrl],
    )
