from typing import Any

import warp as wp
import mujoco_warp as mjw


@wp.kernel
def wp_gather_worlds_2d(
    src: wp.array2d[Any], dst: wp.array2d[Any], world_ids: wp.array[int]
):
    """Gather specific worlds (dim 0) from a 2D Warp array of any dtype.

    This kernel is to be launched with a 2D launch configuration of `dst.shape`.

    Args:
        src (wp.array of shape (n_worlds, n_cols)):
            Source array, e.g. `mjw_data.qpos`.
        dst (wp.array of shape (n_selected_worlds, n_cols), same dtype as `src`):
            Destination array.
        world_ids (wp.array of shape (n_selected_worlds,), type int32):
            Indices of the worlds to gather.
    """
    i, j = wp.tid()
    dst[i, j] = src[world_ids[i], j]


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
def wp_gather_offset_body_positions(
    xpos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    dst: wp.array3d[float],
    bodies: wp.array[int],
    relpos: wp.array[wp.vec3],
):
    """Gather the global positions of frames rigidly attached to given bodies into a
    ``(n_worlds, n_frames, 3)`` ``float32`` destination.

    This is used to get the positions of body segments that `fusestatic` has fused into
    an ancestor body. This kernel is to be launched with a 2D launch configuration of
    `(n_worlds, n_frames)`.

    Args:
        xpos (wp.array of shape (n_worlds, n_bodies), type vec3f):
            Global body positions, i.e. ``mjw_data.xpos``.
        xmat (wp.array of shape (n_worlds, n_bodies), type mat33f):
            Global body orientations, i.e. ``mjw_data.xmat``.
        dst (wp.array of shape (n_worlds, n_frames, 3), type float32):
            Destination array.
        bodies (wp.array of shape (n_frames,), type int32):
            Index of the body that each frame is attached to.
        relpos (wp.array of shape (n_frames,), type vec3f):
            Position of each frame in the frame of its body.
    """
    i, k = wp.tid()
    b = bodies[k]
    p = xpos[i, b] + xmat[i, b] @ relpos[k]
    dst[i, k, 0] = p[0]
    dst[i, k, 1] = p[1]
    dst[i, k, 2] = p[2]


@wp.kernel
def wp_gather_offset_body_rotations(
    xquat: wp.array2d[wp.quat],
    dst: wp.array3d[float],
    bodies: wp.array[int],
    relquat: wp.array[wp.quat],
):
    """Gather the global orientations of frames rigidly attached to given bodies into a
    ``(n_worlds, n_frames, 4)`` ``float32`` destination.

    Counterpart of `wp_gather_offset_body_positions` for orientations. All quaternions
    are stored in MuJoCo's (w, x, y, z) order. This kernel is to be launched with a 2D
    launch configuration of `(n_worlds, n_frames)`.

    Args:
        xquat (wp.array of shape (n_worlds, n_bodies), type quatf):
            Global body orientations, i.e. ``mjw_data.xquat``.
        dst (wp.array of shape (n_worlds, n_frames, 4), type float32):
            Destination array.
        bodies (wp.array of shape (n_frames,), type int32):
            Index of the body that each frame is attached to.
        relquat (wp.array of shape (n_frames,), type quatf):
            Orientation of each frame relative to its body.
    """
    i, k = wp.tid()
    a = xquat[i, bodies[k]]
    b = relquat[k]
    # Hamilton product in (w, x, y, z) order (not Warp's native (x, y, z, w) order)
    dst[i, k, 0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    dst[i, k, 1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    dst[i, k, 2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    dst[i, k, 3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]


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
def wp_masked_copy_rows_2d(
    src: wp.array2d[float], dst: wp.array2d[float], world_mask: wp.array[bool]
):
    """Copy whole rows (dim 0, indexed by world) where a per-world mask is set.

    This kernel is to be launched with a 2D launch configuration of
    `(n_worlds, n_cols)`. Unlike a keyframe reset, `src` may hold a different row per
    world, so this is the building block for per-world reference-state initialization.

    Args:
        src (wp.array of shape (n_worlds, n_cols), type float32):
            Source array; rows for un-masked worlds are ignored.
        dst (wp.array of shape (n_worlds, n_cols), type float32):
            Destination array, written in place for masked worlds only.
        world_mask (wp.array of shape (n_worlds,), type bool):
            Per-world mask; a row is copied only where its entry is True.
    """
    i, k = wp.tid()
    if world_mask[i]:
        dst[i, k] = src[i, k]


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
