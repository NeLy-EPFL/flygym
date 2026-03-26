import warp as wp

from mujoco_warp._src.types import RenderContext


@wp.kernel
def wp_gather_indexed_rows_3d(
    src: wp.array3d(dtype=wp.float32),  # type: ignore
    dst: wp.array3d(dtype=wp.float32),  # type: ignore
    rows: wp.array(dtype=wp.int32),  # type: ignore
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
def wp_scatter_indexed_cols_2d(
    src: wp.array2d(dtype=wp.float32),  # type: ignore
    dst: wp.array2d(dtype=wp.float32),  # type: ignore
    cols: wp.array(dtype=wp.int32),  # type: ignore
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
    src: wp.array2d(dtype=wp.float32),  # type: ignore
    dst: wp.array2d(dtype=wp.float32),  # type: ignore
    cols: wp.array(dtype=wp.int32),  # type: ignore
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
    packed: wp.array2d(dtype=wp.uint32),  # type: ignore
    rgb_adr: wp.array(dtype=int),  # type: ignore
    worldids_to_render: wp.array(dtype=int),  # type: ignore
    camids_to_render: wp.array(dtype=int),  # type: ignore
    # Out:
    rgb_out: wp.array4d(dtype=wp.vec3),  # type: ignore
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
    rc: RenderContext,
    worldids: wp.array(dtype=int),  # type: ignore
    camids: wp.array(dtype=int),  # type: ignore
    rgb_out: wp.array4d(dtype=wp.vec3),  # type: ignore
):
    """Get the RGB data output from the render context buffers for a given camera index.

    Args:
        rc:
            The render context on device.
        worldids: TODO
        camids: TODO
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
