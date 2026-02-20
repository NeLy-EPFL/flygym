import numpy as np
import warp as wp


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
