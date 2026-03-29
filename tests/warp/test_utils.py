"""Tests for flygym.warp.utils (GPU kernels and check_gpu)."""

import pytest
import numpy as np
import warp as wp


# ==============================================================================
# check_gpu
# ==============================================================================


class TestCheckGpu:
    def test_does_not_raise_on_machine_with_gpu(self):
        """check_gpu should succeed when an NVIDIA GPU is available."""
        from flygym.warp.utils import check_gpu

        check_gpu()  # raises ValueError if no GPU found


# ==============================================================================
# wp_gather_indexed_cols_2d
# ==============================================================================


class TestWpGatherIndexedCols2d:
    def test_gathers_correct_columns(self):
        """Selected columns of src should appear in dst in order."""
        from flygym.warp.utils import wp_gather_indexed_cols_2d

        n_rows, n_cols_wide, n_cols_narrow = 3, 6, 2
        src_np = np.arange(n_rows * n_cols_wide, dtype=np.float32).reshape(
            n_rows, n_cols_wide
        )
        col_indices = np.array([1, 4], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((n_rows, n_cols_narrow), dtype=wp.float32)
        cols = wp.array(col_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(n_rows, n_cols_narrow),
            inputs=[src, dst, cols],
        )

        result = dst.numpy()
        expected = src_np[:, col_indices]
        np.testing.assert_array_equal(result, expected)

    def test_single_column(self):
        from flygym.warp.utils import wp_gather_indexed_cols_2d

        src_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )
        col_indices = np.array([2], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((2, 1), dtype=wp.float32)
        cols = wp.array(col_indices, dtype=wp.int32)

        wp.launch(wp_gather_indexed_cols_2d, dim=(2, 1), inputs=[src, dst, cols])
        result = dst.numpy()
        np.testing.assert_array_equal(result, src_np[:, [2]])

    def test_all_columns(self):
        """Gathering all columns should reproduce the full source array."""
        from flygym.warp.utils import wp_gather_indexed_cols_2d

        n_rows, n_cols = 4, 5
        src_np = np.random.rand(n_rows, n_cols).astype(np.float32)
        col_indices = np.arange(n_cols, dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((n_rows, n_cols), dtype=wp.float32)
        cols = wp.array(col_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_cols_2d,
            dim=(n_rows, n_cols),
            inputs=[src, dst, cols],
        )
        np.testing.assert_array_almost_equal(dst.numpy(), src_np)


# ==============================================================================
# wp_scatter_indexed_cols_2d
# ==============================================================================


class TestWpScatterIndexedCols2d:
    def test_scatters_into_correct_columns(self):
        """Values from src should end up in the specified dst columns."""
        from flygym.warp.utils import wp_scatter_indexed_cols_2d

        n_rows, n_cols_wide, n_cols_narrow = 3, 6, 2
        src_np = np.ones((n_rows, n_cols_narrow), dtype=np.float32) * 7.0
        col_indices = np.array([0, 5], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((n_rows, n_cols_wide), dtype=wp.float32)
        cols = wp.array(col_indices, dtype=wp.int32)

        wp.launch(
            wp_scatter_indexed_cols_2d,
            dim=(n_rows, n_cols_narrow),
            inputs=[src, dst, cols],
        )

        result = dst.numpy()
        # Targeted columns should be 7.0; the rest should remain 0.0
        np.testing.assert_array_equal(result[:, 0], np.full(n_rows, 7.0))
        np.testing.assert_array_equal(result[:, 5], np.full(n_rows, 7.0))
        for c in [1, 2, 3, 4]:
            np.testing.assert_array_equal(result[:, c], np.zeros(n_rows))

    def test_scatter_preserves_other_values(self):
        """Scatter should not overwrite columns that are not in the index list."""
        from flygym.warp.utils import wp_scatter_indexed_cols_2d

        n_rows = 2
        src_np = np.array([[1.0], [2.0]], dtype=np.float32)
        dst_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )
        col_indices = np.array([1], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.array(dst_np.copy(), dtype=wp.float32)
        cols = wp.array(col_indices, dtype=wp.int32)

        wp.launch(
            wp_scatter_indexed_cols_2d,
            dim=(n_rows, 1),
            inputs=[src, dst, cols],
        )
        result = dst.numpy()

        # Column 1 gets new values; columns 0 and 2 unchanged
        np.testing.assert_array_equal(result[:, 0], [10.0, 40.0])
        np.testing.assert_array_equal(result[:, 1], [1.0, 2.0])
        np.testing.assert_array_equal(result[:, 2], [30.0, 60.0])


# ==============================================================================
# wp_gather_indexed_rows_3d  (float32 3-D arrays)
# ==============================================================================


class TestWpGatherIndexedRows3d:
    def test_gathers_correct_rows(self):
        """Selected rows (dim-1 indices) should be gathered into dst."""
        from flygym.warp.utils import wp_gather_indexed_rows_3d

        n_worlds, n_rows_wide, n_cols = 2, 5, 3
        src_np = np.arange(
            n_worlds * n_rows_wide * n_cols, dtype=np.float32
        ).reshape(n_worlds, n_rows_wide, n_cols)
        row_indices = np.array([0, 2, 4], dtype=np.int32)
        n_rows_narrow = len(row_indices)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((n_worlds, n_rows_narrow, n_cols), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_3d,
            dim=(n_worlds, n_rows_narrow, n_cols),
            inputs=[src, dst, rows],
        )

        result = dst.numpy()
        expected = src_np[:, row_indices, :]
        np.testing.assert_array_equal(result, expected)

    def test_single_world_single_row(self):
        from flygym.warp.utils import wp_gather_indexed_rows_3d

        src_np = np.array(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32
        )
        row_indices = np.array([2], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros((1, 1, 2), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_3d,
            dim=(1, 1, 2),
            inputs=[src, dst, rows],
        )
        result = dst.numpy()
        np.testing.assert_array_equal(result, [[[5.0, 6.0]]])

    def test_multi_world_all_rows(self):
        """Gathering all rows should reproduce the source array."""
        from flygym.warp.utils import wp_gather_indexed_rows_3d

        n_worlds, n_rows, n_cols = 3, 4, 2
        src_np = np.random.rand(n_worlds, n_rows, n_cols).astype(np.float32)
        row_indices = np.arange(n_rows, dtype=np.int32)

        src = wp.array(src_np, dtype=wp.float32)
        dst = wp.zeros_like(src)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_3d,
            dim=(n_worlds, n_rows, n_cols),
            inputs=[src, dst, rows],
        )
        np.testing.assert_array_almost_equal(dst.numpy(), src_np)


# ==============================================================================
# wp_gather_indexed_rows_vec3f  (2-D vec3f → float32 3-D)
# ==============================================================================


class TestWpGatherIndexedRowsVec3f:
    def test_gathers_correct_rows(self):
        """Selected rows of a vec3f array should appear in dst in order."""
        from flygym.warp.utils import wp_gather_indexed_rows_vec3f

        n_worlds, n_rows_wide = 2, 5
        # Build a (n_worlds, n_rows_wide, 3) numpy array and load as vec3f
        src_np = np.arange(n_worlds * n_rows_wide * 3, dtype=np.float32).reshape(
            n_worlds, n_rows_wide, 3
        )
        row_indices = np.array([1, 3], dtype=np.int32)
        n_rows_narrow = len(row_indices)

        src = wp.array(src_np, dtype=wp.vec3f)
        dst = wp.zeros((n_worlds, n_rows_narrow, 3), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_vec3f,
            dim=(n_worlds, n_rows_narrow),
            inputs=[src, dst, rows],
        )

        result = dst.numpy()
        expected = src_np[:, row_indices, :]
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_rows(self):
        """Gathering all rows should reproduce the full vec3f source."""
        from flygym.warp.utils import wp_gather_indexed_rows_vec3f

        n_worlds, n_rows = 3, 4
        src_np = np.random.rand(n_worlds, n_rows, 3).astype(np.float32)
        row_indices = np.arange(n_rows, dtype=np.int32)

        src = wp.array(src_np, dtype=wp.vec3f)
        dst = wp.zeros((n_worlds, n_rows, 3), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_vec3f,
            dim=(n_worlds, n_rows),
            inputs=[src, dst, rows],
        )
        np.testing.assert_array_almost_equal(dst.numpy(), src_np)

    def test_single_row_single_world(self):
        from flygym.warp.utils import wp_gather_indexed_rows_vec3f

        src_np = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
        row_indices = np.array([1], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.vec3f)
        dst = wp.zeros((1, 1, 3), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_vec3f,
            dim=(1, 1),
            inputs=[src, dst, rows],
        )
        np.testing.assert_array_almost_equal(dst.numpy(), [[[4.0, 5.0, 6.0]]])


# ==============================================================================
# wp_gather_indexed_rows_quatf  (2-D quatf → float32 3-D)
# ==============================================================================


class TestWpGatherIndexedRowsQuatf:
    def test_gathers_correct_rows(self):
        """Selected rows of a quatf array should appear in dst in order."""
        from flygym.warp.utils import wp_gather_indexed_rows_quatf

        n_worlds, n_rows_wide = 2, 4
        src_np = np.random.rand(n_worlds, n_rows_wide, 4).astype(np.float32)
        row_indices = np.array([0, 3], dtype=np.int32)
        n_rows_narrow = len(row_indices)

        src = wp.array(src_np, dtype=wp.quatf)
        dst = wp.zeros((n_worlds, n_rows_narrow, 4), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_quatf,
            dim=(n_worlds, n_rows_narrow),
            inputs=[src, dst, rows],
        )

        result = dst.numpy()
        expected = src_np[:, row_indices, :]
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_rows(self):
        """Gathering all rows should reproduce the full quatf source."""
        from flygym.warp.utils import wp_gather_indexed_rows_quatf

        n_worlds, n_rows = 2, 5
        src_np = np.random.rand(n_worlds, n_rows, 4).astype(np.float32)
        row_indices = np.arange(n_rows, dtype=np.int32)

        src = wp.array(src_np, dtype=wp.quatf)
        dst = wp.zeros((n_worlds, n_rows, 4), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_quatf,
            dim=(n_worlds, n_rows),
            inputs=[src, dst, rows],
        )
        np.testing.assert_array_almost_equal(dst.numpy(), src_np)

    def test_single_row_preserves_all_four_components(self):
        """All four quaternion components should be copied faithfully."""
        from flygym.warp.utils import wp_gather_indexed_rows_quatf

        src_np = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)
        row_indices = np.array([0], dtype=np.int32)

        src = wp.array(src_np, dtype=wp.quatf)
        dst = wp.zeros((1, 1, 4), dtype=wp.float32)
        rows = wp.array(row_indices, dtype=wp.int32)

        wp.launch(
            wp_gather_indexed_rows_quatf,
            dim=(1, 1),
            inputs=[src, dst, rows],
        )
        np.testing.assert_array_almost_equal(
            dst.numpy(), [[[0.1, 0.2, 0.3, 0.4]]], decimal=6
        )


# ==============================================================================
# get_rgb_selected_worlds_and_cameras — shape validation
# ==============================================================================


class TestGetRgbSelectedWorldsAndCamerasValidation:
    def test_worldids_size_mismatch_raises(self):
        """rgb_out world-dim must match len(worldids)."""
        import types
        from flygym.warp.utils import get_rgb_selected_worlds_and_cameras

        rc = types.SimpleNamespace(
            rgb_data=wp.zeros((4, 100), dtype=wp.uint32),
            rgb_adr=wp.zeros(2, dtype=int),
        )

        worldids = wp.array([0, 1, 2], dtype=int)   # size 3
        camids = wp.array([0, 1], dtype=int)         # size 2
        rgb_out = wp.zeros((2, 2, 8, 8), dtype=wp.vec3)  # world-dim=2, mismatch

        with pytest.raises(ValueError, match="worldids"):
            get_rgb_selected_worlds_and_cameras(rc, worldids, camids, rgb_out)

    def test_camids_size_mismatch_raises(self):
        """rgb_out camera-dim must match len(camids)."""
        import types
        from flygym.warp.utils import get_rgb_selected_worlds_and_cameras

        rc = types.SimpleNamespace(
            rgb_data=wp.zeros((4, 100), dtype=wp.uint32),
            rgb_adr=wp.zeros(3, dtype=int),
        )

        worldids = wp.array([0, 1], dtype=int)       # size 2
        camids = wp.array([0, 1, 2], dtype=int)      # size 3
        rgb_out = wp.zeros((2, 2, 8, 8), dtype=wp.vec3)  # cam-dim=2, mismatch

        with pytest.raises(ValueError, match="camids"):
            get_rgb_selected_worlds_and_cameras(rc, worldids, camids, rgb_out)
