"""Tests for flygym.warp.utils (GPU kernels and check_gpu)."""

import pytest
import numpy as np

# These tests require the optional warp (GPU) extra; tag them so they can be
# excluded with ``-m "not warp"``, and skip the whole module if warp is absent.
pytestmark = pytest.mark.warp
wp = pytest.importorskip("warp")
mjw = pytest.importorskip("mujoco_warp")
mj = pytest.importorskip("mujoco")


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

        src_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
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
        dst_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
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
        src_np = np.arange(n_worlds * n_rows_wide * n_cols, dtype=np.float32).reshape(
            n_worlds, n_rows_wide, n_cols
        )
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

        src_np = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
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

        worldids = wp.array([0, 1, 2], dtype=int)  # size 3
        camids = wp.array([0, 1], dtype=int)  # size 2
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

        worldids = wp.array([0, 1], dtype=int)  # size 2
        camids = wp.array([0, 1, 2], dtype=int)  # size 3
        rgb_out = wp.zeros((2, 2, 8, 8), dtype=wp.vec3)  # cam-dim=2, mismatch

        with pytest.raises(ValueError, match="camids"):
            get_rgb_selected_worlds_and_cameras(rc, worldids, camids, rgb_out)


# ==============================================================================
# reset_data_keyframe
# ==============================================================================

# A free joint (nq=7, nv=6) plus a hinge joint (nq=1, nv=1) gives a model
# where nq != nv, which is the case that exposed the qpos/qvel kernel bug.
# One actuator (with activation state) exercises act/ctrl, and one mocap
# body exercises mocap_pos/mocap_quat.
_RESET_KEYFRAME_XML = """
<mujoco>
  <worldbody>
    <body name="free_body" pos="0 0 1">
      <freejoint name="fj"/>
      <geom type="sphere" size="0.1"/>
      <body name="child" pos="0.2 0 0">
        <joint name="hinge1" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05"/>
      </body>
    </body>
    <body name="mocap_body" mocap="true" pos="1 2 3" quat="1 0 0 0">
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="hinge1" dyntype="filter" dynprm="0.5" gaintype="fixed" gainprm="1"/>
  </actuator>
  <keyframe>
    <key name="k0" time="0.5"
         qpos="0.1 0.2 0.3 1 0 0 0 0.5"
         qvel="0.01 0.02 0.03 0.04 0.05 0.06 0.7"
         act="0.33"
         ctrl="0.44"
         mpos="9 8 7" mquat="0 1 0 0"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture
def mj_and_mjw():
    mj_model = mj.MjModel.from_xml_string(_RESET_KEYFRAME_XML)
    mj_data = mj.MjData(mj_model)
    # sanity-check the model actually has nq != nv, which is what this test
    # suite cares about exercising.
    assert mj_model.nq != mj_model.nv
    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.put_data(mj_model, mj_data, nworld=3)
    return mj_model, mjw_model, mjw_data


class TestResetDataKeyframe:
    def test_resets_all_worlds_to_keyframe_values(self, mj_and_mjw):
        from flygym.warp.utils import reset_data_keyframe

        mj_model, mjw_model, mjw_data = mj_and_mjw

        # Perturb data away from both the keyframe and the model defaults so
        # a no-op kernel launch would be caught by the assertions below.
        mjw_data.qpos.fill_(-1.0)
        mjw_data.qvel.fill_(-1.0)
        mjw_data.time.fill_(-1.0)

        reset_data_keyframe(mj_model, mjw_model, mjw_data, key=0)

        n = mjw_data.nworld
        np.testing.assert_allclose(
            mjw_data.qpos.numpy(), np.tile(mj_model.key_qpos[0], (n, 1))
        )
        np.testing.assert_allclose(
            mjw_data.qvel.numpy(), np.tile(mj_model.key_qvel[0], (n, 1))
        )
        np.testing.assert_allclose(
            mjw_data.act.numpy(), np.tile(mj_model.key_act[0], (n, 1))
        )
        np.testing.assert_allclose(
            mjw_data.ctrl.numpy(), np.tile(mj_model.key_ctrl[0], (n, 1))
        )
        np.testing.assert_allclose(
            mjw_data.mocap_pos.numpy()[:, 0, :],
            np.tile(mj_model.key_mpos[0], (n, 1)),
        )
        np.testing.assert_allclose(
            mjw_data.mocap_quat.numpy()[:, 0, :],
            np.tile(mj_model.key_mquat[0], (n, 1)),
        )
        np.testing.assert_allclose(mjw_data.time.numpy(), [0.5] * n)

    def test_qvel_write_does_not_overrun_into_neighboring_world(self, mj_and_mjw):
        """Regression test: qpos has nq=8 columns but qvel only has nv=7.

        A kernel that loops over nq columns and writes both qpos_out and
        qvel_out at the same column index writes one element past the end
        of each world's qvel row. Because qvel is stored as a flat
        (nworld, nv) buffer, that out-of-bounds column aliases column 0 of
        the next world's row. qpos and qvel are reset by separate kernels,
        each launched with its own dimension (nq vs. nv).
        """
        from flygym.warp.utils import reset_data_keyframe

        mj_model, mjw_model, mjw_data = mj_and_mjw

        mjw_data.qvel.fill_(-1.0)
        reset_input = wp.array([True, False, True], dtype=bool)
        reset_data_keyframe(mj_model, mjw_model, mjw_data, key=0, reset=reset_input)

        qvel = mjw_data.qvel.numpy()
        np.testing.assert_allclose(qvel[0], mj_model.key_qvel[0])
        np.testing.assert_allclose(qvel[2], mj_model.key_qvel[0])
        # World 1 was excluded from the reset and must be untouched.
        np.testing.assert_allclose(qvel[1], np.full(mj_model.nv, -1.0))

    def test_partial_reset_leaves_unselected_worlds_untouched(self, mj_and_mjw):
        from flygym.warp.utils import reset_data_keyframe

        mj_model, mjw_model, mjw_data = mj_and_mjw

        mjw_data.qpos.fill_(0.0)
        mjw_data.time.fill_(99.0)
        reset_input = wp.array([True, False, True], dtype=bool)

        reset_data_keyframe(mj_model, mjw_model, mjw_data, key=0, reset=reset_input)

        qpos = mjw_data.qpos.numpy()
        time = mjw_data.time.numpy()
        np.testing.assert_allclose(qpos[0], mj_model.key_qpos[0])
        np.testing.assert_allclose(qpos[2], mj_model.key_qpos[0])
        np.testing.assert_allclose(qpos[1], np.zeros(mj_model.nq))
        assert time[1] == pytest.approx(99.0)
        assert time[0] == pytest.approx(0.5)
        assert time[2] == pytest.approx(0.5)
