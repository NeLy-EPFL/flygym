"""Tests for flygym vision: Retina, Fly.add_vision, and Simulation vision APIs."""

import platform

import pytest
import numpy as np
import yaml

from flygym import assets_dir
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    JointPreset,
    Skeleton,
)
from flygym.compose.fly import Fly, ActuatorType
from flygym.compose.pose import KinematicPosePreset
from flygym.compose.world import TetheredWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym.vision.retina import Retina


# ==============================================================================
# Vision config (used to check derived values)
# ==============================================================================


@pytest.fixture(scope="module")
def vision_config():
    with open(assets_dir / "model/vision.yaml") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Retina (pure: no MuJoCo / rendering required)
# ==============================================================================


class TestRetinaConstruction:
    def test_default_construction(self):
        r = Retina()
        assert r is not None

    def test_default_dimensions_match_config(self, vision_config):
        r = Retina()
        assert r.nrows == vision_config["raw_img_height_px"]
        assert r.ncols == vision_config["raw_img_width_px"]

    def test_default_num_ommatidia_matches_config(self, vision_config):
        r = Retina()
        assert r.num_ommatidia_per_eye == vision_config["num_ommatidia_per_eye"]

    def test_default_distortion_and_zoom_match_config(self, vision_config):
        r = Retina()
        assert r.distortion_coefficient == vision_config["fisheye_distortion_coefficient"]
        assert r.zoom == vision_config["fisheye_zoom"]

    def test_ommatidia_id_map_shape_and_dtype(self):
        r = Retina()
        assert r.ommatidia_id_map.shape == (r.nrows, r.ncols)
        assert r.ommatidia_id_map.dtype == np.int16

    def test_pale_type_mask_shape(self):
        r = Retina()
        assert r.pale_type_mask.shape == (r.num_ommatidia_per_eye,)

    def test_pale_type_mask_values_are_binary(self):
        r = Retina()
        assert set(np.unique(r.pale_type_mask)).issubset({0, 1})

    def test_num_pixels_per_ommatidia_sums_to_covered_pixels(self):
        r = Retina()
        covered = int((r.ommatidia_id_map > 0).sum())
        assert int(r.num_pixels_per_ommatidia.sum()) == covered

    def test_custom_dimensions_override_config(self):
        r = Retina(nrows=64, ncols=80, distortion_coefficient=1.5, zoom=2.0)
        assert r.nrows == 64
        assert r.ncols == 80
        assert r.distortion_coefficient == 1.5
        assert r.zoom == 2.0


class TestRetinaRawImageToHexPxls:
    def test_output_shape(self):
        r = Retina()
        img = np.zeros((r.nrows, r.ncols, 3), dtype=np.uint8)
        out = r.raw_image_to_hex_pxls(img)
        assert out.shape == (r.num_ommatidia_per_eye, 2)

    def test_zero_image_returns_zeros(self):
        r = Retina()
        img = np.zeros((r.nrows, r.ncols, 3), dtype=np.uint8)
        out = r.raw_image_to_hex_pxls(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_uniform_image_normalized_by_255(self):
        """Each ommatidium reads from exactly one channel (G for pale, B for yellow).
        For a uniform image the other channel slot stays 0, so half the entries are 0."""
        r = Retina()
        val = 200
        img = np.full((r.nrows, r.ncols, 3), val, dtype=np.uint8)
        out = r.raw_image_to_hex_pxls(img)
        # Non-zero entries should equal val / 255
        nonzero = out[out > 0]
        np.testing.assert_allclose(nonzero, val / 255.0, atol=1e-6)
        # Every ommatidium has exactly one populated channel
        nonzero_per_om = (out > 0).sum(axis=1)
        assert set(np.unique(nonzero_per_om)).issubset({0, 1})

    def test_pale_ommatidia_use_green_yellow_use_blue(self):
        """Pale-type ommatidia store readings in channel 0 (G); yellow-type in
        channel 1 (B). Use a single-channel image to verify the split."""
        r = Retina()
        # Green-only raw image: only pale-type (mask==1) ommatidia should activate
        green_img = np.zeros((r.nrows, r.ncols, 3), dtype=np.uint8)
        green_img[..., 1] = 255  # G channel
        out_green = r.raw_image_to_hex_pxls(green_img)
        pale_idx = r.pale_type_mask == 1
        yellow_idx = r.pale_type_mask == 0
        # Pale ommatidia (ch_idx=1) read from img[:, 2] = B -> 0 here, so all zero
        # Yellow ommatidia (ch_idx=0) read from img[:, 1] = G -> 255 here, so non-zero
        assert np.all(out_green[pale_idx, 0] == 0)
        assert np.all(out_green[pale_idx, 1] == 0)
        assert np.all(out_green[yellow_idx, 0] > 0)
        assert np.all(out_green[yellow_idx, 1] == 0)


class TestRetinaHexPxlsToHumanReadable:
    def test_output_shape_1d_input(self):
        r = Retina()
        reading = np.zeros(r.num_ommatidia_per_eye, dtype=np.float32)
        out = r.hex_pxls_to_human_readable(reading)
        assert out.shape == (r.nrows, r.ncols)

    def test_output_shape_2d_input(self):
        r = Retina()
        reading = np.zeros((r.num_ommatidia_per_eye, 2), dtype=np.float32)
        out = r.hex_pxls_to_human_readable(reading)
        assert out.shape == (r.nrows, r.ncols, 2)

    def test_default_value_outside_lattice(self):
        r = Retina()
        reading = np.ones(r.num_ommatidia_per_eye, dtype=np.float32)
        out = r.hex_pxls_to_human_readable(reading, default_value=0)
        # Outside-lattice pixels (id == 0 in id_map) should be 0
        outside = r.ommatidia_id_map == 0
        np.testing.assert_array_equal(out[outside], 0)
        # Inside-lattice pixels should be 1 (from the reading)
        inside = r.ommatidia_id_map > 0
        np.testing.assert_array_equal(out[inside], 1)

    def test_color_8bit_returns_uint8_and_scales(self):
        r = Retina()
        reading = np.full((r.num_ommatidia_per_eye, 2), 0.5, dtype=np.float32)
        out = r.hex_pxls_to_human_readable(reading, color_8bit=True)
        assert out.dtype == np.uint8
        # 0.5 * 255 = 127.5, int cast = 127
        inside = r.ommatidia_id_map > 0
        assert out[inside].max() == 127

    def test_dtype_preserved_when_not_8bit(self):
        r = Retina()
        reading = np.zeros(r.num_ommatidia_per_eye, dtype=np.float32)
        out = r.hex_pxls_to_human_readable(reading)
        assert out.dtype == np.float32

    def test_wrong_first_dim_raises(self):
        r = Retina()
        bad = np.zeros(r.num_ommatidia_per_eye + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="0th dimension"):
            r.hex_pxls_to_human_readable(bad)


class TestRetinaCorrectFisheye:
    def test_output_shape_and_dtype(self):
        r = Retina()
        img = np.zeros((r.nrows, r.ncols, 3), dtype=np.uint8)
        out = r.correct_fisheye(img)
        assert out.shape == (r.nrows, r.ncols, 3)
        assert out.dtype == np.uint8

    def test_zero_image_returns_zero_image(self):
        r = Retina()
        img = np.zeros((r.nrows, r.ncols, 3), dtype=np.uint8)
        out = r.correct_fisheye(img)
        np.testing.assert_array_equal(out, 0)

    def test_uniform_image_centre_preserved(self):
        """The centre of the image is the optical axis — the fisheye remap should
        sample from somewhere within the image, so a uniform fill stays uniform
        on the parts of the destination that fall back inside the source bounds."""
        r = Retina()
        img = np.full((r.nrows, r.ncols, 3), 123, dtype=np.uint8)
        out = r.correct_fisheye(img)
        # Centre pixel must remain 123 (fisheye distortion is 0 at the centre)
        cy, cx = r.nrows // 2, r.ncols // 2
        np.testing.assert_array_equal(out[cy, cx], [123, 123, 123])


# ==============================================================================
# Fly.add_vision (pure MJCF assembly; no rendering)
# ==============================================================================


class TestFlyAddVision:
    def test_eye_cameras_registered(self):
        fly = Fly(name="vision_fly_basic")
        fly.add_vision()
        assert set(fly.eyecameraname_to_mjcfcamera.keys()) == {"l_eye_cam", "r_eye_cam"}

    def test_no_markers_by_default(self, vision_config):
        """With draw_sensor_markers=False, only the hidden body segments are added
        to hidden_geoms — no marker spheres."""
        fly = Fly(name="vision_fly_nomarkers")
        fly.add_vision(draw_sensor_markers=False)
        assert len(fly.hidden_geoms) == len(vision_config["hidden_segments"])

    def test_markers_added_when_requested(self, vision_config):
        fly = Fly(name="vision_fly_markers")
        fly.add_vision(draw_sensor_markers=True)
        # One marker per eye camera + all hidden body-segment geoms
        expected = len(vision_config["sensors"]) + len(vision_config["hidden_segments"])
        assert len(fly.hidden_geoms) == expected

    def test_hidden_segments_present_in_hidden_geoms(self, vision_config):
        fly = Fly(name="vision_fly_segs")
        fly.add_vision(draw_sensor_markers=False)
        hidden_names = {g.name for g in fly.hidden_geoms}
        assert set(vision_config["hidden_segments"]).issubset(hidden_names)

    def test_compiles_after_add_vision(self):
        fly = Fly(name="vision_fly_compile")
        fly.add_vision(draw_sensor_markers=True)
        mj_model, _ = fly.compile()
        assert mj_model is not None
        # The two eye cameras should be in the compiled model
        assert mj_model.ncam >= 2

    def test_cameras_compile_in_world(self, neutral_pose, skeleton_ypr):
        fly = Fly(name="vision_fly_world")
        fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
        fly.add_vision()
        world = TetheredWorld(name="vision_world_compile")
        world.add_fly(
            fly,
            spawn_position=[0, 0, 1.5],
            spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        )
        mj_model, _ = world.compile()
        assert mj_model.ncam == 2


# ==============================================================================
# Simulation vision: ID-mapping (no rendering required)
# ==============================================================================


@pytest.fixture(scope="module")
def fly_with_vision(neutral_pose, skeleton_ypr):
    """Standard fly + joints + actuators + vision (with markers)."""
    fly = Fly(name="vision_sim_fly")
    fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
    actuated_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs, ActuatorType.POSITION, neutral_input=neutral_pose, kp=50
    )
    fly.add_vision(draw_sensor_markers=True)
    return fly


@pytest.fixture(scope="module")
def simulation_with_vision(fly_with_vision):
    world = TetheredWorld(name="vision_sim_world")
    world.add_fly(
        fly_with_vision,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    sim = Simulation(world)
    sim.reset()
    return sim


@pytest.fixture(scope="module")
def fly_without_vision(neutral_pose, skeleton_ypr):
    """Fly with joints/actuators but no add_vision call."""
    fly = Fly(name="novision_sim_fly")
    fly.add_joints(skeleton_ypr, neutral_pose=neutral_pose)
    actuated_dofs = skeleton_ypr.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs, ActuatorType.POSITION, neutral_input=neutral_pose, kp=50
    )
    return fly


@pytest.fixture(scope="module")
def simulation_without_vision(fly_without_vision):
    world = TetheredWorld(name="novision_sim_world")
    world.add_fly(
        fly_without_vision,
        spawn_position=[0, 0, 1.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
    )
    sim = Simulation(world)
    sim.reset()
    return sim


class TestSimulationVisionIDMapping:
    def test_eye_camera_ids_present(self, simulation_with_vision, fly_with_vision):
        ids = simulation_with_vision._intern_eye_camera_ids_by_fly[fly_with_vision.name]
        assert ids.shape == (2,)
        assert ids.dtype == np.int32
        assert (ids >= 0).all()

    def test_hidden_geom_ids_present(self, simulation_with_vision, fly_with_vision, vision_config):
        ids = simulation_with_vision._intern_hidden_geom_ids_by_fly[fly_with_vision.name]
        expected = len(vision_config["sensors"]) + len(vision_config["hidden_segments"])
        assert ids.shape == (expected,)
        assert ids.dtype == np.int32
        assert (ids >= 0).all()

    def test_fly_without_vision_has_no_ids(
        self, simulation_without_vision, fly_without_vision
    ):
        assert (
            fly_without_vision.name
            not in simulation_without_vision._intern_eye_camera_ids_by_fly
        )
        assert (
            fly_without_vision.name
            not in simulation_without_vision._intern_hidden_geom_ids_by_fly
        )

    def test_get_raw_vision_raises_for_fly_without_vision(
        self, simulation_without_vision, fly_without_vision
    ):
        with pytest.raises(ValueError, match="add_vision"):
            simulation_without_vision.get_raw_vision(fly_without_vision.name)


# ==============================================================================
# Simulation vision: actual rendering (Linux-only via EGL)
# ==============================================================================


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason=(
        "mujoco hardcodes CGL on macOS and GLFW on Windows; "
        "neither works headlessly in CI without a GPU"
    ),
)
class TestSimulationGetRawVision:
    def test_returns_two_frames(self, simulation_with_vision, fly_with_vision):
        frames = simulation_with_vision.get_raw_vision(fly_with_vision.name)
        assert isinstance(frames, list)
        assert len(frames) == 2

    def test_frame_shape_matches_retina(self, simulation_with_vision, fly_with_vision):
        frames = simulation_with_vision.get_raw_vision(fly_with_vision.name)
        retina = simulation_with_vision.retina
        for frame in frames:
            assert frame.shape == (retina.nrows, retina.ncols, 3)
            assert frame.dtype == np.uint8

    def test_lazy_initializers_set_after_call(
        self, simulation_with_vision, fly_with_vision
    ):
        # Force fresh lazy state then verify it's populated after the call
        simulation_with_vision.eye_renderer = None
        simulation_with_vision.retina = None
        simulation_with_vision.get_raw_vision(fly_with_vision.name)
        assert simulation_with_vision.retina is not None
        assert simulation_with_vision.eye_renderer is not None

    def test_hidden_geom_alpha_restored(self, simulation_with_vision, fly_with_vision):
        """get_raw_vision temporarily zeroes hidden geoms' alpha during rendering;
        the original values must be restored on return."""
        hidden_ids = simulation_with_vision._intern_hidden_geom_ids_by_fly[
            fly_with_vision.name
        ]
        before = simulation_with_vision.mj_model.geom_rgba[hidden_ids, 3].copy()
        simulation_with_vision.get_raw_vision(fly_with_vision.name)
        after = simulation_with_vision.mj_model.geom_rgba[hidden_ids, 3]
        np.testing.assert_array_equal(before, after)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason=(
        "mujoco hardcodes CGL on macOS and GLFW on Windows; "
        "neither works headlessly in CI without a GPU"
    ),
)
class TestSimulationGetOmmatidiaReadouts:
    def test_output_shape(self, simulation_with_vision, fly_with_vision):
        readouts = simulation_with_vision.get_ommatidia_readouts(fly_with_vision.name)
        assert readouts.ndim == 3
        n_cams, n_om, n_ch = readouts.shape
        assert n_cams == 2
        assert n_om == simulation_with_vision.retina.num_ommatidia_per_eye
        assert n_ch == 2

    def test_output_dtype_float32(self, simulation_with_vision, fly_with_vision):
        readouts = simulation_with_vision.get_ommatidia_readouts(fly_with_vision.name)
        assert readouts.dtype == np.float32

    def test_values_in_unit_range(self, simulation_with_vision, fly_with_vision):
        readouts = simulation_with_vision.get_ommatidia_readouts(fly_with_vision.name)
        assert readouts.min() >= 0.0
        assert readouts.max() <= 1.0
