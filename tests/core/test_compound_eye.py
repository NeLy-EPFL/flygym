"""Measured geometry, native cameras and deterministic Gaussian integration."""

import json
import os
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from flygym import assets_dir
from flygym.compose.fly import NeuroMechFly
from flygym.compose.world import TetheredWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym.vision.compound_eye import CompoundEye


requires_rendering = pytest.mark.skipif(
    os.environ.get("SKIP_RENDERING_TESTS") == "1", reason="Headless OpenGL unavailable"
)


@pytest.fixture
def dev_tools(monkeypatch):
    """Make repository-only diagnostic scripts importable."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def small_eye(tmp_path):
    """Five irregular samples, with unequal eye counts."""
    geometry_path = tmp_path / "geometry.json"
    registration_path = tmp_path / "registration.yaml"
    geometry_path.write_text(
        json.dumps(
            {
                "eye": ["right", "left", "left", "right", "left"],
                "position": [
                    [0, 0.2, 0],
                    [0, -0.2, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0.2, 0],
                ],
                "direction": [[1, 0, 0]] * 5,
            }
        )
    )
    registration_path.write_text(
        yaml.safe_dump(
            {
                "source_to_mm": 1,
                "pitch_deg": 0,
                "translation": [0, 0, 0],
                "camera": {"forward": [1, 0, 0], "up": [0, 0, 1], "fovy": 90},
            }
        )
    )

    def make(**kwargs):
        return CompoundEye(
            geometry_path=geometry_path,
            registration_path=registration_path,
            lattice_path=kwargs.pop("lattice_path", None),
            camera_res=kwargs.pop("camera_res", (64, 64)),
            view_resolution=kwargs.pop("view_resolution", 64),
            **kwargs,
        )

    return make


@pytest.fixture
def small_lattice(tmp_path):
    """A small lattice with source rows deliberately opposite to readout order."""
    lattice = {"index_base": 0}
    for side, coordinates in (
        ("left", [[0, 0], [0, 1], [1, 1]]),
        ("right", [[0, 0], [1, 0]]),
    ):
        coordinates = np.array(coordinates)
        lookup = {tuple(xy): i for i, xy in enumerate(coordinates)}
        neighbors = [
            [lookup.get(tuple(xy + offset)) for offset in CompoundEye.neighbor_offsets]
            for xy in coordinates
        ]
        lattice[side] = {
            "coordinates": coordinates.tolist(),
            "neighbors": neighbors,
            "neighbor_angles": [
                [0.1 if index is not None else None for index in row]
                for row in neighbors
            ],
        }
    path = tmp_path / "lattice.json"
    path.write_text(json.dumps(lattice))
    return path


def test_bundled_geometry_registration_and_source_order():
    eye = CompoundEye()
    with open(assets_dir / "model/neuromechfly/vision/ommatidia_microct.json") as f:
        source = json.load(f)
    with open(
        assets_dir / "model/neuromechfly/vision/ommatidia_registration.yaml"
    ) as f:
        registration = yaml.safe_load(f)
    r = Rotation.from_euler("y", registration["pitch_deg"], degrees=True)
    np.testing.assert_allclose(
        eye.positions,
        r.apply(source["position"]) * 0.001 + registration["translation"],
    )
    np.testing.assert_allclose(eye.directions, r.apply(source["direction"]), atol=1e-14)
    assert eye.num_ommatidia_per_eye == (857, 852)
    assert np.count_nonzero(eye.source_indices < 0) == 5
    for i, side in enumerate(("left", "right")):
        np.testing.assert_array_equal(
            np.sort(eye.eye_indices[i]), np.flatnonzero(np.array(source["eye"]) == side)
        )
        assert np.all(eye.positions[eye.eye_indices[i], 1] * (1 if i == 0 else -1) > 0)
    np.testing.assert_allclose(np.linalg.norm(eye.directions, axis=1), 1)


def test_lattice_order_and_neighbor_indices_align_with_readouts(
    small_eye, small_lattice
):
    eye = small_eye(lattice_path=small_lattice)
    np.testing.assert_array_equal(eye.eye_indices[0], [4, 2, 1])
    np.testing.assert_array_equal(eye.eye_indices[1], [3, 0])
    for side, indices in enumerate(eye.eye_indices):
        valid = eye.valid_mask[side]
        np.testing.assert_array_equal(eye.source_indices[side, valid], indices)
        xy = eye.hex_coordinates[side, valid]
        neighbors = eye.neighbor_indices[side, valid]
        present = neighbors >= 0
        offsets = xy[np.maximum(neighbors, 0)] - xy[:, None, :]
        np.testing.assert_array_equal(
            offsets[present],
            np.broadcast_to(eye.neighbor_offsets, offsets.shape)[present],
        )
        np.testing.assert_array_equal(
            np.isnan(eye.neighbor_angles[side, valid]), ~present
        )
        np.testing.assert_array_equal(eye.neighbor_indices[side, ~valid], -1)
    without_lattice = small_eye()
    assert without_lattice.hex_coordinates is None
    assert without_lattice.neighbor_indices is None
    assert without_lattice.neighbor_angles is None
    np.testing.assert_array_equal(without_lattice.eye_indices[0], [1, 2, 4])


def test_bundled_lattice_rows_neighbors_and_source_angles():
    eye = CompoundEye()
    for side, indices in enumerate(eye.eye_indices):
        valid = eye.source_indices[side] >= 0
        xy = eye.hex_coordinates[side, valid]
        assert len(np.unique(xy, axis=0)) == len(indices)
        height, horizontal = xy.sum(axis=1), xy[:, 0] - xy[:, 1]
        assert np.all(np.diff(height) <= 0)
        assert np.all(np.diff(horizontal)[np.diff(height) == 0] > 0)
        assert np.corrcoef(height, eye.positions[indices, 2])[0, 1] > 0.99
        assert np.corrcoef(horizontal, eye.positions[indices, 0])[0, 1] < -0.95
        neighbors = eye.neighbor_indices[side, valid]
        present = neighbors >= 0
        offsets = xy[np.maximum(neighbors, 0)] - xy[:, None, :]
        np.testing.assert_array_equal(
            offsets[present],
            np.broadcast_to(eye.neighbor_offsets, offsets.shape)[present],
        )
        # The original angular distances still refer to these measured axes.
        axes = eye.directions[indices]
        angles = np.arccos(
            np.clip((axes[np.maximum(neighbors, 0)] * axes[:, None, :]).sum(-1), -1, 1)
        )
        np.testing.assert_allclose(
            eye.neighbor_angles[side, valid][present], angles[present], atol=1e-12
        )
        np.testing.assert_array_equal(
            np.isnan(eye.neighbor_angles[side, valid]), ~present
        )
        assert np.any(~present)  # Boundary facets have missing neighbors.


@pytest.mark.parametrize("signal_mode", ("intensity", "pale_yellow"))
def test_anatomical_order_only_permutes_signals(signal_mode):
    options = dict(signal_mode=signal_mode, camera_res=(64, 64), view_resolution=32)
    eye = CompoundEye(**options)
    source_order = CompoundEye(**options, lattice_path=None)
    images = np.random.default_rng(1).integers(
        0, 256, (2, *eye.camera_res, 3), dtype=np.uint8
    )
    ordered = eye.raw_images_to_readouts(images)
    original = source_order.raw_images_to_readouts(images)
    for side, indices in enumerate(eye.eye_indices):
        permutation = np.searchsorted(source_order.eye_indices[side], indices)
        np.testing.assert_array_equal(
            ordered[side, : len(indices)], original[side, permutation]
        )
        np.testing.assert_array_equal(
            eye.valid_mask[side, : len(indices)],
            source_order.valid_mask[side, permutation],
        )
    np.testing.assert_array_equal(eye.positions, source_order.positions)
    np.testing.assert_array_equal(eye.optical_centers, source_order.optical_centers)


@pytest.mark.parametrize("field", ("coordinates", "neighbors", "neighbor_angles"))
def test_invalid_lattice_arrays_are_rejected(small_eye, small_lattice, field):
    lattice = json.loads(small_lattice.read_text())
    if field == "coordinates":
        lattice["left"][field][1] = lattice["left"][field][0]
    elif field == "neighbors":
        lattice["left"][field][0][0] = 3  # Not a lens in this eye.
    else:
        lattice["left"][field][0][0] = 0.1  # This neighbor is missing.
    small_lattice.write_text(json.dumps(lattice))
    with pytest.raises(ValueError, match="lattice"):
        small_eye(lattice_path=small_lattice)


def test_lattice_requires_zero_based_indices(small_eye, small_lattice):
    lattice = json.loads(small_lattice.read_text())
    lattice["index_base"] = 1
    small_lattice.write_text(json.dumps(lattice))
    with pytest.raises(ValueError, match="zero-based"):
        small_eye(lattice_path=small_lattice)


def test_pale_mask_is_reproducible():
    np.testing.assert_array_equal(
        CompoundEye().pale_type_mask, CompoundEye().pale_type_mask
    )
    assert not np.array_equal(
        CompoundEye(seed=0).pale_type_mask, CompoundEye(seed=1).pale_type_mask
    )


def make_simulation(eye, rotation=Rotation3D("quat", (1, 0, 0, 0))):
    fly = NeuroMechFly()
    fly.add_vision(compound_eye=eye)
    world = TetheredWorld()
    world.add_fly(fly, (2, 3, 4), rotation)
    simulation = Simulation(world)
    mj.mj_forward(simulation.mj_model, simulation.mj_data)
    return simulation, fly


def test_sensor_frame_survives_fusion_and_tracks_pose(small_eye):
    eye = small_eye()
    rotation = Rotation.from_euler("xyz", [15, -20, 40], degrees=True)
    simulation, fly = make_simulation(
        eye, Rotation3D("quat", rotation.as_quat(scalar_first=True).tolist())
    )
    try:
        model, data = simulation.mj_model, simulation.mj_data
        head_translation = rotation.apply([0.496 + 0.0287, 0, 1.3 + 0.000698]) + [
            2,
            3,
            4,
        ]
        ids = simulation._intern_eye_camera_ids_by_fly[fly.name]
        for side in range(2):
            np.testing.assert_allclose(
                data.cam_xpos[ids[side]],
                rotation.apply(eye.optical_centers[side]) + head_translation,
                atol=1e-8,
            )
            camera_rotation = data.cam_xmat[ids[side]].reshape(3, 3)
            np.testing.assert_allclose(
                -camera_rotation[:, 2],
                rotation.apply(eye.camera_directions[side]),
                atol=1e-8,
            )
            np.testing.assert_allclose(
                camera_rotation[:, 1],
                rotation.apply(eye.camera_up_vectors[side]),
                atol=1e-8,
            )
        # Native fixed cameras retain their eye bodies through static fusion.
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{fly.name}/l_eye") >= 0
        np.testing.assert_array_equal(model.cam_fovy[ids.ravel()], 90)
        with pytest.raises(ValueError, match="already"):
            fly.add_vision(compound_eye=eye)
    finally:
        simulation.close()


@requires_rendering
def test_retina_self_exclusion_is_scene_only_and_opt_in():
    fly = NeuroMechFly()
    fly.add_vision(exclude_self=True)
    world = TetheredWorld()
    world.add_fly(fly, (0, 0, 0), Rotation3D("quat", (1, 0, 0, 0)))
    simulation = Simulation(world)
    try:
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        before = simulation.mj_model.geom_rgba.copy()
        simulation.get_raw_vision(fly.name)
        own = simulation._internal_geomids_by_fly[fly.name]
        visible = [
            geom
            for geom in simulation.eye_renderer.scene.geoms[
                : simulation.eye_renderer.scene.ngeom
            ]
            if geom.objtype == mj.mjtObj.mjOBJ_GEOM and geom.objid in own
        ]
        assert visible and all(geom.rgba[3] == 0 for geom in visible)
        np.testing.assert_array_equal(simulation.mj_model.geom_rgba, before)
        retina = NeuroMechFly()
        retina.add_vision()
        assert retina.vision_excludes_self is False
    finally:
        simulation.close()


def test_registration_rejects_bad_geometry_and_transform(small_eye, tmp_path):
    # Validation must fail before renderer allocation for corrupted measurements.
    small_eye()
    geometry_path = tmp_path / "geometry.json"
    geometry = json.loads(geometry_path.read_text())
    geometry["direction"][0] = [0, 0, 0]
    geometry_path.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="geometry"):
        small_eye()
    geometry["direction"][0] = [1, 0, 0]
    geometry_path.write_text(json.dumps(geometry))
    (tmp_path / "registration.yaml").write_text(
        yaml.safe_dump(
            {
                "source_to_mm": -1,
                "pitch_deg": 0,
                "translation": [0, 0, 0],
            }
        )
    )
    with pytest.raises(ValueError, match="registration"):
        small_eye()


@requires_rendering
def test_mixed_retina_and_anatomical_flies(small_eye):
    anatomical = NeuroMechFly(name="anatomical")
    anatomical.add_vision(compound_eye=small_eye(), draw_sensor_markers=True)
    retina = NeuroMechFly(name="retina")
    retina.add_vision()
    world = TetheredWorld()
    for fly, position in ((anatomical, (0, 0, 0)), (retina, (10, 0, 0))):
        world.add_fly(fly, position, Rotation3D("quat", (1, 0, 0, 0)))
    simulation = Simulation(world)
    mj.mj_forward(simulation.mj_model, simulation.mj_data)
    try:
        assert simulation.get_ommatidia_readouts(anatomical.name).shape == (2, 3, 1)
        assert simulation.get_ommatidia_readouts(retina.name).shape == (2, 721, 2)
        assert simulation.get_raw_vision(retina.name).dtype == np.uint8
    finally:
        simulation.close()


def test_registered_lenses_and_axes_overlay_target_eyes():
    trimesh = pytest.importorskip("trimesh")
    eye = CompoundEye()
    model_dir = assets_dir / "model/neuromechfly"
    mesh = trimesh.load_mesh(model_dir / "meshes/simplified_max2000faces/l_eye.stl")
    with open(model_dir / "rigging.yaml") as f:
        target = yaml.safe_load(f)["l_eye"]
    mesh.vertices = (
        Rotation.from_quat(target["quat"], scalar_first=True).apply(
            mesh.vertices * NeuroMechFly.SCALE
        )
        + target["pos"]
    )
    for side, indices in enumerate(eye.eye_indices):
        positions = eye.positions[indices].copy()
        directions = eye.directions[indices].copy()
        if side == 1:
            positions[:, 1] *= -1
            directions[:, 1] *= -1
        _, distance, faces = trimesh.proximity.closest_point(mesh, positions)
        assert np.sqrt(np.mean(distance**2)) < 0.016  # <= 16 µm RMS
        assert distance.max() < 0.035  # <= 35 µm maximum
        assert np.mean((mesh.face_normals[faces] * directions).sum(-1) > 0) > 0.98


def test_anatomical_cardinal_frames_are_not_reversed():
    model_dir = assets_dir / "model/neuromechfly"
    with open(model_dir / "vision/ommatidia_registration.yaml") as f:
        registration = yaml.safe_load(f)
    rotation = Rotation.from_euler("y", registration["pitch_deg"], degrees=True)
    # A front or dorsal source unit vector must remain front or dorsal in NMF.
    np.testing.assert_array_less(0.99, np.diag(rotation.as_matrix()))
    assert np.linalg.det(rotation.as_matrix()) == pytest.approx(1)
    with open(model_dir / "rigging.yaml") as f:
        rigging = yaml.safe_load(f)
    assert rigging["c_abdomen12"]["pos"][0] < 0 < rigging["c_head"]["pos"][0]
    assert rigging["l_pedicel"]["pos"][0] > rigging["l_eye"]["pos"][0]
    assert rigging["c_rostrum"]["pos"][2] < 0 < rigging["l_pedicel"]["pos"][2]
    with open(model_dir / "mujoco_globals.yaml") as f:
        globals_ = yaml.safe_load(f)
    assert globals_["option"]["gravity"][2] < 0
    eye = CompoundEye()
    # Optical axes point from the lower and rear cornea toward the lower/rear
    # visual field, and likewise for upper/front samples (not inward or swapped).
    assert np.corrcoef(eye.positions[:, 0], eye.directions[:, 0])[0, 1] > 0.8
    assert np.corrcoef(eye.positions[:, 2], eye.directions[:, 2])[0, 1] > 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"acceptance_angle": 0},
        {"acceptance_angle": -1},
        {"acceptance_angle": np.nan},
        {"acceptance_angle": 180},
        {"view_resolution": 1},
        {"view_resolution": 2.5},
        {"view_resolution": True},
        {"camera_res": (64,)},
        {"camera_res": (1, 64)},
        {"camera_res": (64, 2.5)},
        {"camera_res": (64, 32)},
        {"seed": -1},
        {"seed": True},
        {"signal_mode": "rgb"},
        {"pale_type_mask": [1]},
        {"pale_type_mask": [0, 0, 0, 0, 2]},
    ],
)
def test_invalid_options(small_eye, kwargs):
    with pytest.raises(ValueError):
        small_eye(**kwargs)


def test_display_azimuth_places_left_on_negative_coordinates(dev_tools):
    from outputs.compound_eye.compound_eye.visualization import plot_compound_eye_readouts
    import matplotlib.pyplot as plt

    eye = CompoundEye()
    readings = np.ones((*eye.valid_mask.shape, 1), np.float32)
    readings[0, np.flatnonzero(eye.valid_mask[0])[0]] = np.nan
    figure = plot_compound_eye_readouts(eye, readings)
    try:
        for side, (indices, axis) in enumerate(zip(eye.eye_indices, figure.axes)):
            valid = eye.valid_mask[side, : len(indices)] & np.isfinite(
                readings[side, : len(indices), 0]
            )
            azimuth = axis.collections[0].get_offsets()[:, 0]
            np.testing.assert_allclose(
                azimuth,
                -np.rad2deg(
                    np.arctan2(
                        eye.directions[indices[valid], 1],
                        eye.directions[indices[valid], 0],
                    )
                ),
            )
            assert np.isfinite(axis.collections[0].get_array()).all()
    finally:
        plt.close(figure)


@requires_rendering
def test_comparison_video_saves_synchronized_moving_sensor_data(tmp_path, dev_tools):
    import imageio.v2 as iio
    from outputs.compound_eye.compound_eye.video import render_comparison_video

    path = tmp_path / "comparison.mp4"
    render_comparison_video(
        path, duration=0.3, fps=10, view_resolution=32, camera_res=(128, 128)
    )
    with np.load(path.with_suffix(".npz")) as data:
        np.testing.assert_allclose(data["times"], [0, 0.1, 0.2])
        assert data["anatomical"].shape == (3, 2, 857, 1)
        assert data["retina"].shape == (3, 2, 721, 2)
        assert np.all(np.isnan(data["anatomical"][:, 1, 852:]))
        assert np.all(np.isnan(data["anatomical"][:, ~data["valid_mask"]]))
        assert np.all(np.isfinite(data["anatomical"][:, data["valid_mask"]]))
        np.testing.assert_array_equal(data["valid_mask"].sum(axis=1), [845, 830])
        assert data["camera_fovy"] == 168.04
        assert np.all(np.isfinite(data["retina"]))
        assert data["integration"] == "gaussian_equal_area_pixels"
        assert data["exclude_self"]
        eye = CompoundEye(view_resolution=32)
        np.testing.assert_array_equal(data["hex_coordinates"], eye.hex_coordinates)
        np.testing.assert_array_equal(data["neighbor_indices"], eye.neighbor_indices)
        np.testing.assert_array_equal(data["source_indices"], eye.source_indices)
        np.testing.assert_array_equal(data["light_direction"], [[0, 0, -1], [0, 0, -1]])
        # Prescribed yaw must not introduce a different pitch/roll in either eye.
        np.testing.assert_array_equal(data["body_poses"][:, 4:6], 0)
        # A common prescribed trajectory moves both sensors through the arena.
        assert data["body_poses"][-1, 0] > data["body_poses"][0, 0] + 1
        for name in ("anatomical", "retina"):
            assert not np.allclose(data[name][0], data[name][-1], equal_nan=True)
    with iio.get_reader(str(path)) as reader:
        assert reader.count_frames() == 3
        assert reader.get_data(0).shape == (1280, 1280, 3)
    assert iio.imread(path.with_suffix(".png")).shape[:2] == (1280, 1280)


def test_lattice_plot_pairs_each_signal_with_its_coordinate(
    small_eye, small_lattice, dev_tools
):
    import matplotlib.pyplot as plt
    from outputs.compound_eye.compound_eye.visualization import plot_compound_eye_readouts

    eye = small_eye(lattice_path=small_lattice)
    readings = np.arange(6, dtype=float).reshape(2, 3, 1) / 6
    figure = plot_compound_eye_readouts(eye, readings, projection="lattice")
    try:
        for side, axis in enumerate(figure.axes[:2]):
            p, q = eye.hex_coordinates[side, eye.valid_mask[side]].T
            np.testing.assert_allclose(
                axis.collections[0].get_offsets(),
                np.column_stack((np.sqrt(3) * (p - q) / 2, (p + q) / 2)),
            )
            np.testing.assert_array_equal(
                axis.collections[0].get_array(), readings[side, eye.valid_mask[side], 0]
            )
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="hex_coordinates"):
        plot_compound_eye_readouts(small_eye(), readings, projection="lattice")


def test_optical_centers_fit_axis_concurrency_and_preserve_geometry(small_eye):
    eye = small_eye()
    center = np.array([0.1, 0.2, 0.3])
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    for side, indices in enumerate(eye.eye_indices):
        eye.positions[indices] = center + axes[indices] * 0.2
        eye.directions[indices] = axes[indices]
        if side:
            eye.positions[indices, 1] *= -1
            eye.directions[indices, 1] *= -1
    positions = eye.positions.copy()
    np.testing.assert_allclose(
        eye.optical_centers, [center, center * [1, -1, 1]], atol=1e-15
    )
    np.testing.assert_array_equal(eye.positions, positions)
    # Parallel lines do not constrain the longitudinal coordinate.
    eye.directions[:] = [1, 0, 0]
    assert eye.optical_centers[0, 0] == pytest.approx(eye.positions[:, 0].mean())


def test_bundled_optical_center_residual_is_bounded():
    eye = CompoundEye()
    np.testing.assert_array_equal(
        eye.optical_centers[0] * [1, -1, 1], eye.optical_centers[1]
    )
    offset = np.empty_like(eye.positions)
    for side, indices in enumerate(eye.eye_indices):
        offset[indices] = eye.positions[indices] - eye.optical_centers[side]
    perpendicular = offset - (offset * eye.directions).sum(-1)[:, None] * eye.directions
    assert np.quantile(np.linalg.norm(perpendicular, axis=-1), 0.95) < 0.04


def test_equal_area_fisheye_directions_and_solid_angles(small_eye):
    eye = small_eye()
    assert eye.pixel_directions.shape == (2, 64, 64, 3)
    np.testing.assert_allclose(np.linalg.norm(eye.pixel_directions, axis=-1), 1)
    np.testing.assert_array_equal(
        eye.camera_directions[0] * [1, -1, 1], eye.camera_directions[1]
    )
    assert eye.pixel_solid_angles.sum() == pytest.approx(2 * np.pi / 3, abs=0.03)
    area = eye.pixel_solid_angles
    np.testing.assert_array_equal(area[area > 0], 8 / eye.view_resolution**2)
    assert area[32, 32] > 0 and area[0, 0] == 0


def test_gaussians_are_normalized_over_visible_part(small_eye):
    eye = small_eye(view_resolution=512)
    axes = np.array([[1.0, 0, 0], [1.0, 0, 1], [1.0, 1, 1]])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    eye.directions[eye.eye_indices[0]] = axes
    weights = eye.weights[0]
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=2e-7)
    np.testing.assert_allclose(eye.acceptance_coverage[0], [1, 0.5, 1 / 3], atol=0.02)
    images = np.full((2, *eye.camera_res, 3), 255, np.uint8)
    readouts = eye.raw_images_to_readouts(images)
    np.testing.assert_allclose(readouts[0, :, 0], 1, atol=5e-6)
    np.testing.assert_array_equal(eye.raw_images_to_readouts(images), readouts)
    assert eye.weights[0] is weights  # weights are computed only once


def test_weights_follow_angular_gaussian_and_solid_angle(small_eye):
    eye = small_eye(acceptance_angle=15)
    index = eye.eye_indices[0][0]
    theta = np.arccos(np.clip(eye.pixel_directions[0] @ eye.directions[index], -1, 1))
    expected = (
        np.exp(-4 * np.log(2) * (theta / np.deg2rad(15)) ** 2) * eye.pixel_solid_angles
    )
    expected /= expected.sum()
    np.testing.assert_allclose(
        eye.weights[0][[0]].toarray().reshape(64, 64), expected, atol=1e-9
    )


@pytest.mark.parametrize("angle", (0.01, 5, 179))
def test_angular_normalization_remains_finite(small_eye, angle):
    eye = small_eye(acceptance_angle=angle)
    with np.errstate(divide="raise", invalid="raise"):
        assert np.isfinite(eye.acceptance_coverage[eye.valid_mask]).all()
        np.testing.assert_allclose(eye.weights[0].sum(axis=1), 1, atol=2e-7)


def test_weighted_rgb_readouts_and_input_validation(small_eye):
    eye = small_eye(signal_mode="pale_yellow", pale_type_mask=[0, 1, 0, 1, 1])
    images = np.broadcast_to(
        np.array([25, 100, 200], np.uint8), (2, *eye.camera_res, 3)
    ).copy()
    readings = eye.raw_images_to_readouts(images)
    for side, indices in enumerate(eye.eye_indices):
        for row, source in enumerate(indices):
            channel = int(eye.pale_type_mask[source])
            assert readings[side, row, 1 - channel] == 0
            assert readings[side, row, channel] == pytest.approx(
                (100, 200)[channel] / 255, abs=2e-6
            )
    assert np.isnan(readings[1, 2]).all()
    with pytest.raises(ValueError, match="shape"):
        eye.raw_images_to_readouts(images[0])


@pytest.mark.parametrize("signal_mode", ("intensity", "pale_yellow"))
def test_out_of_view_axes_return_nan_without_changing_geometry(
    small_eye, tmp_path, signal_mode
):
    small_eye()
    path = tmp_path / "geometry.json"
    geometry = json.loads(path.read_text())
    # One left axis is just outside the vertical boundary; both right axes
    # are unobservable (behind the camera and perpendicular to its axis).
    geometry["direction"] = [[-1, 0, 0], [1, 0, 0], [1, 0, 1.1], [0, 1, 0], [1, 0, 0]]
    path.write_text(json.dumps(geometry))
    eye = small_eye(signal_mode=signal_mode)
    np.testing.assert_array_equal(eye.valid_mask, [[True, False, True], [False] * 3])
    np.testing.assert_array_equal(eye.source_indices, [[1, 2, 4], [0, 3, -1]])
    assert eye.num_ommatidia_per_eye == (3, 2)
    images = np.full((2, *eye.camera_res, 3), 255, np.uint8)
    readouts = eye.raw_images_to_readouts(images)
    assert np.isnan(readouts[~eye.valid_mask]).all()
    np.testing.assert_allclose(readouts[eye.valid_mask].sum(-1), 1, atol=2e-6)
    assert np.isnan(eye.acceptance_coverage[~eye.valid_mask]).all()
    for side, indices in enumerate(eye.eye_indices):
        np.testing.assert_allclose(
            eye.weights[side].sum(axis=1),
            eye.valid_mask[side, : len(indices)],
            atol=2e-7,
        )


def test_gaussian_resolution_convergence_against_solid_angle_integral(small_eye):
    from scipy.integrate import quad

    sigma = np.deg2rad(20) / np.sqrt(8 * np.log(2))

    def sensitivity(theta):
        return np.exp(-(theta**2) / (2 * sigma**2))

    expected = (
        quad(lambda t: sensitivity(t) ** 2 * np.sin(t), 0, np.pi)[0]
        / quad(lambda t: sensitivity(t) * np.sin(t), 0, np.pi)[0]
    )
    errors = []
    for resolution in (16, 32, 64):
        eye = small_eye(acceptance_angle=20, view_resolution=resolution)
        theta = np.arccos(np.clip(eye.pixel_directions[0, ..., 0], -1, 1))
        colors = sensitivity(theta).reshape(-1)
        value = (eye.weights[0] @ colors).mean()
        errors.append(abs(value - expected))
    assert errors[-1] < errors[0]
    assert max(errors[1:]) < 1e-6


def test_bilinear_fisheye_remap_preserves_native_image_gradient(small_eye):
    eye = small_eye()
    rows, columns = np.indices(eye.camera_res)
    image = np.stack((rows, columns, rows + columns), axis=-1).astype(np.float32)
    mapped = eye.correct_fisheye(image)
    projected_rows, projected_columns, valid = eye._image_coordinates
    expected = np.stack(
        (projected_rows, projected_columns, projected_rows + projected_columns),
        axis=-1,
    )
    np.testing.assert_allclose(mapped[valid], expected[valid], atol=4e-6)
    np.testing.assert_array_equal(mapped[~valid], 0)


def test_bundled_square_cameras_retain_lower_axes_and_are_symmetric():
    eye = CompoundEye()
    assert eye.fovy == 168.04
    assert eye.camera_res == (1024, 1024)
    np.testing.assert_array_equal(eye.valid_mask.sum(axis=1), [845, 830])
    np.testing.assert_array_equal(
        eye.camera_directions[0] * [1, -1, 1], eye.camera_directions[1]
    )
    np.testing.assert_array_equal(
        eye.camera_up_vectors[0] * [1, -1, 1], eye.camera_up_vectors[1]
    )
    tangent = np.tan(np.deg2rad(eye.fovy / 2))
    for side, indices in enumerate(eye.eye_indices):
        forward, up = eye.camera_directions[side], eye.camera_up_vectors[side]
        right = np.cross(forward, up)
        frame = np.column_stack((right, up, -forward))
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
        assert np.linalg.det(frame) == pytest.approx(1)
        axes = eye.directions[indices]
        depth = axes @ forward
        valid = eye.valid_mask[side, : len(indices)]
        assert np.all(depth[valid] > 0)
        assert np.all(np.abs(axes[valid] @ up) <= depth[valid] * tangent)
        assert np.all(np.abs(axes[valid] @ right) <= depth[valid] * tangent)
        # No optical axis crosses the lower boundary, including excluded axes.
        vertical = np.rad2deg(np.arctan2(axes @ up, depth))
        assert vertical.min() >= -eye.fovy / 2
    images = np.full((2, *eye.camera_res, 3), 128, np.uint8)
    readings = eye.raw_images_to_readouts(images)
    assert np.isnan(readings[~eye.valid_mask]).all()
    np.testing.assert_allclose(readings[eye.valid_mask], 128 / 255, atol=2e-6)


@requires_rendering
def test_raw_camera_views_readouts_reset_and_close(small_eye):
    simulation, fly = make_simulation(small_eye(signal_mode="pale_yellow"))
    try:
        before = simulation.mj_model.geom_rgba.copy()
        images = simulation.get_raw_vision(fly.name)
        assert images.shape == (2, 64, 64, 3)
        assert images.dtype == np.uint8
        readings = simulation.get_ommatidia_readouts(fly.name)
        np.testing.assert_array_equal(
            readings, fly.compound_eye.raw_images_to_readouts(images)
        )
        np.testing.assert_array_equal(simulation.mj_model.geom_rgba, before)
        simulation.reset()
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        np.testing.assert_array_equal(
            simulation.get_ommatidia_readouts(fly.name), readings
        )
        simulation.close()
        simulation.close()
        np.testing.assert_array_equal(
            simulation.get_ommatidia_readouts(fly.name), readings
        )
    finally:
        simulation.close()


@requires_rendering
def test_native_camera_projection_agrees_with_pixel_directions(small_eye):
    simulation, fly = make_simulation(small_eye())
    try:
        eye = fly.compound_eye
        data = simulation.mj_data
        ids = simulation._intern_eye_camera_ids_by_fly[fly.name]
        rows, columns, valid = eye._image_coordinates
        i, j = 31, 31
        assert valid[i, j]
        height, width = eye.camera_res
        focal = height / (2 * np.tan(np.deg2rad(eye.fovy / 2)))
        local = np.array(
            [
                (columns[i, j] + 0.5 - width / 2) / focal,
                (height / 2 - rows[i, j] - 0.5) / focal,
                -1,
            ]
        )
        local /= np.linalg.norm(local)
        for side in range(2):
            rotation = data.cam_xmat[ids[side]].reshape(3, 3)
            # MuJoCo cameras look along local -Z, with local +Y image-up.
            np.testing.assert_allclose(
                rotation @ local, eye.pixel_directions[side, i, j], atol=1e-8
            )
    finally:
        simulation.close()


@requires_rendering
def test_self_exclusion_retains_other_flies_and_native_shadows(small_eye, dev_tools):
    from outputs.compound_eye.compound_eye._scene import _make_world

    observer = NeuroMechFly(name="observer")
    observer.add_vision(compound_eye=small_eye())
    world = _make_world(observer, "corridor")
    other = NeuroMechFly(name="other")
    world.add_fly(other, [4, 0, 0], Rotation3D("quat", [1, 0, 0, 0]))
    simulation = Simulation(world)
    try:
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        before = simulation.mj_model.geom_rgba.copy()
        simulation.get_raw_vision(observer.name)
        own = simulation._internal_geomids_by_fly[observer.name]
        scene = simulation.eye_renderer.scene
        own_scene = [
            g
            for g in scene.geoms[: scene.ngeom]
            if g.objtype == mj.mjtObj.mjOBJ_GEOM and g.objid in own
        ]
        other_scene = [
            g
            for g in scene.geoms[: scene.ngeom]
            if g.objtype == mj.mjtObj.mjOBJ_GEOM and g.objid not in own
        ]
        assert own_scene and all(
            g.rgba[3] == 0 and g.type != mj.mjtGeom.mjGEOM_NONE for g in own_scene
        )
        assert other_scene and any(g.rgba[3] > 0 for g in other_scene)
        assert any(light.castshadow for light in scene.lights[: scene.nlight])
        np.testing.assert_array_equal(simulation.mj_model.geom_rgba, before)
    finally:
        simulation.close()


@requires_rendering
def test_native_self_shadows_retain_ambient_illumination(small_eye, dev_tools):
    from outputs.compound_eye.compound_eye._scene import _make_world

    eye = small_eye()
    eye.positions[:] = 0
    eye.directions[:] = [0, 0, -1]
    eye.camera_directions[:] = [0, 0, -1]
    eye.camera_up_vectors[:] = [1, 0, 0]
    fly = NeuroMechFly()
    fly.bodyseg_to_mjcfbody[fly.root_segment].add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE, pos=[0.03, 0, -0.4], size=[0.25, 0, 0]
    )
    fly.add_vision(compound_eye=eye)
    simulation = Simulation(_make_world(fly, "corridor"))
    try:
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        shadowed = simulation.get_ommatidia_readouts(fly.name)[0, 0, 0]
        # Removing the observer from the native scene also removes its shadow;
        # primary-ray transparency alone retains it.
        own = list(simulation._internal_geomids_by_fly[fly.name])
        simulation.mj_model.geom_group[own] = 1
        unshadowed = simulation.get_ommatidia_readouts(fly.name)[0, 0, 0]
        assert unshadowed > shadowed + 0.1
        assert 0 < shadowed < unshadowed <= 1
    finally:
        simulation.close()


@requires_rendering
def test_hidden_geometry_without_shadows_has_identical_signals(small_eye, monkeypatch):
    from flygym.compose.world import FlatGroundWorld

    fly = NeuroMechFly()
    fly.add_vision(compound_eye=small_eye())
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0],
        Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=False,
    )
    simulation = Simulation(world)
    try:
        mj.mj_forward(simulation.mj_model, simulation.mj_data)
        optimized = simulation.get_ommatidia_readouts(fly.name)
        native_render = mj.mjr_render
        own = simulation._internal_geomids_by_fly[fly.name]

        def render_transparent_triangles(rect, scene, context):
            for geom in scene.geoms[: scene.ngeom]:
                if geom.objtype == mj.mjtObj.mjOBJ_GEOM and geom.objid in own:
                    assert geom.type == mj.mjtGeom.mjGEOM_NONE
                    geom.type = simulation.mj_model.geom_type[geom.objid]
            return native_render(rect, scene, context)

        monkeypatch.setattr(mj, "mjr_render", render_transparent_triangles)
        np.testing.assert_array_equal(
            simulation.get_ommatidia_readouts(fly.name), optimized
        )
    finally:
        simulation.close()


@requires_rendering
@pytest.mark.parametrize("scene_name", ("corridor", "panels", "ground", "near-object"))
def test_anatomical_views_use_native_rendering(small_eye, scene_name, dev_tools):
    from outputs.compound_eye.compound_eye._scene import _make_world

    eye = small_eye()
    fly = NeuroMechFly()
    fly.add_vision(compound_eye=eye)
    simulation = Simulation(_make_world(fly, scene_name))
    baseline = None
    try:
        model = simulation.mj_model
        configured_size = (model.vis.global_.offheight, model.vis.global_.offwidth)
        mj.mj_forward(model, simulation.mj_data)
        images = simulation.get_raw_vision(fly.name)
        assert isinstance(simulation.eye_renderer, mj.Renderer)
        assert (
            model.vis.global_.offheight,
            model.vis.global_.offwidth,
        ) == configured_size
        baseline = mj.Renderer(model, height=64, width=64)
        own = simulation._internal_geomids_by_fly[fly.name]
        frames = []
        for camera in simulation._intern_eye_camera_ids_by_fly[fly.name]:
            baseline.update_scene(
                simulation.mj_data,
                camera,
                scene_option=simulation.eye_renderer_scene_option,
            )
            for geom in baseline.scene.geoms[: baseline.scene.ngeom]:
                if geom.objtype == mj.mjtObj.mjOBJ_GEOM and geom.objid in own:
                    geom.rgba[3] = 0
            frames.append(baseline.render())
        np.testing.assert_array_equal(images, np.array(frames))
    finally:
        if baseline is not None:
            baseline.close()
        simulation.close()


def test_projection_figure_shows_equal_area_camera_images(small_eye, dev_tools):
    import matplotlib.pyplot as plt
    from outputs.compound_eye.compound_eye.projections import plot_compound_eye_projections

    eye = small_eye()
    images = np.zeros((2, *eye.camera_res, 3), np.uint8)
    images[0] = 50
    images[1] = 150
    images += np.arange(64, dtype=np.uint8)[None, :, None, None] // 4
    figure = plot_compound_eye_projections(eye, images)
    try:
        displayed_images = []
        for axis in figure.axes:
            if not axis.images:
                continue
            displayed_images.append(np.asarray(axis.images[0].get_array()))
        assert len(displayed_images) == 6
        for side in range(2):
            for displayed in displayed_images[side * 3 : (side + 1) * 3]:
                np.testing.assert_array_equal(
                    displayed, eye.correct_fisheye(images[side]) / 255
                )
    finally:
        plt.close(figure)


@requires_rendering
def test_compound_eye_example_renders_moving_visual_input(tmp_path):
    import imageio.v2 as iio
    from flygym_demo.compound_eye.__main__ import render_video

    path = tmp_path / "input.mp4"
    render_video(path, duration=0.12, fps=25)
    with iio.get_reader(path) as reader:
        assert reader.count_frames() == 3
        first, last = reader.get_data(0), reader.get_data(2)
        assert first.shape == (720, 1024, 3)
        assert not np.array_equal(first, last)
