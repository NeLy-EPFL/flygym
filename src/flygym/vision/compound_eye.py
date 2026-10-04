"""Compound-eye visual input based on measured ommatidial geometry."""

import json
from functools import cached_property
from os import PathLike
from typing import Literal

import numpy as np
import yaml
from scipy.integrate import quad
from scipy.sparse import csr_array
from scipy.spatial.transform import Rotation

from flygym import assets_dir

__all__ = ["CompoundEye"]

_GEOMETRY_DIR = assets_dir / "model/neuromechfly/vision"


class CompoundEye:
    """This class handles the simulation of the fly's visual input with an
    anatomical compound eye model based on experimentally measured geometry
    (optional).

    The model contains 857 left and 852 right ommatidia, with positions and
    viewing directions measured by
    [Zhao et al. (2025)](https://doi.org/10.1038/s41586-025-09276-5).
    Each ommatidium's signal is a Gaussian-weighted mean of scene intensity
    around its viewing direction. ``acceptance_angle`` sets the Gaussian's
    angular full width at half maximum (FWHM).

    One camera per eye approximates the individual ommatidial views. This omits
    differences in parallax and occlusion between ommatidia, particularly for
    nearby objects. The bundled cameras use a square 168.04-degree field of view
    pitched downward to retain all lower optical axes; the 12 left and 22 right
    out-of-view ommatidia lining the dorsal rim return NaN. Acceptance functions
    crossing the camera boundary are clipped and normalized over the visible
    portion.

    Args:
        geometry_path: File containing ommatidial positions and viewing directions.
        registration_path: File defining alignment to the head and camera settings.
        lattice_path: File containing lattice coordinates and neighbor data.
            Use None to retain per-eye source order in the readouts.
        acceptance_angle: Angular full width at half maximum (FWHM), in degrees.
            The default 5 degrees is an approximation, not a specimen measurement.
            Assumes that acceptance angle is the same for all ommatidia.
        camera_res: Square camera image (height, width), in pixels.
        view_resolution: Pixels per side of the fisheye image used for integration.
        signal_mode: ``"intensity"`` averages RGB; ``"pale_yellow"`` uses the
            existing G/yellow and B/pale abstraction, with inactive channels zero.
        pale_type_mask: Binary pale-type mask in source row order. If None,
            assign pale types with probability 0.3, as in the existing sensor.
        seed: Nonnegative seed for the pale-type assignment only.

    Attributes:
        positions: Measured lens centers, (N, 3), in head-local mm.
        directions: Outward unit optical axes, (N, 3), in the same frame.
        eye_indices: Left/right source row indices in readout order.
        num_ommatidia_per_eye: Left/right counts.
        valid_mask: Boolean (2, max_count) mask excluding out-of-view ommatidia
            and padding, whose readouts are NaN.
        source_indices: Source rows in readout order; padding is -1.
        hex_coordinates: Integer (2, max_count, 2) p/q coordinates aligned with
            readouts. Ignore padding with ``valid_mask``. None without a lattice.
        neighbor_indices: Integer (2, max_count, 6) indices into the same eye's
            readouts, in ``neighbor_offsets`` order. Missing neighbors and padding
            are -1. None without a lattice.
        neighbor_offsets: Six p/q offsets: +p, +(p+q), +q, -p, -(p+q), -q.
        neighbor_angles: (2, max_count, 6) source interommatidial angles in
            radians; missing neighbors and padding are NaN. None without a lattice.
        acceptance_coverage: Fraction of each Gaussian represented in the view,
            (2, max_count), estimated by angular quadrature. Out-of-view
            ommatidia and padding are NaN.

    Examples:
        ```python
        from flygym.compose.fly import NeuroMechFly
        from flygym.vision.compound_eye import CompoundEye

        # Attach the measured compound-eye model to the fly.
        eye = CompoundEye()
        fly = NeuroMechFly()
        fly.add_vision(compound_eye=eye)

        # Left-eye lattice coordinates, ordered to match the visual readouts.
        left_coordinates = eye.hex_coordinates[0, eye.valid_mask[0]]

        # Left-eye lens positions in the head frame, in mm and the same order.
        left_positions = eye.positions[eye.source_indices[0, eye.valid_mask[0]]]

        # Neighbors of the first left-eye ommatidium; -1 means no neighbor.
        neighbors = eye.neighbor_indices[0, 0]
        neighbors = neighbors[neighbors >= 0]  # Keep indices into left-eye readouts.
        ```
    """

    neighbor_offsets = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))

    def __init__(
        self,
        *,
        geometry_path: PathLike = _GEOMETRY_DIR / "ommatidia_microct.json",
        registration_path: PathLike = _GEOMETRY_DIR / "ommatidia_registration.yaml",
        lattice_path: PathLike | None = _GEOMETRY_DIR / "ommatidia_lattice.json",
        acceptance_angle: float = 5.0,
        camera_res: tuple[int, int] = (1024, 1024),
        view_resolution: int = 128,
        signal_mode: Literal["intensity", "pale_yellow"] = "intensity",
        pale_type_mask: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if not np.isfinite(acceptance_angle) or not 0 < acceptance_angle < 180:
            raise ValueError(
                "acceptance_angle must be finite and between 0 and 180 degrees."
            )
        if len(camera_res) != 2:
            raise ValueError("camera_res must contain height and width.")
        for name, value, minimum in (
            ("view_resolution", view_resolution, 2),
            ("seed", seed, 0),
            ("camera height", camera_res[0], 2),
            ("camera width", camera_res[1], 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if camera_res[0] != camera_res[1]:
            raise ValueError(
                "camera_res must be square to give equal camera fields of view."
            )
        if signal_mode not in ("intensity", "pale_yellow"):
            raise ValueError("signal_mode must be 'intensity' or 'pale_yellow'.")
        with open(geometry_path) as f:
            geometry = json.load(f)
        with open(registration_path) as f:
            registration = yaml.safe_load(f)
        positions = np.asarray(geometry["position"], dtype=float)
        directions = np.asarray(geometry["direction"], dtype=float)
        eyes = np.asarray(geometry["eye"])
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or directions.shape != positions.shape
            or eyes.shape != (len(positions),)
            or set(eyes) != {"left", "right"}
            or not np.isfinite(positions).all()
            or not np.isfinite(directions).all()
            or np.any(np.linalg.norm(directions, axis=1) == 0)
        ):
            raise ValueError("Invalid compound-eye geometry arrays.")
        source_to_mm = float(registration["source_to_mm"])
        translation = np.asarray(registration["translation"], dtype=float)
        pitch = float(registration["pitch_deg"])
        if (
            not np.isfinite(source_to_mm)
            or source_to_mm <= 0
            or translation.shape != (3,)
            or not np.isfinite(translation).all()
            or not np.isfinite(pitch)
        ):
            raise ValueError("Invalid compound-eye registration.")
        rotation = Rotation.from_euler("y", pitch, degrees=True).as_matrix()
        self.positions = positions @ rotation.T * source_to_mm + translation
        self.directions = directions @ rotation.T
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.eye_indices = tuple(
            np.flatnonzero(eyes == side) for side in ("left", "right")
        )
        self.num_ommatidia_per_eye = tuple(len(ix) for ix in self.eye_indices)
        shape = (2, max(self.num_ommatidia_per_eye))
        self.hex_coordinates: np.ndarray | None = None
        self.neighbor_indices: np.ndarray | None = None
        self.neighbor_angles: np.ndarray | None = None
        if lattice_path is not None:
            self._load_lattice(lattice_path)
        self.source_indices = np.full(shape, -1, dtype=int)
        for i, indices in enumerate(self.eye_indices):
            self.source_indices[i, : len(indices)] = indices
        self.valid_mask = self.source_indices >= 0
        if pale_type_mask is None:
            pale_type_mask = np.random.default_rng(seed).random(len(positions)) < 0.3
        pale_type_mask = np.asarray(pale_type_mask)
        if (
            pale_type_mask.shape != (len(positions),)
            or not np.isin(pale_type_mask, [0, 1]).all()
        ):
            raise ValueError(
                "pale_type_mask must contain one binary value per source row."
            )
        self.pale_type_mask = pale_type_mask.astype(bool)
        self.acceptance_angle = float(acceptance_angle)
        self.camera_res = tuple(int(x) for x in camera_res)
        self.view_resolution = int(view_resolution)
        self.signal_mode = signal_mode
        camera = registration.get("camera", {})
        self.fovy = float(camera.get("fovy", 157))
        forward = np.asarray(camera.get("forward", [0, 1, 0]), dtype=float)
        up = np.asarray(camera.get("up", [0, 0, 1]), dtype=float)
        if (
            not 0 < self.fovy < 180
            or forward.shape != (3,)
            or up.shape != (3,)
            or not np.isfinite([forward, up]).all()
            or np.linalg.norm(forward) == 0
        ):
            raise ValueError("Invalid compound-eye camera configuration.")
        forward /= np.linalg.norm(forward)
        up -= (up @ forward) * forward
        if np.linalg.norm(up) == 0:
            raise ValueError("Camera up must not be parallel to its viewing direction.")
        up /= np.linalg.norm(up)
        self.camera_directions = np.array([forward, forward * [1, -1, 1]])
        self.camera_up_vectors = np.array([up, up * [1, -1, 1]])
        right = np.cross(self.camera_directions, self.camera_up_vectors)
        tangent = np.tan(np.deg2rad(self.fovy / 2))
        for side, indices in enumerate(self.eye_indices):
            axes = self.directions[indices]
            depth = axes @ self.camera_directions[side]
            self.valid_mask[side, : len(indices)] = (
                (depth > 0)
                & (np.abs(axes @ right[side]) <= depth * tangent)
                & (np.abs(axes @ self.camera_up_vectors[side]) <= depth * tangent)
            )

    def _load_lattice(self, path: PathLike) -> None:
        with open(path) as f:
            lattice = json.load(f)
        if lattice["index_base"] != 0:
            raise ValueError("Lattice neighbor indices must be zero-based.")
        shape = (2, max(self.num_ommatidia_per_eye))
        self.hex_coordinates = np.zeros((*shape, 2), dtype=int)
        self.neighbor_indices = np.full((*shape, 6), -1, dtype=int)
        self.neighbor_angles = np.full((*shape, 6), np.nan)
        ordered_indices = []
        for side_id, side in enumerate(("left", "right")):
            coordinates = np.asarray(lattice[side]["coordinates"], dtype=float)
            neighbors = np.asarray(lattice[side]["neighbors"], dtype=float)
            angles = np.asarray(lattice[side]["neighbor_angles"], dtype=float)
            n = self.num_ommatidia_per_eye[side_id]
            present = ~np.isnan(neighbors)
            if (
                coordinates.shape != (n, 2)
                or not np.isfinite(coordinates).all()
                or np.any(coordinates != np.floor(coordinates))
                or len(np.unique(coordinates, axis=0)) != n
                or neighbors.shape != (n, 6)
                or not np.isfinite(neighbors[present]).all()
                or np.any(neighbors[present] != np.floor(neighbors[present]))
                or np.any((neighbors[present] < 0) | (neighbors[present] >= n))
                or angles.shape != (n, 6)
                or not np.array_equal(np.isnan(angles), ~present)
                or not np.isfinite(angles[present]).all()
                or np.any((angles[present] < 0) | (angles[present] > np.pi))
            ):
                raise ValueError(f"Invalid {side} compound-eye lattice arrays.")
            p, q = coordinates.T
            order = np.lexsort((p - q, -(p + q)))
            inverse = np.empty(n, dtype=int)
            inverse[order] = np.arange(n)
            neighbors = np.where(present, neighbors, 0).astype(int)
            self.hex_coordinates[side_id, :n] = coordinates[order]
            self.neighbor_indices[side_id, :n] = np.where(
                present, inverse[neighbors], -1
            )[order]
            self.neighbor_angles[side_id, :n] = angles[order]
            ordered_indices.append(self.eye_indices[side_id][order])
        self.eye_indices = tuple(ordered_indices)

    @property
    def optical_centers(self) -> np.ndarray:
        """Return the common camera origin for each eye.

        Returns:
            Left/right centers in the head frame, (2, 3), in mm.
        """
        positions, directions = self.positions.copy(), self.directions.copy()
        positions[self.eye_indices[1], 1] *= -1
        directions[self.eye_indices[1], 1] *= -1
        centroid = positions.mean(axis=0)
        projection = np.eye(3) - directions[:, :, None] * directions[:, None, :]
        offset = np.linalg.lstsq(
            projection.sum(axis=0),
            np.einsum("nij,nj->i", projection, positions - centroid),
            rcond=None,
        )[0]
        center = centroid + offset
        return np.array([center, center * [1, -1, 1]])

    @cached_property
    def pixel_directions(self) -> np.ndarray:
        """Head-local directions of equal-area fisheye pixels, (2, R, R, 3)."""
        coordinates = (
            np.arange(self.view_resolution) + 0.5
        ) * 2 / self.view_resolution - 1
        u, v = np.meshgrid(coordinates * np.sqrt(2), -coordinates * np.sqrt(2))
        radius_squared = u**2 + v**2
        scale = np.sqrt(1 - radius_squared / 4)
        right = np.cross(self.camera_directions, self.camera_up_vectors)
        directions = (
            (1 - radius_squared / 2)[None, ..., None]
            * self.camera_directions[:, None, None]
            + (u * scale)[None, ..., None] * right[:, None, None]
            + (v * scale)[None, ..., None] * self.camera_up_vectors[:, None, None]
        )
        return directions

    @cached_property
    def _image_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        directions = self.pixel_directions[0]
        forward = directions @ self.camera_directions[0]
        u = directions @ np.cross(self.camera_directions[0], self.camera_up_vectors[0])
        v = directions @ self.camera_up_vectors[0]
        u = np.divide(u, forward, out=np.zeros_like(u), where=forward > 0)
        v = np.divide(v, forward, out=np.zeros_like(v), where=forward > 0)
        height, width = self.camera_res
        tangent = np.tan(np.deg2rad(self.fovy / 2))
        valid = (
            (forward > 0)
            & (np.abs(u) <= tangent * width / height)
            & (np.abs(v) <= tangent)
        )
        rows = np.clip((1 - v / tangent) * height / 2 - 0.5, 0, height - 1)
        columns = np.clip(width / 2 + u / tangent * height / 2 - 0.5, 0, width - 1)
        return rows, columns, valid

    @cached_property
    def pixel_solid_angles(self) -> np.ndarray:
        """Fisheye pixel solid angles, (R, R), in steradians; unseen pixels zero."""
        # The equal-area map has unit Jacobian; include boundary cells by center.
        return self._image_coordinates[2] * (8 / self.view_resolution**2)

    @cached_property
    def _integration(self) -> tuple[tuple[csr_array, csr_array], np.ndarray]:
        area = self.pixel_solid_angles.ravel()
        valid = np.flatnonzero(area)
        factor = 4 * np.log(2) / np.deg2rad(self.acceptance_angle) ** 2

        def sensitivity_mass(t):
            # Rescale the integral so quadrature resolves narrow Gaussians too.
            theta = t / np.sqrt(factor)
            return t * np.exp(-(t**2)) * np.sinc(theta / np.pi) if theta < np.pi else 0

        normalization = 2 * np.pi / factor * quad(sensitivity_mass, 0, np.inf)[0]
        matrices = []
        coverage = np.full(self.valid_mask.shape, np.nan)
        for side, indices in enumerate(self.eye_indices):
            pixels = self.pixel_directions[side].reshape(-1, 3)[valid]
            values = [np.empty(0, dtype=np.float32)]
            columns = [np.empty(0, dtype=np.int32)]
            indptr = [0]
            for row, axis in enumerate(self.directions[indices]):
                if not self.valid_mask[side, row]:
                    indptr.append(indptr[-1])
                    continue
                theta_squared = np.arccos(np.clip(pixels @ axis, -1, 1)) ** 2
                # Shift squared angles by their minimum to prevent underflow;
                # normalization cancels the common factor.
                weight = (
                    np.exp(-factor * (theta_squared - theta_squared.min()))
                    * area[valid]
                )
                coverage[side, row] = (
                    weight.sum() * np.exp(-factor * theta_squared.min()) / normalization
                )
                # Retain all nonzero float32 weights, without a tail cutoff.
                weight = (weight / weight.sum()).astype(np.float32)
                nonzero = np.flatnonzero(weight)
                values.append(weight[nonzero])
                columns.append(valid[nonzero].astype(np.int32))
                indptr.append(indptr[-1] + len(nonzero))
            matrices.append(
                csr_array(
                    (np.concatenate(values), np.concatenate(columns), np.array(indptr)),
                    shape=(len(indices), len(area)),
                )
            )
        return (matrices[0], matrices[1]), coverage

    @property
    def weights(self) -> tuple[csr_array, csr_array]:
        """Cached normalized Gaussian weights, (eye_count, R**2), per eye.

        Rows follow readout order; columns follow fisheye row/column order.
        """
        return self._integration[0]

    @property
    def acceptance_coverage(self) -> np.ndarray:
        """Estimated fraction of each full angular Gaussian inside the camera."""
        return self._integration[1]

    def correct_fisheye(self, image: np.ndarray) -> np.ndarray:
        """Resample an RGB camera image onto an equal-area fisheye image.

        Args:
            image: Native RGB image, (height, width, 3), in ``camera_res`` order.

        Returns:
            (np.ndarray):
                Float32 RGB image, (R, R, 3), in the original 0–255 range.
                Pixels outside the camera's field of view are zero.
        """
        if image.shape != (*self.camera_res, 3):
            raise ValueError("Expected an RGB image matching camera_res.")
        rows, columns, valid = self._image_coordinates
        i, j = rows.astype(int), columns.astype(int)
        i1 = np.minimum(i + 1, self.camera_res[0] - 1)
        j1 = np.minimum(j + 1, self.camera_res[1] - 1)
        a, b = (rows - i)[..., None], (columns - j)[..., None]
        result = (1 - a) * ((1 - b) * image[i, j] + b * image[i, j1]) + a * (
            (1 - b) * image[i1, j] + b * image[i1, j1]
        )
        result[~valid] = 0
        return result.astype(np.float32)

    def raw_images_to_readouts(self, images: np.ndarray) -> np.ndarray:
        """Apply the geometric fisheye remap and integrate angular Gaussians.

        Args:
            images: Native uint8 RGB images, (2, height, width, 3), left then right.

        Returns:
            (np.ndarray):
                Float32 readouts in 0–1, (2, max_count, 1 or 2), with NaN for
                out-of-view ommatidia and padding.
                Rows align with ``hex_coordinates``, ``neighbor_indices`` and
                ``source_indices``.
        """
        if images.shape != (2, *self.camera_res, 3):
            raise ValueError("Expected RGB images with shape (2, height, width, 3).")
        num_channels = 1 if self.signal_mode == "intensity" else 2
        result = np.full((*self.valid_mask.shape, num_channels), np.nan, np.float32)
        for side, indices in enumerate(self.eye_indices):
            pixels = self.correct_fisheye(images[side]).reshape(-1, 3)
            if self.signal_mode == "intensity":
                # Linearity permits averaging RGB before Gaussian integration.
                values = pixels.mean(axis=1, dtype=np.float32) / 255
                result[side, : len(indices), 0] = self.weights[side] @ values
            else:
                pale_types = self.pale_type_mask[indices].astype(int)
                for channel in (0, 1):
                    values = pixels[:, channel + 1].astype(np.float32) / 255
                    signals = self.weights[side] @ values
                    result[side, : len(indices), channel] = signals * (
                        pale_types == channel
                    )
        result[~self.valid_mask] = np.nan
        return result
