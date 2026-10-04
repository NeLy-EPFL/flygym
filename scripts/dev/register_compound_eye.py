"""Reproduce the symmetric µCT rigid registration against NeuroMechFly eyes.

Run from the repository: python scripts/dev/register_compound_eye.py
Prints the fit and residuals; does not overwrite the bundled configuration.
Requires the dev extra. The bundled registration YAML records provenance and units.
"""

import argparse
import json

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from flygym import assets_dir
from flygym.compose.fly import NeuroMechFly
from flygym.vision.compound_eye import CompoundEye


def fit_camera() -> None:
    """Fit symmetric camera frames, then pitch down to retain lower optical axes."""
    eye = CompoundEye()
    axes = eye.directions.copy()
    axes[eye.eye_indices[1], 1] *= -1
    forward = axes.mean(axis=0)
    forward /= np.linalg.norm(forward)
    # The convex maximum-margin hemisphere fit provides a feasible starting frame.
    hemisphere = minimize(
        lambda x: -x[3],
        np.r_[forward, (axes @ forward).min()],
        jac=lambda x: np.array([0.0, 0, 0, -1]),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: axes @ x[:3] - x[3],
                "jac": lambda x: np.column_stack((axes, -np.ones(len(axes)))),
            },
            {
                "type": "ineq",
                "fun": lambda x: 1 - x[:3] @ x[:3],
                "jac": lambda x: np.r_[-2 * x[:3], 0.0],
            },
        ],
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    if not hemisphere.success or hemisphere.x[3] <= 0:
        raise RuntimeError("Optical axes do not fit in a forward hemisphere.")
    forward = hemisphere.x[:3] / np.linalg.norm(hemisphere.x[:3])
    up = np.array([0.0, 0, 1]) - forward[2] * forward
    up /= np.linalg.norm(up)
    initial = Rotation.from_matrix(
        np.column_stack((np.cross(forward, up), up, -forward))
    )

    def area(rotvec):
        frame = Rotation.from_rotvec(rotvec).as_matrix()
        depth = -axes @ frame[:, 2]
        if np.any(depth <= 0):
            return np.inf
        bounds = np.abs(axes @ frame[:, :2] / depth[:, None]).max(axis=0)
        return np.log(bounds.prod())

    fit = minimize(
        area,
        initial.as_rotvec(),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-9},
    )
    if not fit.success:
        raise RuntimeError(f"Camera fit did not converge: {fit.message}")
    frame = Rotation.from_rotvec(fit.x).as_matrix()
    if frame[2, 1] < 0:
        frame[:, :2] *= -1
    bounds = np.abs(axes @ frame[:, :2] / (-axes @ frame[:, 2])[:, None]).max(axis=0)
    fovy = 168.04
    forward, up = -frame[:, 2], frame[:, 1]
    lowest = np.rad2deg(np.arctan2(axes @ up, axes @ forward)).min()
    pitch_deg = round(-fovy / 2 - lowest, 6)
    pitch = np.deg2rad(pitch_deg)
    forward, up = (
        forward * np.cos(pitch) - up * np.sin(pitch),
        forward * np.sin(pitch) + up * np.cos(pitch),
    )
    print("camera:")
    print(f"  forward: {forward.tolist()}")
    print(f"  up: {up.tolist()}")
    print(f"  fovy: {fovy}")
    print(f"Initial horizontal/vertical bounds: {np.rad2deg(2 * np.arctan(bounds))}")
    print(f"Downward pitch: {pitch_deg:.6f} degrees")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-only", action="store_true")
    if parser.parse_args().camera_only:
        fit_camera()
        return
    model_dir = assets_dir / "model/neuromechfly"
    with open(model_dir / "vision/ommatidia_microct.json") as f:
        source = json.load(f)
    positions = np.asarray(source["position"]) * 0.001
    # Reflect the right eye onto the left only for symmetric distance queries.
    positions[np.asarray(source["eye"]) == "right", 1] *= -1
    mesh = trimesh.load_mesh(model_dir / "meshes/simplified_max2000faces/l_eye.stl")
    with open(model_dir / "rigging.yaml") as f:
        target_eye = yaml.safe_load(f)["l_eye"]
    target_origin = np.asarray(target_eye["pos"])
    # Remove midsagittal rigging roundoff (~1e-9 mm) to enforce symmetry.
    target_origin[1] = 0
    target_rotation = Rotation.from_quat(target_eye["quat"], scalar_first=True)
    mesh.vertices = (
        target_rotation.apply(mesh.vertices * NeuroMechFly.SCALE) + target_origin
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        pitch, x, z = parameters
        rotation = Rotation.from_euler("y", pitch).as_matrix()
        transformed = positions @ rotation.T + [x, 0, z]
        return trimesh.proximity.closest_point(mesh, transformed)[1]

    initial = np.array([0, target_origin[0], target_origin[2]])
    # Try multiple pitches because the faceted mesh can have local minima.
    candidates = [
        least_squares(
            residual,
            [pitch, initial[1], initial[2]],
            bounds=([-0.35, 0.1, -0.1], [0.35, 0.4, 0.1]),
            diff_step=1e-4,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=500,
        )
        for pitch in (-0.15, -0.075, 0.0, 0.075, 0.15)
    ]
    result = min(candidates, key=lambda fit: np.sum(fit.fun**2))
    if not result.success:
        raise RuntimeError(f"Registration did not converge: {result.message}")
    print(f"pitch_deg: {np.rad2deg(result.x[0]):.6f}")
    print(f"translation: [{result.x[1]:.8f}, 0.0, {result.x[2]:.8f}]")
    for name, distances in (
        ("initial", residual(initial)),
        ("rigid", result.fun),
    ):
        print(
            f"{name}: RMS {np.sqrt(np.mean(distances**2)) * 1000:.3f} µm; "
            f"median/p95/max {np.quantile(distances, [0.5, 0.95, 1]) * 1000} µm"
        )
    directions = np.asarray(source["direction"]).copy()
    directions[np.asarray(source["eye"]) == "right", 1] *= -1
    rotation = Rotation.from_euler("y", result.x[0])
    transformed = rotation.apply(positions) + [
        result.x[1],
        0,
        result.x[2],
    ]
    _, distances, faces = trimesh.proximity.closest_point(mesh, transformed)
    dot = np.sum(mesh.face_normals[faces] * rotation.apply(directions), axis=1)
    for side in ("left", "right"):
        mask = np.asarray(source["eye"]) == side
        angles = np.rad2deg(np.arccos(np.clip(dot[mask], -1, 1)))
        print(
            f"{side}: RMS {np.sqrt(np.mean(distances[mask] ** 2)) * 1000:.3f} µm; "
            f"axes outward {np.mean(dot[mask] > 0) * 100:.2f}%; "
            f"axis/mesh-normal angle median/p95 {np.quantile(angles, [0.5, 0.95])} degrees"
        )
    print("Mapped anterior/left/dorsal unit axes (rows):")
    print(rotation.apply(np.eye(3)))
    print(f"Rotation determinant: {np.linalg.det(rotation.as_matrix()):.6f}")
    fit_camera()


if __name__ == "__main__":
    main()
