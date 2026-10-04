import warnings
from os import PathLike
from pathlib import Path
from typing import Any, override

import numpy as np
from scipy.spatial.transform import Rotation

from flygym import assets_dir
from flygym.anatomy import BodySegment, ALL_SEGMENT_NAMES
from flygym.compose.fly.base_fly import BaseFly, MeshType, GeomFittingOption
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
from flygym.utils.mjcf import CAMERA_MODES, GEOM_TYPES
from flygym.vision.compound_eye import CompoundEye

__all__ = ["NeuroMechFly", "Fly"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/neuromechfly/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/neuromechfly/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/neuromechfly/meshes/"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/neuromechfly/visuals.yaml"
DEFAULT_VISION_CONFIG_PATH = assets_dir / "model/neuromechfly/vision.yaml"

# Mesh path relative to the flygym_assets/ dir on the S3 bucket and local cache dir
NEUROMECHFLY_FULLSIZE_MESH_DIR = "neuromechfly_fullsize_meshes_20260623a"


class NeuroMechFly(BaseFly):
    """The NeuroMechFly body model for *Drosophila melanogaster*.

    NeuroMechFly is derived from a micro-CT scan of a real fly. It is the
    default body model in FlyGym and supports all locomotion, sensorimotor,
    and vision experiments. For an alternative anatomically detailed model
    with wing and abdomen degrees of freedom, see `FlyBody`.

    Both `NeuroMechFly` and `FlyBody` inherit from `BaseFly` and expose the
    same composition API, so they can be used interchangeably.

    Args:
        name: Identifier for this fly instance. Defaults to ``"nmf"``.
        rigging_config_path: Path to YAML file defining body segment positions,
            orientations, and masses.
        mesh_basedir: Directory containing STL mesh files for body segments.
        mujoco_globals_path: Path to YAML file with global MuJoCo parameters.
        root_segment: Root body segment for the kinematic tree.
        mirror_left2right: If True, mirror left-side meshes for the right side.
        mesh_type: Mesh resolution to use.
        geom_fitting_option: How to fit collision geometries.
        vision_config_path: Path to YAML file with vision sensor configuration.
    """

    def __init__(
        self,
        name: str = "nmf",
        *,
        rigging_config_path: PathLike = DEFAULT_RIGGING_CONFIG_PATH,
        mesh_basedir: PathLike = DEFAULT_MESH_DIR,
        mujoco_globals_path: PathLike = DEFAULT_MUJOCO_GLOBALS_PATH,
        root_segment: BodySegment | str = "c_thorax",
        mirror_left2right: bool = True,
        mesh_type: MeshType = MeshType.SIMPLIFIED_MAX2000FACES,
        geom_fitting_option: GeomFittingOption = GeomFittingOption.UNMODIFIED,
        vision_config_path: PathLike = DEFAULT_VISION_CONFIG_PATH,
    ) -> None:
        self.compound_eye: CompoundEye | None = None
        self.vision_excludes_self = False
        super().__init__(
            name=name,
            rigging_config_path=rigging_config_path,
            mesh_basedir=mesh_basedir,
            mujoco_globals_path=mujoco_globals_path,
            root_segment=root_segment,
            mirror_left2right=mirror_left2right,
            mesh_type=mesh_type,
            geom_fitting_option=geom_fitting_option,
            vision_config_path=vision_config_path,
        )

    @override
    def add_vision(
        self,
        draw_sensor_markers: bool = False,
        *,
        compound_eye: CompoundEye | None = None,
        exclude_self: bool | None = None,
    ) -> None:
        """Attach fisheye cameras or an anatomical compound-eye sensor.

        Call before adding the fly to a world. Anatomical sensor frames follow
        the eye bodies; custom meshes or rigging require checking registration.

        Args:
            draw_sensor_markers: Show markers in ordinary renders. Anatomical
                eyes show every 16th optical axis, with a length of 0.08 mm.
            compound_eye: Anatomical sensor, or None for the existing Retina model.
            exclude_self: Hide the fly from its own visual input, retaining cast
                shadows and other flies. Defaults to True for anatomical vision
                and False for fisheye vision. Ordinary renders are unaffected.
        """
        if self.compound_eye is not None or self.eyecameraname_to_mjcfcamera:
            raise ValueError("Vision has already been added to this fly.")
        if exclude_self is not None and not isinstance(exclude_self, bool):
            raise TypeError("exclude_self must be a boolean or None.")
        if compound_eye is not None and not isinstance(compound_eye, CompoundEye):
            raise TypeError("compound_eye must be a CompoundEye instance.")
        self.vision_excludes_self = (
            compound_eye is not None if exclude_self is None else exclude_self
        )
        if compound_eye is None:
            return super().add_vision(draw_sensor_markers)
        centers = compound_eye.optical_centers
        for side_id, (side, indices) in enumerate(
            zip(("left", "right"), compound_eye.eye_indices)
        ):
            body = self.mjcf_root.body("l_eye" if side == "left" else "r_eye")
            inverse = Rotation.from_quat(body.quat, scalar_first=True).inv()
            origin = inverse.apply(centers[side_id] - body.pos)
            forward = compound_eye.camera_directions[side_id]
            up = compound_eye.camera_up_vectors[side_id]
            name = f"compound_eye_{side}"
            camera = body.add_camera(
                name=name,
                pos=origin,
                xyaxes=np.concatenate(
                    [inverse.apply(np.cross(forward, up)), inverse.apply(up)]
                ),
                fovy=compound_eye.fovy,
                mode=CAMERA_MODES["fixed"],
                # Keep the eye frame through static-body fusion. MuJoCo
                # 3.10 otherwise drops fixed-camera transforms when fusing.
                targetbody=body.name,
            )
            self.eyecameraname_to_mjcfcamera[name] = camera
            if draw_sensor_markers:
                for index in indices[::16]:
                    start = inverse.apply(compound_eye.positions[index] - body.pos)
                    end = start + inverse.apply(compound_eye.directions[index]) * 0.08
                    body.add_site(
                        name=f"compound_eye_axis_{index}",
                        type=GEOM_TYPES["capsule"],
                        fromto=np.concatenate([start, end]),
                        size=[0.002, 0, 0],
                        rgba=[0.1, 0.8, 0.3, 1]
                        if side == "left"
                        else [0.9, 0.3, 0.1, 1],
                        group=1,
                    )
        self.compound_eye = compound_eye

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
    ) -> None:
        # Simplified meshes are bundled with the package; fullsize meshes are
        # downloaded from S3 and cached on first use.
        if mesh_type == MeshType.FULLSIZE:
            mesh_dir = lazy_load_asset_dir(NEUROMECHFLY_FULLSIZE_MESH_DIR)
        else:
            mesh_dir = Path(mesh_basedir) / mesh_type.value

        for segment_name in ALL_SEGMENT_NAMES:
            if mirror_left2right and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1

            mesh_path = (mesh_dir / f"{mesh_to_use}.stl").resolve()
            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"Mesh file not found for segment {segment_name}: {mesh_path}"
                )

            self.bodyseg_to_mjcfmesh[segment_name] = self.mjcf_root.add_mesh(
                name=segment_name,
                file=str(mesh_path),
                scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
            )

    def colorize(
        self, visuals_config_path: PathLike = DEFAULT_VISUALS_CONFIG_PATH
    ) -> None:
        """Apply colors and textures to the NeuroMechFly model.

        Args:
            visuals_config_path: Path to the YAML file defining per-segment material
                and texture assignments. Defaults to the bundled NeuroMechFly visuals.
        """
        super().colorize(visuals_config_path)


class Fly(NeuroMechFly):
    """Deprecated alias for `NeuroMechFly`. Will be removed in a future release."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "`Fly` is deprecated and will be removed in a future release; "
            "use `NeuroMechFly` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
