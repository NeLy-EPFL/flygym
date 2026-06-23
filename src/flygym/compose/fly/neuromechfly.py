import warnings
from os import PathLike
from pathlib import Path
from typing import Any

from flygym import assets_dir
from flygym.anatomy import BodySegment, ALL_SEGMENT_NAMES
from flygym.compose.fly.base_fly import BaseFly, MeshType, GeomFittingOption
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir

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
