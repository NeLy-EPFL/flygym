import warnings
from os import PathLike
from typing import Any

from flygym import assets_dir
from flygym.anatomy import BodySegment
from flygym.compose.fly.base_fly import BaseFly, MeshType, GeomFittingOption

__all__ = ["NeuroMechFly", "Fly"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/neuromechfly/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/neuromechfly/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/neuromechfly/meshes/"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/neuromechfly/visuals.yaml"
DEFAULT_VISION_CONFIG_PATH = assets_dir / "model/neuromechfly/vision.yaml"


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
