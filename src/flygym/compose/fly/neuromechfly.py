import warnings
from os import PathLike
from pathlib import Path
from typing import Any

from flygym import assets_dir
from flygym.anatomy import BodySegment, ALL_SEGMENT_NAMES
from flygym.compose.fly.base_fly import BaseFly, MeshType, GeomFittingOption
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
from flygym.utils.mjcf import GEOM_TYPES

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

    Note:
        The anatomically fused trochanterfemur segment is rendered as two separate,
        rigidly connected geoms -- a trochanter and a femur -- driven by the
        ``geoms:`` block of each trochanterfemur entry in ``rigging.yaml``. This is
        purely a geometry/mass subdivision of one body: no degree of freedom is
        added and the kinematic chain (including the downstream tibia) is unchanged.
        It requires split ``{leg}_trochanter.stl`` / ``{leg}_femur.stl`` meshes
        (femur authored with its origin at the trochanter-femur joint).
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

        def _add(name: str, source_stem: str) -> None:
            path = (mesh_dir / f"{source_stem}.stl").resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"Mesh file not found for '{name}': {path}"
                )
            y_sign = -1 if (mirror_left2right and name[0] == "r") else 1
            self.bodyseg_to_mjcfmesh[name] = self.mjcf_root.add_mesh(
                name=name,
                file=str(path),
                scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
            )

        for segment_name in ALL_SEGMENT_NAMES:
            mirror = mirror_left2right and segment_name[0] == "r"

            # The (anatomically fused) trochanterfemur is always rendered as two
            # separate rigid geoms -- a trochanter and a femur. This requires split
            # meshes ({leg}_trochanter.stl, {leg}_femur.stl); the fused
            # trochanterfemur mesh is no longer used.
            if segment_name.endswith("_trochanterfemur"):
                leg = segment_name.split("_")[0]
                for piece in ("trochanter", "femur"):
                    name = f"{leg}_{piece}"
                    source = f"l{leg[1:]}_{piece}" if mirror else name
                    _add(name, source)
                continue

            base_stem = f"l{segment_name[1:]}" if mirror else segment_name
            _add(segment_name, base_stem)

    def _add_one_body_and_geoms(
        self,
        parent_body: Any,
        segment: BodySegment,
        my_rigging_config: dict[str, Any],
        geom_group: int,
    ) -> tuple[Any, list[Any]]:
        """Add a body and its geom(s).

        If the rigging config for this segment carries a ``geoms:`` block, the body
        is built with those multiple rigidly attached geoms -- this is how the
        trochanterfemur is rendered as a separate trochanter and femur. Otherwise a
        single geom is added by the base implementation (all other segments). Either
        way the body pose, joints and children are unchanged, so the kinematic chain
        is identical.
        """
        geoms_config = my_rigging_config.get("geoms")
        if not geoms_config:
            return super()._add_one_body_and_geoms(
                parent_body, segment, my_rigging_config, geom_group
            )

        body_element = parent_body.add_body(
            name=segment.name,
            pos=my_rigging_config["pos"],
            quat=my_rigging_config["quat"],
        )
        geom_elements = []
        for geom_name, geom_config in geoms_config.items():
            geom_elements.append(
                body_element.add_geom(
                    name=geom_name,
                    type=GEOM_TYPES["mesh"],
                    meshname=geom_config["mesh"],
                    mass=geom_config["mass"],
                    pos=geom_config.get("pos", [0.0, 0.0, 0.0]),
                    quat=geom_config.get("quat", [1.0, 0.0, 0.0, 0.0]),
                    contype=0,
                    conaffinity=0,
                    group=geom_group,
                )
            )
        return body_element, geom_elements

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
