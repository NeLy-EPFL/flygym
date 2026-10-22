import warnings
from os import PathLike
from pathlib import Path
from typing import Any

from flygym import assets_dir
from flygym.anatomy import (
    BodySegment,
    ALL_SEGMENT_NAMES,
    ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR,
    LEG_LINKS_SEPARATE_TROCHANTERFEMUR,
    SeparateTrochanterFemurJointPreset,
    SeparateTrochanterFemurContactBodiesPreset,
    separate_trochanterfemur_in_dof_name,
)
from flygym.compose.fly.base_fly import BaseFly, MeshType, GeomFittingOption
from flygym.compose.pose import KinematicPose, KinematicPosePreset
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir

__all__ = ["NeuroMechFly", "NeuroMechFlyWithSeparateTrochanterFemur", "Fly"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/neuromechfly/rigging.yaml"
SEPARATE_TROCHANTERFEMUR_RIGGING_CONFIG_PATH = (
    assets_dir / "model/neuromechfly/rigging_separate_trochanterfemur.yaml"
)
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/neuromechfly/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/neuromechfly/meshes/"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/neuromechfly/visuals.yaml"
DEFAULT_VISION_CONFIG_PATH = assets_dir / "model/neuromechfly/vision.yaml"

# Mesh path relative to the flygym_assets/ dir on the S3 bucket and local cache dir
NEUROMECHFLY_FULLSIZE_MESH_DIR = "neuromechfly_fullsize_meshes_20261021a"


class NeuroMechFly(BaseFly):
    """The NeuroMechFly body model for *Drosophila melanogaster*.

    NeuroMechFly is derived from a micro-CT scan of a real fly. It is the
    default body model in FlyGym and supports all locomotion, sensorimotor,
    and vision experiments. For an alternative anatomically detailed model
    with wing and abdomen degrees of freedom, see `FlyBody`.

    Both `NeuroMechFly` and `FlyBody` inherit from `BaseFly` and expose the
    same composition API, so they can be used interchangeably.

    See `NeuroMechFlyWithSeparateTrochanterFemur` for a variant of this model
    where the trochanter and femur are separate body segments (instead of fused
    into a single segment). The user can still effectively fuse the two by not
    adding any joint between them (which is the case by default). When MuJoCo
    compiler parameter `fusestatic` is set to true, the two segment will
    actually be fused at compile time. Tentatively,
    `NeuroMechFlyWithSeparateTrochanterFemur` will fully replace `NeuroMechFly`
    in FlyGym 3.0.

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
        _add_neuromechfly_mesh_assets(
            self, ALL_SEGMENT_NAMES, mesh_basedir, mirror_left2right, mesh_type
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


class NeuroMechFlyWithSeparateTrochanterFemur(BaseFly):
    """NeuroMechFly with the trochanter and the femur as separate body segments.

    In `NeuroMechFly`, the trochanter and the femur form a single `trochanterfemur`
    segment. Here, they are separate segments (e.g. `lf_trochanter` and `lf_femur`),
    so that each has its own body and geom (e.g. for segmentation maps). Use
    `SeparateTrochanterFemurJointPreset` and
    `SeparateTrochanterFemurContactBodiesPreset` instead of `JointPreset` and
    `ContactBodiesPreset` with this model.

    The two are often considered to be fused in real flies (at least in Lobato-Rios et
    al., 2022), so biological joint presets add no DoF between them and the kinematics
    are the same as in `NeuroMechFly`. The physics is close but not bit-for-bit
    identical, because the mass distribution of the two segments is derived from
    separate meshes. Joint DoFs of the fused anatomy are renamed: the
    coxa-trochanterfemur joint becomes the coxa-trochanter joint, and the
    trochanterfemur-tibia joint becomes the femur-tibia joint (see
    `flygym.anatomy.separate_trochanterfemur_in_dof_name`). Poses defined for the fused
    anatomy, such as `KinematicPosePreset.NEUTRAL`, are renamed automatically.

    The user can still effectively fuse the two segments by not adding any joint between
    them (which is the case by default). When MuJoCo compiler parameter `fusestatic`
    is set to true, the two segment will actually be fused at compile time.

    This model will _tentatively_ replace `NeuroMechFly` in FlyGym 3.0.

    Args:
        Same as `NeuroMechFly`.
    """

    LEG_LINKS = LEG_LINKS_SEPARATE_TROCHANTERFEMUR
    CONTACT_BODIES_PRESET_CLASS = SeparateTrochanterFemurContactBodiesPreset

    def __init__(
        self,
        name: str = "nmf",
        *,
        rigging_config_path: PathLike = SEPARATE_TROCHANTERFEMUR_RIGGING_CONFIG_PATH,
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

    def get_pose_lookup(
        self, neutral_pose: KinematicPose | KinematicPosePreset | None
    ) -> dict[str, float]:
        """Same as `BaseFly.get_pose_lookup`, but DoF names of the fused
        trochanterfemur are renamed to their equivalents in this model."""
        lookup = super().get_pose_lookup(neutral_pose)
        return {separate_trochanterfemur_in_dof_name(k): v for k, v in lookup.items()}

    def colorize(
        self, visuals_config_path: PathLike = DEFAULT_VISUALS_CONFIG_PATH
    ) -> None:
        """Same as `NeuroMechFly.colorize`."""
        super().colorize(visuals_config_path)

    def _all_possible_joint_preset(self):
        return SeparateTrochanterFemurJointPreset.ALL_POSSIBLE

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
    ) -> None:
        _add_neuromechfly_mesh_assets(
            self,
            ALL_SEGMENT_NAMES_SEPARATE_TROCHANTERFEMUR,
            mesh_basedir,
            mirror_left2right,
            mesh_type,
        )


def _add_neuromechfly_mesh_assets(
    fly: BaseFly,
    segment_names: list[str],
    mesh_basedir: PathLike,
    mirror_left2right: bool,
    mesh_type: MeshType,
) -> None:
    # Simplified meshes are bundled with the package; fullsize meshes are
    # downloaded from S3 and cached on first use.
    if mesh_type == MeshType.FULLSIZE:
        mesh_dir = lazy_load_asset_dir(NEUROMECHFLY_FULLSIZE_MESH_DIR)
    else:
        mesh_dir = Path(mesh_basedir) / mesh_type.value

    for segment_name in segment_names:
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

        fly.bodyseg_to_mjcfmesh[segment_name] = fly.mjcf_root.add_mesh(
            name=segment_name,
            file=str(mesh_path),
            scale=(fly.SCALE, y_sign * fly.SCALE, fly.SCALE),
        )


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
