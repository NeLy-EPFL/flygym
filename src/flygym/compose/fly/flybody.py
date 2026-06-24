import warnings
from os import PathLike
from fnmatch import filter as filter_with_wildcard
from typing import Iterable, Any

import mujoco as mj
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from flygym import assets_dir
from flygym.anatomy import JointDOF, Skeleton, LEGS
from flygym.flybody.anatomy_flybody import (
    FLYBODY_ALL_SEGMENT_NAMES,
    FLYBODY_LEG_LINKS,
    FlyBodyJointPreset,
    FlyBodySkeleton,
    FlyBodyBodySegment,
    FlyBodyRotationAxis,
    WingFlyBodyRotationAxis,
    FlyBodyJointDOF,
    FlyBodyAxisOrder,
    FlyBodyContactBodiesPreset,
)
from flygym.compose.fly.base_fly import (
    BaseFly,
    ActuatorType,
    MeshType,
    GeomFittingOption,
)
from flygym.compose.pose import KinematicPose, KinematicPosePreset
from flygym.utils.mjcf import add_actuator, GEOM_TYPES, JOINT_TYPES
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir

__all__ = ["FlyBody"]


FLYBODY_RIGGING_CONFIG_PATH = assets_dir / "model/flybody/rigging.yaml"
FLYBODY_MUJOCO_GLOBALS_PATH = assets_dir / "model/flybody/mujoco_globals.yaml"
FLYBODY_MESH_DIR = assets_dir / "model/flybody/meshes/"
FLYBODY_VISUALS_CONFIG_PATH = assets_dir / "model/flybody/visuals.yaml"
FLYBODY_ALL_GEOM_SUFFIXES_PATH = assets_dir / "model/flybody/all_geom_suffixes.yaml"
FLYBODY_JOINT_CONFIG_PATH = assets_dir / "model/flybody/joints.yaml"
FLYBODY_ACTUATOR_CONFIG_PATH = assets_dir / "model/flybody/actuators.yaml"
FLYBODY_DEFAULT_VISION_CONFIG_PATH = assets_dir / "model/flybody/vision.yaml"

# Mesh path relative to the flygym_assets/ dir on the S3 bucket and local cache dir
FLYBODY_FULLSIZE_MESH_DIR = "flybody_fullsize_meshes_20260623a"


class FlyBody(BaseFly):
    """The FlyBody body model for *Drosophila melanogaster*.

    FlyBody is published in:

        Vaxenburg, R., et al. (2025). A whole-body model of *Drosophila* with
        precise neuromuscular connectivity. *Nature*.
        https://doi.org/10.1038/s41586-025-09029-4

    Source code for the original model: https://github.com/TuragaLab/flybody

    FlyBody provides an anatomically detailed model from the Turaga lab with
    wing and abdomen degrees of freedom and biomechanically grounded joint
    parameters, and integrates with the FlyGym API for scene composition,
    simulation, and control.

    !!! warning

        Support for the FlyBody body model is **experimental**. The API may
        change in future releases, and not all features available for the
        default `NeuroMechFly` model are currently supported.

    Both `FlyBody` and `NeuroMechFly` inherit from `BaseFly` and expose the
    same composition API, so they can be used interchangeably for locomotion
    experiments. FlyBody additionally provides:

    - Wing degrees of freedom (pitch, roll, yaw per wing).
    - Abdomen degrees of freedom (pitch and yaw via tendon actuators).
    - Biomechanically calibrated joint stiffness, damping, and ctrl-range
      parameters loaded from per-joint YAML config files.
    - Tendon-coupled tarsus and abdomen joints via `add_tendons()`.

    The FlyBody XML is already in mm units; mesh files use ``.obj`` format.

    Args:
        name: Identifier for this fly instance. Defaults to ``"flybody"``.
        rigging_config_path: Path to YAML file defining body segment positions,
            orientations, and masses.
        mesh_basedir: Directory containing OBJ mesh files for body segments.
        mujoco_globals_path: Path to YAML file with global MuJoCo parameters.
        all_geom_suffixes_path: Path to YAML file listing per-segment geometry
            name suffixes (FlyBody uses multiple geoms per body segment).
        mirror_left2right: If True, mirror left-side meshes for the right side.
            Defaults to ``False`` (FlyBody ships separate left/right meshes).
        root_segment: Root body segment for the kinematic tree.
        mesh_type: Mesh resolution to use.
        geom_fitting_option: How to fit collision geometries.
        joint_config_path: Path to YAML file with per-joint parameters (stiffness,
            damping, range).
        actuator_config_path: Path to YAML file with per-actuator gain parameters.
        vision_config_path: Path to YAML file with vision sensor configuration.
    """

    # The parsed flybody assets are already in mm (FlyGym's convention), so the
    # composer applies no further scaling (SCALE = 1.0). The original flybody XML is
    # authored in cm; parse_flybody.py converts it to mm -- lengths x10, gravity x10
    # (-981 cm/s^2 -> -9810 mm/s^2), and (for the all-hinge skeleton) torque-valued
    # quantities x100: stiffness, damping, armature, forcerange, and actuator gains.
    # See units_mapping and parse_globals in parse_flybody.py.
    SCALE = 1.0

    BODY_SEGMENT_CLASS = FlyBodyBodySegment
    JOINT_DOF_CLASS = FlyBodyJointDOF
    AXIS_ORDER_CLASS = FlyBodyAxisOrder
    BASE_SKELETON_CLASS = FlyBodySkeleton
    CONTACT_BODIES_PRESET_CLASS = FlyBodyContactBodiesPreset
    LEG_LINKS = FLYBODY_LEG_LINKS

    def _all_possible_joint_preset(self):
        return FlyBodyJointPreset.ALL_POSSIBLE

    def __init__(
        self,
        name: str = "flybody",
        *,
        rigging_config_path: PathLike = FLYBODY_RIGGING_CONFIG_PATH,
        mesh_basedir: PathLike = FLYBODY_MESH_DIR,
        mujoco_globals_path: PathLike = FLYBODY_MUJOCO_GLOBALS_PATH,
        all_geom_suffixes_path: PathLike = FLYBODY_ALL_GEOM_SUFFIXES_PATH,
        mirror_left2right: bool = False,
        root_segment: FlyBodyBodySegment | str = "c_thorax",
        mesh_type: MeshType = MeshType.FULLSIZE,
        geom_fitting_option: GeomFittingOption = GeomFittingOption.UNMODIFIED,
        joint_config_path: PathLike = FLYBODY_JOINT_CONFIG_PATH,
        actuator_config_path: PathLike = FLYBODY_ACTUATOR_CONFIG_PATH,
        vision_config_path: PathLike = FLYBODY_DEFAULT_VISION_CONFIG_PATH,
    ) -> None:
        with open(all_geom_suffixes_path) as f:
            self.multi_geom_lookup = yaml.safe_load(f)

        with open(joint_config_path) as f:
            self.joint_config = yaml.safe_load(f)

        with open(actuator_config_path) as f:
            self.actuator_config = yaml.safe_load(f)

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

        self.jointdof_to_mjcftendon = {}
        self._correct_wing_default_pose()

    def _resolve_joint_params(self, joint_name: str) -> dict[str, Any]:
        """Resolve joint parameters from grouped or legacy joint config format."""
        if not isinstance(self.joint_config, dict):
            raise ValueError("Invalid joint config format: expected a dictionary.")

        # Backward compatibility: legacy flat format keyed by joint name.
        if "params" not in self.joint_config or "ranges" not in self.joint_config:
            if joint_name not in self.joint_config:
                raise ValueError(f"Joint {joint_name} not found in joint config.")
            return self.joint_config[joint_name].copy()

        resolved_params = {}

        for group_cfg in self.joint_config.get("params", {}).values():
            apply_to = group_cfg.get("apply_to", [])
            patterns = [apply_to] if isinstance(apply_to, str) else apply_to

            if any(filter_with_wildcard([joint_name], pattern) for pattern in patterns):
                resolved_params.update(
                    {k: v for k, v in group_cfg.items() if k != "apply_to"}
                )

        per_joint_cfg = self.joint_config.get("ranges", {}).get(joint_name, {})
        if isinstance(per_joint_cfg, str):
            resolved_params["range"] = per_joint_cfg
        elif isinstance(per_joint_cfg, dict):
            resolved_params.update(per_joint_cfg)

        if not resolved_params:
            warnings.warn(
                f"No parameters resolved for joint {joint_name} from joint config; "
                "using defaults."
            )
            resolved_params = {
                "range": [-180, 180],
                "stiffness": 0.01,
                "damping": 0.0005,
                "limited": True,
            }
        return resolved_params

    def _is_pitch(self, jointdof):
        if jointdof.child.is_wing():
            return jointdof.axis == WingFlyBodyRotationAxis.PITCH
        else:
            return jointdof.axis == FlyBodyRotationAxis.PITCH

    @staticmethod
    def _coerce_mjcf_value(value: Any) -> Any:
        """Convert YAML-loaded values to MJCF-friendly python types."""
        if isinstance(value, list):
            return tuple(FlyBody._coerce_mjcf_value(v) for v in value)
        if isinstance(value, tuple):
            return tuple(FlyBody._coerce_mjcf_value(v) for v in value)
        if not isinstance(value, str):
            return value

        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            return float(value)
        except ValueError:
            return value

    @classmethod
    def _normalize_mjcf_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        return {k: cls._coerce_mjcf_value(v) for k, v in params.items()}

    # PyMJCF joint attribute names that MjSpec exposes under different identifiers.
    _MJSPEC_JOINT_KEY_RENAMES = {
        "solreflimit": "solref_limit",
        "solimplimit": "solimp_limit",
        "solreffriction": "solref_friction",
        "solimpfriction": "solimp_friction",
    }

    @classmethod
    def _rename_mjspec_joint_keys(cls, params: dict[str, Any]) -> dict[str, Any]:
        return {cls._MJSPEC_JOINT_KEY_RENAMES.get(k, k): v for k, v in params.items()}

    def _add_one_body_and_geoms(
        self,
        parent_body: mj.MjsBody,
        segment: Any,
        my_rigging_config: dict[str, Any],
        geom_group: int,
    ) -> tuple[mj.MjsBody, list[mj.MjsGeom]]:

        body_element = parent_body.add_body(
            name=segment.name,
            pos=self._coerce_mjcf_value(my_rigging_config["pos"]),
            quat=self._coerce_mjcf_value(my_rigging_config["quat"]),
        )

        all_geom_elements = []
        for geom_name, geom_config in my_rigging_config["geoms"].items():
            # PyMJCF used `mesh=`; MjSpec's geom attribute is `meshname=`. Also coerce
            # YAML string values (pos/quat/...) to the numeric types MjSpec expects.
            geom_config = {
                ("meshname" if k == "mesh" else k): self._coerce_mjcf_value(v)
                for k, v in geom_config.items()
            }
            geom_element = body_element.add_geom(
                type=GEOM_TYPES["mesh"],
                name=geom_name,
                contype=0,  # contact pairs to be added explicitly later
                conaffinity=0,  # contact pairs to be added explicitly later
                group=geom_group,
                **geom_config,
            )
            all_geom_elements.append(geom_element)

        return body_element, all_geom_elements

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
    ) -> None:
        # FlyBody's high-resolution meshes are downloaded from S3 and cached on
        # first use; only the (default) fullsize set exists for this model.
        if mesh_type == MeshType.FULLSIZE:
            mesh_dir = lazy_load_asset_dir(FLYBODY_FULLSIZE_MESH_DIR)
        else:
            raise NotImplementedError(
                f"Mesh type {mesh_type} not supported for FlyBody; "
                "only FULLSIZE is available."
            )

        for segment_name in FLYBODY_ALL_SEGMENT_NAMES:
            if mirror_left2right and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1
            for suffix in self.multi_geom_lookup.get(mesh_to_use, [None]):
                if suffix:
                    mesh_name = f"{mesh_to_use}_{suffix}"
                else:
                    warnings.warn(
                        f"No mesh suffix found for segment {mesh_to_use}; "
                        "using segment name as mesh name."
                    )
                    mesh_name = mesh_to_use
                mesh_path = (mesh_dir / f"{mesh_name}.obj").resolve()
                if not mesh_path.exists():
                    raise FileNotFoundError(
                        f"Mesh file not found for segment {segment_name}: {mesh_path}"
                    )

                mesh = self.mjcf_root.add_mesh(
                    name=mesh_name,
                    file=str(mesh_path),
                    scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
                )
                if segment_name not in self.bodyseg_to_mjcfmesh:
                    self.bodyseg_to_mjcfmesh[segment_name] = [mesh]
                else:
                    self.bodyseg_to_mjcfmesh[segment_name].append(mesh)

        # add abdomen8 mesh (just a mesh connected to c_abdomen7)
        self.bodyseg_to_mjcfmesh["c_abdomen7"].append(
            self.mjcf_root.add_mesh(
                name="c_abdomen8_body",
                file=str(mesh_dir / "c_abdomen8_body.obj"),
                scale=(self.SCALE, self.SCALE, self.SCALE),
            )
        )

    def colorize(
        self, visuals_config_path: PathLike = FLYBODY_VISUALS_CONFIG_PATH
    ) -> None:
        """Apply colors and textures to the FlyBody model.

        Args:
            visuals_config_path: Path to the YAML file defining per-segment material
                and texture assignments. Defaults to the bundled FlyBody visuals.
        """
        super().colorize(visuals_config_path)

    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: KinematicPose | KinematicPosePreset | None = None,
        **kwargs: Any,
    ) -> dict[JointDOF, mj.MjsJoint]:
        """Add joints to the FlyBody model based on a skeleton definition.

        Like `BaseFly.add_joints`, but instead of taking uniform ``stiffness``,
        ``damping``, and ``armature`` arguments, the per-joint values are loaded from
        the flybody joint config (via ``_resolve_joint_params``). ``neutral_pose`` sets
        each joint's ``springref`` (the angle at which passive spring forces are zero),
        and ``kwargs`` override the config-derived values.

        Args:
            skeleton:
                Skeleton defining which joints to create and their DOFs.
            neutral_pose:
                Resting angles for joints. If provided, must match the skeleton's axis
                order; otherwise all neutral angles default to 0.
            **kwargs:
                Per-joint MJCF attributes that override the config-derived values. See
                the [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
                for supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF joint elements.
        """

        self.skeleton = skeleton
        neutral_pose_lookup = self.get_pose_lookup(neutral_pose)

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            joint_params = self._resolve_joint_params(jointdof.name)

            # Override default springref with neutral pose value if provided
            joint_params.update(
                {"springref": neutral_pose_lookup.get(jointdof.name, 0.0)}
            )
            # Override any joint config values with values provided in kwargs
            joint_params.update(kwargs)
            joint_params = self._normalize_mjcf_params(joint_params)
            joint_params = self._rename_mjspec_joint_keys(joint_params)

            return_dict[jointdof] = child_body.add_joint(
                name=jointdof.name,
                type=JOINT_TYPES["hinge"],
                axis=jointdof.axis.to_vector(),
                **joint_params,
            )

        self.jointdof_to_mjcfjoint.update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def translate_generaljointparams_to_specificjointparams_simplified(
        self, general_params: dict[str, Any], actuator_type: ActuatorType
    ) -> dict[str, Any]:
        if actuator_type == ActuatorType.POSITION:
            # gainprm comes from YAML as a string (e.g. '30') or list of strings;
            # coerce to a numeric type and use the first entry as kp (per MuJoCo,
            # kp is gainprm[0] for position actuators).
            gainprm = self._coerce_mjcf_value(general_params.get("gainprm", 1.0))
            kp = gainprm[0] if isinstance(gainprm, tuple) else gainprm
            return {
                "kp": kp,
            }
        else:
            return {}

    def add_actuators(
        self,
        jointdofs: Iterable[JointDOF],
        actuator_type: "ActuatorType | str",
        neutral_input: "dict[str, float] | KinematicPose | KinematicPosePreset | None" = None,
        *,
        forcelimited: bool = False,
        forcerange: tuple[float, float] = (-0.3, 0.3),
        **kwargs: Any,
    ) -> dict[JointDOF, mj.MjsActuator]:
        """Add actuators to specified joints.

        Like `BaseFly.add_actuators`, but actuator parameters default to the values in
        the flybody ``actuator_config.yaml`` unless overridden. The FlyBody defaults
        also differ: ``forcelimited`` is False and ``forcerange`` is ``(-0.3, 0.3)``.

        Args:
            jointdofs:
                Joint DOFs to actuate.
            actuator_type:
                Type of actuator (motor, position, velocity, etc.).
            neutral_input:
                Default actuator inputs. Accepts a ``dict`` mapping DoF names to
                values, a `KinematicPose`, or a `KinematicPosePreset`. If None,
                defaults to 0 for all actuators. For position actuators the values
                are joint angles and must match the skeleton axis order.
            forcelimited:
                If True, clamp actuator force to ``forcerange``. If False, fall back to
                the ``forcerange`` from ``actuator_config.yaml`` if one is specified.
            forcerange:
                Force limit as a (min, max) tuple.
            **kwargs:
                MJCF actuator attributes (e.g. ``kp`` for position actuators, ``kv`` for
                velocity actuators) that override the defaults from
                ``actuator_config.yaml``. See the
                [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)
                for supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF actuator elements.
        """
        actuator_type = ActuatorType(actuator_type)
        neutral_input = self._resolve_neutral_input(actuator_type, neutral_input)

        remove_ctrl_limits = False
        if (
            actuator_type == ActuatorType.MOTOR
            or actuator_type == ActuatorType.VELOCITY
        ):
            warnings.warn(
                "Setting ctrllimited=False for MOTOR and VELOCITY actuators; "
                "flybody ctrl limits are meant for POSITION actuators."
            )
            remove_ctrl_limits = True
            if not forcelimited:
                warnings.warn(
                    "Without ctrl limits, MOTOR and VELOCITY actuators can generate "
                    "extreme forces; consider setting forcelimited=True for stability."
                )

        return_dict = {}
        for jointdof in jointdofs:
            self.jointdof_to_neutralaction_by_type[actuator_type][jointdof] = (
                neutral_input.get(jointdof.name, 0.0)
            )

            default_actuator_params = {}
            for _, val in self.actuator_config.items():
                if jointdof.name in val["apply_to"]:
                    default_actuator_params = val["general"]
                    break
            if not default_actuator_params:
                warnings.warn(f"No actuator config found for joint {jointdof.name}.")

            default_actuator_params_specific = (
                self.translate_generaljointparams_to_specificjointparams_simplified(
                    default_actuator_params, actuator_type
                )
            )

            if remove_ctrl_limits:
                default_actuator_params_specific["ctrllimited"] = False
            else:
                # recover ctrllimits from joint limits
                jnt = self.jointdof_to_mjcfjoint[jointdof]
                assert jnt.range is not None, (
                    f"Joint {jointdof.name} must have range specified in order to use "
                    "default ctrlrange for its actuator."
                )
                default_actuator_params_specific["ctrlrange"] = jnt.range

            if forcelimited:
                default_actuator_params_specific["forcelimited"] = forcelimited
                default_actuator_params_specific["forcerange"] = forcerange

            if actuator_type == ActuatorType.POSITION:
                has_kp = "kp" in kwargs
                warning_str = "WARNING: actuator type is POSITION but "
                has_missing_param = False
                for param, has_it in [
                    ("kp", has_kp)
                ]:  # , ("kv", has_kv), ("timeconst", has_timeconst)]:
                    if not has_it:
                        warning_str += f"{param} not specified, using default value "
                        "from general actuator config if specified there, otherwise "
                        "using MuJoCo default. "
                        has_missing_param = True
                if has_missing_param:
                    warnings.warn(warning_str)

            elif actuator_type == ActuatorType.VELOCITY:
                has_kv = "kv" in kwargs
                warning_str = "WARNING: actuator type is VELOCITY but "
                if not has_kv:
                    warning_str += (
                        "kv not specified, using default value from general actuator "
                        "config if specified there, otherwise using MuJoCo default."
                    )
                    warnings.warn(warning_str)

            default_actuator_params_specific.update(kwargs)

            actuator = add_actuator(
                self.mjcf_root,
                actuator_type.value,
                name=f"{jointdof.name}-{actuator_type.value}",
                joint=jointdof.name,
                **default_actuator_params_specific,
            )

            return_dict[jointdof] = actuator
        self.jointdof_to_mjcfactuator_by_type[actuator_type].update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def add_leg_adhesion(
        self,
        gain: float | dict[str, float] = 0.985,
        add_labrum: bool = True,
        labrum_gain: float = 1.0,
    ) -> dict[str, mj.MjsActuator]:
        """Add adhesion actuators to the tarsus5 segments of all legs and optionally to the labrum.

        Adhesion actuators apply a normal attraction force, enabling the fly to grip
        surfaces. The control input per leg ranges from 0 (released) to 1 (full gain).

        Args:
            gain: Adhesion actuator gain. Either a single float applied to all legs,
                or a dict mapping leg position identifiers to per-leg gain values.
            add_labrum: Whether to also add an adhesion actuator for the labrum (mouthpart).
            labrum_gain: Adhesion actuator gain for the labrum (used when ``add_labrum``).

        Returns:
            Dict mapping leg position identifier to the created MJCF adhesion
            actuator element (same as ``self.leg_to_adhesionactuator``).

        Raises:
            ValueError: If adhesion actuators have already been added.
        """
        if len(self.leg_to_adhesionactuator) > 0:
            raise ValueError("Leg adhesion actuators have already been added.")
        for leg in LEGS:
            tarsus5_segment = FlyBodyBodySegment(f"{leg}_tarsus5")
            if isinstance(gain, dict):
                gain_this_leg = gain[leg]
            else:
                gain_this_leg = gain
            self.leg_to_adhesionactuator[leg] = add_actuator(
                self.mjcf_root,
                "adhesion",
                name=f"{tarsus5_segment.name}-adhesion",
                body=self.bodyseg_to_mjcfbody[tarsus5_segment].name,
                gain=gain_this_leg,
                ctrlrange=(0, 1),
            )
        if add_labrum:
            for s in "lr":
                labrum = FlyBodyBodySegment(f"{s}_labrum")
                self.leg_to_adhesionactuator[f"{s}_labrum"] = add_actuator(
                    self.mjcf_root,
                    "adhesion",
                    name=f"{labrum.name}-adhesion",
                    body=self.bodyseg_to_mjcfbody[labrum].name,
                    gain=labrum_gain,
                    ctrlrange=(0, 1),
                )
        return self.leg_to_adhesionactuator

    def add_tendons(
        self, coef: float | dict[str, float] | None = None
    ) -> dict[JointDOF, mj.MjsTendon]:
        # check joints have been added
        if len(self.jointdof_to_mjcfjoint) == 0:
            raise ValueError(
                "Must first add joints via `add_joints` before adding tendons."
            )
        if len(self.jointdof_to_mjcftendon) > 0:
            raise ValueError("Tendons have already been added.")

        def get_coef_for_joint(joint: JointDOF) -> float:
            if coef is None or (isinstance(coef, dict) and joint.name not in coef):
                return 1.0  # default in flybody
            elif isinstance(coef, dict):
                return coef[joint.name]
            else:
                return coef

        abd_bodyseg = FlyBodyBodySegment("c_abdomen1")
        if abd_bodyseg in self.skeleton.body_segments:
            tree = self.skeleton.get_tree()
            for axis in [FlyBodyRotationAxis.PITCH, FlyBodyRotationAxis.YAW]:
                tendon_name = f"abdomen_{axis.name.lower()}"
                tendon = self.mjcf_root.add_tendon(name=tendon_name)
                added_tendon = False
                for parent, child in tree.dfs_edges(abd_bodyseg):
                    joint = FlyBodyJointDOF(parent=parent, child=child, axis=axis)
                    tendon.wrap_joint(
                        self.jointdof_to_mjcfjoint[joint].name,
                        get_coef_for_joint(joint),
                    )
                    if not added_tendon:
                        self.jointdof_to_mjcftendon[joint] = tendon
                        added_tendon = True
        else:
            warnings.warn(
                "abdomen1 not found in skeleton; skipping abdomen tendon creation."
            )

        for leg in LEGS:
            tarsus_bodyseg = FlyBodyBodySegment(f"{leg}_tarsus1")
            if tarsus_bodyseg not in self.skeleton.body_segments:
                warnings.warn(
                    f"{tarsus_bodyseg} not found in skeleton; "
                    "skipping tendon creation for it."
                )
                continue
            else:
                joints = self.skeleton.iter_jointdofs(tarsus_bodyseg)
                first_joint = next(joints)
                tendon_name = f"{leg}_tarsus"
                tendon = self.mjcf_root.add_tendon(name=tendon_name)
                self.jointdof_to_mjcftendon[first_joint] = tendon
                coef_to_use = get_coef_for_joint(first_joint)
                tendon.wrap_joint(
                    self.jointdof_to_mjcfjoint[first_joint].name, coef_to_use
                )
                for joint in joints:
                    coef_to_use = get_coef_for_joint(joint)
                    tendon.wrap_joint(
                        self.jointdof_to_mjcfjoint[joint].name, coef_to_use
                    )

    def add_tendon_actuators(self, **kwargs: Any) -> dict[JointDOF, mj.MjsActuator]:
        if len(self.jointdof_to_mjcftendon) == 0:
            raise ValueError(
                "Must first add tendons via `add_tendons` "
                "before adding tendon actuators."
            )
        if len(self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON]) > 0:
            raise ValueError(
                "Tendon actuators have already been added, cannot add tendon actuators "
                "as MOTOR actuators are used for tendons in this implementation."
            )

        # Motor-actuator gains, in mm units (the original flybody XML, authored in cm,
        # uses affine-servo gains of 0.1 for the abdomen and 0.4 for the tarsus tendons;
        # the tendons drive hinge joints, so these torque-valued gains scale x100).
        ABDOMEN_TENDON_GAIN = 10.0
        TARSUS_TENDON_GAIN = 40.0
        for jointdof, tendon in self.jointdof_to_mjcftendon.items():
            default_params = {}
            if "abdomen" in jointdof.name:
                if jointdof.axis == FlyBodyRotationAxis.PITCH:
                    default_params = {
                        "ctrlrange": [-1.05, 0.7],
                        "gain": ABDOMEN_TENDON_GAIN,
                    }
                elif jointdof.axis == FlyBodyRotationAxis.YAW:
                    default_params = {
                        "ctrlrange": [-0.7, 0.7],
                        "gain": ABDOMEN_TENDON_GAIN,
                    }
                else:
                    warnings.warn(
                        "No default ctrlrange for abdomen tendon joint "
                        f"{jointdof.name} with axis {jointdof.axis}; using no defaults."
                    )
            elif "tarsus" in jointdof.name:
                default_params = {"ctrlrange": [-0.9, 0.9], "gain": TARSUS_TENDON_GAIN}
            else:
                warnings.warn(
                    f"No default tendon actuator params for joint {jointdof.name}; "
                    "using no defaults."
                )

            if jointdof.name in kwargs:
                warnings.warn(
                    "Overriding default tendon actuator params for "
                    f"joint {jointdof.name} with kwargs."
                )
                default_params.update(kwargs[jointdof.name])
            else:
                default_params.update(kwargs)

            actuator = add_actuator(
                self.mjcf_root,
                "general",
                name=f"{tendon.name}-tendon",
                tendon=tendon.name,
                **default_params,
            )

            self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON][jointdof] = (
                actuator
            )
            self.jointdof_to_neutralaction_by_type[ActuatorType.TENDON][jointdof] = 0.0

        return self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON]

    def _correct_wing_default_pose(self) -> None:
        """Rotate the wing bodies into their resting pose.

        In the flybody model the wings are held in place by the joint's spring
        property. With the original general actuators, a zero input applies no force,
        so the wings fall back to this resting pose. Position actuators do not behave
        this way, so we rotate the wing bodies into the resting pose explicitly here.
        """
        for side in ["l", "r"]:
            wing_bodyseg = FlyBodyBodySegment(f"{side}_wing")
            if wing_bodyseg in self.bodyseg_to_mjcfbody:
                mjcf_body = self.bodyseg_to_mjcfbody[wing_bodyseg]
                bquat = R.from_quat(mjcf_body.quat, scalar_first=True)
                correction_quat = R.from_euler(
                    "xyz", [0, 0, -90 if side == "r" else 90], degrees=True
                )
                new_quat = (correction_quat * bquat).as_quat(scalar_first=True)
                mjcf_body.quat = tuple(new_quat)
            else:
                raise ValueError(
                    f"Expected wing body segment {wing_bodyseg} not found in model, "
                    "cannot apply wing default pose correction."
                )
