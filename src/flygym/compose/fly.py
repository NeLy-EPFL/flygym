from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from typing import Iterable, Any, override

import mujoco as mj
import numpy as np
import dm_control.mjcf as mjcf
import yaml

from flygym import assets_dir
from flygym.anatomy import (
    BodySegment,
    JointDOF,
    Skeleton,
    RotationAxis,
    AxisOrder,
    JointPreset,
    ALL_SEGMENT_NAMES,
    LEGS,
    LEG_LINKS,
    )

from flygym.assets.model.flybody.anatomy_flybody import (
    FLYBODY_ALL_SEGMENT_NAMES,
    FLYBODY_LEG_LINKS,
    FlybodyJointPreset,
    FlybodySkeleton,
    FlybodyBodySegment,
    FlybodyRotationAxis,
    WingFlybodyRotationAxis,
    FlybodyJointDOF,
    FlybodyAxisOrder,
    WingFlybodyAxisOrder,
)

from flygym.compose.base import BaseCompositionElement
from flygym.compose.pose import KinematicPose, KinematicPosePreset
from flygym.utils.mjcf import set_mujoco_globals
from flygym.utils.math import Vec3, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError

__all__ = ["Fly", "ActuatorType", "MeshType", "GeomFittingOption"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/meshes/"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/visuals.yaml"


class MeshType(Enum):
    """Mesh resolution to use for fly body geometry.

    Attributes:
        FULLSIZE: Original high-resolution meshes.
        SIMPLIFIED_MAX2000FACES: Simplified meshes with at most 2000 faces per
            segment. Faster to render and simulate. Used by default.
    """

    FULLSIZE = "fullsize"
    SIMPLIFIED_MAX2000FACES = "simplified_max2000faces"


class GeomFittingOption(Enum):
    """How to fit collision geometries to the mesh shapes.

    Attributes:
        UNMODIFIED: Keep the original mesh-based geometries.
        ALL_TO_CAPSULES: Replace all geometries with capsule approximations.
        CLAWS_TO_CAPSULES: Replace only tarsus5 (claw) geometries with capsules.
    """

    UNMODIFIED = "unmodified"
    ALL_TO_CAPSULES = "all_to_capsules"
    CLAWS_TO_CAPSULES = "claws_to_capsules"


class ActuatorType(Enum):
    """Actuator types supported by MuJoCo.
    See `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator>`_
    for details on each type."""

    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"
    TENDON = "tendon"


class Fly(BaseCompositionElement):
    """Represents a complete fly with body segments, joints, actuators, sensors, and
    cameras. The fly is built from mesh assets and configured via config files that
    define rigging (joint positions), visuals (colors/textures), and global MuJoCo
    parameters.

    The fly uses a hierarchical body structure with a root segment (typically the
    thorax) from which all other segments branch. Joints and actuators are added
    separately after initialization to allow flexible model configurations.

    Args:
        name:
            Identifier for this fly instance.
        rigging_config_path:
            Path to YAML file defining body segment positions, orientations, and masses.
        mesh_basedir:
            Directory containing STL mesh files for body segments.
        mujoco_globals_path:
            Path to YAML file with global MuJoCo parameters (timestep, gravity, etc.).
        root_segment:
            Root body segment for the kinematic tree (e.g., ``c_thorax``).
        mirror_left2right:
            If True, mirror left-side meshes for right side instead of loading separate
            mesh files. Reduces asset size and ensures symmetry.
        mesh_type:
            Mesh resolution to use.
        geom_fitting_option:
            How to fit collision geometries.

    Attributes:
        skeleton:
            Joint structure of the fly, set when add_joints() is called.
        bodyseg_to_mjcfmesh:
            Maps body segments to MJCF mesh elements.
        bodyseg_to_mjcfbody:
            Maps body segments to MJCF body elements.
        bodyseg_to_mjcfgeom:
            Maps body segments to MJCF geometry elements.
        jointdof_to_mjcfjoint:
            Maps joint DOFs to MJCF joint elements.
        jointdof_to_mjcfactuator_by_type:
            Maps actuator type to a further dictionary, which maps joint DOFs to MJCF
            actuator elements (only if the actuator exists).
        sensorname_to_mjcfsensor:
            Maps sensor names to MJCF sensor elements.
        cameraname_to_mjcfcamera:
            Maps camera names to MJCF camera elements.
        jointdof_to_neutralangle:
            Neutral (resting) angle for each joint DOF.
        jointdof_to_neutralaction_by_type:
            Neutral actuator input for each (actuator_type, joint_dof) pair. Maps
            actuator type to a further dictionary, which maps joint DOFs to their
            neutral actuator input (only if the actuator exists).
    """

    # For numerical reasons, we simulate length in mm, not m. This changes the units
    # of other quantities as well, for example acceleration is now in mm/s^2.    
    SCALE = 1000
    BODY_SEGMENT_CLASS = BodySegment
    JOINT_DOF_CLASS = JointDOF
    AXIS_ORDER_CLASS = AxisOrder
    BASE_SKELETON_CLASS = Skeleton
    LEG_LINKS = LEG_LINKS

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
    ) -> None:
        self._name = name
        self._mjcf_root = mjcf.RootElement(model=name)
        set_mujoco_globals(self.mjcf_root, mujoco_globals_path)

        self.skeleton: Skeleton | None = None

        self.bodyseg_to_mjcfmesh = {}
        self.bodyseg_to_mjcfbody = {}
        self.bodyseg_to_mjcfgeom = {}
        self.jointdof_to_mjcfjoint = {}
        self.jointdof_to_mjcfactuator_by_type = {ty: {} for ty in ActuatorType}
        self.leg_to_adhesionactuator = {}
        self.sensorname_to_mjcfsensor = {}
        self.cameraname_to_mjcfcamera = {}

        self.jointdof_to_neutralangle = {}
        self.jointdof_to_neutralaction_by_type = {ty: {} for ty in ActuatorType}

        if isinstance(root_segment, str):
            root_segment = self.BODY_SEGMENT_CLASS(root_segment)
        self.root_segment = root_segment

        self._neutral_keyframe = self.mjcf_root.keyframe.add(
            "key", name="neutral", time=0
        )

        self._add_mesh_assets(mesh_basedir, mirror_left2right, mesh_type)
        self._add_bodies_and_geoms(rigging_config_path, geom_fitting_option)

    @override
    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        """Name of this fly instance."""
        return self._name

    def get_bodysegs_order(self) -> list[BodySegment]:
        """Get the canonical order of body segments. The exact order is not important,
        but it should be respected consistently throughout. For example, during
        simulation, the fly body state returned by the simulator will be in this order.
        """
        return list(self.bodyseg_to_mjcfbody.keys())

    def get_jointdofs_order(self) -> list[JointDOF]:
        """Same as `get_bodysegs_order()`, but for joint DoFs instead of body segments."""
        return list(self.jointdof_to_mjcfjoint.keys())

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> list[JointDOF]:
        """Same as `get_jointdofs_order()`, but only for the subset of joint DoFs that
        are actuated by the specified actuator type. During simulation, the user should
        provide control input in this order."""
        actuator_type = ActuatorType(actuator_type)
        return list(self.jointdof_to_mjcfactuator_by_type[actuator_type].keys())

    def get_legs_order(self) -> list[str]:
        """Get the ordered list of leg position identifiers (same as `anatomy.LEGS`)."""
        return LEGS
    
    def get_pose_lookup(self, neutral_pose: KinematicPose | KinematicPosePreset | dict[str, float] | None) -> dict[str, float]:
        """Get a lookup dictionary mapping joint DOF names to neutral angles for a given
        neutral pose."""

        if self.skeleton is None:
            raise FlyGymInternalError("Skeleton must be defined to get pose lookup.")

        if neutral_pose is None:
            return {}
        elif isinstance(neutral_pose, dict):
            return neutral_pose
        elif isinstance(neutral_pose, KinematicPose):
            return neutral_pose.joint_angles_lookup_rad
        elif isinstance(neutral_pose, KinematicPosePreset):
            neutral_pose = neutral_pose.get_pose_by_axis_order(self.skeleton.axis_order)
            return neutral_pose.joint_angles_lookup_rad
        else:
            raise ValueError(
                "When specified, `neutral_pose` must be a "
                "`KinematicPose` or `KinematicPosePreset`."
            )

    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: KinematicPose | KinematicPosePreset | None = None,
        *,
        stiffness: float = 10.0,
        damping: float = 0.5,
        armature: float = 1e-6,
        **kwargs: Any,
    ) -> dict[JointDOF, mjcf.Element]:
        """Add joints to the fly model based on a skeleton definition.

        Creates hinge joints connecting body segments according to the skeleton's
        kinematic tree structure. Each joint is configured with passive spring-damper
        dynamics and a neutral (resting) angle.

        Args:
            skeleton:
                Skeleton defining which joints to create and their DOFs.
            neutral_pose:
                Resting angles for joints. If provided, must match skeleton's axis
                order. If not provided, all neutral angles default to 0.
            stiffness:
                Joint stiffness (spring constant).
            damping:
                Joint damping coefficient.
            armature:
                Additional inertia added to the joint for numerical stability. Should be
                small enough to not affect dynamics.
            **kwargs:
                Additional arguments passed to MJCF joint creation. See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint>`_
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF joint elements.
        """

        self.skeleton = skeleton
        neutral_angle_lookup = self.get_pose_lookup(neutral_pose)


        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            neutral_angle = neutral_angle_lookup.get(jointdof.name, 0.0)
            self.jointdof_to_neutralangle[jointdof] = neutral_angle

            # Flip axis direction for right side's roll and yaw so that axes are defined
            # symmetrically (e.g., positive roll is always "outward").
            vec = np.array(jointdof.axis.to_vector())
            if jointdof.child.pos[0] == "r" and not self._is_pitch(jointdof):
                vec = -vec

            return_dict[jointdof] = child_body.add(
                "joint",
                name=jointdof.name,
                type="hinge",
                axis=vec,
                stiffness=stiffness,
                damping=damping,
                armature=armature,
                springref=neutral_angle,
                **kwargs,
            )

        self.jointdof_to_mjcfjoint.update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def add_actuators(
        self,
        jointdofs: Iterable[JointDOF],
        actuator_type: "ActuatorType | str",
        neutral_input: "dict[str, float] | KinematicPose | KinematicPosePreset | None" = None,
        *,
        forcelimited: bool = True,
        forcerange: tuple[float, float] = (-30.0, 30.0),
        **kwargs: Any,
    ) -> dict[JointDOF, mjcf.Element]:
        """Add actuators to specified joints.

        Creates actuators that can apply forces/torques to joints. Multiple actuator
        types can be added to the same joints.

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
                If True, actuators cannot exceed forcerange.
            forcerange:
                Force limit as a (min, max) tuple.
            **kwargs:
                Additional arguments passed to MJCF actuator creation (e.g., kp for
                position actuators, kv for velocity actuators). See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator>`_
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF actuator elements.
        """
        actuator_type = ActuatorType(actuator_type)

        if actuator_type == ActuatorType.POSITION:
            neutral_input = self.get_pose_lookup(neutral_input)
        else:
            if isinstance(neutral_input, (KinematicPose, KinematicPosePreset)):
                raise ValueError(
                    "When actuator_type is not POSITION, neutral_input cannot be a "
                    "KinematicPose or KinematicPosePreset since those specify joint "
                    "angles, not actuator inputs."
                )
            else:
                neutral_input = {} if neutral_input is None else neutral_input

        return_dict = {}
        for jointdof in jointdofs:
            self.jointdof_to_neutralaction_by_type[actuator_type][jointdof] = (
                neutral_input.get(jointdof.name, 0.0)
            )
            actuator = self.mjcf_root.actuator.add(
                actuator_type.value,
                name=f"{jointdof.name}-{actuator_type.value}",
                joint=jointdof.name,
                forcelimited=forcelimited,
                forcerange=forcerange,
                **kwargs,
            )

            return_dict[jointdof] = actuator
        self.jointdof_to_mjcfactuator_by_type[actuator_type].update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def add_leg_adhesion(
        self, gain: float | dict[str, float] = 1.0
    ) -> dict[str, mjcf.Element]:
        """Add adhesion actuators to the tarsus5 segments of all legs.

        Adhesion actuators apply a normal attraction force, enabling the fly to grip
        surfaces. The control input per leg ranges from 1 to 100.

        Args:
            gain: Adhesion actuator gain. Either a single float applied to all legs,
                or a dict mapping leg position identifiers to per-leg gain values.

        Returns:
            Dict mapping leg position identifier to the created MJCF adhesion
            actuator element (same as ``self.leg_to_adhesionactuator``).

        Raises:
            ValueError: If adhesion actuators have already been added.
        """
        if len(self.leg_to_adhesionactuator) > 0:
            raise ValueError("Leg adhesion actuators have already been added.")
        for leg in LEGS:
            tarsus5 = BodySegment(f"{leg}_tarsus5")
            if isinstance(gain, dict):
                gain_this_leg = gain[leg]
            else:
                gain_this_leg = gain
            self.leg_to_adhesionactuator[leg] = self.mjcf_root.actuator.add(
                "adhesion",
                name=f"{tarsus5.name}-adhesion",
                body=self.bodyseg_to_mjcfbody[tarsus5],
                gain=gain_this_leg,
                ctrlrange=(1, 100),
            )
        return self.leg_to_adhesionactuator

    def colorize(
        self, visuals_config_path: PathLike = DEFAULT_VISUALS_CONFIG_PATH
    ) -> None:
        """Apply colors and textures to the fly model.

        Args:
            visuals_config_path: Path to the YAML file defining per-segment material
                and texture assignments.
        """
        if len(self.bodyseg_to_mjcfgeom) == 0:
            raise ValueError("Must first add geoms via `_add_bodies_and_geoms`.")

        vis_sets_all, lookup = self._parse_visuals_config(visuals_config_path)

        for vis_set_name, params in vis_sets_all.items():
            material = self.mjcf_root.asset.add(
                "material", name=vis_set_name, **params["material"]
            )
            if texture_params := params.get("texture"):
                texture = self.mjcf_root.asset.add(
                    "texture", name=vis_set_name, **texture_params
                )
                material.texture = texture

        for segment, geoms in self.bodyseg_to_mjcfgeom.items():
            for geom in geoms:
                vis_set_name = lookup[segment]
                geom.set_attributes(material=vis_set_name)

    def add_tracking_camera(
        self,
        name: str = "trackcam",
        mode: str = "track",
        pos_offset: Vec3 = (0, -7.5, 6),
        rotation: Rotation3D = Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy: float = 30.0,
        **kwargs: Any,
    ) -> mjcf.Element:
        """Add a camera that tracks the fly's root body.

        Args:
            name: Camera name.
            mode: MuJoCo camera tracking mode (e.g. ``"track"``, ``"targetbody"``).
            pos_offset: Camera position offset from the tracked body in mm.
            rotation: Camera orientation as a `Rotation3D`.
            fovy: Vertical field of view in degrees.
            **kwargs: Additional attributes passed to the MJCF camera element. See
                `MuJoCo XML reference
                <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera>`_.

        Returns:
            The created MJCF camera element.
        """
        camera = self.mjcf_root.worldbody.add(
            "camera",
            name=name,
            mode=mode,
            target=self.root_segment.name,
            pos=pos_offset,
            fovy=fovy,
            **rotation.as_kwargs(),
            **kwargs,
        )
        self.cameraname_to_mjcfcamera[name] = camera
        return camera

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
    ) -> None:
    
        # Decide which folder to load mesh files from
        mesh_dir = mesh_basedir / mesh_type.value
        mesh_fallback_dir = mesh_basedir / MeshType.FULLSIZE.value
        for d in [mesh_dir, mesh_fallback_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Mesh directory not found: {d}")

        for segment_name in ALL_SEGMENT_NAMES:
            if mirror_left2right and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1

            mesh_path = (mesh_dir / f"{mesh_to_use}.stl").resolve()
            if not mesh_path.exists():
                mesh_path = (mesh_fallback_dir / f"{mesh_to_use}.stl").resolve()
                if not mesh_path.exists():
                    raise FileNotFoundError(
                        f"Mesh file not found for segment {segment_name}: "
                        f"tried {mesh_dir} and {mesh_fallback_dir}."
                    )

            self.bodyseg_to_mjcfmesh[segment_name] = self.mjcf_root.asset.add(
                "mesh",
                name=segment_name,
                file=str(mesh_path),
                scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
            )
    
    def _all_possible_joint_preset(self):
        return JointPreset.ALL_POSSIBLE

    def _get_base_skeleton(self) -> Skeleton:
        return self.BASE_SKELETON_CLASS(
            joint_preset=self._all_possible_joint_preset(),
            axis_order=self.AXIS_ORDER_CLASS.DONTCARE,
        )
    
    def _is_pitch(self, jointdof: JointDOF) -> bool:
        return jointdof.axis == RotationAxis.PITCH

    def _add_bodies_and_geoms(
        self, rigging_config_path: PathLike, geom_fitting_option: GeomFittingOption
    ) -> None:
        # Load rigging config
        with open(rigging_config_path) as f:
            rigging_config = yaml.safe_load(f)

        # Add root body and geom
        body, geoms = self._add_one_body_and_geom(
            self.mjcf_root.worldbody,
            self.root_segment,
            rigging_config[self.root_segment.name],
        )
        self.bodyseg_to_mjcfbody[self.root_segment] = body
        self.bodyseg_to_mjcfgeom[self.root_segment] = geoms

        # Add remaining bodies and geoms by traversing the kinematic tree defined by
        # the skeleton
        full_skeleton = self._get_base_skeleton()
        for jointdof in full_skeleton.iter_jointdofs(self.root_segment):
            if not self._is_pitch(jointdof):
                # Look at only 1 DoF per joint as we're still just adding bodies/geoms
                continue
            parent_body = self.bodyseg_to_mjcfbody.get(jointdof.parent)
            if parent_body is None:
                raise FlyGymInternalError("Parent not found during kinematic tree DFS")
            my_rigging_config = rigging_config.get(jointdof.child.name)
            if my_rigging_config is None:
                raise FlyGymInternalError(
                    f"Missing rigging config for body segment {jointdof.child.name}"
                )
            body, geoms = self._add_one_body_and_geom(
                parent_body, jointdof.child, my_rigging_config
            )
            self.bodyseg_to_mjcfbody[jointdof.child] = body
            self.bodyseg_to_mjcfgeom[jointdof.child] = geoms

        # Optionally fit certain geoms to capsule shapes for simpler physics
        for bodyseg, mjcf_elements in self.bodyseg_to_mjcfgeom.items():
            for mjcf_element in mjcf_elements:
                if (geom_fitting_option == GeomFittingOption.ALL_TO_CAPSULES) or (
                    bodyseg.is_leg() and bodyseg.link == "tarsus5"
                ):
                    mjcf_element.type = "capsule"

    def _add_one_body_and_geom(
        self,
        parent_body: mjcf.Element,
        segment: BodySegment,
        my_rigging_config: dict[str, Any],
    ) -> tuple[mjcf.Element, mjcf.Element]:
        body_element = parent_body.add(
            "body",
            name=segment.name,
            pos=my_rigging_config["pos"],
            quat=my_rigging_config["quat"],
        )
        geom_element = body_element.add(
            "geom",
            name=segment.name,
            type="mesh",
            mesh=segment.name,
            mass=my_rigging_config["mass"],
            contype=0,  # contact pairs to be added explicitly later
            conaffinity=0,  # contact pairs to be added explicitly later
        )
        return body_element, [geom_element]

    @staticmethod
    def _parse_visuals_config(
        visuals_config_path: PathLike,
    ) -> tuple[dict[str, dict], dict[BodySegment, dict]]:
        # Load visuals config and assign vis sets to body segments
        with open(visuals_config_path) as f:
            vis_set_params_all = yaml.safe_load(f)
        all_matches_by_segname = {k: [] for k in ALL_SEGMENT_NAMES}
        for vis_set_name, vis_set_params in vis_set_params_all.items():
            apply_to = vis_set_params.get("apply_to")
            material = vis_set_params.get("material")
            if not apply_to or not material:
                raise ValueError(
                    f"Invalid visualization set: {vis_set_name}."
                    "Must specify a non-empty 'apply_to' and 'material'."
                )
            allowed_keys = {"apply_to", "material", "texture"}
            if invalid_keys := (set(vis_set_params.keys()) - allowed_keys):
                raise ValueError(
                    f"Invalid keys in visualization set {vis_set_name}: "
                    f"{invalid_keys}. Must be one of {allowed_keys}."
                )
            target_segnames = set()
            for pattern in [apply_to] if isinstance(apply_to, str) else apply_to:
                target_segnames |= set(filter_with_wildcard(ALL_SEGMENT_NAMES, pattern))
            for segname in target_segnames:
                all_matches_by_segname[segname].append(vis_set_name)
        for segname, vis_set_names in all_matches_by_segname.items():
            if len(vis_set_names) != 1:
                raise ValueError(
                    f"Zero or multiple vis sets matched for body segment {segname}: "
                    f"{vis_set_names}. Only one should apply."
                )
        lookup_by_segname = {
            BodySegment(segname): matches[0]
            for segname, matches in all_matches_by_segname.items()
        }
        return vis_set_params_all, lookup_by_segname

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        self._neutral_keyframe.qpos = self._get_neutral_qpos(mj_model)
        self._neutral_keyframe.ctrl = self._get_neutral_ctrl(mj_model)

    def _get_neutral_qpos(self, mj_model: mj.MjModel) -> np.ndarray:
        neutral_qpos = np.zeros(mj_model.nq)
        for jointdof, angle in self.jointdof_to_neutralangle.items():
            joint_element = self.jointdof_to_mjcfjoint[jointdof]
            internal_jointid = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
            )
            qposadr = mj_model.jnt_qposadr[internal_jointid]
            neutral_qpos[qposadr] = angle
        return neutral_qpos

    def _get_neutral_ctrl(self, mj_model: mj.MjModel) -> np.ndarray:
        neutral_ctrl = np.zeros(mj_model.nu)
        for ty, jointdof_to_actuator in self.jointdof_to_mjcfactuator_by_type.items():
            for jointdof, actuator in jointdof_to_actuator.items():
                internal_actuatorid = mj.mj_name2id(
                    mj_model, mj.mjtObj.mjOBJ_ACTUATOR, actuator.full_identifier
                )
                neutral_input = self.jointdof_to_neutralaction_by_type[ty][jointdof]
                neutral_ctrl[internal_actuatorid] = neutral_input
        return neutral_ctrl
    

FLYBODY_RIGGING_CONFIG_PATH = assets_dir / "model/flybody/flybody_rigging.yaml"
FLYBODY_MUJOCO_GLOBALS_PATH = assets_dir / "model/flybody/flybody_mujoco_globals.yaml"
FLYBODY_MESH_DIR = assets_dir / "model/flybody/meshes/"
FLYBODY_VISUALS_CONFIG_PATH = assets_dir / "model/flybody/flybody_visuals.yaml"
FLYBODY_ALL_GEOM_SUFFIXES_PATH = assets_dir / "model/flybody/flybody_all_geom_suffixes.yaml"
FLYBODY_JOINT_CONFIG_PATH = assets_dir / "model/flybody/flybody_joints.yaml"
FLYBODY_ACTUATOR_CONFIG_PATH = assets_dir / "model/flybody/flybody_actuators.yaml"


class FlybodyFly(Fly):
    """
    Specialized Fly class that uses the flybody XML structure from the turaga lab
    In particular this handles:
        - default classes
        - different naming
        - different joint definitions
        
    """
    SCALE = 0.1  # The flybody XML is already in mm units, so no scaling is needed
    # BUT in the original xml meshes are scaled by 0.1
    # Therefore we need to adjust all other length-realteed quantities by 10 to keep units consitant.
    # density/=1000, viscosity/=10, forcerange*=10, pos*=10, gravity*=10, gainprm*=10, biasprm*=10, etc.
    # see parsing script 
    BODY_SEGMENT_CLASS = FlybodyBodySegment
    JOINT_DOF_CLASS = FlybodyJointDOF
    AXIS_ORDER_CLASS = FlybodyAxisOrder
    BASE_SKELETON_CLASS = FlybodySkeleton
    LEG_LINKS = FLYBODY_LEG_LINKS

    def _all_possible_joint_preset(self):
        return FlybodyJointPreset.ALL_POSSIBLE


    def __init__(
        self,
        name: str = "nmf",
        *,
        rigging_config_path: PathLike = FLYBODY_RIGGING_CONFIG_PATH,
        mesh_basedir: PathLike = FLYBODY_MESH_DIR,
        mujoco_globals_path: PathLike = FLYBODY_MUJOCO_GLOBALS_PATH,
        all_geom_suffixes_path: PathLike = FLYBODY_ALL_GEOM_SUFFIXES_PATH,
        mirror_left2right: bool = False,
        root_segment: FlybodyBodySegment | str = "c_thorax",
        mesh_type: MeshType = MeshType.FULLSIZE,
        geom_fitting_option: GeomFittingOption = GeomFittingOption.UNMODIFIED,
        joint_config_path: PathLike = FLYBODY_JOINT_CONFIG_PATH,
        actuator_config_path: PathLike = FLYBODY_ACTUATOR_CONFIG_PATH,
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
            raise ValueError(f"Joint {joint_name} not found in grouped joint config.")

        return resolved_params

    def _is_pitch(self, jointdof):
        if jointdof.child.is_wing():
            return jointdof.axis == WingFlybodyRotationAxis.PITCH
        else:
            return jointdof.axis == FlybodyRotationAxis.PITCH

    @staticmethod
    def _coerce_mjcf_value(value: Any) -> Any:
        """Convert YAML-loaded values to MJCF-friendly python types."""
        if isinstance(value, list):
            return tuple(FlybodyFly._coerce_mjcf_value(v) for v in value)
        if isinstance(value, tuple):
            return tuple(FlybodyFly._coerce_mjcf_value(v) for v in value)
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

    def _add_one_body_and_geom(
        self,
        parent_body: mjcf.Element,
        segment: BodySegment,
        my_rigging_config: dict[str, Any],
        ) -> tuple[mjcf.Element, mjcf.Element]:

        body_element = parent_body.add(
            "body",
            name=segment.name,
            pos=my_rigging_config["pos"],
            quat=my_rigging_config["quat"],
        )

        all_geom_elements = []
        for fbodygeom_name, geom_config in my_rigging_config["geoms"].items():
            # subtract flybody_name to fbodygeom_name to get suffix
            geom_name = f"{segment.pos}_{fbodygeom_name}"

            geom_element = body_element.add(
                "geom",
                type="mesh",
                name=geom_name,
                contype=0,  # contact pairs to be added explicitly later
                conaffinity=0,  # contact pairs to be added explicitly later
                **geom_config,
            )
            all_geom_elements.append(geom_element)

        return body_element, all_geom_elements

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
        ) -> None:    

        # Decide which folder to load mesh files from
        mesh_dir = mesh_basedir / mesh_type.value
        mesh_fallback_dir = mesh_basedir / MeshType.FULLSIZE.value
        for d in [mesh_dir, mesh_fallback_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Mesh directory not found: {d}")

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
                    print(f"Warning: no mesh suffix found for segment {mesh_to_use}, using segment name as mesh name")
                    mesh_name = mesh_to_use
                mesh_path = (mesh_dir / f"{mesh_name}.obj").resolve()
                if not mesh_path.exists():
                    mesh_path = (mesh_fallback_dir / f"{mesh_name}.obj").resolve()
                    if not mesh_path.exists():
                        raise FileNotFoundError(
                            f"Mesh file not found for segment {segment_name}: "
                            f"tried {mesh_dir} and {mesh_fallback_dir}."
                        )

                mesh = self.mjcf_root.asset.add(
                    "mesh",
                    name=mesh_name,
                    file=str(mesh_path),
                    scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
                )
                if segment_name not in self.bodyseg_to_mjcfmesh:
                    self.bodyseg_to_mjcfmesh[segment_name] = [mesh]
                else:
                    self.bodyseg_to_mjcfmesh[segment_name].append(mesh)
        
        # add abdomen8 mesh (just a mesh connected to c_abdomen7)
        self.bodyseg_to_mjcfmesh["c_abdomen7"].append(self.mjcf_root.asset.add(
            "mesh",
            name="c_abdomen8_body",
            file=str(mesh_dir / "c_abdomen8_body.obj"),
            scale=(self.SCALE, self.SCALE, self.SCALE))
        )
    
    def colorize(
        self, visuals_config_path: PathLike = FLYBODY_VISUALS_CONFIG_PATH
    ) -> None:
        """Apply colors and textures to the fly model.

        Args:
            visuals_config_path: Path to the YAML file defining per-segment material
                and texture assignments.
        """
        if len(self.bodyseg_to_mjcfgeom) == 0:
            raise ValueError("Must first add geoms via `_add_bodies_and_geoms`.")

        with open(visuals_config_path) as f:
            vis_config = yaml.safe_load(f)

        for vis_set_name, params in vis_config.items():
            material = self.mjcf_root.asset.add(
                "material", name=vis_set_name, **params["material"]
            )
            if texture_params := params.get("texture"):
                texture = self.mjcf_root.asset.add(
                    "texture", name=vis_set_name, **texture_params
                )
                material.texture = texture

        for _, geoms in self.bodyseg_to_mjcfgeom.items():
            for geom in geoms:
                found_match = False
                for mat_name in vis_config.keys():
                    if geom.name.endswith(mat_name):
                        geom.set_attributes(material=mat_name)
                        found_match = True
                        break
                if not found_match:
                    if "claw" in geom.name:
                        geom.set_attributes(material="brown")
                    else:
                        geom.set_attributes(material="body")

        
    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: KinematicPose | KinematicPosePreset | None = None,
        **kwargs: Any,
    ) -> dict[JointDOF, mjcf.Element]:
        """Add joints to the fly model based on a skeleton definition.

        Creates hinge joints connecting body segments according to the skeleton's
        kinematic tree structure. Each joint is configured with passive spring-damper
        dynamics and a neutral (resting) angle.

        Args:
            skeleton:
                Skeleton defining which joints to create and their DOFs.
            neutral_pose:
                Resting angles for joints. If provided, must match skeleton's axis
                order. If not provided, all neutral angles default to 0.
                Will be used to set the springref attribute of the joints, which defines the angle at
                which the passive spring forces are zero.
            **kwargs:
                Additional arguments passed to MJCF joint creation. See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint>`_
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF joint elements.
        """

        self.skeleton = skeleton
        neutral_pose_lookup = self.get_pose_lookup(neutral_pose)

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            joint_params = self._resolve_joint_params(jointdof.name)

            joint_params.update({
                "springref": neutral_pose_lookup.get(jointdof.name, 0.0)
            })# override default springref with neutral pose value if provided
            joint_params.update(kwargs) # override any joint config values with values provided in kwargs
            joint_params = self._normalize_mjcf_params(joint_params)
            
            return_dict[jointdof] = child_body.add(
                "joint",
                name=jointdof.name,
                type="hinge",
                axis = jointdof.axis.to_vector(),
                **joint_params,
            )

        self.jointdof_to_mjcfjoint.update(return_dict)        
        self._rebuild_neutral_keyframe()        
        return return_dict
    
    def translate_generalactparams_to_specificactparams(self, general_params: dict[str, Any],
                                                         actuator_type: ActuatorType) -> dict[str, Any]:
        def _parse_param_values(raw_value: Any, name: str) -> list[float]:
            if isinstance(raw_value, str):
                tokens = raw_value.split()
            elif isinstance(raw_value, (list, tuple, np.ndarray)):
                tokens = list(raw_value)
            else:
                tokens = [raw_value]

            try:
                return [float(x) for x in tokens]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {name} value {raw_value!r}. Expected a whitespace-delimited string or a numeric list."
                ) from exc

        specific_params = general_params.copy()
        if actuator_type == ActuatorType.POSITION:
            # set kp and kv from gainprm, biasprm and dynprm as specified here: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position
            if "gainprm" in general_params:
                gainprm = general_params["gainprm"]
                gainprm_parsed = _parse_param_values(gainprm, "gainprm")
                kp = gainprm_parsed[0]
                if len(gainprm_parsed) > 1 and np.sum(gainprm_parsed[1:]) != 0:
                    print("WARNING: gainprm has more than one value and non-zero values after the first one, but only the first value is used as kp for position actuators according to MuJoCo docs.")
            else:
                kp = 1.0
            if "biasprm" in general_params:
                biasprm = general_params["biasprm"]
                biasprm_parsed = _parse_param_values(biasprm, "biasprm")
                assert len(biasprm_parsed) >= 2 and biasprm_parsed[0] == 0.0 and biasprm_parsed[1] == -1*kp, "Conflicting kp: according to MuJoCo docs biasprm is [0, -kp, -kv]"
                if len(biasprm_parsed) > 2:
                    kv = -biasprm_parsed[2]
                else:
                    kv = 0.0 # default for position according to mujoco
                if len(biasprm_parsed) > 3 and np.sum(biasprm_parsed[3:]) != 0:
                    print("WARNING: biasprm has more than three values and non-zero values after the third one, but only the first three values are used as biasprm for position actuators according to MuJoCo docs.")
            else:
                kv = 0.0
            if "dynprm" in general_params:
                dynprm = general_params["dynprm"]
                dynprm_parsed = _parse_param_values(dynprm, "dynprm")
                timeconst = dynprm_parsed[0]
                if len(dynprm_parsed) > 1 and np.sum(dynprm_parsed[1:]) != 0:
                    print("WARNING: dynprm has more than one value and non-zero values after the first one, but only the first value is used as timeconst for position actuators according to MuJoCo docs.")
            else:
                timeconst = 1.0 # default for general according to mujoco
            specific_params = {
                "kp": kp,
                "kv": kv,
                "timeconst": timeconst,
            }
        elif actuator_type == ActuatorType.VELOCITY:
            # Setting parameters values according to https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-velocity
            if "gainprm" in general_params:
                gainprm = general_params["gainprm"]
                gainprm_parsed = _parse_param_values(gainprm, "gainprm")
                kv = gainprm_parsed[0]
                if len(gainprm_parsed) > 1 and np.sum(gainprm_parsed[1:]) != 0:
                    print("WARNING: gainprm has more than one value and non-zero values after the first one, but only the first value is used as kv for velocity actuators according to MuJoCo docs.")
            else:
                kv = 1.0
            if "biasprm" in general_params:
                biasprm = general_params["biasprm"]
                biasprm_parsed = _parse_param_values(biasprm, "biasprm")
                assert len(biasprm_parsed) >= 3 and biasprm_parsed[0] == 0.0 and biasprm_parsed[1] == 0.0 and biasprm_parsed[2] == -1*kv, "Conflicting kvs: according to MuJoCo docs biasprm is [0, 0, -kv]"
                if len(biasprm_parsed) > 3 and np.sum(biasprm_parsed[3:]) != 0:
                    print("WARNING: biasprm has more than four values and non-zero values after the fourth one, but only the first four values are used as biasprm for velocity actuators according to MuJoCo docs.")
            if "dynprm" in general_params:
                print("WARNING: dynprm is not used for velocity actuators according to MuJoCo docs, but dynprm is specified in the general actuator config. Ignoring dynprm values.")
            specific_params = {
                "kv": kv,
            }            
        elif actuator_type == ActuatorType.MOTOR:
            # Setting parameters values according to https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
            if "gainprm" in general_params:
                print("WARNING: ignoring default gainprm as it is not used in classical motor actuators according to MuJoCo docs, but gainprm is specified in the general actuator config.")
            if "biasprm" in general_params:
                print("WARNING: ignoring default biasprm as it is not used in classical motor actuators according to MuJoCo docs, but biasprm is specified in the general actuator config.") 
            if "dynprm" in general_params:
                print("WARNING: ignoring default dynprm as it is not used in classical motor actuators according to MuJoCo docs, but dynprm is specified in the general actuator config.")
            specific_params = {}
        else:
            raise ValueError(f"Unsupported actuator type: {actuator_type}")
        
        for params in ["gainprm", "biasprm", "dynprm"]:
            if params in specific_params:
                # remove them from specific params
                specific_params.pop(params)

        return specific_params
    
    def translate_generaljointparams_to_specificjointparams_simplified(self, general_params: dict[str, Any], actuator_type: ActuatorType) -> dict[str, Any]:
        if actuator_type == ActuatorType.POSITION:
            return {
                "kp": general_params.get("gainprm", 1.0),
            }
        else:
            return {}

    def add_actuators(
        self,
        jointdofs: Iterable[JointDOF],
        actuator_type: "ActuatorType | str",
        *,
        forcelimited: bool = False,
        forcerange: tuple[float, float] = (-0.3, 0.3),
        **kwargs: Any,
    ) -> dict[JointDOF, mjcf.Element]:
        """Add actuators to specified joints.

        Creates actuators that can apply forces/torques to joints. Multiple actuator
        types can be added to the same joints.

        Args:
            jointdofs:
                Joint DOFs to actuate.
            actuator_type:
                Type of actuator (motor, position, velocity, etc.).
            forcelimited:
                If True, actuators cannot exceed set forcerange otherwise uses
                default forcerange if specified in actuator_config.yaml.
                default is False.
            forcerange:
                Force limit as a (min, max) tuple.
            **kwargs:
                Additional arguments passed to MJCF actuator creation (e.g., kp for
                position actuators, kv for velocity actuators). See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator>`_
                for details on supported attributes.
                Overrides any default values specified in actuator_config.yaml.

        Returns:
            Dictionary mapping JointDOF to created MJCF actuator elements.
        """
        actuator_type = ActuatorType(actuator_type)


        remove_ctrl_limits = False
        if (actuator_type == ActuatorType.MOTOR or actuator_type == ActuatorType.VELOCITY):
            print("WARNING: setting ctrllimit to false for MOTOR and VELOCITY actuators, as flybody as limits are ment for POSITION actuators.")
            remove_ctrl_limits = True
            if not forcelimited:
                print("WARNING: without ctrl limits, MOTOR and VELOCITY actuators can generate extreme forces for stability you might want to use force limits (e.g set forcelimited to TRUE).")

        return_dict = {}
        for jointdof in jointdofs:
            self.jointdof_to_neutralaction_by_type[actuator_type][jointdof] = 0.0

            default_actuator_params = {}
            for _, val in self.actuator_config.items():
                if jointdof.name in val["apply_to"]:
                    default_actuator_params = val["general"]
                    break
            if not default_actuator_params:
                print(f"WARNING: no actuator config found for joint {jointdof.name}")
            
            default_actuator_params_specific = self.translate_generaljointparams_to_specificjointparams_simplified(default_actuator_params, actuator_type)
    
            if remove_ctrl_limits:
                default_actuator_params_specific["ctrllimited"] = False
            else:
                # recover ctrllimits from joint limits
                jnt = self.jointdof_to_mjcfjoint[jointdof]
                assert jnt.range is not None, f"Joint {jointdof.name} must have range specified in order to use default ctrlrange for its actuator."
                default_actuator_params_specific["ctrlrange"] = jnt.range
            
            if forcelimited:
                default_actuator_params_specific["forcelimited"] = forcelimited
                default_actuator_params_specific["forcerange"] = forcerange
        
            if actuator_type == ActuatorType.POSITION:
                has_kp = "kp" in kwargs
                has_kv = "kv" in kwargs
                has_timeconst = "timeconst" in kwargs
                warning_str = f"WARNING: actuator type is POSITION but "
                has_missing_param = False
                for param, has_it in [("kp", has_kp), ("kv", has_kv), ("timeconst", has_timeconst)]:
                    if not has_it:
                        warning_str += f"{param} not specified, using default value from general actuator config if specified there, otherwise using MuJoCo default. "
                        has_missing_param = True
                if has_missing_param:
                    print(warning_str)
                
            elif actuator_type == ActuatorType.VELOCITY:
                has_kv = "kv" in kwargs
                warning_str = f"WARNING: actuator type is VELOCITY but "
                if not has_kv:
                    warning_str += "kv not specified, using default value from general actuator config if specified there, otherwise using MuJoCo default. "
                    print(warning_str)
                
            default_actuator_params_specific.update(kwargs)

            actuator = self.mjcf_root.actuator.add(
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
        self, gain: float | dict[str, float] = 0.985, add_labrum: bool = True,
        labrum_gain: float = 1.0
        ) -> dict[str, mjcf.Element]:
        """Add adhesion actuators to the tarsus5 segments of all legs.

        Adhesion actuators apply a normal attraction force, enabling the fly to grip
        surfaces. The control input per leg ranges from 1 to 100.

        Args:
            gain: Adhesion actuator gain. Either a single float applied to all legs,
                or a dict mapping leg position identifiers to per-leg gain values.
            add_labrum: Whether to also add an adhesion actuator for the labrum (mouthpart).

        Returns:
            Dict mapping leg position identifier to the created MJCF adhesion
            actuator element (same as ``self.leg_to_adhesionactuator``).

        Raises:
            ValueError: If adhesion actuators have already been added.
        """
        if len(self.leg_to_adhesionactuator) > 0:
            raise ValueError("Leg adhesion actuators have already been added.")
        for leg in LEGS:
            claw = FlybodyBodySegment(f"{leg}_claw")
            if isinstance(gain, dict):
                gain_this_leg = gain[leg]
            else:
                gain_this_leg = gain
            self.leg_to_adhesionactuator[leg] = self.mjcf_root.actuator.add(
                "adhesion",
                name=f"{claw.name}-adhesion",
                body=self.bodyseg_to_mjcfbody[claw],
                gain=gain_this_leg,
                ctrlrange=(0, 1),
            )
        if add_labrum:
            for s in "lr":
                labrum = FlybodyBodySegment(f"{s}_labrum")
                self.leg_to_adhesionactuator[f"{s}_labrum"] = self.mjcf_root.actuator.add(
                    "adhesion",
                    name=f"{labrum.name}-adhesion",
                    body=self.bodyseg_to_mjcfbody[labrum],
                    gain=labrum_gain,
                    ctrlrange=(0, 1),
                )
        return self.leg_to_adhesionactuator
    
    def add_tendons(self, coef: float| dict[str, float]| None = None) -> dict[JointDOF, mjcf.Element]:
        # check joints have been added
        if len(self.jointdof_to_mjcfjoint) == 0:
            raise ValueError("Must first add joints via `add_joints` before adding tendons.")
        if len(self.jointdof_to_mjcftendon) > 0:
            raise ValueError("Tendons have already been added.")

        def get_coef_for_joint(joint: JointDOF) -> float:
            if coef is None or (isinstance(coef, dict) and joint.name not in coef):
                return 1.0 # default in flybody
            elif isinstance(coef, dict):
                return coef[joint.name]
            else:
                return coef

        abd_bodyseg = FlybodyBodySegment("c_abdomen1")
        if abd_bodyseg in self.skeleton.body_segments:
            tree = self.skeleton.get_tree()
            for axis in [FlybodyRotationAxis.PITCH, FlybodyRotationAxis.YAW]:
                tendon_name = f"abdomen_{axis.name.lower()}"
                tendon = self.mjcf_root.tendon.add("fixed", name=tendon_name)
                added_tendon = False
                for parent, child in tree.dfs_edges(abd_bodyseg):
                        joint = FlybodyJointDOF(parent=parent, child=child, axis=axis)
                        tendon.add("joint",
                                    joint=self.jointdof_to_mjcfjoint[joint],
                                    coef=get_coef_for_joint(joint)
                                    )
                        if not added_tendon:
                            self.jointdof_to_mjcftendon[joint] = tendon
                            added_tendon = True
        else:
            print("Warning: abdomen1 not found in skeleton, skipping abdomen tendon creation.")
        
        for leg in LEGS:
            tarsus_bodyseg = FlybodyBodySegment(f"{leg}_tarsus1")
            if tarsus_bodyseg not in self.skeleton.body_segments:
                print(f"Warning: {tarsus_bodyseg} not found in skeleton, skipping tendon creation for {tarsus_bodyseg}.")
                continue
            else:
                joints = self.skeleton.iter_jointdofs(tarsus_bodyseg)
                first_joint = next(joints)
                tendon_name = f"{leg}_tarsus"
                tendon = self.mjcf_root.tendon.add("fixed",
                                                    name=tendon_name)
                self.jointdof_to_mjcftendon[first_joint] = tendon
                coef_to_use = get_coef_for_joint(first_joint)
                tendon.add("joint",
                            joint=self.jointdof_to_mjcfjoint[first_joint],
                            coef=coef_to_use
                            )
                for joint in joints:
                    coef_to_use = get_coef_for_joint(joint)
                    tendon.add("joint",
                                joint=self.jointdof_to_mjcfjoint[joint],
                                coef=coef_to_use
                                )
                    
    def add_tendon_actuators(self, **kwargs: Any) -> dict[JointDOF, mjcf.Element]:
        if len(self.jointdof_to_mjcftendon) == 0:
            raise ValueError("Must first add tendons via `add_tendons` before adding tendon actuators.")
        if len(self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON]) > 0:
            raise ValueError("Tendon actuators have already been added, cannot add tendon actuators as MOTOR actuators are used for tendons in this implementation.")
        
        for jointdof, tendon in self.jointdof_to_mjcftendon.items():
            if "abdomen" in jointdof.name:
                if jointdof.axis == FlybodyRotationAxis.PITCH:
                    default_params = {
                        "ctrlrange": [-1.05, 0.7]
                    }
                elif jointdof.axis == FlybodyRotationAxis.YAW:
                    default_params = {
                        "ctrlrange": [-0.7, 0.7]
                    }
            elif "tarsus" in jointdof.name:
                default_params = {
                    "ctrlrange": [-0.9, 0.9]
                }

            if jointdof.name in kwargs:
                print(f"WARNING: overriding default tendon actuator params for joint {jointdof.name} with params provided in kwargs.")
                default_params.update(kwargs[jointdof.name])
            else:
                default_params.update(kwargs)

            actuator = self.mjcf_root.actuator.add(
                "general",
                name=f"{tendon.name}-tendon",
                tendon=tendon,
                **default_params
            )
            
            self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON][jointdof] = actuator
            self.jointdof_to_neutralaction_by_type[ActuatorType.TENDON][jointdof] = 0.0 

        return self.jointdof_to_mjcfactuator_by_type[ActuatorType.TENDON]
    
    def _correct_wing_default_pose(self) -> float:
        """
            In flybody the wings are put in place by the spring property of the joint
            As they use general actuators an input of 0 means no forces leading to the wings
            going to their default pose. For us with position actuators, this does not happen.

            For that reason we position the wings bodies"
        """
        from scipy.spatial.transform import Rotation as R
        print("Applying wing default pose correction.")
        for side in ["l", "r"]:
            wing_bodyseg = FlybodyBodySegment(f"{side}_wing")
            if wing_bodyseg in self.bodyseg_to_mjcfbody:
                mjcf_body = self.bodyseg_to_mjcfbody[wing_bodyseg]
                bquat = R.from_quat(mjcf_body.quat, scalar_first=True)
                correction_quat = R.from_euler("xyz", [0, 0, -88 if side == "r" else 88], degrees=True)
                new_quat = (correction_quat * bquat).as_quat(scalar_first=True)
                mjcf_body.quat = tuple(new_quat)
            else:
                raise ValueError(f"Expected wing body segment {wing_bodyseg} not found in model, cannot apply wing default pose correction.")
