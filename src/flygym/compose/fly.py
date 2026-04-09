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
    AnatomicalJoint,
    JointDOF,
    Skeleton,
    RotationAxis,
    AxisOrder,
    JointPreset,
    ALL_SEGMENT_NAMES,
    LEGS,
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
        anatomicaljoint_to_mjcfsites:
            Maps anatomical joints to MJCF site elements.
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
        self.anatomicaljoint_to_mjcfsites = {}
        self.sensorname_to_mjcfsensor = {}
        self.cameraname_to_mjcfcamera = {}

        self.jointdof_to_neutralangle = {}
        self.jointdof_to_neutralaction_by_type = {ty: {} for ty in ActuatorType}

        if isinstance(root_segment, str):
            root_segment = BodySegment(root_segment)
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

    def get_sites_order(self) -> list[AnatomicalJoint]:
        """Get the canonical order of anatomical joints with associated MJCF sites.

        This is the order used by simulation site-state readout methods such as
        ``Simulation.get_site_positions``.
        """
        return list(self.anatomicaljoint_to_mjcfsites.keys())

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
        if neutral_pose is None:
            neutral_angle_lookup = {}
        elif isinstance(neutral_pose, KinematicPose):
            neutral_angle_lookup = neutral_pose.joint_angles_lookup_rad
        elif isinstance(neutral_pose, KinematicPosePreset):
            neutral_pose = neutral_pose.get_pose_by_axis_order(skeleton.axis_order)
            neutral_angle_lookup = neutral_pose.joint_angles_lookup_rad
        else:
            raise ValueError(
                "When specified, `neutral_pose` must be a "
                "`KinematicPose` or `KinematicPosePreset`."
            )

        self.skeleton = skeleton

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            neutral_angle = neutral_angle_lookup.get(jointdof.name, 0.0)
            self.jointdof_to_neutralangle[jointdof] = neutral_angle

            # Flip axis direction for right side's roll and yaw so that axes are defined
            # symmetrically (e.g., positive roll is always "outward").
            vec = np.array(jointdof.axis.to_vector())
            if jointdof.child.pos[0] == "r" and jointdof.axis != RotationAxis.PITCH:
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

        if neutral_input is None:
            neutral_input = {}

        if actuator_type == ActuatorType.POSITION:
            if isinstance(neutral_input, KinematicPose):
                neutral_input = neutral_input.joint_angles_lookup_rad
            elif isinstance(neutral_input, KinematicPosePreset):
                neutral_pose = neutral_input.get_pose_by_axis_order(
                    self.skeleton.axis_order
                )
                neutral_input = neutral_pose.joint_angles_lookup_rad

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

    def add_joint_sites(
        self, anatomical_joints: list[AnatomicalJoint]
    ) -> dict[AnatomicalJoint, mjcf.Element]:
        """Add MJCF sites at the origins of selected anatomical joints.

        Each site is placed at ``(0, 0, 0)`` in the child body frame. Since body
        origins are defined at their parent-child joint locations in this model,
        these sites track anatomical joint positions in world coordinates during
        simulation.

        Args:
            anatomical_joints: Anatomical joints to materialize as MJCF sites.

        Returns:
            Dictionary mapping each anatomical joint to its created MJCF site
            element (same entries added into ``self.anatomicaljoint_to_mjcfsites``).

        Raises:
            ValueError: If a site for a requested anatomical joint already exists.
        """
        return_dict = {}
        for joint in anatomical_joints:
            if joint in self.anatomicaljoint_to_mjcfsites:
                raise ValueError(
                    f"A site has already been added for anatomical joint '{joint.name}'."
                )
            child_body_element = self.bodyseg_to_mjcfbody[joint.child]
            site = child_body_element.add(
                "site",
                name=joint.name,
                pos=(0, 0, 0),  # origin of child body is defined at joint to parent
            )
            # child_body_element.add(
            #     "geom",
            #     name=f"{joint.name}_sitegeom",
            #     type="sphere",
            #     size=(0.05,),
            #     rgba=(1, 0, 0, 1),
            #     density=0,
            # )
            return_dict[joint] = site
        self.anatomicaljoint_to_mjcfsites.update(return_dict)
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

        for segment, geom in self.bodyseg_to_mjcfgeom.items():
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
        # For numerical reasons, we simulate length in mm, not m. This changes the units
        # of other quantities as well, for example acceleration is now in mm/s^2.
        SCALE = 1000

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
                scale=(SCALE, y_sign * SCALE, SCALE),
            )

    def _add_bodies_and_geoms(
        self, rigging_config_path: PathLike, geom_fitting_option: GeomFittingOption
    ) -> None:
        # Load rigging config
        with open(rigging_config_path) as f:
            rigging_config = yaml.safe_load(f)

        # Add root body and geom
        body, geom = self._add_one_body_and_geom(
            self.mjcf_root.worldbody,
            self.root_segment,
            rigging_config[self.root_segment.name],
        )
        self.bodyseg_to_mjcfbody[self.root_segment] = body
        self.bodyseg_to_mjcfgeom[self.root_segment] = geom

        # Add remaining bodies and geoms by traversing the kinematic tree defined by
        # the skeleton
        full_skeleton = Skeleton(
            joint_preset=JointPreset.ALL_POSSIBLE, axis_order=AxisOrder.DONTCARE
        )
        for jointdof in full_skeleton.iter_jointdofs(self.root_segment):
            if jointdof.axis != RotationAxis.PITCH:
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
            body, geom = self._add_one_body_and_geom(
                parent_body, jointdof.child, my_rigging_config
            )
            self.bodyseg_to_mjcfbody[jointdof.child] = body
            self.bodyseg_to_mjcfgeom[jointdof.child] = geom

        # Optionally fit certain geoms to capsule shapes for simpler physics
        for bodyseg, mjcf_element in self.bodyseg_to_mjcfgeom.items():
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
        return body_element, geom_element

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
